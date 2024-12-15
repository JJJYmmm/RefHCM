"""This file contains the model definition of TiTok.
    modified from bytedance/1d-tokenizer
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

from models.taming.modules.tiktok.blocks import TiTokEncoder, TiTokDecoder
from models.taming.modules.tiktok.quantizer import VectorQuantizer
from omegaconf import OmegaConf

import pytorch_lightning as pl
from models.taming.util import instantiate_from_config

class TiTok(pl.LightningModule):
    def __init__(self, 
                 config,                  
                 monitor=None,
                 image_key="image"):
        super().__init__()
        self.config = config
        self.image_size = config.dataset.preprocessing.crop_size 
        self.patch_size = config.model.vq_model.vit_enc_patch_size
        self.encoder = TiTokEncoder(config)
        self.decoder = TiTokDecoder(config)
        self.image_key = image_key
        self.num_labels = config.model.vq_model.out_channels
        
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        scale = self.encoder.width ** -0.5

        self.quantize = VectorQuantizer(
            codebook_size=config.model.vq_model.codebook_size,
            token_size=config.model.vq_model.token_size,
            commitment_cost=config.model.vq_model.commitment_cost,
            use_l2_norm=config.model.vq_model.use_l2_norm,)

        self.apply(self._init_weights)

        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width).requires_grad_())

        
        self.loss = instantiate_from_config(config.loss)


        if monitor is not None:
            self.monitor = monitor

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x):
        z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
        z_quantized, result_dict = self.quantize(z)
        return z_quantized, result_dict
    
    def decode(self, z_quantized):
        return self.decoder(z_quantized)

    def forward(self, input):
        quant, result = self.encode(input)
        dec = self.decode(quant)
        return dec, result['quantizer_loss']
    
    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()
    
    def decode_tokens(self, tokens):
        tokens = tokens.squeeze(1)
        batch, seq_len = tokens.shape # B x N
        z_quantized = self.quantize.get_codebook_entry(
            tokens.reshape(-1)).reshape(batch, 1, seq_len, -1)
        if self.quantize.use_l2_norm:
            z_quantized = torch.nn.functional.normalize(z_quantized, dim=-1)
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        decoded = self.decode(z_quantized)
        return decoded
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(  list(self.encoder.parameters())+
                                    list(self.decoder.parameters())+
                                    list(self.quantize.parameters())+
                                    [self.latent_tokens],
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        quant, _ = self.encode(x)
        xrec = self.decode(quant)
        # convert logits to indices
        xrec = torch.argmax(xrec, dim=1).cpu().numpy()
        x = torch.argmax(x, dim=1).cpu().numpy()
        return self.calculate_confusion_matrix(xrec, x)

    def validation_epoch_end(self, test_step_outputs):
        import numpy as np
        # eval for parsing map (miou)
        confusion_matrix = np.zeros((self.num_labels, self.num_labels))
        for out in test_step_outputs:
            confusion_matrix += out
        acc = np.full(self.num_labels, np.nan, dtype=np.float64)
        iou = np.full(self.num_labels, np.nan, dtype=np.float64)
        tp = confusion_matrix.diagonal().astype(np.float64)
        pos_gt = np.sum(confusion_matrix, axis=0).astype(np.float64)
        pos_pred = np.sum(confusion_matrix, axis=1).astype(np.float64)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        pacc = np.sum(tp) / np.sum(pos_gt)
        self.log("macc", macc)
        self.log("miou", miou)
        self.log("pacc", pacc)
    
    def calculate_confusion_matrix(self, pred, gt, ignore = -1):
        import numpy as np
        valid = gt != ignore
        pred = pred[valid]
        gt = gt[valid]
        index = (gt * self.num_labels + pred).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((self.num_labels, self.num_labels))

        for i_label in range(self.num_labels):
            for i_pred in range(self.num_labels):
                cur_index = i_label * self.num_labels + i_pred
                if cur_index < len(label_count):
                    confusion_matrix[i_label,
                                    i_pred] = label_count[cur_index]
        return confusion_matrix

    @torch.no_grad()
    def log_images(self, x, xrec):
        log = dict()
        x = x.to(self.device)
        xrec = xrec.to(self.device)
        # colorize with random projection
        x = F.one_hot(x.to(torch.long), num_classes=self.num_labels).permute(0,3,1,2).float()
        xrec = F.one_hot(xrec.to(torch.long), num_classes=self.num_labels).permute(0,3,1,2).float()
        x = self.to_rgb(x)
        xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log
    
    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x