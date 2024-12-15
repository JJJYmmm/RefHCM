# Copyright 2024 JJJYmmm
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import json
import logging
from typing import Optional
from argparse import Namespace

import os
import re
import numpy as np
import torch
from torch.nn import functional as F
import torchvision.utils as vutils
import importlib

from fairseq import metrics
from fairseq.tasks import register_task

from tasks.ofa_task import OFATask, OFAConfig
from data.mm_data.rpar_dataset_full_mask import RParDatasetFullMask, PARSING_BEGIN_TOKEN, PARSING_END_TOKEN, PARSING_PLACEHOLDER_IDS, PARSING_BEGIN_ID, PARSING_END_ID
from data.file_dataset import FileDataset

from models.taming.models.vqgan import VQSegmentationModel
from models.taming.models.titok import TiTok


logger = logging.getLogger(__name__)


@dataclass
class RParFullMaskConfig(OFAConfig):
    eval_acc: bool = field(
        default=False, metadata={"help": "evaluation with accuracy"}
    )
    eval_args: Optional[str] = field(
        default='{}',
        metadata={
            "help": 'generation args, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )

    max_image_size: int = field(
        default=512, metadata={"help": "max image size for normalization"}
    )
    scst: bool = field(
        default=False, metadata={"help": "Self-critical sequence training"}
    )
    scst_args: str = field(
        default='{}',
        metadata={
            "help": 'generation args for Self-critical sequence training, as JSON string'
        },
    )
    vq_config: str = field(
        default="",
        metadata={
            "help": 'path to vq config'
        },
    )
    vq_ckpt: str = field(
        default="",
        metadata={
            "help": 'path to vq checkpoint'
        }
    )
    titok_config: str = field(
        default="",
        metadata={
            "help": 'path to titok config'
        },
    )
    titok_ckpt: str = field(
        default="",
        metadata={
            "help": 'path to titok checkpoint'
        }
    )
    vq_n_embed: int = field(
        default=32,
        metadata={
            "help": 'num of vq embeddings'
        }
    )


@register_task("rpar_full_mask", dataclass=RParFullMaskConfig)
class RParFullMaskTask(OFATask):
    def __init__(self, cfg: RParFullMaskConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

        self.num_labels = 20 # 19 parts + 1 background

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        if split == 'train':
            file_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            file_path = paths[-1]
        dataset = FileDataset(file_path, self.cfg.selected_cols)

        self.datasets[split] = RParDatasetFullMask(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            patch_image_size=self.cfg.patch_image_size,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
            num_bins=self.cfg.num_bins,
            max_image_size=self.cfg.max_image_size
        )

    def load_vq_model(self, cfg):
        from omegaconf import OmegaConf
        device = torch.cuda.current_device()

        config = OmegaConf.load(cfg.vq_config)
        vqgan = VQSegmentationModel(**config.model.params)
        sd = torch.load(self.cfg.vq_ckpt, map_location="cpu")["state_dict"]
        missing, unexpected = vqgan.load_state_dict(sd, strict=False)
        for k, v in vqgan.named_parameters():
            v.requires_grad = False
        self.vq = vqgan
        self.vq.to(device)
        self.vq.eval()
    
    def load_titok_model(self, cfg):
        from omegaconf import OmegaConf
        device = torch.cuda.current_device()

        config = OmegaConf.load(cfg.titok_config)
        titok = TiTok(**config.model.params)
        sd = torch.load(self.cfg.titok_ckpt, map_location="cpu")["state_dict"]
        missing, unexpected = titok.load_state_dict(sd, strict=False)
        for k, v in titok.named_parameters():
            v.requires_grad = False
        self.vq = titok
        self.vq.to(device)
        self.vq.eval()

    def build_model(self, cfg):
        model = super().build_model(cfg)

        if self.cfg.eval_acc:
            gen_args = json.loads(self.cfg.eval_args)
        if self.cfg.scst:
            scst_args = json.loads(self.cfg.scst_args)
            self.scst_generator = self.build_generator(
                [model], Namespace(**scst_args)
            )

        if len(self.cfg.vq_config):
            self.load_vq_model(self.cfg)

        if len(self.cfg.titok_config):
            self.load_titok_model(self.cfg)

        return model
    
    def _get_parsing_map(self, codes):
        # code [BS, 48]
        # import pdb
        # pdb.set_trace()
        B = codes.shape[0]
        #  check
        codes = codes.clamp(0,self.cfg.vq_n_embed - 1).to(self.vq.device)
        codes = F.one_hot(codes, num_classes = self.vq.quantize.embedding.weight.size(0))
        quant  = codes.to(self.vq.dtype) @ self.vq.quantize.embedding.weight
        quant = quant.reshape(B, 8, 6, 256).permute(0,3,1,2) # hard code for 8*6 latent code
        output = self.vq.decode(quant)
        output = torch.argmax(output, dim=1)
        return output
    
    def _get_parsing_map_titok(self, codes):
        # code [BS, 32]
        # import pdb
        # pdb.set_trace()
        B = codes.shape[0]
        codes = codes.clamp(0,self.cfg.vq_n_embed - 1).to(self.vq.device)
        codes = F.one_hot(codes, num_classes = self.vq.quantize.embedding.weight.size(0))
        quant  = codes.to(self.vq.dtype) @ self.vq.quantize.embedding.weight
        quant = quant.reshape(B, 1, 32, 256).permute(0,3,1,2) # hard code for 8*6 latent code
        output = self.vq.decode(quant)
        output = torch.argmax(output, dim=1)
        return output

    def _calculate_ap_score(self, hyps, refs, thresh=0.5):
        interacts = torch.cat(
            [torch.where(hyps[:, :2] < refs[:, :2], refs[:, :2], hyps[:, :2]),
             torch.where(hyps[:, 2:] < refs[:, 2:], hyps[:, 2:], refs[:, 2:])],
            dim=1
        )
        area_predictions = (hyps[:, 2] - hyps[:, 0]) * (hyps[:, 3] - hyps[:, 1])
        area_targets = (refs[:, 2] - refs[:, 0]) * (refs[:, 3] - refs[:, 1])
        interacts_w = interacts[:, 2] - interacts[:, 0]
        interacts_h = interacts[:, 3] - interacts[:, 1]
        area_interacts = interacts_w * interacts_h
        ious = area_interacts / (area_predictions + area_targets - area_interacts + 1e-6)
        return ((ious >= thresh) & (interacts_w > 0) & (interacts_h > 0)).float()

    def _calculate_confusion_matrix(self, pred, gt, ignore = -1):
        '''
        hyps [B, H, W] value 0 ~ num_class - 1
        gt [B, H, W]    value 0 ~ num_class - 1
        '''
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


    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = criterion(model, sample)

        model.eval()
        if self.cfg.eval_acc:
            hyps, refs = self._gready_inference(sample, model)
            pred_coords, gt_coords = hyps[:,:4], refs[:,:4]

            pred_coords = pred_coords / (self.cfg.num_bins - 1) * self.cfg.max_image_size

            if len(self.cfg.vq_config):
                parsing_map = self._get_parsing_map(hyps[:,4:])
            else:
                parsing_map = self._get_parsing_map_titok(hyps[:,4:])
            

            # align bbox
            N = hyps.size(0)
            pred = torch.zeros(N, 512, 512).to(hyps.device)
            for i in range(N):
                x1, y1, x2, y2 = list(map(int, pred_coords[i].tolist()))
                pred[i, y1:y2, x1:x2] = F.interpolate(parsing_map[i].to(dtype=torch.float32).unsqueeze(0).unsqueeze(0), size=(y2 - y1, x2 - x1), mode='nearest')
            pred = pred.to(dtype=torch.int32)
            
            # it doesn't matter for a fixed size of parsing maps for both gt and prediction (512x512), if necessary, you can transform back to the origital image size
            logging_output['_confusion_matrix'] = self._calculate_confusion_matrix(pred.cpu().numpy(), sample['gt'].cpu().numpy())


            # calculate box ap
            pred_coords[:, ::2] /= sample['w_resize_ratios'].unsqueeze(1)
            pred_coords[:, 1::2] /= sample['h_resize_ratios'].unsqueeze(1)
            bbox_scores = self._calculate_ap_score(pred_coords, sample['region_coords'].float())
            logging_output["_box_score_sum"] = bbox_scores.sum().item()
            logging_output["_box_score_cnt"] = bbox_scores.size(0)

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        def compute_box_score(meters):
            score = meters["_box_score_sum"].sum / meters["_box_score_cnt"].sum
            score = score if isinstance(score, float) else score.item()
            return round(score, 4)
        
        def compute_miou(confusion_matrix):
            if torch.is_tensor(confusion_matrix):
                confusion_matrix = confusion_matrix.numpy()
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

            return miou, macc, pacc

        if sum_logs("_box_score_cnt") > 0:
            metrics.log_scalar("_box_score_sum", sum_logs("_box_score_sum"))
            metrics.log_scalar("_box_score_cnt", sum_logs("_box_score_cnt"))
            metrics.log_derived("box_score", compute_box_score)

            miou, pacc, macc = compute_miou(sum_logs("_confusion_matrix"))        
            metrics.log_scalar("miou", miou)
            metrics.log_scalar("pacc", pacc)
            metrics.log_scalar("macc", macc)
    

    def _inference(self, generator, sample, model):
        # assert not True
        gen_out = self.inference_step(generator, [model], sample)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(self._decode_parsing(gen_out[i][0]["tokens"][:-1]))
            refs.append(self._decode_parsing(sample["target"][i][:-1]))
        if self.cfg.eval_print_samples:
            logger.info("example hypothesis: ", hyps[0])
            logger.info("example reference: ", refs[0])

        return torch.stack(hyps, dim=0), torch.stack(refs, dim=0)
    
    def _gready_inference(self, sample, model):
        '''
        fake gready search for Query Parallel Generation (QPG), currently only support batch = 1
        '''

        bsz = sample["target"].size(0)

        gen_parsing_code_flag = False
        next_tokens = torch.tensor([self.tgt_dict.bos()] * bsz).unsqueeze(-1).to(sample["target"].device)

        while True:
            sample["net_input"]["prev_output_tokens"] = next_tokens
            sample["net_input"]["full_mask_position"] = None if next_tokens.size(1) <= 6 else (6, next_tokens.size(1))
            net_output = model(**sample["net_input"])

            if not gen_parsing_code_flag and next_tokens.size(1) <= 4:
                net_output[0][:, -1:, :58457] = -torch.inf # bin tokens range from 58457 to 59456
                net_output[0][:, -1:, 59457:] = -torch.inf
                last_item = F.softmax(net_output[0][:, -1:, :], dim = -1).argmax(dim = -1)
            elif not gen_parsing_code_flag: # must begin parsing
                net_output[0][:, -1:, :50305] = -torch.inf 
                net_output[0][:, -1:, 50306:] = -torch.inf
                last_item = F.softmax(net_output[0][:, -1:, :], dim = -1).argmax(dim = -1)
            else:
                net_output[0][:, -49:-1, :50265] = - torch.inf
                net_output[0][:, -49:-1, 58457:] = - torch.inf
                parsing_codes = F.softmax(net_output[0][:, -49:-1, :], dim = -1).argmax(dim = -1) # 49 = 48 placeholder + 1 end token
                next_tokens = torch.cat([next_tokens[:, :-50], parsing_codes], dim = 1) # 50 = 1 + 48 + 1, the final seqence excludes bos/eos of parsing codes for simplicity

                # jump the check of eos
                # last_item = F.softmax(net_output[0][:, -1:, :], dim = -1).argmax(dim = -1)
                # assert last_item.eq(self.tgt_dict.eos()).sum() == bsz
                
                break
            
            if last_item.eq(PARSING_BEGIN_ID).sum() == bsz:
                gen_parsing_code_flag = True
                next_tokens = torch.cat([next_tokens, last_item, torch.tensor(PARSING_PLACEHOLDER_IDS).unsqueeze(0).repeat(bsz, 1).to(next_tokens.device), torch.tensor([PARSING_END_ID]).unsqueeze(0).repeat(bsz, 1).to(next_tokens.device)], dim = 1)
                continue

            next_tokens = torch.cat([next_tokens, last_item], dim = 1)

        hyps, refs = [], []
        for i in range(bsz):
            hyps.append(self._decode_parsing(next_tokens[i][1:])) # jump bos instead of eos
            refs.append(self._decode_parsing(sample["target"][i][:-1]))
        
        return torch.stack(hyps, dim=0), torch.stack(refs, dim=0)

    
    def _decode_parsing(self, gen_out):
        gen_out[:4] = gen_out[:4] - len(self.src_dict) + self.cfg.num_bins
        gen_out[4:] = gen_out[4:] - len(self.src_dict) + self.cfg.num_bins + self.cfg.code_dict_size
        return gen_out