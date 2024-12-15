# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import json
import logging
from typing import Optional
from argparse import Namespace

import re
import torch
from torch.nn import functional as F
from fairseq import metrics
from fairseq.tasks import register_task

from tasks.ofa_task import OFATask, OFAConfig
from data.mm_data.rkpt_dataset import RKptDataset
from data.file_dataset import FileDataset


logger = logging.getLogger(__name__)


@dataclass
class RKptConfig(OFAConfig):
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


@register_task("rkpt", dataclass=RKptConfig)
class RKptTask(OFATask):
    def __init__(self, cfg: RKptConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

        self.kpt_name = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
        self.kpt2idx = {item: idx for idx, item in enumerate(self.kpt_name)}


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        if split == 'train':
            file_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            file_path = paths[-1]
        dataset = FileDataset(file_path, self.cfg.selected_cols)

        self.datasets[split] = RKptDataset(
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

    def build_model(self, cfg):
        model = super().build_model(cfg)
        if self.cfg.eval_acc:
            gen_args = json.loads(self.cfg.eval_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        if self.cfg.scst:
            scst_args = json.loads(self.cfg.scst_args)
            self.scst_generator = self.build_generator(
                [model], Namespace(**scst_args)
            )

        return model

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

    def _calculate_oks_score(self, hyps, refs, bbox, thresh=0.5):

        if hyps.shape[1] < refs.shape[1]:
            hyps = F.pad(hyps,(0,refs.shape[1] - hyps.shape[1], 0, 0))
        
        assert hyps.shape == refs.shape

        B = refs.shape[0]
        refs = refs.reshape(B, -1, 2)
        hyps = hyps.reshape(B, -1, 2)

        xg, yg = refs[:, :, 0], refs[:, :, 1]
        vg =  (refs[:, :, 0] != 0) | (refs[:, :, 1] != 0)
        xd, yd = hyps[:, :, 0], hyps[:, :, 1]

        gt_areas = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
        # double gt bbox
        x0, y0, x1, y1 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        w = bbox[:, 2] - bbox[:, 0]
        h = bbox[:, 3] - bbox[:, 1]
        x0 -= w
        y0 -= h
        x1 += w
        y1 += h

        kpt_oks_sigmas = torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
        kpt_oks_sigmas = kpt_oks_sigmas.to(refs.device)

        k = len(kpt_oks_sigmas)
        vars = (kpt_oks_sigmas * 2) ** 2

        k1 = torch.sum(vg > 0, dim = 1)
        mask = (k1 > 0).to(refs.device)

        e = torch.zeros((B,k)).to(refs.device)
        oks = torch.zeros(B).to(refs.device)

        dx = xd - xg
        dy = yd - yg
        e[mask] = ( (dx**2 + dy**2) / vars.unsqueeze(0) / (gt_areas[:, None] + torch.finfo(torch.float32).eps) / 2 )[mask]
        # filter unvisible kpt
        for i, flag in enumerate(mask):
            if flag == True:
                tmp = e[i][(vg > 0)[i]]
                oks[i] = torch.sum(torch.exp(-tmp) ,dim = 0) / tmp.shape[0]
                 

        z = torch.zeros((B,k)).to(refs.device)
        dx = torch.max(torch.stack((z, x0[:, None] - xd)), dim=0)[0] + torch.max(torch.stack((z, xd - x1[:, None])), dim = 0)[0]
        dy = torch.max(torch.stack((z, y0[:, None] - yd)), dim=0)[0] + torch.max(torch.stack((z, yd - y1[:, None])), dim = 0)[0]
        e[~mask] = ( (dx**2 + dy**2) / vars.unsqueeze(0) / (gt_areas[:, None] + torch.finfo(torch.float32).eps) / 2 )[~mask]

        oks[~mask] = torch.sum(torch.exp(-e[~mask]) ,dim = 1) / e[~mask].shape[1] # [B]
       
        return (oks >= thresh).float()
       

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = criterion(model, sample)

        model.eval()
        if self.cfg.eval_acc:
            hyps, refs = self._inference(self.sequence_generator, sample, model)
            hyps = hyps / (self.cfg.num_bins - 1) * self.cfg.max_image_size
            refs = refs / (self.cfg.num_bins - 1) * self.cfg.max_image_size
            hyps[:, ::2] /= sample['w_resize_ratios'].unsqueeze(1)
            hyps[:, 1::2] /= sample['h_resize_ratios'].unsqueeze(1)
            refs[:, ::2] /= sample['w_resize_ratios'].unsqueeze(1)
            refs[:, 1::2] /= sample['h_resize_ratios'].unsqueeze(1)

            # calculate box ap
            bbox_scores = self._calculate_ap_score(hyps[:, :4], sample['region_coords'].float())
            logging_output["_box_score_sum"] = bbox_scores.sum().item()
            logging_output["_box_score_cnt"] = bbox_scores.size(0)

            # calculate kpt oks
            kpt_scores = self._calculate_oks_score(hyps[:, 4:], sample['keypoints'].float(), sample['region_coords'].float())
            logging_output["_oks_score_sum"] = kpt_scores.sum().item()
            logging_output["_oks_score_cnt"] = kpt_scores.size(0)

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

        def compute_oks_score(meters):
            score = meters["_oks_score_sum"].sum / meters["_oks_score_cnt"].sum
            score = score if isinstance(score, float) else score.item()
            return round(score, 4)

        if sum_logs("_box_score_cnt") > 0:
            metrics.log_scalar("_box_score_sum", sum_logs("_box_score_sum"))
            metrics.log_scalar("_box_score_cnt", sum_logs("_box_score_cnt"))
            metrics.log_derived("box_score", compute_box_score)
            metrics.log_scalar("_oks_score_sum", sum_logs("_oks_score_sum"))
            metrics.log_scalar("_oks_score_cnt", sum_logs("_oks_score_cnt"))
            metrics.log_derived("oks_score", compute_oks_score)

    def _inference(self, generator, sample, model):
        gen_out = self.inference_step(generator, [model], sample)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            # hyps.append(gen_out[i][0]["tokens"][:-1] - len(self.src_dict) + self.cfg.num_bins) # -1 jump eos
            hyps.append(self._decode_keypoints(gen_out[i][0]["tokens"][:-1]))
            refs.append(sample["target"][i][:-1] - len(self.src_dict) + self.cfg.num_bins)
        if self.cfg.eval_print_samples:
            logger.info("example hypothesis: ", hyps[0])
            logger.info("example reference: ", refs[0])

        return torch.stack(hyps, dim=0), torch.stack(refs, dim=0)
    
    def extract_number(self, string):
        pattern = r'<bin_(\d+)>'
        match = re.search(pattern, string)
        if match:
            return int(match.group(1))
        else:
            return None

    def _decode_keypoints(self, gen_out):
        recover_kpt = self.tgt_dict.string(gen_out.int().cpu())
        result = torch.zeros(38).to(gen_out.device) # 4 bbox + 17x2 keypoints

        names = []
        coords = []
        for token in recover_kpt.strip().split():
            if token.startswith('<bin_'):
                coords.append(self.extract_number(token))
            else:
                if self.bpe is not None:
                    token = self.bpe.decode('{}'.format(token))
                if token.startswith(' ') or len(names) == 0:
                    names.append(token.strip())
                else:
                    names[-1] += token
        
            
        # bbox
        shift = 4
        if len(coords) < 2 * len(names) + shift:
            coords = [0 for _ in range(2 * len(names) + shift - len(coords))] + coords

        result[:shift] = torch.tensor(coords[:shift]).to(gen_out.device)
        for idx, name in enumerate(names):
            pos = self.kpt2idx.get(name)
            if pos == None:
                continue
            result[shift + pos * 2] = coords[shift + idx * 2] 
            result[shift + pos * 2 + 1] = coords[shift + idx * 2 + 1] 

        return result