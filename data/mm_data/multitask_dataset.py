# Copyright 2024 JJJYmmm
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import os
from io import BytesIO

import math
import logging
import random
import warnings

import numpy as np
import torch
import base64
import utils.transforms as T

from PIL import Image, ImageFile

from data import data_utils
from data.ofa_dataset import OFADataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

root_anno_path = '../../dataset/rpar/annos' # your anno path

def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
        )

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source")
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
    patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

    w_resize_ratios = torch.stack([s["w_resize_ratio"] for s in samples], dim=0)
    h_resize_ratios = torch.stack([s["h_resize_ratio"] for s in samples], dim=0)
    region_coords = torch.stack([s['region_coord'] for s in samples], dim=0)

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target")
        tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples])
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens")
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "prev_output_tokens": prev_output_tokens
        },
        "target": target,
        "w_resize_ratios": w_resize_ratios,
        "h_resize_ratios": h_resize_ratios,
        "region_coords": region_coords
    }

    return batch

class MultiTaskDataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=100,
        max_tgt_length=100,
        patch_image_size=512,
        imagenet_default_mean_and_std=False,
        num_bins=1000,
        max_image_size=512,
        refpose_dataset=None,
        refparsing_dataset=None,
        regcaption_dataset=None,
        seed=7,
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        self.num_bins = num_bins
        self.seed = seed

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        
        self.refpose_dataset = refpose_dataset
        self.refparsing_dataset = refparsing_dataset
        self.regcaption_dataset = regcaption_dataset

        # for rec
        self.rec_transform = T.Compose([
            T.RandomResize([patch_image_size], max_size=patch_image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std, max_image_size=max_image_size)
        ])
        self.rec_prompt = ' which region does the text " {} " describe?'

        # for reference pose
        self.refpose_transform = T.Compose([
            T.RandomResize([patch_image_size], max_size=patch_image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std, max_image_size=max_image_size)
        ])
        self.refpose_prompt = ' which region does the text " {} " describe? Provide the bounding box and keypoints.'
        self.kpt_name = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]

        # for reference parsing
        self.refparsing_transform = T.Compose([
            T.RandomResize([patch_image_size], max_size=patch_image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std, max_image_size=max_image_size)
        ])
        self.refparsing_prompt = ' which region does the text " {} " describe? Provide the bounding box and the parsing map.'

        # for region caption
        self.regcaption_transform = T.Compose([
            T.RandomResize([patch_image_size], max_size=patch_image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std, max_image_size=max_image_size)
        ])
        self.regcaption_prompt = ' what does the region {} describe?'.split()

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed, self.epoch):
            # do not need to add rec task, because refpose/refparsing can cover
            # samples = self.process_rec(index)
            samples = self.process_rkpt(0) if self.refpose_dataset else []
            samples += self.process_rpar(0) if self.refparsing_dataset else []
            samples += self.process_rhrc(0) if self.regcaption_dataset else []
        return samples

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch
    
    def process_rec(self, index):
        uniq_id, base64_str, text, region_coord = self.dataset[index]

        image = Image.open(BytesIO(base64.urlsafe_b64decode(base64_str))).convert("RGB")
        w, h = image.size
        boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
        x0, y0, x1, y1 = region_coord.strip().split(',')
        region = torch.tensor([float(x0), float(y0), float(x1), float(y1)])
        boxes_target["boxes"] = torch.tensor([[float(x0), float(y0), float(x1), float(y1)]])
        boxes_target["labels"] = np.array([0])
        boxes_target["area"] = torch.tensor([(float(x1) - float(x0)) * (float(y1) - float(y0))])

        patch_image, patch_boxes = self.rec_transform(image, boxes_target)
        resize_h, resize_w = patch_boxes["size"][0], patch_boxes["size"][1]
        patch_mask = torch.tensor([True])
        quant_x0 = "<bin_{}>".format(int((patch_boxes["boxes"][0][0] * (self.num_bins - 1)).round()))
        quant_y0 = "<bin_{}>".format(int((patch_boxes["boxes"][0][1] * (self.num_bins - 1)).round()))
        quant_x1 = "<bin_{}>".format(int((patch_boxes["boxes"][0][2] * (self.num_bins - 1)).round()))
        quant_y1 = "<bin_{}>".format(int((patch_boxes["boxes"][0][3] * (self.num_bins - 1)).round()))
        region_coord = "{} {} {} {}".format(quant_x0, quant_y0, quant_x1, quant_y1)
        src_caption = self.pre_caption(text, self.max_src_length)
        src_item = self.encode_text(self.rec_prompt.format(src_caption))
        tgt_item = self.encode_text(region_coord, use_bpe=False)

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "w_resize_ratio": resize_w / w,
            "h_resize_ratio": resize_h / h,
            "region_coord": region
        }
        return [example]
    
    def process_rhrc(self, index):
        uniq_id, image_path, region_coord, caption = self.regcaption_dataset[index]

        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        x0, y0, x1, y1 = region_coord.strip().split(',')
        region = torch.tensor([float(x0), float(y0), float(x1), float(y1)])
        boxes_target = {"size": torch.tensor([h, w])}
        boxes_target["boxes"] = torch.tensor([[float(x0), float(y0), float(x1), float(y1)]])

        patch_image, patch_boxes = self.regcaption_transform(image, boxes_target)
        resize_h, resize_w = patch_boxes["size"][0], patch_boxes["size"][1]
        patch_mask = torch.tensor([True])
        quant_x0 = "<bin_{}>".format(int((patch_boxes["boxes"][0][0] * (self.num_bins - 1)).round()))
        quant_y0 = "<bin_{}>".format(int((patch_boxes["boxes"][0][1] * (self.num_bins - 1)).round()))
        quant_x1 = "<bin_{}>".format(int((patch_boxes["boxes"][0][2] * (self.num_bins - 1)).round()))
        quant_y1 = "<bin_{}>".format(int((patch_boxes["boxes"][0][3] * (self.num_bins - 1)).round()))
        region_coord = "{} {} {} {}".format(quant_x0, quant_y0, quant_x1, quant_y1)
        
        caption_token_list = caption.strip().split()
        tgt_caption = ' '.join(caption_token_list[:self.max_tgt_length])

        # process text
        text_list = []
        for item in self.regcaption_prompt:
            if item != "{}":
                text_list.append(self.bpe.encode(' {}'.format(item)))
            else:
                text_list.append(region_coord)
        
        src_item = self.encode_text(' '.join(text_list), use_bpe=False)
        tgt_item = self.encode_text(" {}".format(tgt_caption))

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "region_coord": region,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "w_resize_ratio": resize_w / w,
            "h_resize_ratio": resize_h / h,
        }
        return [example]

    def process_rkpt(self, index):

        uniq_id, base64_str, text, region_coord, keypoints_coord = self.refpose_dataset[index]

        image = Image.open(BytesIO(base64.urlsafe_b64decode(base64_str))).convert("RGB")
        w, h = image.size

        boxes_target = {"boxes": [], "keypoints": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
        x0, y0, x1, y1 = region_coord.strip().split(',')
        region = torch.tensor([float(x0), float(y0), float(x1), float(y1)])
        boxes_target["boxes"] = torch.tensor([[float(x0), float(y0), float(x1), float(y1)]])
        boxes_target["labels"] = np.array([0])
        boxes_target["area"] = torch.tensor([(float(x1) - float(x0)) * (float(y1) - float(y0))])

        keypoints_coord = list(map(float,keypoints_coord.strip().split(',')))
        boxes_target['keypoints'] = torch.tensor([keypoints_coord])

        patch_image, patch_boxes = self.refpose_transform(image, boxes_target)
        resize_h, resize_w = patch_boxes["size"][0], patch_boxes["size"][1]
        patch_mask = torch.tensor([True])
        quant_x0 = "<bin_{}>".format(int((patch_boxes["boxes"][0][0] * (self.num_bins - 1)).round()))
        quant_y0 = "<bin_{}>".format(int((patch_boxes["boxes"][0][1] * (self.num_bins - 1)).round()))
        quant_x1 = "<bin_{}>".format(int((patch_boxes["boxes"][0][2] * (self.num_bins - 1)).round()))
        quant_y1 = "<bin_{}>".format(int((patch_boxes["boxes"][0][3] * (self.num_bins - 1)).round()))
        region_coord = "{} {} {} {}".format(quant_x0, quant_y0, quant_x1, quant_y1)

        kpts = patch_boxes['keypoints'][0].reshape(-1, 2)
        quant_kpt_coord = []
        for name, coords in zip(self.kpt_name, kpts):
            if coords.max():
                quant_kpt_coord.append(self.bpe.encode(' {}'.format(name)))
                quant_kpt_coord.append("<bin_{}>".format(int((coords[0] * (self.num_bins - 1)).round())))
                quant_kpt_coord.append("<bin_{}>".format(int((coords[1] * (self.num_bins - 1)).round())))
        keypoints_coord = " ".join(quant_kpt_coord)

        src_caption = self.pre_caption(text, self.max_src_length)
        src_item = self.encode_text(self.refpose_prompt.format(src_caption))
        tgt_item = self.encode_text(region_coord + " " + keypoints_coord, use_bpe=False)

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "w_resize_ratio": resize_w / w,
            "h_resize_ratio": resize_h / h,
            "region_coord": region
        }
        return [example]
    
    def process_rpar(self, index):

        # reference
        uniq_id, image_path, text, region_coord, parsing_code, gt_path = self.refparsing_dataset[index]

        # find absolute path
        image_path = os.path.join(root_anno_path, 'imgs/', image_path)
        gt_path = os.path.join(root_anno_path, 'gts/', gt_path)

        image = Image.open(image_path).convert("RGB")

        w, h = image.size

        boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
        x0, y0, x1, y1 = region_coord.strip().split(',')
        region = torch.tensor([float(x0), float(y0), float(x1), float(y1)])

        boxes_target["boxes"] = torch.tensor([[float(x0), float(y0), float(x1), float(y1)]])
        boxes_target["labels"] = np.array([0])
        boxes_target["area"] = torch.tensor([(float(x1) - float(x0)) * (float(y1) - float(y0))])

        patch_image, patch_boxes = self.refparsing_transform(image, boxes_target)
        resize_h, resize_w = patch_boxes["size"][0], patch_boxes["size"][1]
        patch_mask = torch.tensor([True])
        quant_x0 = "<bin_{}>".format(int((patch_boxes["boxes"][0][0] * (self.num_bins - 1)).round()))
        quant_y0 = "<bin_{}>".format(int((patch_boxes["boxes"][0][1] * (self.num_bins - 1)).round()))
        quant_x1 = "<bin_{}>".format(int((patch_boxes["boxes"][0][2] * (self.num_bins - 1)).round()))
        quant_y1 = "<bin_{}>".format(int((patch_boxes["boxes"][0][3] * (self.num_bins - 1)).round()))
        region_coord = "{} {} {} {}".format(quant_x0, quant_y0, quant_x1, quant_y1)
        
        parsing_code = list(map(int,parsing_code.strip().split(',')))

        parsing_code_tokens = []
        for code in parsing_code:
            parsing_code_tokens.append("<code_{}>".format(code))
        parsing_code = " ".join(parsing_code_tokens)

        # reference
        src_caption = self.pre_caption(text, self.max_src_length)
        src_item = self.encode_text(self.refparsing_prompt.format(src_caption))
        tgt_item = self.encode_text(region_coord + " " + parsing_code, use_bpe=False)

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": uniq_id,
            "source": src_item,
            "text": text,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "w_resize_ratio": resize_w / w,
            "h_resize_ratio": resize_h / h,
            "region_coord": region,
        }
        return [example]

    def collater(self, samples, pad_to_length=None):
        all_samples = []
        for sample in samples:
            all_samples += sample
        return collate(all_samples, pad_idx=self.pad, eos_idx=self.eos)
