# Copyright 2024 JJJYmmm
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import os
from io import BytesIO

import logging
import warnings

import numpy as np
import torch
import base64
from scipy.ndimage import zoom
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

root_anno_path = '../../dataset/rpar/anno' # your anno path

# <code_0~31> for image code, <code_40> for parsing begin token, <code_41> for parsing end token, <code_42~48> for parsing queries
PARSING_BEGIN_TOKEN = "<code_40>"
PARSING_END_TOKEN = "<code_41>"
PARSING_PLACEHOLDER_TOKENS = " ".join([f"<code_{k}>" for k in range(42, 42+48)])
PARSING_BEGIN_ID = 50305
PARSING_END_ID = 50306
PARSING_PLACEHOLDER_IDS = [50307, 50308, 50309, 50310,
         50311, 50312, 50313, 50314, 50315, 50316, 50317, 50318, 50319, 50320,
         50321, 50322, 50323, 50324, 50325, 50326, 50327, 50328, 50329, 50330,
         50331, 50332, 50333, 50334, 50335, 50336, 50337, 50338, 50339, 50340,
         50341, 50342, 50343, 50344, 50345, 50346, 50347, 50348, 50349, 50350,
         50351, 50352, 50353, 50354]

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
    text = np.array([s["text"] for s in samples])
    src_tokens = merge("source")
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
    patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

    w_resize_ratios = torch.stack([s["w_resize_ratio"] for s in samples], dim=0)
    h_resize_ratios = torch.stack([s["h_resize_ratio"] for s in samples], dim=0)
    region_coords = torch.stack([s['region_coord'] for s in samples], dim=0)
    parsing_codes = torch.stack([s['parsing_code'] for s in samples], dim=0)
    gt = torch.stack([s['gt'] for s in samples], dim=0)

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
        "text": text,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "prev_output_tokens": prev_output_tokens,
            "full_mask_position": (6, 53) # hard code, ofa arch only support 2d attn mask now, <bos> <x1> <y1> <x2> <y2> <beg> <parsing_code> (6 ~ 53) <end>
        },
        "target": target,
        "w_resize_ratios": w_resize_ratios,
        "h_resize_ratios": h_resize_ratios,
        "region_coords": region_coords,
        "parsing_codes": parsing_codes,
        "gt": gt
    }

    return batch


class RParDatasetFullMask(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=80,
        max_tgt_length=30,
        patch_image_size=512,
        imagenet_default_mean_and_std=False,
        num_bins=1000,
        max_image_size=512
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        self.num_bins = num_bins

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        # for positioning
        self.positioning_transform = T.Compose([
            T.RandomResize([patch_image_size], max_size=patch_image_size),
            # T.ResizeWithBoxes(patch_image_size), # for crop image
            T.ToTensor(),
            T.Normalize(mean=mean, std=std, max_image_size=max_image_size)
        ])

        if type(bpe).__name__ == 'GPT2BPE':
            self.prompt = ' which region does the text " {} " describe? Provide the bounding box and the parsing map.'
        elif type(bpe).__name__ == 'BertBPE':
            self.prompt = '这段文字" {} "描述的是哪个人？提供他/她的边界框和人体解析图.'
        
        
    def __getitem__(self, index):

        uniq_id, image_path, text, region_coord, parsing_code, gt_path = self.dataset[index]


        # find absolute path
        image_path = os.path.join(root_anno_path, 'imgs/', image_path)
        gt_path = os.path.join(root_anno_path, 'gts/', gt_path)

        image = Image.open(image_path).convert("RGB")

        w, h = image.size

        array = np.load(gt_path)

        # resize gt to 512x512 for batching, in eval mode, we will resize parsing map to original size
        gt = torch.tensor(zoom(array, (512/array.shape[0], 512/array.shape[1]), order=0), dtype=torch.int32) 

        boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
        x0, y0, x1, y1 = region_coord.strip().split(',')
        region = torch.tensor([float(x0), float(y0), float(x1), float(y1)])

        boxes_target["boxes"] = torch.tensor([[float(x0), float(y0), float(x1), float(y1)]])
        boxes_target["labels"] = np.array([0])
        boxes_target["area"] = torch.tensor([(float(x1) - float(x0)) * (float(y1) - float(y0))])

        patch_image, patch_boxes = self.positioning_transform(image, boxes_target)
        resize_h, resize_w = patch_boxes["size"][0], patch_boxes["size"][1]
        patch_mask = torch.tensor([True])
        quant_x0 = "<bin_{}>".format(int((patch_boxes["boxes"][0][0] * (self.num_bins - 1)).round()))
        quant_y0 = "<bin_{}>".format(int((patch_boxes["boxes"][0][1] * (self.num_bins - 1)).round()))
        quant_x1 = "<bin_{}>".format(int((patch_boxes["boxes"][0][2] * (self.num_bins - 1)).round()))
        quant_y1 = "<bin_{}>".format(int((patch_boxes["boxes"][0][3] * (self.num_bins - 1)).round()))
        region_coord = "{} {} {} {}".format(quant_x0, quant_y0, quant_x1, quant_y1)
        
        parsing_code = list(map(int,parsing_code.strip().split(',')))
        parsing_code_tensor = torch.tensor(parsing_code)

        parsing_code_tokens = []
        for code in parsing_code:
            parsing_code_tokens.append("<code_{}>".format(code))
        parsing_code = " ".join(parsing_code_tokens)

        # <x1>  <y1> <x2> <y2> <beg> <pad> xxxxxxxxxxxxx <eos>
        # <bos> <x1> <y1> <x2> <y2> <beg> <parsing_code> <end>

        src_caption = self.pre_caption(text, self.max_src_length)
        src_item = self.encode_text(self.prompt.format(src_caption))
        tgt_item = self.encode_text(" ".join([region_coord, PARSING_BEGIN_TOKEN, parsing_code]), use_bpe=False)
        tgt_item = torch.cat([tgt_item[:5], torch.tensor([self.pad]), tgt_item[5:]], dim = 0)

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, self.encode_text(" ".join([region_coord, PARSING_BEGIN_TOKEN, PARSING_PLACEHOLDER_TOKENS, PARSING_END_TOKEN]), use_bpe=False)])

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
            "parsing_code": parsing_code_tensor,
            "gt": gt
        }
        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos)
