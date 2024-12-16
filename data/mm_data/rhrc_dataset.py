# Copyright 2024 JJJYmmm
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from io import BytesIO

import re
import logging
import warnings
import string

import numpy as np
import torch
import base64
from torchvision import transforms

from PIL import Image, ImageFile

import utils.transforms as T
from data import data_utils
from data.ofa_dataset import OFADataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


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
        "region_coords": region_coords,
    }

    return batch


class RHrcDataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=128,
        max_tgt_length=30,
        patch_image_size=224,
        imagenet_default_mean_and_std=False,
        scst=False,
        num_bins=1000
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        self.scst = scst
        self.num_bins = num_bins

        self.transtab = str.maketrans({key: None for key in string.punctuation})

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.positioning_transform = T.Compose([
            T.RandomResize([patch_image_size], max_size=patch_image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std, max_image_size=patch_image_size)
        ])

        if type(bpe).__name__ == 'GPT2BPE':
            self.prompt = " what does the region {} describe?"
        elif type(bpe).__name__ == 'BertBPE':
            self.prompt = "区域{}描述了什么内容?"
        
        self.splited_prompt = self.prompt.split() # support english
        
        self.pattern = r'\s*\[\d+(?:, \d+)*\]\s*'

    def __getitem__(self, index):
        uniq_id, base64_str, region_coord, caption = self.dataset[index]

        image = Image.open(BytesIO(base64.urlsafe_b64decode(base64_str))).convert("RGB")
        w, h = image.size

        x0, y0, x1, y1 = region_coord.strip().split(',')
        region = torch.tensor([float(x0), float(y0), float(x1), float(y1)])
        boxes_target = {"size": torch.tensor([h, w])}
        boxes_target["boxes"] = torch.tensor([[float(x0), float(y0), float(x1), float(y1)]])

        patch_image, patch_boxes = self.positioning_transform(image, boxes_target)
        patch_mask = torch.tensor([True])
        quant_x0 = "<bin_{}>".format(int((patch_boxes["boxes"][0][0] * (self.num_bins - 1)).round()))
        quant_y0 = "<bin_{}>".format(int((patch_boxes["boxes"][0][1] * (self.num_bins - 1)).round()))
        quant_x1 = "<bin_{}>".format(int((patch_boxes["boxes"][0][2] * (self.num_bins - 1)).round()))
        quant_y1 = "<bin_{}>".format(int((patch_boxes["boxes"][0][3] * (self.num_bins - 1)).round()))
        region_coord = "{} {} {} {}".format(quant_x0, quant_y0, quant_x1, quant_y1)
        
        caption = self.process_caption(caption)
        if self.split == 'train' and not self.scst:
            caption_token_list = caption.strip().split()
            tgt_caption = ' '.join(caption_token_list[:self.max_tgt_length])
        else:
            caption = ' '.join(caption.strip().split())
            caption_list = [cap.translate(self.transtab).strip() for cap in caption.strip().split('&&')]
            tgt_caption = '&&'.join(caption_list)

        # process text
        text_list = []
        for item in self.splited_prompt:
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
            "prev_output_tokens": prev_output_item
        }
        return example

    def process_caption(self, caption):
        caption = re.sub(self.pattern, " ", caption)
        return caption

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos)
