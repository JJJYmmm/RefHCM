# Copyright 2024 JJJYmmm
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from io import BytesIO

import logging
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
    keypoints = torch.stack([s['keypoints'] for s in samples], dim=0)

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
        "region_coords": region_coords,
        "keypoints": keypoints
    }

    return batch


class RKptDataset(OFADataset):
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
            T.ToTensor(),
            T.Normalize(mean=mean, std=std, max_image_size=max_image_size)
        ])

        self.kpt_name = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]

        if type(bpe).__name__ == 'GPT2BPE':
            self.prompt = ' which region does the text " {} " describe? Provide the bounding box and keypoints.'
        elif type(bpe).__name__ == 'BertBPE':
            self.prompt = '这段文字" {} "描述的是哪个人？提供他/她的边界框和关键点'
        
    def __getitem__(self, index):
        uniq_id, base64_str, text, region_coord, keypoints_coord = self.dataset[index]

        image = Image.open(BytesIO(base64.urlsafe_b64decode(base64_str))).convert("RGB")
        w, h = image.size

        boxes_target = {"boxes": [], "keypoints": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
        x0, y0, x1, y1 = region_coord.strip().split(',')
        region = torch.tensor([float(x0), float(y0), float(x1), float(y1)])
        boxes_target["boxes"] = torch.tensor([[float(x0), float(y0), float(x1), float(y1)]])
        boxes_target["labels"] = np.array([0])
        boxes_target["area"] = torch.tensor([(float(x1) - float(x0)) * (float(y1) - float(y0))])

        keypoints_coord = list(map(float,keypoints_coord.strip().split(',')))
        keypoints = torch.tensor(keypoints_coord)
        boxes_target['keypoints'] = torch.tensor([keypoints_coord])

        patch_image, patch_boxes = self.positioning_transform(image, boxes_target)
        resize_h, resize_w = patch_boxes["size"][0], patch_boxes["size"][1]
        patch_mask = torch.tensor([True])
        x0 = int((patch_boxes["boxes"][0][0] * (self.num_bins - 1)).round())
        y0 = int((patch_boxes["boxes"][0][1] * (self.num_bins - 1)).round())
        x1 = int((patch_boxes["boxes"][0][2] * (self.num_bins - 1)).round())
        y1 = int((patch_boxes["boxes"][0][3] * (self.num_bins - 1)).round())
        quant_x0 = "<bin_{}>".format(x0)
        quant_y0 = "<bin_{}>".format(y0)
        quant_x1 = "<bin_{}>".format(x1)
        quant_y1 = "<bin_{}>".format(y1)
        region_coord = "{} {} {} {}".format(quant_x0, quant_y0, quant_x1, quant_y1)

        region_coord_language = f" {x0} {y0} {x1} {y1}"
        
        kpts = patch_boxes['keypoints'][0].reshape(-1, 2)
        quant_kpt_coord = []
        kpt_coord_language = []
        for name, coords in zip(self.kpt_name, kpts):
            if coords.max():
                quant_kpt_coord.append(self.bpe.encode(' {}'.format(name)))
                quant_kpt_coord.append("<bin_{}>".format(int((coords[0] * (self.num_bins - 1)).round())))
                quant_kpt_coord.append("<bin_{}>".format(int((coords[1] * (self.num_bins - 1)).round())))
                kpt_coord_language.append(name)
                kpt_coord_language.append(str(int((coords[0] * (self.num_bins - 1)).round())))
                kpt_coord_language.append(str(int((coords[1] * (self.num_bins - 1)).round())))

        keypoints_coord = " ".join(quant_kpt_coord)
        kpt_coord_language = " ".join(kpt_coord_language)

        src_caption = self.pre_caption(text, self.max_src_length)
        src_item = self.encode_text(self.prompt.format(src_caption))
        
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
            "region_coord": region,
            "keypoints": keypoints
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
