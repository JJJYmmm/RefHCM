# Copyright 2022 The OFA-Sys Team. 
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

    target_boxes = [s["target_boxes"] for s in samples]
    target_action_labels = [s["target_action_labels"] for s in samples]
    target_obj_labels = [s["target_obj_labels"] for s in samples]

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
        "target_boxes": target_boxes,
        "target_action_labels": target_action_labels,
        "target_obj_labels": target_obj_labels
    }

    return batch


class HOIDataset(OFADataset):
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

        if type(bpe).__name__ == 'GPT2BPE':
            self.prompt = ' what does the person at {} interact with?'
        elif type(bpe).__name__ == 'BertBPE':
            raise NotImplementedError("BertBPE is not supported for HOI dataset")
        
        self.splited_prompt = self.prompt.split()

        self.INTERACTIONS = [
            'hold obj', 'sit instr', 'ride instr', 'look obj', 'hit instr', 'hit obj', 'eat obj',
            'eat instr', 'jump instr', 'lay instr', 'talk_on_phone instr', 'carry obj', 'throw obj',
            'catch obj', 'cut instr', 'cut obj', 'work_on_computer instr', 'ski instr', 'surf instr',
            'skateboard instr', 'drink instr', 'kick obj', 'read obj', 'snowboard instr'
        ]

        self.OBJECTS = [
            'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]

    def __getitem__(self, index):
        uniq_id, base64_str, region_coord, action_objects = self.dataset[index]

        image = Image.open(BytesIO(base64.urlsafe_b64decode(base64_str))).convert("RGB")
        w, h = image.size
        boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}

        action_objects = action_objects.split(';')[:-1]
        action_objects = [action_object.split(',') for action_object in action_objects]

        x0, y0, x1, y1 = region_coord.strip().split(',')

        action_labels = [0]
        object_labels = [-1]
        boxes = [[float(x0), float(y0), float(x1), float(y1)]]
        areas = [(float(x1) - float(x0)) * (float(y1) - float(y0))]

        for action_object in action_objects:
            if len(action_object) != 6:
                continue
            action_label, object_label, x0, y0, x1, y1 = action_object
            action_labels.append(int(action_label))
            object_labels.append(int(object_label))
            boxes.append([float(x0), float(y0), float(x1), float(y1)])
            areas.append((float(x1) - float(x0)) * (float(y1) - float(y0)))
        
        target_boxes = torch.tensor(boxes)
        target_obj_labels = torch.tensor(object_labels)
        target_action_labels = torch.tensor(action_labels)

        boxes_target["boxes"] = torch.tensor(boxes)
        boxes_target["labels"] = np.array(object_labels)
        boxes_target["area"] = torch.tensor(areas)

        patch_image, patch_boxes = self.positioning_transform(image, boxes_target)
        resize_h, resize_w = patch_boxes["size"][0], patch_boxes["size"][1]
        patch_mask = torch.tensor([True])

        quant_boxes = []
        for box in patch_boxes["boxes"]:
            quant_x0 = "<bin_{}>".format(int((box[0] * (self.num_bins - 1)).round()))
            quant_y0 = "<bin_{}>".format(int((box[1] * (self.num_bins - 1)).round()))
            quant_x1 = "<bin_{}>".format(int((box[2] * (self.num_bins - 1)).round()))
            quant_y1 = "<bin_{}>".format(int((box[3] * (self.num_bins - 1)).round()))
            quant_boxes.append([quant_x0, quant_y0, quant_x1, quant_y1])

        region_coord = " {} {} {} {}".format(*quant_boxes[0])

        # process text
        text_list = []
        for item in self.splited_prompt:
            if item != "{}":
                text_list.append(self.bpe.encode(' {}'.format(item)))
            else:
                text_list.append(region_coord)
        
        tgt_list = []
        for idx, quant_box in enumerate(quant_boxes):
            if idx == 0:
                continue
            tgt_list.append(self.bpe.encode(' {}'.format(self.INTERACTIONS[action_labels[idx]])))
            # tgt_list.append(self.bpe.encode(' {}'.format(action_labels[idx])))
            tgt_list.append(self.bpe.encode(' {}'.format(self.OBJECTS[object_labels[idx]])))
            # tgt_list.append(self.bpe.encode(' {}'.format(object_labels[idx])))
            tgt_list.append(' '.join(quant_box))
            tgt_list.append(self.bpe.encode(' ;'))
        
        src_item = self.encode_text(' '.join(text_list), use_bpe=False)
        tgt_item = self.encode_text(' '.join(tgt_list), use_bpe=False)

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
            "target_boxes": target_boxes,
            "target_obj_labels": target_obj_labels,
            "target_action_labels": target_action_labels
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
