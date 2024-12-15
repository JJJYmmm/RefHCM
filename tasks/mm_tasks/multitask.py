# Copyright 2024 JJJYmmm
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import json
import math
import logging
from typing import Optional
from argparse import Namespace

from fairseq.tasks import register_task
from fairseq.tasks import register_task
from fairseq.data import FairseqDataset, iterators

from tasks.ofa_task import OFATask, OFAConfig
from data.mm_data.multitask_dataset import MultiTaskDataset
from data.file_dataset import FileDataset

logger = logging.getLogger(__name__)


@dataclass
class MultiTaskConfig(OFAConfig):
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

    seed : int = field(
        default=7, metadata={"help":"random seed"} 
    )

    refpose_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "refpose data"},
    )
    refparsing_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "refparsing data"},
    )
    regcaption_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "region caption data"},
    )

    rec_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "rec data selected cols"},
    )

    refpose_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "refpose data selected cols"},
    )

    refparsing_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "refparsing data selected cols"},
    )

    regcaption_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "regcaption data selected cols"},
    )


@register_task("multitask", dataclass=MultiTaskConfig)
class MultiTaskTask(OFATask):
    def __init__(self, cfg: MultiTaskConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

        self.refpose_dataset = None
        self.refparsing_dataset = None

        if self.cfg.refpose_dataset is not None:
            self.refpose_dataset = FileDataset(self.cfg.refpose_dataset, self.cfg.refpose_selected_cols)

        if self.cfg.refparsing_dataset is not None:
            self.refparsing_dataset = FileDataset(self.cfg.refparsing_dataset, self.cfg.refparsing_selected_cols)

        if self.cfg.regcaption_dataset is not None:
            self.regcaption_dataset = FileDataset(self.cfg.regcaption_dataset, self.cfg.regcaption_selected_cols)
            
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        file_path = paths[(epoch - 1) % len(paths)]
        dataset = FileDataset(file_path, self.cfg.rec_selected_cols)

        self.datasets[split] = MultiTaskDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            seed=self.cfg.seed,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            patch_image_size=self.cfg.patch_image_size,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
            num_bins=self.cfg.num_bins,
            max_image_size=self.cfg.max_image_size,
            refparsing_dataset=self.refparsing_dataset,
            refpose_dataset=self.refpose_dataset,
            regcaption_dataset=self.regcaption_dataset
        )

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):
        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # create mini-batches with given size constraints
        batch_sampler = [
            [j for j in range(i, min(i + max_sentences, len(dataset)))]
            for i in range(0, len(dataset), max_sentences)
        ]
        total_row_count = dataset.dataset.get_total_row_count()
        num_batches = math.ceil(math.ceil(total_row_count / num_shards) / max_sentences)
        if len(batch_sampler) < num_batches:
            batch_sampler.append([1])

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=1,
            shard_id=0,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size
        )

        return epoch_iter