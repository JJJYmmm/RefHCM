#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6085
export CUDA_VISIBLE_DEVICES=0
export GPUS_PER_NODE=1


########################## Evaluate Refcoco ##########################
user_dir=../../ofa_module
bpe_dir=../../utils/BPE
selected_cols=0,5,2,3
patch_image_size=512

data=../../dataset/rkpt/refcoco_pose_testA.tsv
path=../../checkpoints/refhcm.pt

result_path=../../results/rec
split='refcoco_testA'

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --bpe-dir=${bpe_dir} \
    --task=refcoco \
    --selected-cols=${selected_cols} \
    --batch-size=32 \
    --log-format=simple --log-interval=10 \
    --seed=77 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --min-len=4 \
    --max-len-a=0 \
    --max-len-b=4 \
    --patch-image-size=${patch_image_size} \
    --constraint-range="58457,59457" \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0