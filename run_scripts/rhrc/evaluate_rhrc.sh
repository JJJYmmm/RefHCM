#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6085
export CUDA_VISIBLE_DEVICES=0
export GPUS_PER_NODE=1

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

data=../../dataset/rhrc/capcihp_val.tsv
path=../../checkpoints/refhcm.pt
result_path=../../results/rhrc
max_src_length=80
max_tgt_length=100
selected_cols=0,3,2,1
split='val'
eval_cider_cached=../../dataset/rhrc/capcihp-words.p

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --eval-cider \
    --eval-cider-cached-tokens=${eval_cider_cached} \
    --path=${path} \
    --user-dir=${user_dir} \
    --selected-cols=${selected_cols} \
    --bpe-dir=${bpe_dir} \
    --task=rhrc \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --max-src-length=${max_src_length} \
    --max-tgt-length=${max_tgt_length} \
    --eval-args='{"beam":5,"max_len_b":100,"no_repeat_ngram_size":3}' \
    --fp16 \
    --num-workers=0 