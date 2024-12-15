#!/usr/bin/env

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6088
export CUDA_VISIBLE_DEVICES=2
export GPUS_PER_NODE=1

log_dir=./rpar_logs
save_dir=./rpar_checkpoints
mkdir -p $log_dir $save_dir

bpe_dir=../../utils/BPE
user_dir=../../ofa_module
selected_cols=0,5,1,2,3,4

patch_image_size=512
vq_config=../../checkpoints/vqgan/model.yaml
vq_ckpt=../../checkpoints/vqgan/model.ckpt
vq_n_embed=32

path=../../checkpoints/refhcm.pt

dataset=rpar
data_dir=../../dataset/rpar
data=${data_dir}/refcihp_val.tsv

result_path=../../results/rpar

subset=rpar
log_file=${log_dir}/"evaluate_rpar.log"

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=rpar \
    --bpe-dir=${bpe_dir} \
    --selected-cols=${selected_cols} \
    --batch-size=32 \
    --log-format=simple --log-interval=10 \
    --seed=77 \
    --gen-subset=${subset} \
    --results-path=${result_path} \
    --beam=5 \
    --min-len=52 \
    --max-len-a=0 \
    --max-len-b=52 \
    --patch-image-size=${patch_image_size} \
    --vq-config=${vq_config} \
    --vq-ckpt=${vq_ckpt} \
    --vq-n-embed=${vq_n_embed} \
    --fp16 \
    --num-workers=0 \
    # > ${log_file} 2>&1