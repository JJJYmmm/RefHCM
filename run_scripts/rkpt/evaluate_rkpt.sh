#!/usr/bin/env

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6083
export CUDA_VISIBLE_DEVICES=0
export GPUS_PER_NODE=1

log_dir=./rkpt_logs
save_dir=./rkpt_checkpoints
mkdir -p $log_dir $save_dir

bpe_dir=../../utils/BPE
user_dir=../../ofa_module
selected_cols=0,5,2,3,4
patch_image_size=512

data=../../dataset/rkpt/refcoco_pose_testA.tsv
path=../../checkpoints/refhcm.pt
result_path=../../results/rkpt

subset=${dataset}'_pose_'${split}
log_file=${log_dir}/"evaluate_rkpt.log"

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=rkpt \
    --bpe-dir=${bpe_dir} \
    --selected-cols=${selected_cols} \
    --batch-size=32 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${subset} \
    --results-path=${result_path} \
    --patch-image-size=${patch_image_size} \
    --beam=5 \
    --min-len=50 \
    --fp16 \
    --num-workers=0 \
    # > ${log_file} 2>&1