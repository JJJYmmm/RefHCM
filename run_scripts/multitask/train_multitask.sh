#!/usr/bin/env

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=1056
export CUDA_VISIBLE_DEVICES=1,2
export GPUS_PER_NODE=2


bpe_dir=../../utils/BPE
user_dir=../../ofa_module

restore_file=../../checkpoints/refhcm.pt

data_dir=../../dataset


# multitask training, form refhcm
# data=${data_dir}/rec/refcoco_train.tsv
# refpose_dataset=${data_dir}/rkpt/rkpt_train.tsv
# refparsing_dataset=${data_dir}/rpar/refcihp_train.tsv
# regcaption_dataset=${data_dir}/rhrc/capcihp_train.tsv

# tune on reason data, form refhcm-tuned
reason_data_dir=${data_dir}/reasonref
data=${reason_data_dir}/reasondec/total_reasondec_train.tsv
refpose_dataset=${reason_data_dir}/reasonpose/total_reasonpose_train.tsv
refparsing_dataset=${reason_data_dir}/reasonpar/total_reasonpar_train.tsv
regcaption_dataset=${data_dir}/CIHP/caption_data/regcaption_train_ferret_13b.tsv

rec_selected_cols=0,4,2,3
refpose_selected_cols=0,5,2,3,4
refparsing_selected_cols=0,5,1,2,3,4
regcaption_selected_cols=0,3,2,1

task=multitask
arch=ofa_large
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
lr=4e-5
max_epoch=5 # modify default: 50
warmup_ratio=0.01 # default: 0.01
batch_size=2
update_freq=4
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.2
decoder_drop_path_rate=0.2
dropout=0.1
attention_dropout=0.0
max_src_length=100
max_tgt_length=100
num_bins=1000
patch_image_size=512

save_path=./multitask_checkpoints
log_dir=./multitask_logs
mkdir -p $log_dir
log_file=${log_dir}/${max_epoch}"_"${lr}"_"${patch_image_size}".log"


# train with only the ckpt
# --reset-optimizer --reset-dataloader --reset-meters \ 
# --restore-file=${restore_file} \

# train with ckpt, lr scheduler...
# --finetune-from-model=${restore_file} \

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../train.py \
  $data \
  --refpose-dataset=${refpose_dataset} \
  --refparsing-dataset=${refparsing_dataset} \
  --regcaption-dataset=${regcaption_dataset} \
  --rec-selected-cols=${rec_selected_cols} \
  --refpose-selected-cols=${refpose_selected_cols} \
  --refparsing-selected-cols=${refparsing_selected_cols} \
  --regcaption-selected-cols=${regcaption_selected_cols} \
  --bpe-dir=${bpe_dir} \
  --user-dir=${user_dir} \
  --restore-file=${restore_file} \
  --reset-optimizer --reset-dataloader --reset-meters \
  --save-dir=${save_path} \
  --task=${task} \
  --arch=${arch} \
  --criterion=${criterion} \
  --label-smoothing=${label_smoothing} \
  --batch-size=${batch_size} \
  --update-freq=${update_freq} \
  --encoder-normalize-before \
  --decoder-normalize-before \
  --share-decoder-input-output-embed \
  --share-all-embeddings \
  --layernorm-embedding \
  --patch-layernorm-embedding \
  --code-layernorm-embedding \
  --resnet-drop-path-rate=${resnet_drop_path_rate} \
  --encoder-drop-path-rate=${encoder_drop_path_rate} \
  --decoder-drop-path-rate=${decoder_drop_path_rate} \
  --dropout=${dropout} \
  --attention-dropout=${attention_dropout} \
  --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=5.0 \
  --lr-scheduler=polynomial_decay --lr=${lr} \
  --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
  --log-format=simple --log-interval=10 \
  --fixed-validation-seed=7 \
  --keep-last-epochs=15 \
  --save-interval=1 \
  --save-interval-updates=10000 \
  --disable-validation \
  --max-src-length=${max_src_length} \
  --max-tgt-length=${max_tgt_length} \
  --add-type-embedding \
  --scale-attn \
  --scale-fc \
  --scale-heads \
  --disable-entangle \
  --num-bins=${num_bins} \
  --patch-image-size=${patch_image_size} \
  --fp16 \
  --fp16-scale-window=512 \
  --num-workers=0 \
  --ddp-backend=no_c10d \
  --tensorboard-logdir=${log_dir} > ${log_file} 2>&1