#!/usr/bin/env

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6060

log_dir=./rpar_logs
save_dir=./rpar_checkpoints
mkdir -p $log_dir $save_dir

bpe_dir=../../utils/BPE
user_dir=../../ofa_module

data_dir=../../dataset/rpar
data=${data_dir}/refcihp_train.tsv,${data_dir}/refcihp_val.tsv
restore_file=../../checkpoints/ofa_large.pt
selected_cols=0,5,1,2,3,4

task=rpar
arch=refhcm
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
lr=3e-5
max_epoch=20
warmup_ratio=0.06
batch_size=8
update_freq=8
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.2
decoder_drop_path_rate=0.2
dropout=0.1
attention_dropout=0.0
max_src_length=80
max_tgt_length=260
num_bins=1000
code_dict_size=8192
patch_image_size=512

vq_config=../../checkpoints/vqgan/model.yaml
vq_ckpt=../../checkpoints/vqgan/model.ckpt
vq_n_embed=32

# train with only the ckpt
# --reset-optimizer --reset-dataloader --reset-meters \ 
# --restore-file=${restore_file} \

# train with ckpt, lr scheduler...
# --finetune-from-model=${restore_file} \

for max_epoch in {15,}; do
  echo "max_epoch "${max_epoch}
  for lr in {3e-5,}; do
    echo "lr "${lr}
    for patch_image_size in {512,}; do
      echo "patch_image_size "${patch_image_size}

      log_file=${log_dir}/${max_epoch}"_"${lr}"_"${patch_image_size}".log"
      save_path=${save_dir}/${max_epoch}"_"${lr}"_"${patch_image_size}
      mkdir -p $save_path

      CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=${MASTER_PORT} ../../train.py \
          $data \
          --selected-cols=${selected_cols} \
          --bpe-dir=${bpe_dir} \
          --user-dir=${user_dir} \
          --finetune-from-model=${restore_file} \
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
          --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=1.0 \
          --lr-scheduler=polynomial_decay --lr=${lr} \
          --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
          --log-format=simple --log-interval=10 \
          --fixed-validation-seed=7 \
          --save-interval=1 --validate-interval=4 \
          --eval-acc \
          --eval-args='{"beam":1,"min_len":36,"max_len_a":0,"max_len_b":36}' \
          --vq-config=${vq_config} \
          --vq-ckpt=${vq_ckpt} \
          --vq-n-embed=${vq_n_embed} \
          --best-checkpoint-metric=miou --maximize-best-checkpoint-metric \
          --max-src-length=${max_src_length} \
          --max-tgt-length=${max_tgt_length} \
          --find-unused-parameters \
          --add-type-embedding \
          --scale-attn \
          --scale-fc \
          --scale-heads \
          --disable-entangle \
          --num-bins=${num_bins} \
          --code-dict-size=${code_dict_size} \
          --patch-image-size=${patch_image_size} \
          --fp16 \
          --fp16-scale-window=512 \
          --num-workers=0 \
          --tensorboard-logdir=${log_dir} > ${log_file} 2>&1
    done
  done
done