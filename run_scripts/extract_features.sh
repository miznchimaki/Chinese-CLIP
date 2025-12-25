#!/usr/bin/bash


source ${HOME}/.bashrc
source ${HOME}/depends/anaconda3/etc/profile.d/conda.sh
cd ${HOME}/projects/Chinese-CLIP/
conda activate qwen3vl

DATAPATH=${HOME}/projects/Chinese-CLIP
device=${1:-'0'}
split=${2:-'test'}
dataset_name=${3:-'concat_laion_cn_wukong_zero_aic_and_aic_coco'}
exp_name=${4:-'concat_laion_cn_and_wukong_and_zero_aic_and_aic_and_coco_finetune_vit_large_336_lr_8e-6_bs512_epochs1_gradaccum_1_wd0.008_warmup_18_gpu8_nodes4'}
resume_ckpt_name=${5:-'epoch1.pt'}

export CUDA_VISIBLE_DEVICES=${device}
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip


python -u cn_clip/eval/extract_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --image-data="${DATAPATH}/datasets/${dataset_name}/lmdb/${split}/imgs" \
    --text-data="${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl" \
    --img-batch-size=32 \
    --text-batch-size=32 \
    --context-length=512 \
    --resume=${DATAPATH}/experiments/${exp_name}/checkpoints/${resume_ckpt_name} \
    --vision-model=ViT-L-14-336 \
    --text-model=RoBERTa-wwm-ext-base-chinese

