#!/usr/bin/bash


source ${HOME}/.bashrc
source ${HOME}/depends/anaconda3/etc/profile.d/conda.sh
cd ${HOME}/projects/Chinese-CLIP/
conda activate qwen3vl

DATAPATH=${HOME}/projects/Chinese-CLIP
split=${1:-'test'}
dataset_name=${2:-'concat_wukong_zero_aic_and_aic_and_coco'}
exp_name=${3:-'concat_wukong_zero_aic_and_aic_and_coco_finetune_vit_large_336_lr_8e-6_bs512_epochs1_gradaccum_4_wd0.001_warmup_10_gpu8_nodes1'}
resume_ckpt_name=${4:-'epoch1.pt'}
resume_ckpt_stem=`echo ${resume_ckpt_name} | cut -d'.' -f1-1`

output_dir=${DATAPATH}/experiments/${exp_name}/${resume_ckpt_stem}_results
if [[ -d ${output_dir} ]]; then
    rm -rf ${output_dir}
fi
mkdir -p ${output_dir}


python -u cn_clip/eval/make_topk_predictions.py \
    --image-feats="${DATAPATH}/datasets/${dataset_name}/${split}_imgs.img_feat.jsonl" \
    --text-feats="${DATAPATH}/datasets/${dataset_name}/${split}_texts.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="${output_dir}/${split}_predictions.jsonl"

python -u cn_clip/eval/make_topk_predictions_tr.py \
    --image-feats="${DATAPATH}/datasets/${dataset_name}/${split}_imgs.img_feat.jsonl" \
    --text-feats="${DATAPATH}/datasets/${dataset_name}/${split}_texts.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="${output_dir}/${split}_tr_predictions.jsonl"
