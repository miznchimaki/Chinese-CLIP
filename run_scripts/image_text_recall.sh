#!/usr/bin/bash


source ${HOME}/.bashrc
source ${HOME}/depends/anaconda3/etc/profile.d/conda.sh
cd ${HOME}/projects/Chinese-CLIP/
conda activate qwen3vl

DATAPATH=${HOME}/projects/Chinese-CLIP
split=${1:-'test'}
dataset_name=${2:-'concat_laion_cn_wukong_zero_aic_and_aic_coco'}
exp_name=${3:-'concat_laion_cn_and_wukong_and_zero_aic_and_aic_and_coco_finetune_vit_large_336_lr_8e-6_bs512_epochs1_gradaccum_4_wd0.001_warmup_18_gpu8_nodes1'}
resume_ckpt_name=${4:-'epoch1.pt'}
resume_ckpt_stem=`echo ${resume_ckpt_name} | cut -d'.' -f1-1`

output_dir=${DATAPATH}/experiments/${exp_name}/${resume_ckpt_stem}_results


python cn_clip/eval/evaluation.py \
    ${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl \
    ${output_dir}/${split}_predictions.jsonl \
    ${output_dir}/image_retrieval_output.json
cat --number ${output_dir}/image_retrieval_output.json; printf '\n'

python cn_clip/eval/transform_ir_annotation_to_tr.py \
    --input ${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl

python cn_clip/eval/evaluation_tr.py \
    ${DATAPATH}/datasets/${dataset_name}/${split}_texts.tr.jsonl \
    ${output_dir}/${split}_tr_predictions.jsonl \
    ${output_dir}/text_retrieval_output.json
cat --number ${output_dir}/text_retrieval_output.json; echo

