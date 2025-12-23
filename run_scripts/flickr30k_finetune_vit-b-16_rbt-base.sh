#!/usr/bin/bash


export CUDA_LAUNCH_BLOCKING=1

# Number of GPUs per GPU worker
GPUS_PER_NODE=${1:-8}

# Number of GPU workers, for single-worker training, please set to 1
WORKER_CNT=${2:-1}

# The ip address of the rank-0 worker, for single-worker training, please set to localhost
MASTER_ADDR_VAR=${3:-localhost}
export MASTER_ADDR=${MASTER_ADDR_VAR}

# The port for communication
MASTER_PORT_VAR=${4:-34970}
export MASTER_PORT=${MASTER_PORT_VAR}

# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
RANK_VAR=${5:-0}
export RANK=${RANK_VAR}

DATAPATH=${6:-${HOME}/projects/Chinese-CLIP}

source ${HOME}/.bashrc
source ${HOME}/depends/anaconda3/etc/profile.d/conda.sh
cd ${HOME}/projects/Chinese-CLIP/
conda activate qwen3vl
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip/

# data options
train_data=${DATAPATH}/datasets/concat_laion_cn_wukong_zero_aic_and_aic_coco/lmdb/train
val_data=${DATAPATH}/datasets/concat_laion_cn_wukong_zero_aic_and_aic_coco/lmdb/test # if val_data is not specified, the validation will be automatically disabled

# restore options
resume=${DATAPATH}/pretrained_weights/chinese-clip-vit-large-patch14-336px/clip_cn_vit-l-14-336.pt  # or specify your customed ckpt path to resume
reset_data_offset="--reset-data-offset"
reset_optimizer="--reset-optimizer"
# reset_optimizer=""

# output options
output_base_dir=${DATAPATH}/experiments/
save_step_frequency=999999 # disable it
save_epoch_frequency=1
log_interval=1
report_training_batch_acc="--report-training-batch-acc"
# report_training_batch_acc=""

# training hyper-params
context_length=512
# warmup=100
warmup=18  # warmup ratio 0.01 is ok
batch_size=512
valid_batch_size=128
accum_freq=4
lr=8e-6  # learning rate 3e-6 is best for 1 node; 1.2e-5 is best for 4 nodes
# wd=0.001
wd=0.003 # weight decay 0.001 is best
# epoch 1 is best
max_epochs=1  # or you can alternatively specify --max-steps
valid_step_interval=2000
valid_epoch_interval=1
vision_model=ViT-L-14-336
# vision_model=ViT-H-14-336
text_model=RoBERTa-wwm-ext-base-chinese
# text_model=RoBERTa-wwm-ext-large-chinese
use_augment="--use-augment"
# use_augment=""
name=concat_laion_cn_and_wukong_and_zero_aic_and_aic_and_coco_finetune_vit_large_336_lr_${lr}_bs${batch_size}_epochs${max_epochs}_gradaccum_${accum_freq}_wd${wd}_warmup_${warmup}_gpu${GPUS_PER_NODE}_nodes${WORKER_CNT}

python3 -m torch.distributed.launch --use_env --nproc_per_node=${GPUS_PER_NODE} --nnodes=${WORKER_CNT} --node_rank=${RANK} \
          --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} cn_clip/training/main.py \
          --train-data=${train_data} \
          --val-data=${val_data} \
          --resume=${resume} \
          ${reset_data_offset} \
          ${reset_optimizer} \
          --logs=${output_base_dir} \
          --name=${name} \
          --save-step-frequency=${save_step_frequency} \
          --save-epoch-frequency=${save_epoch_frequency} \
          --log-interval=${log_interval} \
          ${report_training_batch_acc} \
          --context-length=${context_length} \
          --warmup=${warmup} \
          --batch-size=${batch_size} \
          --valid-batch-size=${valid_batch_size} \
          --valid-step-interval=${valid_step_interval} \
          --valid-epoch-interval=${valid_epoch_interval} \
          --accum-freq=${accum_freq} \
          --lr=${lr} \
          --wd=${wd} \
          --max-epochs=${max_epochs} \
          --vision-model=${vision_model} \
          --grad-checkpointing \
          ${use_augment} \
          --text-model=${text_model} \
          --text-mask-ratio=0.15 \
          --mlm-loss-weight=1.0
