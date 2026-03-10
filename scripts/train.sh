#!/bin/bash

STUDENT="ViT-T-16"
TEACHER="ViT-B-32"   
TEACHER_PRETRAINED="openai"            
LOSS="clipkd"
NAME="clipkd_${STUDENT}_from_${TEACHER}"

# hyperparameters for Loss weights (CLIP-KD paper values)
ALPHA_CKD=1.0
ALPHA_ICL=1.0
ALPHA_FD=2000.0
LR=1e-3
WD=0.1
WARMUP=10000
EPOCHS=1
BATCH_SIZE=64                       

# Base project directory on cluster
BASE_DIR="/project/lt200394-thllmV/mkd-exp"
TRAIN_DATA="${BASE_DIR}/datasets/cc12m-wds/{0000..0010}.tar"
TRAIN_NUM_SAMPLES=50000        
IMAGENET_VAL="${BASE_DIR}/datasets/imagenet/val"
LOG_DIR="${BASE_DIR}/open_clip/exp1"
WANDB_PROJECT="multilingual-vl-kd-clipkd-exp1"

export WANDB_MODE=offline

ml Mamba/23.11.0-0
conda activate openclip_venv

cd "${BASE_DIR}/open_clip"

torchrun \
    -m open_clip_train.main \
    --model                "${STUDENT}" \
    --distill-model        "${TEACHER}" \
    --distill-pretrained   "${TEACHER_PRETRAINED}" \
    --distill-loss         "${LOSS}" \
    --alpha-ckd-loss       ${ALPHA_CKD} \
    --alpha-icl-loss       ${ALPHA_ICL} \
    --alpha-fd-loss        ${ALPHA_FD} \
    --train-data           "${TRAIN_DATA}" \
    --train-num-samples    ${TRAIN_NUM_SAMPLES} \
    --dataset-type         webdataset \
    --batch-size           ${BATCH_SIZE} \
    --epochs               ${EPOCHS} \
    --lr                   ${LR} \
    --wd                   ${WD} \
    --warmup               ${WARMUP} \
    --workers              16 \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --grad-clip-norm       10.0 \
    --imagenet-val         "${IMAGENET_VAL}" \
    --zeroshot-frequency   1 \
    --save-frequency       5 \
    --log-every-n-steps    100 \
    --seed                 42 \
    --logs                 "${LOG_DIR}" \
    --name                 "${NAME}" \
    --wandb-project-name   "${WANDB_PROJECT}" \
    --report-to            wandb
