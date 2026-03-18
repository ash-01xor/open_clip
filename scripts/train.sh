#!/bin/bash
#SBATCH --job-name=clipkd-debug
#SBATCH --partition=gpu
#SBATCH --account=lt200394
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"    
echo "Time: $(date)"

# -------------------------------
# Experiment configuration
# -------------------------------

BASE_DIR="/project/lt200394-thllmV/mkd-exp"

STUDENT="ViT-T-16"
TEACHER="ViT-B-32"
TEACHER_PRETRAINED="${BASE_DIR}/pretrained/vit_b_32_openai.safetensors"
LOSS="clipkd"
NAME="clipkd_${STUDENT}_from_${TEACHER}"

ALPHA_CKD=1.0
ALPHA_ICL=1.0
ALPHA_FD=2000.0

LR=1e-3
WD=0.1
WARMUP=5000
EPOCHS=1
BATCH_SIZE=256

# Debug dataset (10 shards)
TRAIN_DATA="${BASE_DIR}/datasets/cc12m-wds/cc12m-train-{0000..0009}.tar"
TRAIN_NUM_SAMPLES=50000

LOG_DIR="${BASE_DIR}/open_clip/exp1"
WANDB_PROJECT="multilingual-vl-kd-clipkd-exp1"

IMAGENET_VAL="${BASE_DIR}/datasets/imagenet/val"

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# -------------------------------
# Environment setup
# -------------------------------

ml Mamba/23.11.0-0
conda activate openclip_venv

cd "${BASE_DIR}/open_clip"

echo "Starting training..."

torchrun --nproc_per_node=2 -m open_clip_train.main \
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
    --workers              4 \
    --imagenet-val         "${IMAGENET_VAL}" \
    --gather-with-grad \
    --grad-checkpointing \
    --grad-clip-norm       10.0 \
    --save-frequency       1 \
    --log-every-n-steps    100 \
    --seed                 42 \
    --logs                 "${LOG_DIR}" \
    --name                 "${NAME}" \
    --wandb-project-name   "${WANDB_PROJECT}" \
    --report-to            wandb

echo "Training finished at $(date)"