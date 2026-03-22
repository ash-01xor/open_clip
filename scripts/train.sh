#!/bin/bash
#SBATCH --job-name=clipkd-cc12m-baseline
#SBATCH --partition=gpu
#SBATCH --account=lt200394
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
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
NAME="clipkd_${STUDENT}_from_${TEACHER}_${SLURM_JOB_ID}"

ALPHA_CKD=1.0
ALPHA_ICL=1.0
ALPHA_FD=2000.0

LR=2e-3
WD=0.1
WARMUP=2000
EPOCHS=32
BATCH_SIZE=128 

# Full CC12M dataset (2176 shards)
TRAIN_DATA="${BASE_DIR}/datasets/cc12m-wds/cc12m-train-{0000..2175}.tar"
TRAIN_NUM_SAMPLES=10968539

LOG_DIR="${BASE_DIR}/open_clip/experiments/$(date +%Y%m%d_%H%M%S)"
WANDB_PROJECT="multilingual-vl-kd-clipkd-exp1"

IMAGENET_VAL="${BASE_DIR}/datasets/imagenet/val"

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# -------------------------------
# Multi-node distributed setup
# -------------------------------

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Nodes: ${SLURM_NNODES}, GPUs per node: 4, Total GPUs: $((SLURM_NNODES * 4))"
echo "Global batch size: $((BATCH_SIZE * SLURM_NNODES * 4))"


# -------------------------------
# Environment setup
# -------------------------------

ml Mamba/23.11.0-0
eval "$(conda shell.bash hook)"
conda activate openclip_venv

cd "${BASE_DIR}/open_clip"
export PYTHONPATH="${BASE_DIR}/open_clip/src:${PYTHONPATH}"

echo "Starting training..."

srun python -m open_clip_train.main \
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
    --workers              8 \
    --imagenet-val         "${IMAGENET_VAL}" \
    --gather-with-grad \
    --grad-checkpointing \
    --grad-clip-norm       10.0 \
    --save-frequency        8 \
    --save-most-recent \
    --log-every-n-steps    50 \
    --seed                 42 \
    --logs                 "${LOG_DIR}" \
    --name                 "${NAME}" \
    --wandb-project-name   "${WANDB_PROJECT}" \
    --report-to            wandb

echo "Training finished at $(date)"