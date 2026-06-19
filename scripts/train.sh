#!/usr/bin/env bash
#SBATCH --job-name=mclipkd-cc12m-baseline
#SBATCH --partition=gpu
#SBATCH --account=lt200394
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

echo "Job started on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Time: $(date)"

# -------------------------------
# Experiment configuration
# -------------------------------

BASE_DIR="/project/lt200394-thllmV/multilingual-clip-kd"

STUDENT="ViT-T-16"
TEACHER="ViT-B-16-SigLIP2"
TEACHER_PRETRAINED="${BASE_DIR}/open_clip/pretrained/siglip2/open_clip_model.safetensors"
LOSS="clipkd"
NAME="clipkd_ViT-T-16_from_ViT-B-16-SigLIP2_v2"

ALPHA_CKD=1.0
ALPHA_ICL=1.0
ALPHA_FD=2000.0

LR=2e-3
WD=0.1
WARMUP=2000
EPOCHS=32
BATCH_SIZE=128
WORKERS=8
SEED=42

# Full CC12M dataset (2176 shards)
TRAIN_DATA="/project/lt200394-thllmV/mkd-exp/datasets/cc12m-wds/cc12m-train-{0000..2175}.tar"
TRAIN_NUM_SAMPLES=10968539

LOG_DIR="${BASE_DIR}/open_clip/experiments/siglip2_kd"
WANDB_PROJECT="multilingual-vl-kd-siglip2"

IMAGENET_VAL="/project/lt200394-thllmV/mkd-exp/datasets/imagenet/val"
IMAGENET_A=""
IMAGENET_R=""
IMAGENET_SKETCH=""
FLICKR_VAL_IMAGES=""
FLICKR_VAL_ANNOTATIONS=""
FLICKR_VAL_SPLIT="test"
MSCOCO_VAL_IMAGES=""
MSCOCO_VAL_ANNOTATIONS=""
MSCOCO_VAL_SPLIT=""

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# -------------------------------
# Multi-node distributed setup
# -------------------------------

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=29500

echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Nodes: ${SLURM_NNODES}, GPUs per node: 4, Total GPUs: $((SLURM_NNODES * 4))"
echo "Global batch size: $((BATCH_SIZE * SLURM_NNODES * 4))"

# -------------------------------
# Environment setup
# -------------------------------

ml Mamba/23.11.0-0
eval "$(conda shell.bash hook)"
conda activate habibienv

cd "${BASE_DIR}/open_clip"
export PYTHONPATH="${BASE_DIR}/open_clip/src:${PYTHONPATH}"

CMD=(
    python -m open_clip_train.main
    --model "${STUDENT}"
    --distill-model "${TEACHER}"
    --distill-pretrained "${TEACHER_PRETRAINED}"
    --distill-loss "${LOSS}"
    --alpha-ckd-loss "${ALPHA_CKD}"
    --alpha-icl-loss "${ALPHA_ICL}"
    --alpha-fd-loss "${ALPHA_FD}"
    --train-data "${TRAIN_DATA}"
    --train-num-samples "${TRAIN_NUM_SAMPLES}"
    --dataset-type webdataset
    --batch-size "${BATCH_SIZE}"
    --epochs "${EPOCHS}"
    --lr "${LR}"
    --wd "${WD}"
    --warmup "${WARMUP}"
    --workers "${WORKERS}"
    --imagenet-val "${IMAGENET_VAL}"
    --gather-with-grad
    --grad-checkpointing
    --grad-clip-norm 10.0
    --save-frequency 8
    --save-most-recent
    --log-every-n-steps 50
    --seed "${SEED}"
    --logs "${LOG_DIR}"
    --name "${NAME}"
    --wandb-project-name "${WANDB_PROJECT}"
    --report-to wandb
    --resume latest
)

if [ -n "${IMAGENET_A}" ]; then
    CMD+=(--imagenet-a "${IMAGENET_A}")
fi

if [ -n "${IMAGENET_R}" ]; then
    CMD+=(--imagenet-r "${IMAGENET_R}")
fi

if [ -n "${IMAGENET_SKETCH}" ]; then
    CMD+=(--imagenet-sketch "${IMAGENET_SKETCH}")
fi

if [ -n "${FLICKR_VAL_IMAGES}" ] && [ -n "${FLICKR_VAL_ANNOTATIONS}" ]; then
    CMD+=(
        --flickr-val-images "${FLICKR_VAL_IMAGES}"
        --flickr-val-annotations "${FLICKR_VAL_ANNOTATIONS}"
        --flickr-val-split "${FLICKR_VAL_SPLIT}"
    )
fi

if [ -n "${MSCOCO_VAL_IMAGES}" ] && [ -n "${MSCOCO_VAL_ANNOTATIONS}" ]; then
    CMD+=(
        --mscoco-val-images "${MSCOCO_VAL_IMAGES}"
        --mscoco-val-annotations "${MSCOCO_VAL_ANNOTATIONS}"
    )
    if [ -n "${MSCOCO_VAL_SPLIT}" ]; then
        CMD+=(--mscoco-val-split "${MSCOCO_VAL_SPLIT}")
    fi
fi

echo "Starting training..."
srun "${CMD[@]}"
echo "Training finished at $(date)"
