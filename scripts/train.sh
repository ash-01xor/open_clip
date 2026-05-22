#!/usr/bin/env bash
#SBATCH --job-name=openclip-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

STUDENT="ViT-B-32"
TEACHER="ViT-L-14"
TEACHER_PRETRAINED="openai"   # or /path/to/teacher_checkpoint.pt
TRAIN_DATA="/path/to/train_data"
TRAIN_NUM_SAMPLES=0           # set >0 for webdataset epoch sizing
IMAGENET_VAL=""  # optional: set to /path/to/imagenet/val
IMAGENET_A=""    # optional: set to /path/to/imagenet-a
IMAGENET_R=""    # optional: set to /path/to/imagenet-r

# Distillation: default | clipkd
DISTILL_LOSS="clipkd"
ALPHA_CKD=1.0
ALPHA_ICL=1.0
ALPHA_FD=2000.0

# Training
BATCH_SIZE=128
EPOCHS=32
LR=1e-3
WD=0.1
WARMUP=10000
WORKERS=8
PRECISION="amp"
SEED=42

# Logging
RUN_NAME="clipkd_${STUDENT}_from_${TEACHER}"
LOG_DIR="./logs"
WANDB_PROJECT="openclip"

CMD=(
    python -m open_clip_train.main
    --model "${STUDENT}"
    --distill-model "${TEACHER}"
    --distill-pretrained "${TEACHER_PRETRAINED}"
    --distill-loss "${DISTILL_LOSS}"
    --alpha-ckd-loss "${ALPHA_CKD}"
    --alpha-icl-loss "${ALPHA_ICL}"
    --alpha-fd-loss "${ALPHA_FD}"
    --train-data "${TRAIN_DATA}"
    --dataset-type webdataset
    --batch-size "${BATCH_SIZE}"
    --epochs "${EPOCHS}"
    --lr "${LR}"
    --wd "${WD}"
    --warmup "${WARMUP}"
    --workers "${WORKERS}"
    --precision "${PRECISION}"
    --seed "${SEED}"
    --logs "${LOG_DIR}"
    --name "${RUN_NAME}"
    --report-to wandb
    --wandb-project-name "${WANDB_PROJECT}"
    --save-frequency 1
    --save-most-recent
    --log-every-n-steps 50
    --gather-with-grad
    --grad-checkpointing
    --grad-clip-norm 10.0
)

if [ "${TRAIN_NUM_SAMPLES}" -gt 0 ]; then
    CMD+=(--train-num-samples "${TRAIN_NUM_SAMPLES}")
fi

if [ -n "${IMAGENET_VAL}" ]; then
    CMD+=(--imagenet-val "${IMAGENET_VAL}")
fi

if [ -n "${IMAGENET_A}" ]; then
    CMD+=(--imagenet-a "${IMAGENET_A}")
fi

if [ -n "${IMAGENET_R}" ]; then
    CMD+=(--imagenet-r "${IMAGENET_R}")
fi

srun "${CMD[@]}"
