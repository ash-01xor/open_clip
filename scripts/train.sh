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
IMAGENET_SKETCH=""  # optional: set to /path/to/imagenet-sketch
FLICKR_VAL_IMAGES=""  # optional: set to /path/to/flickr30k/images
FLICKR_VAL_ANNOTATIONS=""  # optional: set to Flickr30k Karpathy/COCO-style JSON, CSV, or TSV
FLICKR_VAL_SPLIT="test"
MSCOCO_VAL_IMAGES=""  # optional: set to /path/to/coco/val2014 or val2017
MSCOCO_VAL_ANNOTATIONS=""  # optional: set to COCO captions/Karpathy JSON, CSV, or TSV
MSCOCO_VAL_SPLIT=""

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

srun "${CMD[@]}"
