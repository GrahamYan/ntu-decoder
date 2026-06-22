#!/bin/bash
#
# Distributed training launcher for AlphaQubit V2 surface-code decoder.
#
# Supports two modes:
#   scratch  -- train from randomly initialized weights.
#   transfer -- resume from an existing checkpoint for transfer learning.
#
# Usage:
#   bash train.sh --mode scratch --d 25 --train_p 0.007 --eval_p 0.007 \
#       --target_high 0.986 --target_low 0.986 --batch_size 32 --lr 2e-5 \
#       --max_steps 150000 --output_dir ./experiments
#
#   bash train.sh --mode transfer --ckpt ./checkpoint_d23.pth --d 25 \
#       --train_p 0.007 --eval_p 0.007 --target_high 0.986 --target_low 0.986 \
#       --batch_size 32 --lr 2e-5 --max_steps 150000 --output_dir ./experiments
#
#   bash train.sh --mode transfer --hf_ckpt Dreamworldsmile/ntu-surface-code-decoder/surface/d7.pth --d 11 \
#       --train_p 0.005 --eval_p 0.005 --target_high 0.98 --target_low 0.98 \
#       --batch_size 32 --lr 3e-5 --max_steps 80000 --output_dir ./experiments

set -euo pipefail

# ------------------------------------------------------------------
# 1. Argument parsing
# ------------------------------------------------------------------
MODE=""
CKPT_PATH=""
HF_CKPT=""
D=""
TRAIN_P=""
EVAL_P=""
TARGET_HIGH=""
TARGET_LOW=""
BATCH_SIZE=""
LR=""
MAX_STEPS=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)         MODE="$2"; shift 2 ;;
        --ckpt)         CKPT_PATH="$2"; shift 2 ;;
        --hf_ckpt)      HF_CKPT="$2"; shift 2 ;;
        --d)            D="$2"; shift 2 ;;
        --train_p)      TRAIN_P="$2"; shift 2 ;;
        --eval_p)       EVAL_P="$2"; shift 2 ;;
        --target_high)  TARGET_HIGH="$2"; shift 2 ;;
        --target_low)   TARGET_LOW="$2"; shift 2 ;;
        --batch_size)   BATCH_SIZE="$2"; shift 2 ;;
        --lr)           LR="$2"; shift 2 ;;
        --max_steps)    MAX_STEPS="$2"; shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ------------------------------------------------------------------
# 2. Validate required arguments
# ------------------------------------------------------------------
if [[ "$MODE" != "scratch" && "$MODE" != "transfer" ]]; then
    echo "Error: --mode must be 'scratch' or 'transfer'."
    exit 1
fi

if [[ "$MODE" == "transfer" && -z "$CKPT_PATH" && -z "$HF_CKPT" ]]; then
    echo "Error: --ckpt or --hf_ckpt is required for transfer mode."
    exit 1
fi

if [[ -z "$D" || -z "$TRAIN_P" || -z "$EVAL_P" || -z "$TARGET_HIGH" || \
      -z "$TARGET_LOW" || -z "$BATCH_SIZE" || -z "$LR" || \
      -z "$MAX_STEPS" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: Missing required arguments."
    echo "Required: --mode --d --train_p --eval_p --target_high --target_low --batch_size --lr --max_steps --output_dir"
    exit 1
fi

# ------------------------------------------------------------------
# 3. Conda environment activation (user-customizable)
# ------------------------------------------------------------------
# Set this to the path of your conda.sh before running.
# Example: CONDA_SH_PATH="/home/user/miniconda3/etc/profile.d/conda.sh"
CONDA_SH_PATH="${CONDA_SH_PATH:-}"

if [[ -n "$CONDA_SH_PATH" && -f "$CONDA_SH_PATH" ]]; then
    source "$CONDA_SH_PATH"
    conda activate tennis
    echo "[OK] Activated conda environment: tennis"
else
    echo "[WARN] CONDA_SH_PATH not set or not found. Skipping conda activation."
fi

# ------------------------------------------------------------------
# 4. Platform environment variables
# ------------------------------------------------------------------
# These are typically injected by the cluster scheduler (e.g., SLURM).
# Set defaults for single-node debugging.
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
PET_NNODES="${PET_NNODES:-1}"
PET_NODE_RANK="${PET_NODE_RANK:-0}"

# ------------------------------------------------------------------
# 5. Output directory setup
# ------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}"
CURRENT_OUTPUT="${OUTPUT_DIR}/checkpoint_d${D}.pth"

# ------------------------------------------------------------------
# 6. Build resume argument
# ------------------------------------------------------------------
RESUME_ARG=""
if [[ "$MODE" == "transfer" ]]; then
    if [[ -n "$HF_CKPT" ]]; then
        RESUME_ARG="--hf_resume ${HF_CKPT}"
        echo "[Transfer] Will download from HF: ${HF_CKPT}"
    elif [[ -f "$CKPT_PATH" ]]; then
        RESUME_ARG="--resume ${CKPT_PATH}"
        echo "[Transfer] Will resume from: ${CKPT_PATH}"
    else
        echo "Error: Checkpoint not found at ${CKPT_PATH}"
        exit 1
    fi
else
    echo "[Scratch] Training from randomly initialized weights."
fi

# ------------------------------------------------------------------
# 7. Launch distributed training
# ------------------------------------------------------------------
echo "Launching distributed training: d=${D}, nodes=${PET_NNODES}, node_rank=${PET_NODE_RANK}"

torchrun \
    --nnodes=${PET_NNODES} \
    --node_rank=${PET_NODE_RANK} \
    --nproc_per_node=8 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    transformer.py \
    --d ${D} \
    --train_p ${TRAIN_P} \
    --eval_p ${EVAL_P} \
    --target_high ${TARGET_HIGH} \
    --target_low ${TARGET_LOW} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --max_steps ${MAX_STEPS} \
    --output ${CURRENT_OUTPUT} \
    ${RESUME_ARG}

if [ ! -f "${CURRENT_OUTPUT}" ]; then
    echo "Error: Training for d=${D} failed -- no checkpoint generated."
    exit 1
fi

echo "[Done] Training completed. Checkpoint saved to ${CURRENT_OUTPUT}"
