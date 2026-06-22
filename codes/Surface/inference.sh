#!/bin/bash
#
# Neural decoder inference / evaluation launcher.
#
# Loads a pre-trained AlphaQubit V2 checkpoint and evaluates its logical
# error rate (LER) on surface code syndromes using multi-GPU sampling.
#
# Usage:
#   bash inference.sh --d 19 --ckpt models/Surface/d19.pth --shots 100000000 \
#       --eval_p 0.003 --batch_size 256
#
#   bash inference.sh --d 7 --hf_repo Dreamworldsmile/ntu-surface-code-decoder \
#       --shots 100000 --eval_p 0.003

set -euo pipefail

# ------------------------------------------------------------------
# 1. Argument parsing
# ------------------------------------------------------------------
D=""
CKPT_PATH=""
HF_REPO=""
SHOTS=""
EVAL_P="0.003"
BATCH_SIZE="256"

while [[ $# -gt 0 ]]; do
    case $1 in
        --d)          D="$2"; shift 2 ;;
        --ckpt)       CKPT_PATH="$2"; shift 2 ;;
        --hf_repo)    HF_REPO="$2"; shift 2 ;;
        --shots)      SHOTS="$2"; shift 2 ;;
        --eval_p)     EVAL_P="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ------------------------------------------------------------------
# 2. Validate required arguments
# ------------------------------------------------------------------
if [[ -z "$D" || -z "$SHOTS" ]]; then
    echo "Error: --d and --shots are required."
    exit 1
fi

if [[ -z "$CKPT_PATH" && -z "$HF_REPO" ]]; then
    echo "Error: Either --ckpt or --hf_repo is required."
    echo "Usage: bash inference.sh --d <d> (--ckpt <path> | --hf_repo <repo>) --shots <n>"
    exit 1
fi

if [[ -n "$CKPT_PATH" && ! -f "$CKPT_PATH" ]]; then
    echo "Error: Checkpoint not found at ${CKPT_PATH}"
    exit 1
fi

# ------------------------------------------------------------------
# 3. Conda environment activation (user-customizable)
# ------------------------------------------------------------------
CONDA_SH_PATH="${CONDA_SH_PATH:-}"

if [[ -n "$CONDA_SH_PATH" && -f "$CONDA_SH_PATH" ]]; then
    source "$CONDA_SH_PATH"
    conda activate tennis
    echo "[OK] Activated conda environment: tennis"
else
    echo "[WARN] CONDA_SH_PATH not set or not found. Skipping conda activation."
fi

# ------------------------------------------------------------------
# 4. Launch inference
# ------------------------------------------------------------------
if [[ -n "$HF_REPO" ]]; then
    echo "Launching inference: d=${D}, hf_repo=${HF_REPO}, shots=${SHOTS}, eval_p=${EVAL_P}"
    HF_ARG="--hf_repo ${HF_REPO}"
    CKPT_ARG=""
else
    echo "Launching inference: d=${D}, ckpt=${CKPT_PATH}, shots=${SHOTS}, eval_p=${EVAL_P}"
    HF_ARG=""
    CKPT_ARG="--ckpt_path ${CKPT_PATH}"
fi

python inference.py \
    --d ${D} \
    ${CKPT_ARG} \
    ${HF_ARG} \
    --shots ${SHOTS} \
    --eval_p ${EVAL_P} \
    --batch_size ${BATCH_SIZE}

echo "[Done] Inference completed. Results saved to eval_d${D}_results.csv"
