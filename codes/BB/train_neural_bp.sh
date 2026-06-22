#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

BLOCK_SIZE="${BLOCK_SIZE:-72}"
P="${P:-0.005}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
MAX_STEPS="${MAX_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-}"
TARGET_BS="${TARGET_BS:-2048}"
LR="${LR:-3e-4}"

if [[ "$BLOCK_SIZE" == "72" ]]; then
  ROUNDS="${ROUNDS:-6}"
  NUM_ITER="${NUM_ITER:-12}"
  BATCH_SIZE="${BATCH_SIZE:-256}"
elif [[ "$BLOCK_SIZE" == "144" ]]; then
  ROUNDS="${ROUNDS:-12}"
  NUM_ITER="${NUM_ITER:-24}"
  BATCH_SIZE="${BATCH_SIZE:-128}"
else
  echo "Unsupported BLOCK_SIZE=$BLOCK_SIZE" >&2
  exit 1
fi

DEM_PATH="${DEM_PATH:-data/ldpc/${BLOCK_SIZE}_12_${ROUNDS}_${P}.dem}"
OUT_DIR="${OUT_DIR:-experiments/neural_bp_n${BLOCK_SIZE}_p${P}}"
mkdir -p "$OUT_DIR"

if [[ ! -f "$DEM_PATH" ]]; then
  echo "DEM file not found: $DEM_PATH"
  echo "Generating BB detector error model: block_size=$BLOCK_SIZE p=$P rounds=$ROUNDS"
  python "$ROOT_DIR/neural_bp.py" generate-dems \
    --out_dir "$(dirname "$DEM_PATH")" \
    --block_size "$BLOCK_SIZE" \
    --p "$P" \
    --rounds "$ROUNDS"
fi

if [[ ! -f "$DEM_PATH" ]]; then
  echo "DEM generation did not create expected file: $DEM_PATH" >&2
  exit 1
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_GPUS="$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')"
else
  NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
fi
[[ "$NUM_GPUS" -eq 0 ]] 2>/dev/null && NUM_GPUS=1

torchrun --standalone --nproc_per_node="$NUM_GPUS" "$ROOT_DIR/neural_bp.py" train \
  --block_size "$BLOCK_SIZE" \
  --p "$P" \
  --lr "$LR" \
  --hidden_dim "$HIDDEN_DIM" \
  --num_iter "$NUM_ITER" \
  --max_steps "$MAX_STEPS" \
  --batch_size "$BATCH_SIZE" \
  --target_bs "$TARGET_BS" \
  --dem_path "$DEM_PATH" \
  --output "$OUT_DIR/neural_bp_bb${BLOCK_SIZE}.pt" \
  2>&1 | tee "$OUT_DIR/train_$(date +%m%d_%H%M).log"
