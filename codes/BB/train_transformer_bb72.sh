#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

STAMP="$(date +%m%d_%H%M)"
OUT_ROOT="${OUT_ROOT:-experiments/bb72_transformer_${STAMP}}"
mkdir -p "$OUT_ROOT/ckpts" "$OUT_ROOT/logs"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MAX_STEPS="${MAX_STEPS:-16001}"
LR_SCHEDULE_STEPS="${LR_SCHEDULE_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
TARGET_BS="${TARGET_BS:-2048}"
LR="${LR:-5e-4}"
WARMUP="${WARMUP:-200}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
EVAL_SAMPLES="${EVAL_SAMPLES:-20000}"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" "$ROOT_DIR/transformer.py" train \
  --torus_l 6 --torus_m 6 --rounds 6 --p 0.005 \
  --A_x 3 --A_y 1 2 --B_x 1 2 --B_y 3 \
  --d_model 512 --n_heads 8 \
  --logical_anchor_mode representative \
  --batch_size "$BATCH_SIZE" \
  --target_bs "$TARGET_BS" \
  --max_steps "$MAX_STEPS" \
  --lr_schedule_steps "$LR_SCHEDULE_STEPS" \
  --lr "$LR" \
  --warmup "$WARMUP" \
  --eval_every "$EVAL_EVERY" \
  --eval_samples "$EVAL_SAMPLES" \
  --save_every "$SAVE_EVERY" \
  --skip_oom_probe \
  --output "$OUT_ROOT/ckpts/bb72_transformer.pt" \
  2>&1 | tee "$OUT_ROOT/logs/train_transformer_bb72.log"
