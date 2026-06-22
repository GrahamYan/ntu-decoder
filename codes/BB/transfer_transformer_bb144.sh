#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

SOURCE_CKPT="${SOURCE_CKPT:-}"
if [[ -z "$SOURCE_CKPT" ]]; then
  SOURCE_CKPT="$(find experiments -path '*/ckpts/bb72_transformer_best.pt' -type f 2>/dev/null | sort | tail -n 1 || true)"
fi
if [[ -z "$SOURCE_CKPT" ]]; then
  echo "Set SOURCE_CKPT=/path/to/bb72_transformer_checkpoint.pt" >&2
  exit 1
fi

STAMP="$(date +%m%d_%H%M)"
OUT_ROOT="${OUT_ROOT:-experiments/bb72_to_bb144_transformer_${STAMP}}"
mkdir -p "$OUT_ROOT/ckpts" "$OUT_ROOT/logs"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MAX_STEPS="${MAX_STEPS:-15001}"
LR_SCHEDULE_STEPS="${LR_SCHEDULE_STEPS:-15000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
TARGET_BS="${TARGET_BS:-1024}"
LR="${LR:-1e-4}"
WARMUP="${WARMUP:-200}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SAVE_EVERY="${SAVE_EVERY:-500}"
EVAL_SAMPLES="${EVAL_SAMPLES:-20000}"

echo "SOURCE_CKPT=$SOURCE_CKPT"
echo "OUT_ROOT=$OUT_ROOT"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" "$ROOT_DIR/transformer.py" train \
  --torus_l 12 --torus_m 6 --rounds 12 --p 0.005 \
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
  --resume "$SOURCE_CKPT" \
  --skip_oom_probe \
  --output "$OUT_ROOT/ckpts/bb144_transformer.pt" \
  2>&1 | tee "$OUT_ROOT/logs/transfer_transformer_bb144.log"
