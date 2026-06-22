#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

METHOD="${METHOD:-bposd}"
P_VALUES="${P_VALUES:-0.005 0.004 0.003 0.002 0.001}"
NUM_SAMPLES="${NUM_SAMPLES:-50000}"
DEM_DIR="${DEM_DIR:-data/dems}"
OUT_DIR="${OUT_DIR:-experiments/baselines}"
SEED="${SEED:-20260528}"

mkdir -p "$OUT_DIR"

case "$METHOD" in
  bposd)
    WORKERS="${WORKERS:-24}"
    CHUNK_SAMPLES="${CHUNK_SAMPLES:-1000}"
    MAX_ITER="${MAX_ITER:-12}"
    OSD_ORDER="${OSD_ORDER:-10}"
    OUT="${OUT:-$OUT_DIR/bposd_bb72.csv}"

    python "$ROOT_DIR/baseline.py" bposd \
      --dem_dir "$DEM_DIR" \
      --ps $P_VALUES \
      --num_samples "$NUM_SAMPLES" \
      --workers "$WORKERS" \
      --chunk_samples "$CHUNK_SAMPLES" \
      --max_iter "$MAX_ITER" \
      --osd_order "$OSD_ORDER" \
      --seed "$SEED" \
      --out "$OUT"
    ;;

  relaybp)
    PRESET="${PRESET:-2d-reduced}"
    CHUNK_SIZE="${CHUNK_SIZE:-10000}"
    OUT="${OUT:-$OUT_DIR/relaybp_${PRESET}_bb72.csv}"

    python "$ROOT_DIR/baseline.py" relaybp \
      --preset "$PRESET" \
      --dem_dir "$DEM_DIR" \
      --ps $P_VALUES \
      --num_samples "$NUM_SAMPLES" \
      --chunk_size "$CHUNK_SIZE" \
      --seed "$SEED" \
      --out "$OUT"
    ;;

  *)
    echo "Unknown METHOD=$METHOD; expected bposd or relaybp" >&2
    exit 1
    ;;
esac
