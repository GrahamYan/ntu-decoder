#!/bin/bash
#
# PyMatching baseline evaluation launcher.
#
# Runs the correlated or standard PyMatching decoder on surface code
# circuits to produce logical error rate (LER) baselines for comparison
# with the neural decoder.
#
# Usage:
#   bash baseline.sh --d 11 15 --p 0.002 --shots 5000000 --mode correlated
#   bash baseline.sh --d 7 11 15 --p 0.01 --shots 1000000 --mode standard

set -euo pipefail

# ------------------------------------------------------------------
# 1. Argument parsing
# ------------------------------------------------------------------
DISTANCES=""
P_VAL=""
SHOTS="5000000"
MODE="correlated"

while [[ $# -gt 0 ]]; do
    case $1 in
        --d)       DISTANCES="$2"; shift 2 ;;
        --p)       P_VAL="$2"; shift 2 ;;
        --shots)   SHOTS="$2"; shift 2 ;;
        --mode)    MODE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ------------------------------------------------------------------
# 2. Validate required arguments
# ------------------------------------------------------------------
if [[ -z "$DISTANCES" || -z "$P_VAL" ]]; then
    echo "Error: --d and --p are required."
    echo "Usage: bash baseline.sh --d <d1 d2 ...> --p <p> [--shots <n>] [--mode correlated|standard]"
    exit 1
fi

if [[ "$MODE" != "correlated" && "$MODE" != "standard" ]]; then
    echo "Error: --mode must be 'correlated' or 'standard'."
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
# 4. Launch baseline evaluation
# ------------------------------------------------------------------
echo "Launching baseline evaluation: d=${DISTANCES}, p=${P_VAL}, shots=${SHOTS}, mode=${MODE}"

python baseline.py \
    --d ${DISTANCES} \
    --p ${P_VAL} \
    --shots ${SHOTS} \
    --mode ${MODE}

echo "[Done] Baseline evaluation completed."
