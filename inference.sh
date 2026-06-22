#!/bin/bash
#
# NTU Decoder — Unified inference / evaluation launcher.
#
# Loads a pre-trained neural decoder checkpoint (locally or from the
# Hugging Face Hub) and evaluates its logical error rate (LER) on
# surface code or bivariate-bicycle (BB) code syndromes.
#
# Usage:
#   # Surface code (AlphaQubit V2)
#   bash inference.sh --code surface --d 19 --shots 100000000 --eval_p 0.003
#   bash inference.sh --code surface --d 7 \
#       --hf_repo Dreamworldsmile/ntu-surface-code-decoder --shots 100000
#
#   # BB code — Transformer (AlphaQubitV2_BB)
#   bash inference.sh --code bb --model transformer --block_size 72 \
#       --shots 100000 --p 0.005
#
#   # BB code — Neural Belief Propagation
#   bash inference.sh --code bb --model neural_bp --block_size 72 \
#       --shots 100000 --p 0.005

set -euo pipefail

# ------------------------------------------------------------------
# 1. Argument parsing
# ------------------------------------------------------------------
CODE=""            # "surface" or "bb"
MODEL=""           # "transformer" or "neural_bp" (BB only)
D=""               # surface code distance
BLOCK_SIZE=""      # BB code block size (72 or 144)
SHOTS=""           # total evaluation samples (required)
P=""               # physical error rate
EVAL_P=""          # surface evaluation error rate (default 0.003)
HF_REPO=""         # Hugging Face Hub repository ID
CKPT=""            # local checkpoint path
BATCH_SIZE=""      # batch size override

while [[ $# -gt 0 ]]; do
    case $1 in
        --code)        CODE="$2"; shift 2 ;;
        --model)       MODEL="$2"; shift 2 ;;
        --d)           D="$2"; shift 2 ;;
        --block_size)  BLOCK_SIZE="$2"; shift 2 ;;
        --shots)       SHOTS="$2"; shift 2 ;;
        --p)           P="$2"; shift 2 ;;
        --eval_p)      EVAL_P="$2"; shift 2 ;;
        --hf_repo)     HF_REPO="$2"; shift 2 ;;
        --ckpt)        CKPT="$2"; shift 2 ;;
        --batch_size)  BATCH_SIZE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ------------------------------------------------------------------
# 2. Validate required arguments
# ------------------------------------------------------------------
if [[ -z "$CODE" ]]; then
    echo "Error: --code is required (surface | bb)."
    exit 1
fi

if [[ "$CODE" != "surface" && "$CODE" != "bb" ]]; then
    echo "Error: --code must be 'surface' or 'bb', got '${CODE}'."
    exit 1
fi

if [[ -z "$SHOTS" ]]; then
    echo "Error: --shots is required."
    exit 1
fi

# ------------------------------------------------------------------
# 3. Defaults
# ------------------------------------------------------------------
HF_REPO="${HF_REPO:-Dreamworldsmile/ntu-surface-code-decoder}"

# ------------------------------------------------------------------
# 4. Conda environment activation (user-customizable)
# ------------------------------------------------------------------
CONDA_SH_PATH="${CONDA_SH_PATH:-}"

if [[ -n "$CONDA_SH_PATH" && -f "$CONDA_SH_PATH" ]]; then
    source "$CONDA_SH_PATH"
    conda activate tennis
    echo "[OK] Activated conda environment: tennis"
else
    echo "[WARN] CONDA_SH_PATH not set or not found. Skipping conda activation."
fi

# ==================================================================
# Surface code inference
# ==================================================================
if [[ "$CODE" == "surface" ]]; then
    if [[ -z "$D" ]]; then
        echo "Error: --d <distance> is required for surface code."
        exit 1
    fi

    EVAL_P="${EVAL_P:-0.003}"
    BATCH_SIZE="${BATCH_SIZE:-256}"

    echo "============================================================"
    echo " Surface Code Inference"
    echo " d=${D}  shots=${SHOTS}  eval_p=${EVAL_P}"
    echo "============================================================"

    if [[ -n "$CKPT" ]]; then
        if [[ ! -f "$CKPT" ]]; then
            echo "Error: Checkpoint not found at ${CKPT}"
            exit 1
        fi
        echo "Using local checkpoint: ${CKPT}"
        python codes/Surface/inference.py \
            --d "$D" \
            --ckpt_path "$CKPT" \
            --shots "$SHOTS" \
            --eval_p "$EVAL_P" \
            --batch_size "$BATCH_SIZE"
    elif [[ -n "$HF_REPO" ]]; then
        echo "Downloading from Hugging Face Hub: ${HF_REPO}"
        python codes/Surface/inference.py \
            --d "$D" \
            --hf_repo "$HF_REPO" \
            --shots "$SHOTS" \
            --eval_p "$EVAL_P" \
            --batch_size "$BATCH_SIZE"
    else
        echo "Error: Either --ckpt or --hf_repo is required."
        exit 1
    fi

    echo "[Done] Surface code inference completed."

# ==================================================================
# BB code inference
# ==================================================================
elif [[ "$CODE" == "bb" ]]; then
    if [[ -z "$BLOCK_SIZE" ]]; then
        echo "Error: --block_size <72|144> is required for BB code."
        exit 1
    fi

    if [[ "$BLOCK_SIZE" != "72" && "$BLOCK_SIZE" != "144" ]]; then
        echo "Error: --block_size must be 72 or 144, got '${BLOCK_SIZE}'."
        exit 1
    fi

    MODEL="${MODEL:-transformer}"
    if [[ "$MODEL" != "transformer" && "$MODEL" != "neural_bp" ]]; then
        echo "Error: --model must be 'transformer' or 'neural_bp', got '${MODEL}'."
        exit 1
    fi

    # Resolve BB code parameters from block size.
    if [[ "$BLOCK_SIZE" == "72" ]]; then
        L=6; M=6; ROUNDS="${ROUNDS:-6}"
        P="${P:-0.005}"
        TRANSFORMER_BATCH="${BATCH_SIZE:-64}"
        NEURALBP_BATCH="${BATCH_SIZE:-128}"
        NEURALBP_HIDDEN=64
        NEURALBP_ITER=8
    elif [[ "$BLOCK_SIZE" == "144" ]]; then
        L=12; M=6; ROUNDS="${ROUNDS:-12}"
        P="${P:-0.005}"
        TRANSFORMER_BATCH="${BATCH_SIZE:-32}"
        NEURALBP_BATCH="${BATCH_SIZE:-64}"
        NEURALBP_HIDDEN=64
        NEURALBP_ITER=12
    fi

    echo "============================================================"
    echo " BB Code Inference"
    echo " block_size=${BLOCK_SIZE}  l=${L}  m=${M}  rounds=${ROUNDS}"
    echo " model=${MODEL}  p=${P}  shots=${SHOTS}"
    echo "============================================================"

    if [[ "$MODEL" == "transformer" ]]; then
        # Build arguments for BB Transformer eval.
        TRANSFORMER_ARGS=(
            --torus_l "$L" --torus_m "$M"
            --A_x 3 --A_y 1 2 --B_x 1 2 --B_y 3
            --rounds "$ROUNDS" --p "$P"
            --d_model 512 --n_heads 8
            --logical_anchor_mode representative
            --shots "$SHOTS"
            --batch_size "$TRANSFORMER_BATCH"
        )

        if [[ -n "$CKPT" ]]; then
            if [[ ! -f "$CKPT" ]]; then
                echo "Error: Checkpoint not found at ${CKPT}"
                exit 1
            fi
            echo "Using local checkpoint: ${CKPT}"
            TRANSFORMER_ARGS+=(--ckpt_path "$CKPT")
        elif [[ -n "$HF_REPO" ]]; then
            echo "Downloading from Hugging Face Hub: ${HF_REPO}"
            TRANSFORMER_ARGS+=(--hf_repo "$HF_REPO")
        else
            echo "Error: Either --ckpt or --hf_repo is required."
            exit 1
        fi

        python codes/BB/transformer.py eval "${TRANSFORMER_ARGS[@]}"

    elif [[ "$MODEL" == "neural_bp" ]]; then
        # Build arguments for Neural-BP eval.
        NEURALBP_ARGS=(
            --block_size "$BLOCK_SIZE"
            --p "$P"
            --rounds "$ROUNDS"
            --hidden_dim "$NEURALBP_HIDDEN"
            --num_iter "$NEURALBP_ITER"
            --shots "$SHOTS"
            --batch_size "$NEURALBP_BATCH"
        )

        if [[ -n "$CKPT" ]]; then
            if [[ ! -f "$CKPT" ]]; then
                echo "Error: Checkpoint not found at ${CKPT}"
                exit 1
            fi
            echo "Using local checkpoint: ${CKPT}"
            NEURALBP_ARGS+=(--ckpt_path "$CKPT")
        elif [[ -n "$HF_REPO" ]]; then
            echo "Downloading from Hugging Face Hub: ${HF_REPO}"
            NEURALBP_ARGS+=(--hf_repo "$HF_REPO")
        else
            echo "Error: Either --ckpt or --hf_repo is required."
            exit 1
        fi

        python codes/BB/neural_bp.py eval "${NEURALBP_ARGS[@]}"
    fi

    echo "[Done] BB code inference completed."
fi
