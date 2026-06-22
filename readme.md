# Efficient Foundation Decoders for Fault-Tolerant Quantum Computing

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://grahamyan.github.io/ntu-decoder/)
[![Hugging Face Hub](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/Dreamworldsmile/ntu-surface-code-decoder)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official implementation of **Neural Transfer Unification (NTU)**, an
architecture-agnostic transfer-learning framework for scalable neural quantum
error correction (QEC) decoders. By exploiting the algebraic scale invariance
inherent in structured QEC code families, NTU enables error knowledge learned on
small codes to transfer directly to large-scale fault-tolerant regimes —
dramatically reducing training overhead and eliminating the cold-start problem.

📄 **Paper**: *Efficient Foundation Decoders for Fault-Tolerant Quantum Computing*
🌐 **Project page**: [https://grahamyan.github.io/ntu-decoder/](https://grahamyan.github.io/ntu-decoder/)
🤗 **Model weights**: [Dreamworldsmile/ntu-surface-code-decoder](https://huggingface.co/Dreamworldsmile/ntu-surface-code-decoder)

---

## Overview

Neural decoders for QEC process syndrome data to infer whether a logical error has
occurred. Training high-capacity neural decoders directly at large code distances
is prohibitively expensive: models initialized from scratch encounter prolonged
cold-start plateaus where accuracy remains near random guessing for thousands of
optimization steps.

NTU addresses this bottleneck through cross-distance transfer. Because the local
topological neighborhood of each detector remains invariant under code scaling
(spatial, temporal, and scale invariance), the heavy feature-extraction backbone
can be pre-trained on a small code and seamlessly transferred to larger codes.
Only the lightweight global decision layer requires brief fine-tuning, reducing
the computational scaling exponent from Γ ≈ 1.64 (scratch) to Γ ≈ 1.51 (NTU).

The framework is instantiated with two backbone architectures:

- **NTU-Transformer** — Transformer-based decoder with interleaved
  GRU and self-attention layers, QEC-aware 2D rotary position embeddings (RoPE),
  and a cross-attention logical readout module.
- **NTU-Neural-BP** — Graph-neural-network belief-propagation decoder operating
  on the Tanner graph of the code with gated recurrent message updates.

Evaluated on two code families:

| Code family | Distances / Sizes | Decoder |
|---|---|---|
| Rotated surface code | *d* = 7, 11, 15, 19, 25 | NTU-Transformer |
| Bivariate-bicycle (BB) code | [[72, 12, 6]], [[144, 12, 12]] | NTU-Transformer, NTU-Neural-BP |

For surface codes under circuit-level depolarizing noise, NTU-Transformer
surpasses standard PyMatching at *d* = 25 within a ~10³ GPU-hour training budget.
For the [[72, 12, 6]] BB code, it outperforms BP+OSD across all tested physical
error rates and remains competitive with multi-stage Relay BP. Transfer from
[[72, 12, 6]] to [[144, 12, 12]] reaches 93.1% block accuracy within 2,500
steps (NTU-Transformer) and 95.3% within 500 steps (NTU-Neural-BP).

---

## Repository Structure

```
ntu-decoder/
├── inference.sh                          ← Unified inference launcher
├── readme.md                             ← This file
├── requirements.txt                      ← Pip dependency list
├── environment.yml                       ← Conda environment specification
├── codes/
│   ├── Surface/
│   │   ├── transformer.py                ← NTU-Transformer (surface code)
│   │   ├── inference.py                  ← Multi-GPU evaluation harness
│   │   ├── baseline.py                   ← PyMatching baselines
│   │   ├── train.sh                      ← Training launcher
│   │   ├── inference.sh                  ← Inference launcher
│   │   └── baseline.sh                   ← Baseline launcher
│   └── BB/
│       ├── transformer.py                ← NTU-Transformer (BB code)
│       ├── neural_bp.py                  ← NTU-Neural-BP decoder
│       ├── baseline.py                   ← BP-OSD & Relay BP baselines
│       ├── train_transformer_bb72.sh     ← BB72 Transformer training
│       ├── transfer_transformer_bb144.sh ← BB144 transfer learning
│       ├── train_neural_bp.sh            ← Neural-BP training
│       └── run_baseline_bb72.sh          ← BB72 baseline launcher
└── webpage/                              ← Project page (GitHub Pages)
```

---

## Installation

### Requirements

- Python ≥ 3.10
- CUDA-compatible GPU (recommended; CPU-only inference is also supported)

### Option 1 — Conda (recommended)

```bash
git clone https://github.com/GrahamYan/ntu-decoder.git
cd ntu-decoder
conda env create -f environment.yml
conda activate tennis
```

To update an existing environment:

```bash
conda env update -f environment.yml --prune
```

### Option 2 — pip

First install PyTorch matching your CUDA version from
[pytorch.org](https://pytorch.org/get-started/locally/), then:

```bash
pip install -r requirements.txt
```

### Optional dependencies

| Package | Needed for |
|---|---|
| `ldpc` | BP-OSD baseline (BB code) |
| `relay_bp` | Relay BP baseline (BB code) |

### Verifying the installation

```bash
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA', torch.cuda.is_available())"
python -c "import stim; print('Stim', stim.__version__)"
python -c "from huggingface_hub import hf_hub_download; print('Hugging Face Hub OK')"
```

---

## Quick Start — Inference

The top-level `inference.sh` script provides a unified interface for evaluating
pre-trained decoders. It automatically downloads the required checkpoint from the
Hugging Face Hub unless a local path is given.

### Surface code

```bash
# Download d=7 checkpoint from the Hub and evaluate on 100K samples.
bash inference.sh --code surface --d 7 \
    --hf_repo Dreamworldsmile/ntu-surface-code-decoder \
    --shots 100000 --eval_p 0.003

# Use a local checkpoint.
bash inference.sh --code surface --d 19 \
    --ckpt models/Surface/d19.pth \
    --shots 100000000 --eval_p 0.003
```

### BB code

```bash
# NTU-Transformer on [[72,12,6]].
bash inference.sh --code bb --model transformer --block_size 72 \
    --shots 100000 --p 0.005

# NTU-Neural-BP on [[72,12,6]].
bash inference.sh --code bb --model neural_bp --block_size 72 \
    --shots 100000 --p 0.005

# With a local checkpoint.
bash inference.sh --code bb --model transformer --block_size 72 \
    --ckpt bb72_transformer.pt --shots 100000 --p 0.005
```

### Available pre-trained checkpoints

| Code | Distance / Size | Architecture | Hugging Face Hub path |
|---|---|---|---|
| Surface | *d* = 7 | NTU-Transformer | `surface/d7.pth` |
| Surface | *d* = 11 | NTU-Transformer | `surface/d11.pth` |
| Surface | *d* = 15 | NTU-Transformer | `surface/d15.pth` |
| Surface | *d* = 19 | NTU-Transformer | `surface/d19.pth` |
| Surface | *d* = 23 | NTU-Transformer | `surface/d23.pth` |
| Surface | *d* = 25 | NTU-Transformer | `surface/d25.pth` |
| BB | [[72, 12, 6]] | NTU-Transformer | `bb/bb72_transformer.pt` |
| BB | [[72, 12, 6]] | NTU-Neural-BP | `bb/neural_bp_bb72.pt` |

---

## Training

### Surface code

```bash
cd codes/Surface

# Train from scratch on d=7.
bash train.sh --mode scratch --d 7 --train_p 0.005 --eval_p 0.003 \
    --target_high 0.90 --target_low 0.10 \
    --batch_size 256 --lr 2e-4 --max_steps 100000 \
    --output_dir ./output/d7

# Transfer from d=7 to d=11.
bash train.sh --mode transfer --d 11 \
    --ckpt ./output/d7/ckpts/model_best.pt --train_p 0.005 --eval_p 0.003 \
    --target_high 0.90 --target_low 0.10 \
    --batch_size 128 --lr 5e-5 --max_steps 50000 \
    --output_dir ./output/d11
```

### BB code

```bash
cd codes/BB

# Train NTU-Transformer on BB72.
bash train_transformer_bb72.sh

# Transfer from BB72 to BB144.
bash transfer_transformer_bb144.sh

# Train NTU-Neural-BP.
BLOCK_SIZE=72 bash train_neural_bp.sh
```

---

## Baselines

### Surface code — PyMatching

```bash
cd codes/Surface
bash baseline.sh --d 7 --p 0.003 --shots 100000 --mode correlated
```

### BB code — BP-OSD & Relay BP

```bash
cd codes/BB

# BP-OSD.
METHOD=bposd bash run_baseline_bb72.sh

# Relay BP.
METHOD=relaybp bash run_baseline_bb72.sh
```

---

## Authors

[Ge Yan](https://grahamyan.github.io)<sup>1</sup>,
Shanchuan Li<sup>1, 2</sup>,
Shiyi Xiao<sup>1, 3</sup>,
Pengyue Ma<sup>1</sup>,
Hanyan Cao<sup>4</sup>,
[Feng Pan](https://scholar.google.com/citations?user=Vp6hFhUAAAAJ)<sup>4,\*</sup>,
[Yuxuan Du](https://yuxuan-du.github.io)<sup>1,\*</sup>

<sup>1</sup> College of Computing and Data Science, Nanyang Technological University, Singapore<br>
<sup>2</sup> Department of Electrical Engineering and Computer Science, Tokyo University of Agriculture and Technology, Japan<br>
<sup>3</sup> School of Artificial Intelligence, Shanghai Jiao Tong University, China<br>
<sup>4</sup> Science, Mathematics and Technology Cluster, Singapore University of Technology and Design, Singapore

<small><sup>\*</sup> Corresponding authors</small>

---

## Citation

If you use this code or the NTU framework in your research, please cite:

```bibtex
@article{ntu2026,
  title={Efficient Foundation Decoders for Fault-Tolerant Quantum Computing},
  author={Yan, Ge and Li, Shanchuan and Xiao, Shiyi and Ma, Pengyue and
          Cao, Hanyan and Pan, Feng and Du, Yuxuan},
  year={2026},
}
```

---

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).
