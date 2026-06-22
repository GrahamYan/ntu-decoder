"""Evaluation harness for pre-trained AlphaQubit V2 neural decoders.

Loads a model checkpoint and evaluates the logical error rate (LER) on
rotated surface code syndromes using multi-GPU data-parallel sampling.
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from transformer import (
    AlphaQubitV2,
    FullMapper,
    OnlineSurfaceCodeDataset,
    download_from_hf,
)


def run_single_eval(
    rank,
    world_size,
    return_dict,
    d,
    global_n,
    bs,
    ckpt_path,
    eval_p,
    seed_offset,
):
    """Evaluate LER on a single GPU rank.

    Loads the model checkpoint onto the assigned GPU, runs inference on
    the allocated share of the evaluation samples, and returns the error
    count.

    Args:
        rank: GPU device index.
        world_size: Total number of GPUs.
        return_dict: Shared dict for aggregating per-GPU results.
        d: Surface code distance.
        global_n: Total number of evaluation samples across all GPUs.
        bs: Per-GPU batch size.
        ckpt_path: Path to the model checkpoint (.pth).
        eval_p: Physical error rate for evaluation.
        seed_offset: Base seed offset for reproducibility.
    """
    device = torch.device(f"cuda:{rank}")

    # Deterministic seed to ensure orthogonal data across GPUs.
    seed = int(time.time() * 1000) % (2**32 - 1) + seed_offset + rank * 1000
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    local_n = global_n // world_size

    mapper = FullMapper(d, d)
    model = AlphaQubitV2(mapper, d_model=512, n_heads=8).to(device)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(
            {
                k.replace("_orig_mod.", "").replace("module.", ""): v
                for k, v in state_dict.items()
            },
            strict=False,
        )

    model.eval()
    dataset = OnlineSurfaceCodeDataset(d, d, eval_p, bs, is_eval=True)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=True)

    local_errors = 0
    processed = 0

    # Progress reporting interval (every 100K samples).
    print_step = 100000
    next_print = print_step

    with torch.no_grad():
        for batch in dataloader:
            if processed >= local_n:
                break
            x, y = batch[0], batch[2]
            current_bs = min(x.shape[0], local_n - processed)
            x_gpu = x[:current_bs].to(device)
            y_gpu = y[:current_bs].to(device)

            logits = model(x_gpu)
            if logits.dim() == 1:
                preds = (logits >= 0).float()
            else:
                preds = (logits[:, -1] >= 0).float()

            local_errors += (preds.view(-1) != y_gpu.view(-1)).sum().item()
            processed += current_bs

            # Real-time progress monitoring on rank 0.
            if rank == 0 and processed >= next_print:
                percent = (processed / local_n) * 100
                print(
                    f"   [Progress | d={d}] GPU 0: {processed} /"
                    f" {local_n} samples ({percent:.1f}%)"
                )
                next_print += print_step

    return_dict[rank] = {"samples": processed, "errors": local_errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate AlphaQubit V2 logical error rate."
    )
    parser.add_argument(
        "--d",
        type=int,
        required=True,
        help="Surface code distance.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        required=True,
        help="Total number of evaluation samples.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Path to the model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default="",
        help="Hugging Face repo (e.g. 'user/repo') to download checkpoint from.",
    )
    parser.add_argument(
        "--hf_filename",
        type=str,
        default="",
        help="Filename in HF repo (default: surface/d{d}.pth).",
    )
    parser.add_argument(
        "--eval_p",
        type=float,
        default=0.003,
        help="Physical error rate for evaluation (default: 0.003).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Per-GPU batch size (default: 256).",
    )
    args = parser.parse_args()

    # Resolve checkpoint path from Hugging Face if requested.
    if args.hf_repo:
        hf_filename = args.hf_filename if args.hf_filename else f"surface/d{args.d}.pth"
        print(f"Downloading from HF: {args.hf_repo}/{hf_filename}")
        args.ckpt_path = download_from_hf(args.hf_repo, hf_filename)

    if not args.ckpt_path:
        parser.error("Either --ckpt_path or --hf_repo must be provided.")

    mp.set_start_method("spawn", force=True)
    world_size = torch.cuda.device_count()

    print("=" * 60)
    print(
        f"Evaluation: d={args.d} | GPUs={world_size} |"
        f" shots={args.shots} | eval_p={args.eval_p}"
    )
    print("=" * 60)

    manager = mp.Manager()
    return_dict = manager.dict()

    mp.spawn(
        run_single_eval,
        args=(
            world_size,
            return_dict,
            args.d,
            args.shots,
            args.batch_size,
            args.ckpt_path,
            args.eval_p,
            10000,
        ),
        nprocs=world_size,
        join=True,
    )

    # Aggregate results across all GPUs.
    total_samples = sum(return_dict[i]["samples"] for i in range(world_size))
    total_errors = sum(return_dict[i]["errors"] for i in range(world_size))
    ler = total_errors / total_samples
    ler_per_round = ler / args.d

    # Write results to CSV.
    csv_file = f"eval_d{args.d}_results.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, "w") as f:
            f.write("d,total_samples,total_errors,ler,ler_per_round\n")
    with open(csv_file, "a") as f:
        f.write(
            f"{args.d},{total_samples},{total_errors},"
            f"{ler:.8f},{ler_per_round:.8f}\n"
        )

    print(f"[Complete] d={args.d} LER: {ler:.6f} |" f" Results saved to {csv_file}")
