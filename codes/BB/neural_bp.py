#!/usr/bin/env python3
"""Neural-BP decoder for bivariate-bicycle code detector-error models."""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
from functools import reduce

import numpy as np
import scipy.sparse
import stim
from scipy.sparse import identity, kron
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import MessagePassing


def download_from_hf(repo_id: str, filename: str, cache_dir: str | None = None) -> str:
    """Download a checkpoint from the Hugging Face Hub and return the local path.

    Args:
        repo_id: Hugging Face Hub repository ID (e.g. ``Dreamworldsmile/ntu-surface-code-decoder``).
        filename: File path within the repository (e.g. ``bb/neural_bp_bb72.pt``).
        cache_dir: Optional custom cache directory. Defaults to
            ``~/.cache/huggingface/hub/``.

    Returns:
        Absolute path to the downloaded file on local disk.
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)


# ============================================================================
# BB code construction
# ============================================================================


def row_echelon(mat, reduced=False):
    m, n = np.shape(mat)
    # Don't do "m<=n" check, allow over-complete matrices
    mat = np.copy(mat)
    # Convert to bool for faster arithmetics
    mat = mat.astype(bool)
    transform = np.identity(m).astype(bool)
    pivot_row = 0
    pivot_cols = []

    # Allow all-zero column. Row operations won't induce all-zero columns, if they are not present originally.
    # The make_systematic method will swap all-zero columns with later non-all-zero columns.
    # Iterate over cols, for each col find a pivot (if it exists)
    for col in range(n):
        # Select the pivot - if not in this row, swap rows to bring a 1 to this row, if possible
        if not mat[pivot_row, col]:
            # Find a row with a 1 in this column
            swap_row_index = pivot_row + np.argmax(mat[pivot_row:m, col])
            # If an appropriate row is found, swap it with the pivot. Otherwise, all zeroes - will loop to next col
            if mat[swap_row_index, col]:
                # Swap rows
                mat[[swap_row_index, pivot_row]] = mat[[pivot_row, swap_row_index]]
                # Transformation matrix update to reflect this row swap
                transform[[swap_row_index, pivot_row]] = transform[[pivot_row, swap_row_index]]

        if mat[pivot_row, col]:  # will evaluate to True if this column is not all-zero
            if not reduced:  # clean entries below the pivot
                elimination_range = [k for k in range(pivot_row + 1, m)]
            else:  # clean entries above and below the pivot
                elimination_range = [k for k in range(m) if k != pivot_row]
            for idx_r in elimination_range:
                if mat[idx_r, col]:
                    mat[idx_r] ^= mat[pivot_row]
                    transform[idx_r] ^= transform[pivot_row]
            pivot_row += 1
            pivot_cols.append(col)

        if pivot_row >= m:  # no more rows to search
            break

    rank = pivot_row
    row_ech_form = mat.astype(int)

    return [row_ech_form, rank, transform.astype(int), pivot_cols]


def kernel(mat):
    transpose = mat.T
    m, _ = transpose.shape
    _, rank, transform, pivot_cols = row_echelon(transpose)
    ker = transform[rank:m]
    return ker, rank, pivot_cols


class css_code:  # a refactored version of Roffe's package
    # do as less row echelon form calculation as possible.
    def __init__(
        self, hx=np.array([[]]), hz=np.array([[]]), name=None, name_prefix="", check_css=False
    ):

        self.hx = hx  # hx pcm
        self.hz = hz  # hz pcm

        self.lx = np.array([[]])  # x logicals
        self.lz = np.array([[]])  # z logicals

        self.N = np.nan  # block length
        self.K = np.nan  # code dimension
        self.L = np.nan  # max column weight
        self.Q = np.nan  # max row weight

        _, nx = self.hx.shape
        _, nz = self.hz.shape

        assert nx == nz, "hx and hz should have equal number of columns!"
        assert nx != 0, "number of variable nodes should not be zero!"
        if check_css:  # For performance reason, default to False
            assert not np.any(hx @ hz.T % 2), "CSS constraint not satisfied"

        self.N = nx
        self.hx_perp, self.rank_hx, self.pivot_hx = kernel(hx)  # orthogonal complement
        self.hz_perp, self.rank_hz, self.pivot_hz = kernel(hz)
        self.hx_basis = self.hx[self.pivot_hx]  # same as calling row_basis(self.hx)
        self.hz_basis = self.hz[self.pivot_hz]  # but saves one row echelon calculation
        self.K = self.N - self.rank_hx - self.rank_hz

        self.compute_ldpc_params()
        self.compute_logicals()

        self.name = f"{name_prefix}_n{self.N}_k{self.K}" if name is None else name

    def compute_ldpc_params(self):

        # column weights
        hx_l = np.max(np.sum(self.hx, axis=0))
        hz_l = np.max(np.sum(self.hz, axis=0))
        self.L = np.max([hx_l, hz_l]).astype(int)

        # row weights
        hx_q = np.max(np.sum(self.hx, axis=1))
        hz_q = np.max(np.sum(self.hz, axis=1))
        self.Q = np.max([hx_q, hz_q]).astype(int)

    def compute_logicals(self):

        def compute_lz(ker_hx, im_hzT):
            # lz logical operators
            # lz\in ker{hx} AND \notin Im(hz.T)
            # in the below we row reduce to find vectors in kx that are not in the image of hz.T.
            log_stack = np.vstack([im_hzT, ker_hx])
            pivots = row_echelon(log_stack.T)[3]
            log_op_indices = [i for i in range(im_hzT.shape[0], log_stack.shape[0]) if i in pivots]
            log_ops = log_stack[log_op_indices]
            return log_ops

        self.lx = compute_lz(self.hz_perp, self.hx_basis)
        self.lz = compute_lz(self.hx_perp, self.hz_basis)

        return self.lx, self.lz


def create_circulant_matrix(l, pows):
    h = np.zeros((l, l), dtype=int)
    for i in range(l):
        for c in pows:
            h[(i + c) % l, i] = 1
    return h


def create_bivariate_bicycle_codes(l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows, name=None):
    S_l = create_circulant_matrix(l, [-1])
    S_m = create_circulant_matrix(m, [-1])
    x = kron(S_l, identity(m, dtype=int))
    y = kron(identity(l, dtype=int), S_m)
    A_list = [x**p for p in A_x_pows] + [y**p for p in A_y_pows]
    B_list = [y**p for p in B_y_pows] + [x**p for p in B_x_pows]
    A = reduce(lambda x, y: x + y, A_list).toarray()
    B = reduce(lambda x, y: x + y, B_list).toarray()
    hx = np.hstack((A, B))
    hz = np.hstack((B.T, A.T))
    return css_code(hx, hz, name=name, name_prefix="BB", check_css=True), A_list, B_list


def build_circuit(code, A_list, B_list, p, num_repeat, z_basis=True, use_both=False, HZH=False):
    n = code.N
    a1, a2, a3 = A_list
    b1, b2, b3 = B_list

    def nnz(m):
        a, b = m.nonzero()
        return b[np.argsort(a)]

    A1, A2, A3 = nnz(a1), nnz(a2), nnz(a3)  # a1[i,A1[i]] == 1 else 0 (each row have a entry 1)
    B1, B2, B3 = nnz(b1), nnz(b2), nnz(b3)

    A1_T, A2_T, A3_T = nnz(a1.T), nnz(a2.T), nnz(a3.T)
    B1_T, B2_T, B3_T = nnz(b1.T), nnz(b2.T), nnz(b3.T)

    # |+> ancilla: 0 ~ n/2-1. Control in CNOTs.
    X_check_offset = 0
    # L data qubits: n/2 ~ n-1.
    L_data_offset = n // 2
    # R data qubits: n ~ 3n/2-1.
    R_data_offset = n
    # |0> ancilla: 3n/2 ~ 2n-1. Target in CNOTs.
    Z_check_offset = 3 * n // 2

    p_after_clifford_depolarization = p
    p_after_reset_flip_probability = p
    p_before_measure_flip_probability = p
    p_before_round_data_depolarization = p

    def append_detector_initial(roundsid):
        detector_circuit_str = ""
        for i in range(n // 2):
            detector_circuit_str += f"DETECTOR(0, 0, {roundsid}) rec[{-n//2+i}]\n"
        detector_circuit = stim.Circuit(detector_circuit_str)
        return detector_circuit

    def append_detector_repeat(roundsid):
        detector_repeat_circuit_str = ""
        for i in range(n // 2):
            detector_repeat_circuit_str += (
                f"DETECTOR(0, 0, {roundsid}) rec[{-n//2+i}] rec[{-n-n//2+i}]\n"
            )
        detector_repeat_circuit = stim.Circuit(detector_repeat_circuit_str)
        return detector_repeat_circuit

    def append_blocks(circuit, roundsid, repeat=False):
        # Round 1
        if repeat:
            for i in range(n // 2):
                # measurement preparation errors
                circuit.append("X_ERROR", Z_check_offset + i, p_after_reset_flip_probability)
                if HZH:
                    circuit.append("X_ERROR", X_check_offset + i, p_after_reset_flip_probability)
                    circuit.append("H", [X_check_offset + i])
                    circuit.append(
                        "DEPOLARIZE1", X_check_offset + i, p_after_clifford_depolarization
                    )
                else:
                    circuit.append("Z_ERROR", X_check_offset + i, p_after_reset_flip_probability)
                circuit.append("DEPOLARIZE1", R_data_offset + i, p_before_round_data_depolarization)
        else:
            for i in range(n // 2):
                circuit.append("H", [X_check_offset + i])
                if HZH:
                    circuit.append(
                        "DEPOLARIZE1", X_check_offset + i, p_after_clifford_depolarization
                    )

        for i in range(n // 2):
            # CNOTs from R data to to Z-checks
            circuit.append("CNOT", [R_data_offset + A1_T[i], Z_check_offset + i])
            circuit.append(
                "DEPOLARIZE2",
                [R_data_offset + A1_T[i], Z_check_offset + i],
                p_after_clifford_depolarization,
            )
            # identity gate on L data
            circuit.append("DEPOLARIZE1", L_data_offset + i, p_before_round_data_depolarization)

        # tick
        circuit.append("TICK")

        # Round 2
        for i in range(n // 2):
            # CNOTs from X-checks to L data
            circuit.append("CNOT", [X_check_offset + i, L_data_offset + A2[i]])
            circuit.append(
                "DEPOLARIZE2",
                [X_check_offset + i, L_data_offset + A2[i]],
                p_after_clifford_depolarization,
            )
            # CNOTs from R data to Z-checks
            circuit.append("CNOT", [R_data_offset + A3_T[i], Z_check_offset + i])
            circuit.append(
                "DEPOLARIZE2",
                [R_data_offset + A3_T[i], Z_check_offset + i],
                p_after_clifford_depolarization,
            )

        # tick
        circuit.append("TICK")

        # Round 3
        for i in range(n // 2):
            # CNOTs from X-checks to R data
            circuit.append("CNOT", [X_check_offset + i, R_data_offset + B2[i]])
            circuit.append(
                "DEPOLARIZE2",
                [X_check_offset + i, R_data_offset + B2[i]],
                p_after_clifford_depolarization,
            )
            # CNOTs from L data to Z-checks
            circuit.append("CNOT", [L_data_offset + B1_T[i], Z_check_offset + i])
            circuit.append(
                "DEPOLARIZE2",
                [L_data_offset + B1_T[i], Z_check_offset + i],
                p_after_clifford_depolarization,
            )

        # tick
        circuit.append("TICK")

        # Round 4
        for i in range(n // 2):
            # CNOTs from X-checks to R data
            circuit.append("CNOT", [X_check_offset + i, R_data_offset + B1[i]])
            circuit.append(
                "DEPOLARIZE2",
                [X_check_offset + i, R_data_offset + B1[i]],
                p_after_clifford_depolarization,
            )
            # CNOTs from L data to Z-checks
            circuit.append("CNOT", [L_data_offset + B2_T[i], Z_check_offset + i])
            circuit.append(
                "DEPOLARIZE2",
                [L_data_offset + B2_T[i], Z_check_offset + i],
                p_after_clifford_depolarization,
            )

        # tick
        circuit.append("TICK")

        # Round 5
        for i in range(n // 2):
            # CNOTs from X-checks to R data
            circuit.append("CNOT", [X_check_offset + i, R_data_offset + B3[i]])
            circuit.append(
                "DEPOLARIZE2",
                [X_check_offset + i, R_data_offset + B3[i]],
                p_after_clifford_depolarization,
            )
            # CNOTs from L data to Z-checks
            circuit.append("CNOT", [L_data_offset + B3_T[i], Z_check_offset + i])
            circuit.append(
                "DEPOLARIZE2",
                [L_data_offset + B3_T[i], Z_check_offset + i],
                p_after_clifford_depolarization,
            )

        # tick
        circuit.append("TICK")

        # Round 6
        for i in range(n // 2):
            # CNOTs from X-checks to L data
            circuit.append("CNOT", [X_check_offset + i, L_data_offset + A1[i]])
            circuit.append(
                "DEPOLARIZE2",
                [X_check_offset + i, L_data_offset + A1[i]],
                p_after_clifford_depolarization,
            )
            # CNOTs from R data to Z-checks
            circuit.append("CNOT", [R_data_offset + A2_T[i], Z_check_offset + i])
            circuit.append(
                "DEPOLARIZE2",
                [R_data_offset + A2_T[i], Z_check_offset + i],
                p_after_clifford_depolarization,
            )

        # tick
        circuit.append("TICK")

        # Round 7
        for i in range(n // 2):
            # CNOTs from X-checks to L data
            circuit.append("CNOT", [X_check_offset + i, L_data_offset + A3[i]])
            circuit.append(
                "DEPOLARIZE2",
                [X_check_offset + i, L_data_offset + A3[i]],
                p_after_clifford_depolarization,
            )
            # Measure Z-checks
            circuit.append("X_ERROR", Z_check_offset + i, p_before_measure_flip_probability)
            circuit.append("MR", [Z_check_offset + i])

        # Z check detectors
        if z_basis:
            if repeat:
                circuit += append_detector_repeat(roundsid)
            else:
                circuit += append_detector_initial(roundsid)
        elif use_both and repeat:
            circuit += append_detector_repeat(roundsid)

        # tick
        circuit.append("TICK")

        # Round 8
        for i in range(n // 2):
            if HZH:
                circuit.append("H", [X_check_offset + i])
                circuit.append("DEPOLARIZE1", X_check_offset + i, p_after_clifford_depolarization)
                circuit.append("X_ERROR", X_check_offset + i, p_before_measure_flip_probability)
                circuit.append("MR", [X_check_offset + i])
            else:
                circuit.append("Z_ERROR", X_check_offset + i, p_before_measure_flip_probability)
                circuit.append("MRX", [X_check_offset + i])

        # X basis detector
        if not z_basis:
            if repeat:
                circuit += append_detector_repeat(roundsid)
            else:
                circuit += append_detector_initial(roundsid)
        elif use_both and repeat:
            circuit += append_detector_repeat(roundsid)

        # tick
        circuit.append("TICK")

    circuit = stim.Circuit()
    for i in range(n // 2):  # ancilla initialization
        circuit.append("R", X_check_offset + i)
        circuit.append("R", Z_check_offset + i)
        circuit.append("X_ERROR", X_check_offset + i, p_after_reset_flip_probability)
        circuit.append("X_ERROR", Z_check_offset + i, p_after_reset_flip_probability)
    for i in range(n):
        circuit.append("R" if z_basis else "RX", L_data_offset + i)
        circuit.append(
            "X_ERROR" if z_basis else "Z_ERROR", L_data_offset + i, p_after_reset_flip_probability
        )

    # begin round tick
    circuit.append("TICK")
    append_blocks(circuit, roundsid=0, repeat=False)  # encoding round

    for i in range(1, num_repeat):
        rep_circuit = stim.Circuit()
        append_blocks(rep_circuit, roundsid=i, repeat=True)
        circuit += rep_circuit

    for i in range(0, n):
        circuit.append("M" if z_basis else "MX", L_data_offset + i)

    pcm = code.hz if z_basis else code.hx
    logical_pcm = code.lz if z_basis else code.lx
    stab_detector_circuit_str = ""  # stabilizers
    for i, s in enumerate(pcm):
        nnz = np.nonzero(s)[0]  # nonzero entries in s-th row
        det_str = f"DETECTOR(0, 0, {num_repeat})"
        for ind in nnz:
            det_str += f" rec[{-n+ind}]"
        det_str += f" rec[{-n-n+i}]" if z_basis else f" rec[{-n-n//2+i}]"
        det_str += "\n"
        stab_detector_circuit_str += det_str
    stab_detector_circuit = stim.Circuit(stab_detector_circuit_str)
    circuit += stab_detector_circuit

    log_detector_circuit_str = ""  # logical operators
    for i, l in enumerate(logical_pcm):
        nnz = np.nonzero(l)[0]
        det_str = f"OBSERVABLE_INCLUDE({i})"
        for ind in nnz:
            det_str += f" rec[{-n+ind}]"
        det_str += "\n"
        log_detector_circuit_str += det_str
    log_detector_circuit = stim.Circuit(log_detector_circuit_str)
    circuit += log_detector_circuit

    return circuit


def qcc_circuit(
    error_rate=0.005,
    l=6,
    m=6,
    A_x_pows=[3],
    A_y_pows=[1, 2],
    B_x_pows=[1, 2],
    B_y_pows=[3],
    rounds=6,
    **kwargs,
):
    code, A_list, B_list = create_bivariate_bicycle_codes(
        l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows
    )
    circuit = build_circuit(
        code,
        A_list,
        B_list,
        p=error_rate,  # physical error rate
        num_repeat=rounds,  # usually set to code distance
        z_basis=True,  # whether in the z-basis or x-basis
        use_both=True,  # whether use measurement results in both basis to decode one basis
    )
    return circuit


# ============================================================================
# Neural-BP utilities
# ============================================================================

try:
    from torch_scatter import scatter
except ModuleNotFoundError:

    def scatter(src, index, dim=0, reduce="sum", dim_size=None):
        """Small torch-native fallback for the reductions used in this file."""
        if dim != 0:
            raise NotImplementedError("fallback scatter only supports dim=0")
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() else 0

        if index.dim() == 1:
            expand_shape = [index.numel()] + [1] * (src.dim() - 1)
            index = index.view(expand_shape).expand_as(src)

        out_shape = list(src.shape)
        out_shape[dim] = dim_size

        if reduce in ("sum", "add"):
            out = src.new_zeros(out_shape)
            out.scatter_add_(dim, index, src)
            return out

        if reduce == "mul":
            out = src.new_ones(out_shape)
            return out.scatter_reduce(dim, index, src, reduce="prod", include_self=True)

        raise NotImplementedError(f"fallback scatter does not support reduce={reduce!r}")


# =============================================================================
# 1. Parity Check Matrix Extraction
# =============================================================================
def dem_to_check_matrix(dem):
    """Convert a Stim DEM into parity-check and logical-observable matrices."""
    num_detectors = dem.num_detectors
    num_observables = dem.num_observables

    h_rows, h_cols = [], []
    l_rows, l_cols = [], []
    probs = []

    var_idx = 0
    for instruction in dem:
        if instruction.type == "error":
            p = instruction.args_copy()[0]
            targets = instruction.targets_copy()

            dets = []
            logits = []
            for t in targets:
                if t.is_logical_observable_id():
                    logits.append(t.val)
                elif not t.is_separator():
                    dets.append(t.val)

            if len(dets) > 0:
                for d_id in dets:
                    h_rows.append(d_id)
                    h_cols.append(var_idx)
                for l_id in logits:
                    l_rows.append(l_id)
                    l_cols.append(var_idx)
                probs.append(p)
                var_idx += 1

    num_vars = var_idx

    if num_detectors == 0 or num_vars == 0:
        raise ValueError(f"Empty matrix! Detectors: {num_detectors}, Vars: {num_vars}")

    data_h = np.ones(len(h_rows), dtype=np.uint8)
    data_l = np.ones(len(l_rows), dtype=np.uint8)

    H = scipy.sparse.csr_matrix((data_h, (h_rows, h_cols)), shape=(num_detectors, num_vars))
    L = scipy.sparse.csr_matrix((data_l, (l_rows, l_cols)), shape=(num_observables, num_vars))

    return H, L, np.array(probs)


# =============================================================================
# 2. Advanced Loss Functions
# =============================================================================
class FocalLoss(nn.Module):
    """Focal loss for sparse hard-error targets."""

    def __init__(self, alpha=0.95, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        pt = torch.exp(-bce_loss)

        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)

        focal_loss = alpha_t * (1.0 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class SyndromeConsistencyLoss(nn.Module):
    """Syndrome-consistency loss computed in the log domain."""

    def __init__(self):
        super().__init__()

    def forward(self, logits_flat, true_syndrome_flat, edge_index_v2c):
        probs = torch.sigmoid(logits_flat)  # [B*N, 1]
        v_val = 1.0 - 2.0 * probs  # in (-1, 1)

        sign_val = v_val.sign()  # +1 or -1
        log_abs_val = v_val.abs().clamp(min=0.01).log()

        var_idx, check_idx = edge_index_v2c[0], edge_index_v2c[1]

        sign_edges = sign_val[var_idx]
        pred_sign = scatter(
            sign_edges, check_idx, dim=0, reduce="mul", dim_size=true_syndrome_flat.size(0)
        )

        log_edges = log_abs_val[var_idx]
        log_pred_abs = scatter(
            log_edges, check_idx, dim=0, reduce="sum", dim_size=true_syndrome_flat.size(0)
        )
        pred_abs = torch.exp(log_pred_abs.clamp(max=0.0))

        pred_syndrome_val = pred_sign * pred_abs  # [B*M, 1]
        true_syndrome_val = 1.0 - 2.0 * true_syndrome_flat

        return F.mse_loss(pred_syndrome_val, true_syndrome_val)


# ============================================================================
# Neural-BP training
# ============================================================================


def setup_ddp():
    if "RANK" not in os.environ:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


class ScaledTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        clamp = torch.clamp(self.scale, min=0.1, max=5.0)
        return torch.tanh(x * clamp)


class NeuralBPLayer(MessagePassing):
    def __init__(self, hidden_dim=64, dropout_rate=0.0):
        super().__init__(aggr="add", flow="source_to_target")

        self.c2v_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            ScaledTanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.v2c_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            ScaledTanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.check_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.var_gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, h_v, h_c, h_c_initial, edge_index_v2c, edge_index_c2v):
        v_msg = self.propagate(edge_index_v2c, x=h_v, size=(h_v.size(0), h_c.size(0)))
        v_msg_cat = torch.cat([v_msg, h_c_initial], dim=1)
        v_msg_processed = self.v2c_mlp(v_msg_cat)
        h_c_new = self.check_gru(v_msg_processed, h_c)

        c_msg = self.propagate(edge_index_c2v, x=h_c_new, size=(h_c.size(0), h_v.size(0)))
        c_msg = self.c2v_mlp(c_msg)
        h_v_new = self.var_gru(c_msg, h_v)

        return h_v_new, h_c_new

    def message(self, x_j):
        return x_j


class NeuralBPDecoder(nn.Module):
    def __init__(self, hidden_dim=64, num_iterations=8):
        super().__init__()
        self.num_iterations = num_iterations
        self.check_encoder = nn.Linear(1, hidden_dim)
        self.var_encoder = nn.Linear(1, hidden_dim)

        self.processor = NeuralBPLayer(hidden_dim, dropout_rate=0.0)

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(0.0), nn.Linear(hidden_dim, 1)
        )
        nn.init.constant_(self.readout[-1].bias, -2.0)

        self.criterion_cls = FocalLoss(alpha=0.95, gamma=2.0)
        self.criterion_syn = SyndromeConsistencyLoss()
        self.lambda_syn = 0.2

    def forward(self, data):
        x_c_flat = data.x_c
        x_v_flat = data.x_v
        edge_index_v2c = data.edge_index_v2c
        edge_index_c2v = data.edge_index_c2v

        h_c_centered = 1.0 - 2.0 * x_c_flat
        h_c = self.check_encoder(h_c_centered)
        h_v = self.var_encoder(x_v_flat)
        h_c_initial = h_c.clone()

        for _ in range(self.num_iterations):
            h_v, h_c = self.processor(h_v, h_c, h_c_initial, edge_index_v2c, edge_index_c2v)

        output_logits = self.readout(h_v)

        batch_size = data.batch_size
        output = output_logits.view(batch_size, -1)

        if self.training:
            y = data.y
            loss_cls = self.criterion_cls(output, y)
            loss_syn = self.criterion_syn(output_logits, data.x_c, edge_index_v2c)
            loss = loss_cls + self.lambda_syn * loss_syn
            return loss, output, {"cls": loss_cls.detach(), "syn": loss_syn.detach()}

        return output


class BipartiteData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_c2v":
            return torch.tensor([[self.x_c.size(0)], [self.x_v.size(0)]])
        elif key == "edge_index_v2c":
            return torch.tensor([[self.x_v.size(0)], [self.x_c.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


class OnlineBBPDataset(IterableDataset):
    def __init__(self, H, L, probs, p, rank=0):
        super().__init__()
        self.H = torch.from_numpy(H).float()
        self.L = torch.from_numpy(L).float()
        self.probs = torch.from_numpy(probs).float()
        self.p = p
        self.rank = rank
        self.num_checks, self.num_vars = self.H.shape

        check_idx, var_idx = torch.where(self.H == 1)
        self.edge_index_c2v = torch.stack([check_idx, var_idx], dim=0)
        self.edge_index_v2c = torch.stack([var_idx, check_idx], dim=0)

        safe_probs = torch.clamp(self.probs, 1e-10, 1.0 - 1e-10)
        self.initial_llrs = torch.log((1.0 - safe_probs) / safe_probs).unsqueeze(1)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        entropy = int.from_bytes(os.urandom(4), byteorder="little")
        time_ns = time.time_ns()
        final_seed = (time_ns + entropy + worker_id * 10000 + self.rank * 100000) % (2**32 - 1)
        torch.manual_seed(final_seed)

        while True:
            e = (torch.rand(self.num_vars) < self.probs).float()
            s = torch.matmul(self.H, e) % 2
            l_flip = torch.matmul(self.L, e) % 2

            data = BipartiteData(
                x_c=s.unsqueeze(1),
                x_v=self.initial_llrs.clone(),
                y=e.unsqueeze(0),
                edge_index_c2v=self.edge_index_c2v,
                edge_index_v2c=self.edge_index_v2c,
                logical_flip=l_flip.unsqueeze(0),
                num_nodes=self.num_vars + self.num_checks,
            )
            yield data


def has_bad_grad(params):
    """Return whether any gradient contains NaN or Inf values."""
    for p in params:
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return True
    return False


def compute_grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += float(p.grad.detach().norm() ** 2)
    return math.sqrt(total)


def fmt_time(sec: float) -> str:
    """Format elapsed seconds as a compact string."""
    if sec < 60:
        return f"{sec:.0f}s"
    if sec < 3600:
        return f"{int(sec // 60)}m{int(sec % 60):02d}s"
    return f"{int(sec // 3600)}h{int((sec % 3600) // 60):02d}m"


def _is_oom(exc: BaseException) -> bool:
    """Recognize CUDA out-of-memory errors across PyTorch versions."""
    if (
        hasattr(torch, "cuda")
        and hasattr(torch.cuda, "OutOfMemoryError")
        and isinstance(exc, torch.cuda.OutOfMemoryError)
    ):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def probe_max_bs(args, H, L, probs, device, rank):
    """Probe the largest micro-batch size that fits on rank 0."""
    bs = args.batch_size
    while bs >= 1:
        torch.cuda.empty_cache()
        try:
            ds = OnlineBBPDataset(H, L, probs, args.p, rank=rank)
            loader = PyGDataLoader(ds, batch_size=bs, num_workers=0)
            data = next(iter(loader)).to(device)
            data.batch_size = data.num_graphs
            m = NeuralBPDecoder(
                hidden_dim=args.hidden_dim,
                num_iterations=args.num_iter,
            ).to(device)
            m.train()
            loss, _, _ = m(data)
            if loss.dim() > 0:
                loss = loss.mean()
            loss.backward()
            del m, data, loader, ds, loss
            torch.cuda.empty_cache()
            return bs
        except BaseException as e:
            if not _is_oom(e):
                raise
            torch.cuda.empty_cache()
            if rank == 0:
                print(f"[OOM probe] bs={bs} OOM -> halving to {bs // 2}", flush=True)
            bs //= 2
    raise RuntimeError("[OOM probe] even bs=1 does not fit")


def run_training(args):
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"\n=== Neural-BP for BB Codes (N={args.block_size}) ===")
        print(f"    clip=1.0 | num_iter={args.num_iter}")

    dem = stim.DetectorErrorModel.from_file(args.dem_path).flattened()
    H, L, probs = dem_to_check_matrix(dem)
    H, L = H.toarray(), L.toarray()
    if rank == 0:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    original_bs = args.batch_size
    if rank == 0:
        probed_bs = probe_max_bs(args, H, L, probs, device, rank)
    else:
        probed_bs = args.batch_size
    if dist.is_initialized():
        bs_tensor = torch.tensor([probed_bs], device=device, dtype=torch.long)
        dist.broadcast(bs_tensor, src=0)
        probed_bs = int(bs_tensor.item())
    args.batch_size = probed_bs
    if rank == 0 and args.batch_size != original_bs:
        print(f"--> [OOM probe] batch_size {original_bs} -> {args.batch_size}", flush=True)

    worker_per_gpu = args.num_workers if args.num_workers is not None else max(4, 50 // world_size)
    val_workers = args.val_workers if args.val_workers is not None else 4
    if rank == 0:
        print(f"--> [Performance] train_workers={worker_per_gpu}, val_workers={val_workers}")

    train_loader = PyGDataLoader(
        OnlineBBPDataset(H, L, probs, args.p, rank=rank),
        batch_size=args.batch_size,
        num_workers=worker_per_gpu,
        pin_memory=True,
    )
    val_loader = PyGDataLoader(
        OnlineBBPDataset(H, L, probs, args.p, rank=rank),
        batch_size=args.batch_size,
        num_workers=val_workers,
        pin_memory=True,
    )

    model = NeuralBPDecoder(
        hidden_dim=args.hidden_dim,
        num_iterations=args.num_iter,
    ).to(device)

    best_acc = -1.0

    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            print(f"--> Transfer weights from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        clean = {k.replace("module.", ""): v for k, v in ckpt.items()}
        model.load_state_dict(clean, strict=True)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_lambda(current_step):
        warmup_steps = 100
        decay_steps = args.max_steps
        min_lr_ratio = 0.05
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        if current_step < decay_steps:
            progress = float(current_step - warmup_steps) / float(
                max(1, decay_steps - warmup_steps)
            )
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    model.train()
    ACCUM = max(1, args.target_bs // (args.batch_size * world_size))
    micro = 0
    iterator = iter(train_loader)
    update_step = 0
    L_tensor = torch.from_numpy(L).float().to(device)

    if rank == 0:
        print(
            f"--> Target Total BS: {args.target_bs} | Micro BS: {args.batch_size} | World Size: {world_size} | ACCUM: {ACCUM}"
        )

    running_cls = 0.0
    running_syn = 0.0
    running_loss = 0.0
    running_count = 0
    nan_skips = 0

    t_start = time.time()
    t_last_print = t_start

    while update_step < args.max_steps:
        try:
            data = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            data = next(iterator)

        data = data.to(device)
        data.batch_size = data.num_graphs

        my_context = (
            model.no_sync()
            if dist.is_initialized() and (micro + 1) % ACCUM != 0
            else contextlib.nullcontext()
        )

        with my_context:
            loss, _, parts = model(data)
            if loss.dim() > 0:
                loss = loss.mean()
            loss = loss / ACCUM

            if not torch.isfinite(loss):
                if rank == 0:
                    print(
                        f"[WARN] non-finite loss at update_step={update_step} micro={micro}, skipping this micro-batch",
                        flush=True,
                    )
                nan_skips += 1
                optimizer.zero_grad(set_to_none=True)
                micro = 0
                continue

            loss.backward()

        running_loss += float(loss) * ACCUM
        running_cls += float(parts["cls"])
        running_syn += float(parts["syn"])
        running_count += 1
        micro += 1

        if micro % ACCUM == 0:
            if has_bad_grad(model.parameters()):
                if rank == 0:
                    print(
                        f"[WARN] non-finite grad at update_step={update_step}, skipping", flush=True
                    )
                nan_skips += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            grad_norm_before = compute_grad_norm(model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if update_step % 100 == 0:
                block_acc, per_log_mean, per_log_worst, per_logical = validate(
                    model, val_loader, L_tensor, device, 50000, world_size
                )
                metrics = torch.tensor(
                    [block_acc, per_log_mean, per_log_worst] + per_logical.tolist(),
                    device=device,
                )
                if dist.is_initialized():
                    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                    metrics /= world_size

                if rank == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    avg_loss = running_loss / max(running_count, 1)
                    avg_cls = running_cls / max(running_count, 1)
                    avg_syn = running_syn / max(running_count, 1)

                    now = time.time()
                    elapsed_total = now - t_start
                    if update_step > 0:
                        delta = now - t_last_print
                        rate = 100.0 / max(delta, 1e-6)
                        eta = (args.max_steps - update_step) / max(rate, 1e-6)
                        rate_str = f"{rate:.2f}s/s"
                        eta_str = fmt_time(eta)
                    else:
                        rate_str = "  --- "
                        eta_str = " --- "
                    t_last_print = now
                    wall = time.strftime("%H:%M:%S")

                    cur_iter = (model.module if dist.is_initialized() else model).num_iterations
                    print(
                        f"[{wall}] Step {update_step:5d} | LR {lr:.2e} | iter={cur_iter} | "
                        f"Loss {avg_loss:.4f} (cls {avg_cls:.4f} syn {avg_syn:.4f}) | "
                        f"GN {grad_norm_before:.3f} | "
                        f"block {metrics[0]*100:6.3f}% | "
                        f"per-log mean {metrics[1]*100:6.3f}% worst {metrics[2]*100:6.3f}% | "
                        f"rate={rate_str} elapsed={fmt_time(elapsed_total)} ETA={eta_str} | "
                        f"skips={nan_skips}",
                        flush=True,
                    )
                    per_log_str = " ".join(
                        f"L{i}={metrics[3 + i].item() * 100:.3f}%"
                        for i in range(metrics.numel() - 3)
                    )
                    print(
                        f"           PER_LOGICAL_ACC step={update_step} {per_log_str}", flush=True
                    )
                    state = (
                        model.module.state_dict() if dist.is_initialized() else model.state_dict()
                    )
                    torch.save(state, args.output)

                    cur_acc = float(metrics[0])
                    if cur_acc > best_acc:
                        best_acc = cur_acc
                        best_path = args.output.replace(".pt", "_best.pt")
                        torch.save(state, best_path)
                        print(
                            f"  -> new best block {best_acc*100:.3f}% "
                            f"(per-log mean {metrics[1]*100:.3f}%) saved to {best_path}",
                            flush=True,
                        )

                running_loss = running_cls = running_syn = 0.0
                running_count = 0

            update_step += 1

    cleanup_ddp()


# ============================================================================
# Standalone evaluation (no DDP required)
# ============================================================================


def run_eval(args):
    """Evaluate a pre-trained Neural-BP decoder on a single GPU.

    Automatically generates the detector error model (DEM) from the BB circuit
    if it is not already present on disk.  Loads a checkpoint (from a local
    path or downloaded from the Hugging Face Hub) and computes the block
    accuracy and logical error rate.

    Args:
        args: Parsed command-line arguments with attributes:
            block_size, p, rounds (code parameters),
            hidden_dim, num_iter (model parameters),
            ckpt_path, dem_path, shots, batch_size.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Code table mapping block sizes to BB polynomial parameters.
    code_table = {
        72: dict(l=6, m=6, A_x_pows=[3], A_y_pows=[1, 2], B_x_pows=[1, 2], B_y_pows=[3], rounds=6),
        144: dict(
            l=12, m=6, A_x_pows=[3], A_y_pows=[1, 2], B_x_pows=[1, 2], B_y_pows=[3], rounds=12
        ),
    }
    if args.block_size not in code_table:
        raise SystemExit(
            f"Unsupported block_size={args.block_size}. " f"Supported: {list(code_table.keys())}"
        )
    cfg = code_table[args.block_size]
    rounds = args.rounds if args.rounds is not None else cfg["rounds"]

    print(
        f"=== Neural-BP Evaluation | "
        f"block_size={args.block_size} rounds={rounds} p={args.p} ==="
    )
    print(
        f"    device={device}  shots={args.shots}  "
        f"hidden_dim={args.hidden_dim}  num_iter={args.num_iter}"
    )

    # Ensure the detector error model exists.
    dem_path = args.dem_path or os.path.join(
        "data", "ldpc", f"{args.block_size}_12_{rounds}_{args.p}.dem"
    )
    if not os.path.exists(dem_path):
        os.makedirs(os.path.dirname(dem_path) or ".", exist_ok=True)
        print(f"    Generating DEM: {dem_path}")
        circuit = qcc_circuit(error_rate=args.p, rounds=rounds, **cfg)
        dem = circuit.detector_error_model(
            flatten_loops=True,
            decompose_errors=False,
        )
        dem.to_file(dem_path)
        print(f"    DEM saved to {dem_path}")
    else:
        print(f"    Using existing DEM: {dem_path}")

    # Load DEM and build parity-check / logical matrices.
    dem = stim.DetectorErrorModel.from_file(dem_path).flattened()
    H, L, probs = dem_to_check_matrix(dem)
    H_dense = torch.from_numpy(H.toarray()).float()
    L_dense = torch.from_numpy(L.toarray()).float()
    L_tensor = L_dense.to(device)
    n_logicals = L_dense.size(0)
    print(f"    DEM: {H_dense.size(0)} checks × {H_dense.size(1)} vars, " f"{n_logicals} logicals")

    # Instantiate the model.
    model = NeuralBPDecoder(
        hidden_dim=args.hidden_dim,
        num_iterations=args.num_iter,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    model params: {n_params / 1e3:.1f}K")

    # Load checkpoint.
    print(f"    loading checkpoint: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=True)
    clean = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(clean, strict=True)
    model.eval()

    # Build online dataset.
    ds = OnlineBBPDataset(H, L, probs, p=args.p, rank=0)
    loader = PyGDataLoader(ds, batch_size=args.batch_size, num_workers=0)

    # Inference loop.
    block_correct = 0
    total = 0
    per_logical_correct = None
    t_start = time.time()
    print_step = max(args.shots // 10, 1) if args.shots >= 10 else 1

    with torch.no_grad():
        for data in loader:
            if total >= args.shots:
                break
            data = data.to(device)
            data.batch_size = data.num_graphs
            batch_n = data.num_graphs
            if total + batch_n > args.shots:
                # Trim the last batch.
                excess = total + batch_n - args.shots
                keep = batch_n - excess
                if keep <= 0:
                    continue
                data = data[:keep]
                data.batch_size = keep
                batch_n = keep

            outputs = model(data)
            predicted = (outputs > 0).float()
            predicted_logical_flip = torch.matmul(predicted, L_tensor.t()) % 2
            true_logical_flip = data.logical_flip.view(-1, n_logicals)

            match = (predicted_logical_flip == true_logical_flip).float()
            block_correct += match.all(dim=1).sum().item()
            per_logical_correct = (
                match.sum(dim=0)
                if per_logical_correct is None
                else per_logical_correct + match.sum(dim=0)
            )
            total += batch_n

            if total % print_step < args.batch_size or total >= args.shots:
                pct = total / args.shots * 100
                cur_acc = block_correct / total * 100 if total > 0 else 0.0
                elapsed = time.time() - t_start
                rate = total / max(elapsed, 1e-6)
                print(
                    f"    [{total}/{args.shots} samples | {pct:.1f}%]  "
                    f"block_acc={cur_acc:.3f}%  "
                    f"rate={rate:.0f} samp/s"
                )

    # Compute final metrics.
    block_acc = block_correct / max(total, 1)
    ler = 1.0 - block_acc
    per_log_mean = (
        float(per_logical_correct.mean() / max(total, 1))
        if per_logical_correct is not None
        else float("nan")
    )
    per_log_worst = (
        float(per_logical_correct.min() / max(total, 1))
        if per_logical_correct is not None
        else float("nan")
    )

    elapsed = time.time() - t_start
    print(f"    === Results ===")
    print(f"    total_samples: {total}")
    print(f"    block_acc:     {block_acc * 100:.6f}%")
    print(f"    LER:           {ler:.6e}")
    print(f"    per_log_mean:  {per_log_mean * 100:.3f}%")
    print(f"    per_log_worst: {per_log_worst * 100:.3f}%")
    print(f"    elapsed:       {elapsed:.1f}s  " f"rate: {total / max(elapsed, 1e-6):.0f} samp/s")

    # Write CSV results.
    csv_file = f"bb{args.block_size}_neuralbp_eval_r{rounds}_p{args.p}.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, "w") as f:
            f.write(
                "block_size,rounds,p,total_samples,correct,errors,"
                "block_acc,ler,per_log_mean,per_log_worst\n"
            )
    with open(csv_file, "a") as f:
        errors = total - block_correct
        f.write(
            f"{args.block_size},{rounds},{args.p},"
            f"{total},{block_correct},{errors},"
            f"{block_acc:.8f},{ler:.8f},"
            f"{per_log_mean:.8f},{per_log_worst:.8f}\n"
        )
    print(f"    Results saved to {csv_file}")


def validate(model, loader, L_tensor, device, target_samples=50000, world_size=1):
    """Return block and per-logical accuracy metrics."""
    model.eval()
    block_correct = 0
    total = 0
    per_logical_correct = None
    target_per_rank = target_samples // world_size

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            data.batch_size = data.num_graphs
            outputs = model(data)
            predicted = (outputs > 0).float()

            predicted_logical_flip = torch.matmul(predicted, L_tensor.t()) % 2
            true_logical_flip = data.logical_flip.view(-1, L_tensor.size(0))

            match = (predicted_logical_flip == true_logical_flip).float()  # [B, K]
            block_correct += match.all(dim=1).sum().item()
            if per_logical_correct is None:
                per_logical_correct = match.sum(dim=0)  # [K]
            else:
                per_logical_correct = per_logical_correct + match.sum(dim=0)
            total += data.batch_size

            if total >= target_per_rank:
                break

    model.train()
    total = max(total, 1)
    block_acc = block_correct / total
    per_logical = (per_logical_correct / total).cpu()
    return block_acc, per_logical.mean().item(), per_logical.min().item(), per_logical


def main_train_neural_bp():

    parser = argparse.ArgumentParser()
    parser.add_argument("--block_size", type=int, default=72)
    parser.add_argument("--p", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument(
        "--num_iter", type=int, default=8, help="Fixed number of Neural-BP iterations."
    )
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dem_path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--target_bs", type=int, default=2048, help="Target effective global batch size."
    )
    parser.add_argument(
        "--num_workers", type=int, default=None, help="DataLoader training workers per rank."
    )
    parser.add_argument(
        "--val_workers", type=int, default=None, help="DataLoader validation workers per rank."
    )
    args = parser.parse_args()

    run_training(args)


def main_generate_dems():
    parser = argparse.ArgumentParser(description="Generate BB detector error models for Neural-BP.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing DEM files.")
    parser.add_argument("--out_dir", default="data/ldpc")
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="Generate only this BB block size. Defaults to all release sizes.",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=None,
        help="Generate only this physical error rate. Defaults to 0.001...0.005.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override the default round count for the selected block size.",
    )
    args = parser.parse_args()
    code_table = {
        72: dict(l=6, m=6, A_x_pows=[3], A_y_pows=[1, 2], B_x_pows=[1, 2], B_y_pows=[3], rounds=6),
        144: dict(
            l=12, m=6, A_x_pows=[3], A_y_pows=[1, 2], B_x_pows=[1, 2], B_y_pows=[3], rounds=12
        ),
    }
    if args.block_size is not None:
        if args.block_size not in code_table:
            raise SystemExit(f"unsupported block_size={args.block_size}")
        code_table = {args.block_size: code_table[args.block_size]}
    if args.rounds is not None:
        if args.block_size is None:
            raise SystemExit("--rounds requires --block_size")
        code_table[args.block_size]["rounds"] = args.rounds

    target_ps = [args.p] if args.p is not None else [0.001, 0.002, 0.003, 0.004, 0.005]
    os.makedirs(args.out_dir, exist_ok=True)
    for n, cfg in code_table.items():
        for p in target_ps:
            filename = os.path.join(args.out_dir, f"{n}_12_{cfg['rounds']}_{p}.dem")
            if os.path.exists(filename) and not args.force:
                print(f"[skip] {filename} already exists (use --force to overwrite)")
                continue
            print(f"[gen] N={n} p={p} rounds={cfg['rounds']} -> {filename}")
            circuit = qcc_circuit(error_rate=p, **cfg)
            dem = circuit.detector_error_model(flatten_loops=True, decompose_errors=False)
            dem.to_file(filename)
    print("Done.")


def main_eval_neural_bp():
    """Parse command-line arguments and launch Neural-BP evaluation.

    Supports downloading the checkpoint from the Hugging Face Hub or loading
    from a local path.  Automatically generates the detector error model if
    not already present on disk.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a pre-trained Neural-BP decoder for BB codes."
    )
    parser.add_argument(
        "--block_size", type=int, required=True, help="BB code block size (72 or 144)."
    )
    parser.add_argument(
        "--p", type=float, default=0.005, help="Physical error rate for evaluation."
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Number of syndrome extraction rounds " "(default: auto-derived from block_size).",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="Hidden dimension of the Neural-BP decoder."
    )
    parser.add_argument(
        "--num_iter", type=int, default=8, help="Number of Neural-BP message-passing iterations."
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="", help="Local path to the checkpoint (.pt file)."
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default="",
        help="Hugging Face Hub repository ID (e.g. " "Dreamworldsmile/ntu-surface-code-decoder).",
    )
    parser.add_argument(
        "--hf_filename",
        type=str,
        default="",
        help="File path within the Hugging Face Hub repository "
        "(default: bb/neural_bp_bb{block_size}.pt).",
    )
    parser.add_argument(
        "--dem_path",
        type=str,
        default="",
        help="Path to the detector error model (.dem) file. " "Generated automatically if missing.",
    )
    parser.add_argument("--shots", type=int, required=True, help="Number of evaluation samples.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for inference.")
    args = parser.parse_args()

    if args.block_size not in (72, 144):
        parser.error(f"Unsupported block_size={args.block_size}. " f"Supported: 72, 144")

    # Resolve checkpoint path from the Hugging Face Hub if requested.
    if args.hf_repo and not args.ckpt_path:
        hf_filename = args.hf_filename or f"bb/neural_bp_bb{args.block_size}.pt"
        print(f"Downloading from Hugging Face Hub: " f"{args.hf_repo}/{hf_filename}")
        args.ckpt_path = download_from_hf(args.hf_repo, hf_filename)

    if not args.ckpt_path:
        parser.error("Either --ckpt_path or --hf_repo must be provided.")
    if not os.path.exists(args.ckpt_path):
        parser.error(f"Checkpoint not found: {args.ckpt_path}")

    run_eval(args)


def main():
    command = "train"
    if len(sys.argv) > 1 and sys.argv[1] in {"train", "generate-dems", "eval"}:
        command = sys.argv.pop(1)
    if command == "train":
        main_train_neural_bp()
    elif command == "generate-dems":
        main_generate_dems()
    elif command == "eval":
        main_eval_neural_bp()
    else:
        raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main()
