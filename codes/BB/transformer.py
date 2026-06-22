#!/usr/bin/env python3
"""BB Transformer decoder for bivariate-bicycle code memory experiments."""
from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
from dataclasses import dataclass
from functools import reduce
from typing import List, Tuple

import numpy as np
import stim
from scipy.sparse import identity, kron
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset



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

        if mat[pivot_row, col]: # will evaluate to True if this column is not all-zero
            if not reduced: # clean entries below the pivot 
                elimination_range = [k for k in range(pivot_row + 1, m)]
            else:           # clean entries above and below the pivot
                elimination_range = [k for k in range(m) if k != pivot_row]
            for idx_r in elimination_range:
                if mat[idx_r, col]:    
                    mat[idx_r] ^= mat[pivot_row]
                    transform[idx_r] ^= transform[pivot_row]
            pivot_row += 1
            pivot_cols.append(col)

        if pivot_row >= m: # no more rows to search
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

class css_code(): # a refactored version of Roffe's package
    # do as less row echelon form calculation as possible.
    def __init__(self, hx=np.array([[]]), hz=np.array([[]]), name=None, name_prefix="", check_css=False):

        self.hx = hx # hx pcm
        self.hz = hz # hz pcm

        self.lx = np.array([[]]) # x logicals
        self.lz = np.array([[]]) # z logicals

        self.N = np.nan # block length
        self.K = np.nan # code dimension
        self.L = np.nan # max column weight
        self.Q = np.nan # max row weight

        _, nx = self.hx.shape
        _, nz = self.hz.shape

        assert nx == nz, "hx and hz should have equal number of columns!"
        assert nx != 0,  "number of variable nodes should not be zero!"
        if check_css: # For performance reason, default to False
            assert not np.any(hx @ hz.T % 2), "CSS constraint not satisfied"
        
        self.N = nx
        self.hx_perp, self.rank_hx, self.pivot_hx = kernel(hx) # orthogonal complement
        self.hz_perp, self.rank_hz, self.pivot_hz = kernel(hz)
        self.hx_basis = self.hx[self.pivot_hx] # same as calling row_basis(self.hx)
        self.hz_basis = self.hz[self.pivot_hz] # but saves one row echelon calculation
        self.K = self.N - self.rank_hx - self.rank_hz

        self.compute_ldpc_params()
        self.compute_logicals()

        self.name = f"{name_prefix}_n{self.N}_k{self.K}" if name is None else name

    def compute_ldpc_params(self):

        #column weights
        hx_l = np.max(np.sum(self.hx, axis=0))
        hz_l = np.max(np.sum(self.hz, axis=0))
        self.L = np.max([hx_l, hz_l]).astype(int)

        #row weights
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
    h = np.zeros((l,l), dtype=int)
    for i in range(l):
        for c in pows:
            h[(i+c)%l, i] = 1
    return h

def create_bivariate_bicycle_codes(l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows, name=None):
    S_l=create_circulant_matrix(l, [-1])
    S_m=create_circulant_matrix(m, [-1])
    x = kron(S_l, identity(m, dtype=int))
    y = kron(identity(l, dtype=int), S_m)
    A_list = [x**p for p in A_x_pows] + [y**p for p in A_y_pows]
    B_list = [y**p for p in B_y_pows] + [x**p for p in B_x_pows] 
    A = reduce(lambda x,y: x+y, A_list).toarray()
    B = reduce(lambda x,y: x+y, B_list).toarray()
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
    L_data_offset = n//2
    # R data qubits: n ~ 3n/2-1.
    R_data_offset = n
    # |0> ancilla: 3n/2 ~ 2n-1. Target in CNOTs.
    Z_check_offset = 3*n//2

    p_after_clifford_depolarization = p
    p_after_reset_flip_probability = p
    p_before_measure_flip_probability = p
    p_before_round_data_depolarization = p

    def append_detector_initial(roundsid):
        detector_circuit_str = ""
        for i in range(n//2):
            detector_circuit_str += f"DETECTOR(0, 0, {roundsid}) rec[{-n//2+i}]\n"
        detector_circuit = stim.Circuit(detector_circuit_str)   
        return detector_circuit
    
    def append_detector_repeat(roundsid):
        detector_repeat_circuit_str = ""
        for i in range(n//2):
            detector_repeat_circuit_str += f"DETECTOR(0, 0, {roundsid}) rec[{-n//2+i}] rec[{-n-n//2+i}]\n"
        detector_repeat_circuit = stim.Circuit(detector_repeat_circuit_str)
        return detector_repeat_circuit



    def append_blocks(circuit, roundsid, repeat=False):
        # Round 1
        if repeat:        
            for i in range(n//2):
                # measurement preparation errors
                circuit.append("X_ERROR", Z_check_offset + i, p_after_reset_flip_probability)
                if HZH:
                    circuit.append("X_ERROR", X_check_offset + i, p_after_reset_flip_probability)
                    circuit.append("H", [X_check_offset + i])
                    circuit.append("DEPOLARIZE1", X_check_offset + i, p_after_clifford_depolarization)
                else:
                    circuit.append("Z_ERROR", X_check_offset + i, p_after_reset_flip_probability)
                circuit.append("DEPOLARIZE1", R_data_offset + i, p_before_round_data_depolarization)
        else:
            for i in range(n//2):
                circuit.append("H", [X_check_offset + i])
                if HZH:
                    circuit.append("DEPOLARIZE1", X_check_offset + i, p_after_clifford_depolarization)

        for i in range(n//2):
            # CNOTs from R data to to Z-checks
            circuit.append("CNOT", [R_data_offset + A1_T[i], Z_check_offset + i])
            circuit.append("DEPOLARIZE2", [R_data_offset + A1_T[i], Z_check_offset + i], p_after_clifford_depolarization)
            # identity gate on L data
            circuit.append("DEPOLARIZE1", L_data_offset + i, p_before_round_data_depolarization)

        # tick
        circuit.append("TICK")

        # Round 2
        for i in range(n//2):
            # CNOTs from X-checks to L data
            circuit.append("CNOT", [X_check_offset + i, L_data_offset + A2[i]])
            circuit.append("DEPOLARIZE2", [X_check_offset + i, L_data_offset + A2[i]], p_after_clifford_depolarization)
            # CNOTs from R data to Z-checks
            circuit.append("CNOT", [R_data_offset + A3_T[i], Z_check_offset + i])
            circuit.append("DEPOLARIZE2", [R_data_offset + A3_T[i], Z_check_offset + i], p_after_clifford_depolarization)

        # tick
        circuit.append("TICK")

        # Round 3
        for i in range(n//2):
            # CNOTs from X-checks to R data
            circuit.append("CNOT", [X_check_offset + i, R_data_offset + B2[i]])
            circuit.append("DEPOLARIZE2", [X_check_offset + i, R_data_offset + B2[i]], p_after_clifford_depolarization)
            # CNOTs from L data to Z-checks
            circuit.append("CNOT", [L_data_offset + B1_T[i], Z_check_offset + i])
            circuit.append("DEPOLARIZE2", [L_data_offset + B1_T[i], Z_check_offset + i], p_after_clifford_depolarization)

        # tick
        circuit.append("TICK")

        # Round 4
        for i in range(n//2):
            # CNOTs from X-checks to R data
            circuit.append("CNOT", [X_check_offset + i, R_data_offset + B1[i]])
            circuit.append("DEPOLARIZE2", [X_check_offset + i, R_data_offset + B1[i]], p_after_clifford_depolarization)
            # CNOTs from L data to Z-checks
            circuit.append("CNOT", [L_data_offset + B2_T[i], Z_check_offset + i])
            circuit.append("DEPOLARIZE2", [L_data_offset + B2_T[i], Z_check_offset + i], p_after_clifford_depolarization)

        # tick
        circuit.append("TICK")

        # Round 5
        for i in range(n//2):
            # CNOTs from X-checks to R data
            circuit.append("CNOT", [X_check_offset + i, R_data_offset + B3[i]])
            circuit.append("DEPOLARIZE2", [X_check_offset + i, R_data_offset + B3[i]], p_after_clifford_depolarization)
            # CNOTs from L data to Z-checks
            circuit.append("CNOT", [L_data_offset + B3_T[i], Z_check_offset + i])
            circuit.append("DEPOLARIZE2", [L_data_offset + B3_T[i], Z_check_offset + i], p_after_clifford_depolarization)

        # tick
        circuit.append("TICK")

        # Round 6
        for i in range(n//2):
            # CNOTs from X-checks to L data
            circuit.append("CNOT", [X_check_offset + i, L_data_offset + A1[i]])
            circuit.append("DEPOLARIZE2", [X_check_offset + i, L_data_offset + A1[i]], p_after_clifford_depolarization)
            # CNOTs from R data to Z-checks
            circuit.append("CNOT", [R_data_offset + A2_T[i], Z_check_offset + i])
            circuit.append("DEPOLARIZE2", [R_data_offset + A2_T[i], Z_check_offset + i], p_after_clifford_depolarization)

        # tick
        circuit.append("TICK")

        # Round 7
        for i in range(n//2):
            # CNOTs from X-checks to L data
            circuit.append("CNOT", [X_check_offset + i, L_data_offset + A3[i]])
            circuit.append("DEPOLARIZE2", [X_check_offset + i, L_data_offset + A3[i]], p_after_clifford_depolarization)
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
        for i in range(n//2):
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
    for i in range(n//2): # ancilla initialization
        circuit.append("R", X_check_offset + i)
        circuit.append("R", Z_check_offset + i)
        circuit.append("X_ERROR", X_check_offset + i, p_after_reset_flip_probability)
        circuit.append("X_ERROR", Z_check_offset + i, p_after_reset_flip_probability)
    for i in range(n):
        circuit.append("R" if z_basis else "RX", L_data_offset + i)
        circuit.append("X_ERROR" if z_basis else "Z_ERROR", L_data_offset + i, p_after_reset_flip_probability)

    # begin round tick
    circuit.append("TICK") 
    append_blocks(circuit, roundsid = 0, repeat=False) # encoding round

    for i in range(1, num_repeat):
        rep_circuit = stim.Circuit()
        append_blocks(rep_circuit, roundsid=i, repeat=True)
        circuit += rep_circuit
        


    for i in range(0, n):
        circuit.append("M" if z_basis else "MX", L_data_offset + i)
        
    pcm = code.hz if z_basis else code.hx
    logical_pcm = code.lz if z_basis else code.lx
    stab_detector_circuit_str = "" # stabilizers
    for i, s in enumerate(pcm):
        nnz = np.nonzero(s)[0]      # nonzero entries in s-th row 
        det_str = f"DETECTOR(0, 0, {num_repeat})"
        for ind in nnz:
            det_str += f" rec[{-n+ind}]"       
        det_str += f" rec[{-n-n+i}]" if z_basis else f" rec[{-n-n//2+i}]"
        det_str += "\n"
        stab_detector_circuit_str += det_str
    stab_detector_circuit = stim.Circuit(stab_detector_circuit_str)
    circuit += stab_detector_circuit

    log_detector_circuit_str = "" # logical operators
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
    **kwargs):
    code, A_list, B_list = create_bivariate_bicycle_codes(l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows)
    circuit = build_circuit(code, A_list, B_list, 
                        p=error_rate, # physical error rate
                        num_repeat=rounds, # usually set to code distance
                        z_basis=True,   # whether in the z-basis or x-basis
                        use_both=True, # whether use measurement results in both basis to decode one basis
                       )
    return circuit




# ============================================================================
# Cartesian RoPE
# ============================================================================

class CartesianRoPE(nn.Module):
    def __init__(self, head_dim: int, base: float = 100.0):
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
        self.head_dim = head_dim
        self.half_dim = head_dim // 2
        self.quarter_dim = head_dim // 4

        inv_freq = 1.0 / (base ** (torch.arange(self.quarter_dim).float() / self.quarter_dim))
        self.register_buffer("inv_freq_i", inv_freq.clone())
        self.register_buffer("inv_freq_j", inv_freq.clone())

    def get_freqs(self, coords: torch.Tensor, device):
        """Return rotary frequencies for integer coordinate tensors."""
        i = coords[:, 0].to(device).float()
        j = coords[:, 1].to(device).float()

        fi = torch.einsum("n,k->nk", i, self.inv_freq_i)   # [N, quarter_dim]
        fj = torch.einsum("n,k->nk", j, self.inv_freq_j)

        freqs_i = torch.cat([fi, fi], dim=-1)
        freqs_j = torch.cat([fj, fj], dim=-1)
        return freqs_i, freqs_j


def apply_rope_2d_cart(q, k, freqs_i, freqs_j):
    """Apply 2D rotary phases to query and key tensors."""
    half = q.shape[-1] // 2
    q_i, q_j = q[..., :half], q[..., half:]
    k_i, k_j = k[..., :half], k[..., half:]

    fi = freqs_i.unsqueeze(0).unsqueeze(2)   # [1, N, 1, half]
    fj = freqs_j.unsqueeze(0).unsqueeze(2)

    def _rotate_half(t, f):
        t_rot = torch.cat((-t[..., t.shape[-1] // 2:], t[..., :t.shape[-1] // 2]), dim=-1)
        return t * f.cos() + t_rot * f.sin()

    q_i = _rotate_half(q_i, fi)
    k_i = _rotate_half(k_i, fi)
    q_j = _rotate_half(q_j, fj)
    k_j = _rotate_half(k_j, fj)
    return torch.cat([q_i, q_j], dim=-1), torch.cat([k_i, k_j], dim=-1)


# ============================================================================
# Transformer input mapping
# ============================================================================

@dataclass
class BBMappingInfo:
    gather_z: torch.Tensor
    valid_z: torch.Tensor
    z_neighbors: torch.Tensor
    z_hint_neighbors: torch.Tensor
    gather_x: torch.Tensor
    valid_x: torch.Tensor
    x_neighbors: torch.Tensor
    x_hint_neighbors: torch.Tensor
    spatial_coords_z: torch.Tensor
    spatial_coords_x: torch.Tensor
    xx_offsets: List[Tuple[int, int]]
    xz_offsets: List[Tuple[int, int]]
    l: int
    m: int
    lm: int
    num_t: int
    rounds: int
    total_detectors: int
    K_same: int
    K_cross: int


def _polynomial_terms(x_pows: List[int], y_pows: List[int]) -> List[Tuple[int, int]]:
    return [(a, 0) for a in x_pows] + [(0, b) for b in y_pows]


def _derive_xx_offsets(a_terms: List[Tuple[int, int]],
                       b_terms: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    offsets = set()
    for terms in (a_terms, b_terms):
        for t1 in terms:
            for t2 in terms:
                delta = (t1[0] - t2[0], t1[1] - t2[1])
                if delta != (0, 0):
                    offsets.add(delta)
    return sorted(offsets)


def _derive_xz_offsets(a_terms: List[Tuple[int, int]],
                       b_terms: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    offsets = set()
    for t_a in a_terms:
        for t_b in b_terms:
            offsets.add((t_a[0] + t_b[0], t_a[1] + t_b[1]))
    return sorted(offsets)


def _neighbors_by_offsets(coords: torch.Tensor,
                          offsets: List[Tuple[int, int]],
                          l: int,
                          m: int) -> torch.Tensor:
    out = torch.zeros((coords.shape[0], len(offsets)), dtype=torch.long)
    for i in range(coords.shape[0]):
        ci, cj = coords[i].tolist()
        for k, (di, dj) in enumerate(offsets):
            out[i, k] = ((ci + di) % l) * m + ((cj + dj) % m)
    return out


class BBMapper(nn.Module):
    """Build detector-index and spatial-neighbor buffers for BB circuits."""

    def __init__(self, l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows,
                 rounds, p=0.005):
        super().__init__()
        self.l = l
        self.m = m
        self.lm = l * m
        self.rounds = rounds
        self.p = p

        a_terms = _polynomial_terms(A_x_pows, A_y_pows)
        b_terms = _polynomial_terms(B_x_pows, B_y_pows)
        self.xx_offsets = _derive_xx_offsets(a_terms, b_terms)
        self.xz_offsets = _derive_xz_offsets(a_terms, b_terms)
        self.K_same = len(self.xx_offsets)
        self.K_cross = len(self.xz_offsets)

        code, A_list, B_list = create_bivariate_bicycle_codes(
            l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows
        )
        self.circuit = build_circuit(
            code, A_list, B_list,
            p=p, num_repeat=rounds,
            z_basis=True, use_both=True, HZH=False,
        )
        self.code = code
        self.mapping_info = self._build_mapping()

        for name in (
            "gather_z", "valid_z", "z_neighbors", "z_hint_neighbors",
            "gather_x", "valid_x", "x_neighbors", "x_hint_neighbors",
            "spatial_coords_z", "spatial_coords_x",
        ):
            self.register_buffer(name, getattr(self.mapping_info, name))

    def _build_mapping(self) -> BBMappingInfo:
        l, m, lm = self.l, self.m, self.lm
        rounds = self.rounds
        num_t = rounds + 1
        total_detectors = rounds * 2 * lm

        gather_z = torch.zeros((num_t, lm), dtype=torch.long)
        gather_x = torch.zeros((num_t, lm), dtype=torch.long)
        valid_z = torch.zeros((num_t, lm), dtype=torch.float32)
        valid_x = torch.zeros((num_t, lm), dtype=torch.float32)

        gather_z[0] = torch.arange(lm, dtype=torch.long)
        valid_z[0] = 1.0

        for t in range(1, rounds):
            base = lm + (t - 1) * 2 * lm
            ids = torch.arange(lm, dtype=torch.long)
            gather_z[t] = base + ids
            gather_x[t] = base + lm + ids
            valid_z[t] = 1.0
            valid_x[t] = 1.0

        base = lm + (rounds - 1) * 2 * lm
        gather_z[rounds] = base + torch.arange(lm, dtype=torch.long)
        valid_z[rounds] = 1.0
        assert base + lm == total_detectors, (base + lm, total_detectors)

        spatial_coords = torch.tensor([[i // m, i % m] for i in range(lm)], dtype=torch.long)
        z_neighbors = _neighbors_by_offsets(spatial_coords, self.xx_offsets, l, m)
        x_neighbors = _neighbors_by_offsets(spatial_coords, self.xx_offsets, l, m)
        z_to_x_stab = _neighbors_by_offsets(spatial_coords, self.xz_offsets, l, m)
        x_to_z_stab = _neighbors_by_offsets(spatial_coords, self.xz_offsets, l, m)

        z_hint = torch.full((num_t, lm, self.K_cross), total_detectors, dtype=torch.long)
        x_hint = torch.full((num_t, lm, self.K_cross), total_detectors, dtype=torch.long)
        for t in range(1, rounds):
            for i in range(lm):
                for k in range(self.K_cross):
                    z_hint[t, i, k] = gather_x[t, z_to_x_stab[i, k].item()]
                    x_hint[t, i, k] = gather_z[t, x_to_z_stab[i, k].item()]

        return BBMappingInfo(
            gather_z=gather_z.flatten(),
            valid_z=valid_z.flatten(),
            z_neighbors=z_neighbors,
            z_hint_neighbors=z_hint.view(-1, self.K_cross),
            gather_x=gather_x.flatten(),
            valid_x=valid_x.flatten(),
            x_neighbors=x_neighbors,
            x_hint_neighbors=x_hint.view(-1, self.K_cross),
            spatial_coords_z=spatial_coords,
            spatial_coords_x=spatial_coords.clone(),
            xx_offsets=self.xx_offsets,
            xz_offsets=self.xz_offsets,
            l=l, m=m, lm=lm,
            num_t=num_t, rounds=rounds,
            total_detectors=total_detectors,
            K_same=self.K_same,
            K_cross=self.K_cross,
        )




# ============================================================================
# Output observables
# ============================================================================

@dataclass
class LogicalBasis:
    name: str
    representatives: np.ndarray       # [K, N] binary Z-logical reps
    basis_transform: np.ndarray       # [K, K], new_obs = T @ old_obs
    detector_transform: np.ndarray    # [total_detectors, K], detector correction
    stabilizer_coeffs: np.ndarray     # [K, lm], reps = T @ code.lz + C @ H_Z

def gf2_rank(matrix: np.ndarray) -> int:
    a = np.asarray(matrix, dtype=np.uint8).copy() % 2
    if a.ndim == 1:
        a = a[None, :]
    rank = 0
    rows, cols = a.shape
    for col in range(cols):
        pivots = np.flatnonzero(a[rank:, col])
        if len(pivots) == 0:
            continue
        pivot = rank + int(pivots[0])
        if pivot != rank:
            a[[rank, pivot]] = a[[pivot, rank]]
        for row in range(rows):
            if row != rank and a[row, col]:
                a[row] ^= a[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def solve_rows(rows: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    """Return coeff such that coeff @ rows == target over GF(2), if possible."""
    rows = np.asarray(rows, dtype=np.uint8) % 2
    target = np.asarray(target, dtype=np.uint8) % 2
    num_rows, width = rows.shape
    aug = np.concatenate([rows.T.copy(), target[:, None]], axis=1)
    rank = 0
    pivots: list[int] = []
    for col in range(num_rows):
        pivot_rows = np.flatnonzero(aug[rank:, col])
        if len(pivot_rows) == 0:
            continue
        pivot = rank + int(pivot_rows[0])
        if pivot != rank:
            aug[[rank, pivot]] = aug[[pivot, rank]]
        for row in range(width):
            if row != rank and aug[row, col]:
                aug[row] ^= aug[rank]
        pivots.append(col)
        rank += 1
        if rank == width:
            break
    for row in range(rank, width):
        if aug[row, :num_rows].sum() == 0 and aug[row, num_rows]:
            return None
    coeff = np.zeros(num_rows, dtype=np.uint8)
    for row, col in enumerate(pivots):
        coeff[col] = aug[row, num_rows]
    if not np.array_equal((coeff @ rows) % 2, target):
        return None
    return coeff


def _raw_logical_rep(lz: np.ndarray, coord: np.ndarray) -> np.ndarray:
    return (np.asarray(coord, dtype=np.uint8) @ np.asarray(lz, dtype=np.uint8)) % 2


def _build_logical_basis(mapper,
                         representatives: np.ndarray,
                         basis_transform: np.ndarray,
                         name: str,
                         allow_dependent: bool = False) -> LogicalBasis:
    code = mapper.code
    representatives = np.asarray(representatives, dtype=np.uint8) % 2
    basis_transform = np.asarray(basis_transform, dtype=np.uint8) % 2
    detector_transform, stabilizer_coeffs = _build_detector_transform(
        mapper, representatives, basis_transform
    )
    if not allow_dependent and gf2_rank(basis_transform) != code.K:
        raise RuntimeError(f"{name} output transform is not full rank")
    if np.any((np.asarray(code.hx, dtype=np.uint8) @ representatives.T) % 2):
        raise RuntimeError(f"{name} representatives are not in ker(H_X)")
    return LogicalBasis(
        name=name,
        representatives=representatives,
        basis_transform=basis_transform,
        detector_transform=detector_transform,
        stabilizer_coeffs=stabilizer_coeffs,
    )


def build_default_observables(mapper) -> LogicalBasis:
    """Build the default BB output observables used by this release."""
    code = mapper.code
    if code.K != 12:
        raise ValueError("default BB output convention expects K=12")
    unit = np.eye(code.K, dtype=np.uint8)
    lz = np.asarray(code.lz, dtype=np.uint8) % 2
    coords_arr = np.stack([
        *unit[:6],
        unit[7],
        unit[7] ^ unit[8] ^ unit[11],
        unit[6],
        unit[8] ^ unit[9] ^ unit[10],
        unit[9] ^ unit[10] ^ unit[11],
        unit[9] ^ unit[10] ^ unit[11],
    ])
    representatives = np.stack([_raw_logical_rep(lz, row) for row in coords_arr])
    return _build_logical_basis(
        mapper,
        representatives,
        coords_arr,
        name="bb_default",
        allow_dependent=True,
    )


def _build_detector_transform(mapper, representatives: np.ndarray, basis_transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    code = mapper.code
    hz = np.asarray(code.hz, dtype=np.uint8) % 2
    lz = np.asarray(code.lz, dtype=np.uint8) % 2
    reps = np.asarray(representatives, dtype=np.uint8) % 2
    transform = np.asarray(basis_transform, dtype=np.uint8) % 2
    coeffs = []
    info = mapper.mapping_info
    gather_z = info.gather_z.reshape(info.num_t, info.lm).cpu().numpy()
    valid_z = info.valid_z.reshape(info.num_t, info.lm).cpu().numpy().astype(bool)
    detector_transform = np.zeros((info.total_detectors, code.K), dtype=np.uint8)
    for k in range(code.K):
        base = (transform[k] @ lz) % 2
        diff = reps[k] ^ base
        coeff = solve_rows(hz, diff)
        if coeff is None:
            raise RuntimeError(f"logical representative {k} is not in transform(code.lz)+span(H_Z)")
        coeffs.append(coeff.astype(np.uint8))
        for row in np.flatnonzero(coeff):
            for t in range(info.num_t):
                if valid_z[t, row]:
                    detector_transform[gather_z[t, row], k] ^= 1
    return detector_transform, np.stack(coeffs, axis=0)


def transform_observables(det: np.ndarray, obs: np.ndarray, basis: LogicalBasis) -> np.ndarray:
    """Apply the output-observable transform to a sampled batch."""
    if basis.name == "current":
        return obs
    obs_u8 = np.asarray(obs, dtype=np.uint8)
    det_u8 = np.asarray(det, dtype=np.uint8)
    out = (obs_u8 @ basis.basis_transform.T) % 2
    correction = (det_u8 @ basis.detector_transform) % 2
    out ^= correction
    return out.astype(np.float32, copy=False)


# ============================================================================
# Transformer model
# ============================================================================

# ============================================================================
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x.to(torch.float32).pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt() * x


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RecurrentBlock(nn.Module):
    """Per-stabilizer GRU residual block with persistent temporal state."""
    def __init__(self, d_model):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.gru = nn.GRUCell(d_model, d_model)
        self.d_model = d_model

    def forward(self, x, h):
        """
        x: [B, N, D]
        h: [B, N, D]
        returns: (out [B,N,D], new_h [B,N,D])
        """
        B, N, D = x.shape
        x_normed = self.norm(x).reshape(B * N, D)
        h_flat = h.reshape(B * N, D)
        new_h = self.gru(x_normed, h_flat).reshape(B, N, D)
        return x + new_h, new_h


def _rotate_half(t):
    return torch.cat((-t[..., t.shape[-1] // 2:], t[..., :t.shape[-1] // 2]), dim=-1)


class SpatialTransformerBlock(nn.Module):
    """Spatial self-attention block with signed-wrap relative RoPE."""
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.mlp = SwiGLU(dim, 2 * dim)

    def forward(self, x, folded_trig):
        B, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h)
        q, k, v = (t.view(B, N, self.n_heads, self.head_dim) for t in qkv.chunk(3, dim=-1))

        half = self.head_dim // 2
        q_i, q_j = q[..., :half], q[..., half:]
        k_i, k_j = k[..., :half], k[..., half:]
        q_i = q_i.permute(0, 2, 1, 3).contiguous()  # [B,H,N,half]
        q_j = q_j.permute(0, 2, 1, 3).contiguous()
        k_i = k_i.permute(0, 2, 1, 3).contiguous()
        k_j = k_j.permute(0, 2, 1, 3).contiguous()

        cos_i, sin_i, cos_j, sin_j = folded_trig

        # q dot R(delta) k with delta = signed_wrap(coord_k - coord_q, current L).
        # The einsum form avoids materializing [B,H,N,N,head_dim/2].
        attn = torch.einsum("bhnd,bhmd,nmd->bhnm", q_i, k_i, cos_i)
        attn = attn + torch.einsum("bhnd,bhmd,nmd->bhnm", q_i, _rotate_half(k_i), sin_i)
        attn = attn + torch.einsum("bhnd,bhmd,nmd->bhnm", q_j, k_j, cos_j)
        attn = attn + torch.einsum("bhnd,bhmd,nmd->bhnm", q_j, _rotate_half(k_j), sin_j)
        attn = attn * (1.0 / math.sqrt(self.head_dim))

        weights = F.softmax(attn, dim=-1)
        v = v.permute(0, 2, 1, 3).contiguous()
        out = torch.einsum("bhnm,bhmd->bhnd", weights, v)
        x = x + self.proj(out.transpose(1, 2).reshape(B, N, D))
        x = x + self.mlp(self.norm2(x))
        return x


class AQCrossAttentionLayer(nn.Module):
    """Cross-attention block where logical queries attend to stabilizer features."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, q, kv, padding_mask, attn_mask=None):
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        if attn_mask is not None:
            attn_mask = attn_mask.to(device=q_norm.device, dtype=q_norm.dtype)
        attn_out, _ = self.cross_attn(
            q_norm, kv_norm, kv_norm,
            key_padding_mask=padding_mask,
            attn_mask=attn_mask,
        )
        q = q + attn_out
        q = q + self.mlp(self.norm_mlp(q))
        return q



# ============================================================================
# ============================================================================

class AlphaQubitV2_BB(nn.Module):
    """BB Transformer decoder with recurrent and spatial-attention blocks."""
    def __init__(self, mapper: BBMapper, n_logicals: int, d_model: int = 512, n_heads: int = 8,
                 rope_delta_mode: str = "signed_neg_half",
                 logical_representatives=None):
        super().__init__()
        self.d_model = d_model
        self.n_logicals = n_logicals
        self.n_heads = n_heads
        if rope_delta_mode not in {"signed_neg_half", "signed_pos_half", "raw_mod"}:
            raise ValueError(f"unknown rope_delta_mode: {rope_delta_mode}")
        self.rope_delta_mode = rope_delta_mode

        info = mapper.mapping_info
        self.num_z = info.lm
        self.num_x = info.lm
        self.num_stab = self.num_z + self.num_x
        self.num_t = info.num_t
        self.rounds = info.rounds
        self.total_detectors = info.total_detectors
        self.K_same = info.K_same
        self.K_cross = info.K_cross

        # Z mapping buffers
        self.register_buffer('gather_z', info.gather_z)
        self.register_buffer('valid_z', info.valid_z.view(self.num_t, self.num_z))
        self.register_buffer('z_neighbors', info.z_neighbors)
        self.register_buffer('z_hint_neighbors', info.z_hint_neighbors)
        # X mapping buffers
        self.register_buffer('gather_x', info.gather_x)
        self.register_buffer('valid_x', info.valid_x.view(self.num_t, self.num_x))
        self.register_buffer('x_neighbors', info.x_neighbors)
        self.register_buffer('x_hint_neighbors', info.x_hint_neighbors)
        self.l = info.l
        self.m = info.m
        spatial_coords = torch.cat([info.spatial_coords_z, info.spatial_coords_x], dim=0)
        self.register_buffer('spatial_coords', spatial_coords)

        logical_readout_bias = self._build_logical_readout_bias(mapper, logical_representatives)
        self.register_buffer('logical_readout_bias', logical_readout_bias)
        self.logical_anchor_attn_scale = nn.Parameter(torch.tensor(1.0))
        self.logical_anchor_context_scale = nn.Parameter(torch.tensor(0.1))
        self.logical_anchor_norm = nn.LayerNorm(d_model)

        # ---- Full-pattern input embeddings ----
        # Keep the expressive local syndrome-pattern lookup, but override
        # nn.Embedding's default N(0,1) initialization with a transformer-scale
        # std so the large 8192 x d_model table does not dominate activations.
        n_space = 2 ** (1 + self.K_same)           # center + K_same same-type
        self.emb_space = nn.Embedding(n_space, d_model)
        self.emb_temp = nn.Embedding(4, d_model)    # T_prev * 2 + T_curr
        n_hints = 2 ** self.K_cross
        self.emb_x_hints = nn.Embedding(n_hints, d_model)
        self._reset_input_embeddings()

        self.stem_norm = RMSNorm(d_model)
        self.stem_resnet = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model),
        )

        self.type_emb = nn.Parameter(torch.randn(2, d_model) * 0.02)

        # ---- RoPE frequencies ----
        # Keep CartesianRoPE's code-size-agnostic inv_freq/checkpoint keys.
        # Periodicity is restored by signed-wrap pairwise relative phases in forward.
        self.rope_gen = CartesianRoPE(d_model // n_heads)
        self._folded_trig_cache = {}

        # ---- Backbone: AQ2 RNN+TF alternating ----
        self.n_rnn = 5
        self.rnn_layers = nn.ModuleList([RecurrentBlock(d_model) for _ in range(self.n_rnn)])
        self.n_tf = 6
        self.tf_layers = nn.ModuleList([SpatialTransformerBlock(d_model, n_heads) for _ in range(self.n_tf)])

        # ---- Readout (K logical queries) ----
        self.logical_query_embed = nn.Parameter(torch.randn(1, n_logicals, d_model) * 0.02)
        self.readout_layers = nn.ModuleList([AQCrossAttentionLayer(d_model, n_heads) for _ in range(2)])
        self.res_dense1 = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.res_dense2 = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.head_norm = nn.LayerNorm(d_model)
        self.head_weight = nn.Parameter(torch.empty(n_logicals, d_model))
        self.head_bias = nn.Parameter(torch.zeros(n_logicals))
        nn.init.kaiming_uniform_(self.head_weight, a=math.sqrt(5))

    def _reset_input_embeddings(self):
        # Keep learned input embeddings on the same scale as the rest of the
        # transformer stack. PyTorch nn.Embedding defaults to N(0,1), which is
        # too large for high-cardinality detector-pattern tables.
        for emb in (self.emb_space, self.emb_temp, self.emb_x_hints):
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    @staticmethod
    def _time_sinusoidal(num_t: int, d_model: int, device):
        position = torch.arange(num_t, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, num_t, 1, d_model, device=device)
        pe[0, :, 0, 0::2] = torch.sin(position * div_term)
        pe[0, :, 0, 1::2] = torch.cos(position * div_term)
        return pe

    @staticmethod
    def _build_logical_readout_bias(mapper: BBMapper, logical_representatives=None) -> torch.Tensor:
        """
        Static [K, 2lm] logical-query to stabilizer-token prior.

        The bias uses only code-defined logical/stabilizer overlap. It gives the
        K output heads a legal basis anchor while leaving spatial attention
        torus-periodic.
        """
        code = mapper.code
        if logical_representatives is None:
            logical = torch.as_tensor(code.lz, dtype=torch.float32)  # [K, 2lm]
        else:
            logical = torch.as_tensor(logical_representatives, dtype=torch.float32)
        hz = torch.as_tensor(code.hz, dtype=torch.float32)       # [lm, 2lm]
        hx = torch.as_tensor(code.hx, dtype=torch.float32)       # [lm, 2lm]

        z_overlap = logical @ hz.t()
        x_overlap = logical @ hx.t()
        overlap = torch.cat([z_overlap, x_overlap], dim=1)
        bias = (overlap > 0).float()

        expected = mapper.mapping_info.lm * 2
        if bias.shape != (code.K, expected):
            raise ValueError(
                f"logical_readout_bias shape {tuple(bias.shape)} != {(code.K, expected)}"
            )

        bias = bias - bias.mean(dim=1, keepdim=True)
        std = bias.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return bias / std

    def _embed_stabs(self, det_raw, gather_idx, valid_mask, neighbors_same,
                     hint_neighbors, num_spatial):
        """
        Build per-stab embedding for one type (Z or X).

        Indexing convention for discrete full-pattern embeddings:
          idx_space = C * 2^K_same + sum_k N_k * 2^(K_same-1-k)
          idx_temp  = T_prev * 2   + T_curr
          idx_hint  =               sum_k H_k * 2^(K_cross-1-k)

        Returns [B, num_t, num_spatial, d_model]
        """
        B = det_raw.shape[0]
        device = det_raw.device

        # Gather this-type detection events at every (t, spatial) slot
        X_3d = det_raw.gather(1, gather_idx.unsqueeze(0).expand(B, -1)).view(B, self.num_t, num_spatial)
        X_3d = (X_3d * valid_mask).long()

        # --- spatial idx: center + K_same same-type neighbors ---
        # Advanced indexing: X_sp[:, :, neighbors_same] gives [B, num_t, num_spatial, K_same].
        X_sp = F.pad(X_3d, (0, 1), value=0)  # trailing padded slot (not actually indexed on torus)
        N_vals = X_sp[:, :, neighbors_same]
        idx_space = X_3d * (2 ** self.K_same)
        for k in range(self.K_same):
            idx_space = idx_space + N_vals[..., k] * (2 ** (self.K_same - 1 - k))

        # --- temporal idx: T_prev * 2 + T_curr ---
        X_t = F.pad(X_3d, (0, 0, 1, 0), value=0)
        T_prev = X_t[:, 0:self.num_t, :]
        idx_temp = T_prev * 2 + X_3d

        # --- cross-type hint idx ---
        # hint_neighbors values are in [0, total_detectors], where total_detectors = padding.
        det_raw_padded = F.pad(det_raw, (0, 1), value=0.0)
        X_hints = det_raw_padded.gather(
            1, hint_neighbors.view(-1).unsqueeze(0).expand(B, -1)
        ).view(B, self.num_t, num_spatial, self.K_cross).long()
        idx_hint = torch.zeros_like(X_hints[..., 0])
        for k in range(self.K_cross):
            idx_hint = idx_hint + X_hints[..., k] * (2 ** (self.K_cross - 1 - k))

        emb_time = self._time_sinusoidal(self.num_t, self.d_model, device)
        emb = self.emb_space(idx_space) + self.emb_temp(idx_temp) + self.emb_x_hints(idx_hint) + emb_time
        emb = emb + self.stem_resnet(self.stem_norm(emb))
        return emb  # [B, num_t, num_spatial, d_model]

    def _encode_features(self, x):
        """Run detector embeddings and the recurrent/attention backbone."""
        B = x.shape[0]
        device = x.device

        emb_z = self._embed_stabs(
            x, gather_idx=self.gather_z, valid_mask=self.valid_z,
            neighbors_same=self.z_neighbors, hint_neighbors=self.z_hint_neighbors,
            num_spatial=self.num_z,
        )
        emb_x = self._embed_stabs(
            x, gather_idx=self.gather_x, valid_mask=self.valid_x,
            neighbors_same=self.x_neighbors, hint_neighbors=self.x_hint_neighbors,
            num_spatial=self.num_x,
        )
        emb_z = emb_z + self.type_emb[0]
        emb_x = emb_x + self.type_emb[1]
        emb = torch.cat([emb_z, emb_x], dim=2)  # [B, num_t, num_stab, D]


        folded_trig = self._folded_relative_trig(device)

        rnn_states = [torch.zeros(B, self.num_stab, self.d_model, device=device)
                      for _ in range(self.n_rnn)]

        for t in range(self.num_t):
            curr = emb[:, t]

            curr, rnn_states[0] = self.rnn_layers[0](curr, rnn_states[0])
            curr, rnn_states[1] = self.rnn_layers[1](curr, rnn_states[1])

            curr = self.tf_layers[0](curr, folded_trig)
            curr = self.tf_layers[1](curr, folded_trig)

            curr, rnn_states[2] = self.rnn_layers[2](curr, rnn_states[2])

            curr = self.tf_layers[2](curr, folded_trig)
            curr = self.tf_layers[3](curr, folded_trig)

            curr, rnn_states[3] = self.rnn_layers[3](curr, rnn_states[3])

            curr = self.tf_layers[4](curr, folded_trig)
            curr = self.tf_layers[5](curr, folded_trig)

            curr, rnn_states[4] = self.rnn_layers[4](curr, rnn_states[4])

        return curr

    def forward(self, x):
        """Return logical logits for raw detector events [B, total_detectors]."""
        feat = self._encode_features(x)
        return self._readout(feat, x.shape[0])

    @staticmethod
    def _fold_signed(delta: torch.Tensor, L: int) -> torch.Tensor:
        half = L // 2
        return ((delta + half) % L) - half

    @staticmethod
    def _fold_signed_pos_half(delta: torch.Tensor, L: int) -> torch.Tensor:
        # Alternative even-L convention: fold into (-L/2, L/2], so +L/2
        # stays positive. This is an ablation, not a full BB edge-label fix.
        half = L // 2
        return ((delta + half - 1) % L) - half + 1

    def _canonical_delta(self, delta: torch.Tensor, L: int) -> torch.Tensor:
        if self.rope_delta_mode == "signed_neg_half":
            return self._fold_signed(delta, L)
        if self.rope_delta_mode == "signed_pos_half":
            return self._fold_signed_pos_half(delta, L)
        if self.rope_delta_mode == "raw_mod":
            return delta % L
        raise RuntimeError(f"unknown rope_delta_mode: {self.rope_delta_mode}")

    def _folded_relative_phases(self, device):
        """
        Pairwise relative RoPE phases [Nq, Nk, head_dim//2].

        Direction convention matches absolute RoPE:
            theta_k - theta_q = (coord_k - coord_q) * inv_freq.
        """
        coords = self.spatial_coords.to(device)
        di = coords[None, :, 0] - coords[:, None, 0]  # key - query
        dj = coords[None, :, 1] - coords[:, None, 1]
        di = self._canonical_delta(di, self.l)
        dj = self._canonical_delta(dj, self.m)

        inv_i = self.rope_gen.inv_freq_i.to(device)
        inv_j = self.rope_gen.inv_freq_j.to(device)
        phase_i = di.float().unsqueeze(-1) * inv_i
        phase_j = dj.float().unsqueeze(-1) * inv_j
        return torch.cat([phase_i, phase_i], dim=-1), torch.cat([phase_j, phase_j], dim=-1)

    def _folded_relative_trig(self, device):
        if device.type == "cuda" and torch.is_autocast_enabled("cuda"):
            dtype = torch.get_autocast_dtype("cuda")
        elif device.type == "cpu" and torch.is_autocast_enabled("cpu"):
            dtype = torch.get_autocast_dtype("cpu")
        else:
            dtype = self.rope_gen.inv_freq_i.dtype
        device_key = device.index if device.index is not None else -1
        cache_key = (device.type, device_key, dtype)
        cached = self._folded_trig_cache.get(cache_key)
        if cached is not None:
            return cached

        phase_i, phase_j = self._folded_relative_phases(device)
        folded_trig = (
            phase_i.cos().to(dtype=dtype),
            phase_i.sin().to(dtype=dtype),
            phase_j.cos().to(dtype=dtype),
            phase_j.sin().to(dtype=dtype),
        )
        self._folded_trig_cache[cache_key] = folded_trig
        return folded_trig

    def _build_readout_inputs(self, feat, B):
        """Build logical query inputs and the static anchor attention bias."""
        pooled = feat.mean(dim=1, keepdim=True)
        q = pooled.expand(-1, self.n_logicals, -1) + self.logical_query_embed.expand(B, -1, -1)
        anchor_logits = self.logical_anchor_attn_scale * self.logical_readout_bias
        anchor_weights = torch.softmax(anchor_logits.to(device=feat.device, dtype=feat.dtype), dim=-1)
        anchor_context = torch.einsum('kn,bnd->bkd', anchor_weights, feat)
        anchor_context = self.logical_anchor_norm(anchor_context)
        q = q + self.logical_anchor_context_scale.to(dtype=q.dtype) * anchor_context.to(dtype=q.dtype)
        return q, anchor_logits

    def _readout_features(self, feat, B):
        """Return per-logical embeddings before the final linear head."""
        q, anchor_logits = self._build_readout_inputs(feat, B)
        for layer in self.readout_layers:
            q = layer(q, feat, padding_mask=None, attn_mask=anchor_logits)
        q = q + self.res_dense1(q)
        q = q + self.res_dense2(q)
        q_normed = self.head_norm(q)     # [B, K, D]
        return q_normed

    def _readout(self, feat, B):
        """Return logical logits from stabilizer features."""
        q_normed = self._readout_features(feat, B)
        # Per-logical linear head: logit[b, k] = q_normed[b, k, :] dot head_weight[k, :] + bias[k]
        return torch.einsum('bkd,kd->bk', q_normed, self.head_weight) + self.head_bias



# ============================================================================
# Transformer training
# ============================================================================

def build_anchor_representatives(mapper, logical_basis, mode: str):
    """Return the logical vectors used only for the static readout anchor."""
    if mode == "representative":
        return logical_basis.representatives
    if mode == "base_transform":
        lz = np.asarray(mapper.code.lz, dtype=np.uint8) % 2
        transform = np.asarray(logical_basis.basis_transform, dtype=np.uint8) % 2
        return (transform @ lz) % 2
    raise ValueError(f"unknown logical_anchor_mode: {mode}")


# ---- OOM-aware batch-size probe ----
def _is_oom(exc: BaseException) -> bool:
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "OutOfMemoryError") \
            and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def probe_max_bs(args, mapper, n_logicals, device, rank, logical_representatives=None):
    """Probe the largest micro-batch size that fits on rank 0."""
    bs = args.batch_size
    total_det = mapper.mapping_info.total_detectors
    while bs >= 1:
        torch.cuda.empty_cache()
        try:
            m = AlphaQubitV2_BB(mapper, n_logicals=n_logicals,
                                d_model=args.d_model, n_heads=args.n_heads,
                                rope_delta_mode=args.rope_delta_mode,
                                logical_representatives=logical_representatives).to(device)
            m.train()
            x = torch.randint(0, 2, (bs, total_det), device=device, dtype=torch.float32)
            y = torch.randint(0, 2, (bs, n_logicals), device=device, dtype=torch.float32)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = m(x)
                if out.dim() == 3:
                    y_exp = y.unsqueeze(1).expand(-1, out.size(1), -1)
                    loss = F.binary_cross_entropy_with_logits(out, y_exp)
                else:
                    loss = F.binary_cross_entropy_with_logits(out, y)
            loss.backward()
            del m, x, y, out, loss
            torch.cuda.empty_cache()
            return bs
        except BaseException as e:
            if not _is_oom(e):
                raise
            torch.cuda.empty_cache()
            if rank == 0:
                print(f"[OOM probe] bs={bs} OOM -> halving to {bs // 2}", flush=True)
            bs //= 2
    raise RuntimeError("[OOM probe] even bs=1 doesn't fit")


# ---- DDP infra ----
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


# ---- Dataset ----
class OnlineBBDataset(IterableDataset):
    """Generate online BB detector samples with independent worker seeds."""
    def __init__(self, l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows,
                 rounds, p, batch_size, rank=0):
        super().__init__()
        self.rank = rank
        self.batch_size = batch_size
        self.mapper = BBMapper(l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows,
                               rounds=rounds, p=p)
        self.circuit = self.mapper.circuit
        self.n_logicals = self.mapper.code.K
        self.logical_basis = None

    def set_logical_basis(self, logical_basis):
        self.logical_basis = logical_basis

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        entropy = int.from_bytes(os.urandom(4), byteorder="little")
        time_ns = time.time_ns()
        seed = (time_ns + entropy + worker_id * 10000 + self.rank * 100000) % (2**32 - 1)
        sampler = self.circuit.compile_detector_sampler(seed=seed)
        while True:
            det, obs = sampler.sample(self.batch_size, separate_observables=True)
            if self.logical_basis is not None and self.logical_basis.name != "current":
                obs = transform_observables(det, obs, self.logical_basis)
            x = torch.from_numpy(det).float()
            y = torch.from_numpy(obs).float()
            yield x, y


# ---- Optimizer helpers ----
def _uses_no_decay(name: str) -> bool:
    name = name.replace("_orig_mod.", "").replace("module.", "")
    lname = name.lower()
    return lname.endswith("bias")


def build_param_groups(m):
    groups = {True: [], False: []}
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        use_decay = not _uses_no_decay(n)
        groups[use_decay].append(p)

    param_groups = []
    for use_decay, params in groups.items():
        if not params:
            continue
        param_groups.append({
            "params": params,
            "weight_decay": 1e-2 if use_decay else 0.0,
        })
    return param_groups


def lr_factor(step: int, warmup: int, max_steps: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer, args, step: int):
    schedule_steps = args.lr_schedule_steps if args.lr_schedule_steps is not None else args.max_steps
    factor = lr_factor(step, args.warmup, schedule_steps)
    for group in optimizer.param_groups:
        group["lr"] = args.lr * factor


def _format_float_list(values, scale: float = 1.0, digits: int = 1) -> str:
    vals = values.detach().float().cpu().tolist()
    return "[" + ", ".join(f"{v * scale:.{digits}f}" for v in vals) + "]"


# ---- Validation ----
#
def validate(model, loader, device, target_samples, world_size):
    """Return block accuracy and per-logical accuracies."""
    model.eval()
    correct_all = 0.0
    total = 0
    per_logical_correct = None
    target_per_rank = target_samples // max(world_size, 1)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x).float()
            pred = (logits > 0).float()
            match = (pred == y).float()
            correct_all += float(match.all(dim=1).sum().item())
            per_logical_correct = match.sum(dim=0) if per_logical_correct is None else per_logical_correct + match.sum(dim=0)
            total += x.size(0)
            if total >= target_per_rank:
                break
    model.train()
    if per_logical_correct is None:
        raise RuntimeError("validation loader produced no batches")
    return {
        "total": torch.tensor(float(total), device=device),
        "correct_all": torch.tensor(correct_all, device=device),
        "per_logical_correct": per_logical_correct,
    }


def run_training(args):
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"\n=== BB Transformer | l={args.l} m={args.m} rounds={args.rounds} p={args.p} ===")
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    ds_kwargs = dict(
        l=args.l, m=args.m,
        A_x_pows=args.A_x, A_y_pows=args.A_y,
        B_x_pows=args.B_x, B_y_pows=args.B_y,
        rounds=args.rounds, p=args.p,
        rank=rank,
    )
    train_ds = OnlineBBDataset(batch_size=args.batch_size, **ds_kwargs)
    val_ds = OnlineBBDataset(batch_size=args.batch_size, **ds_kwargs)
    mapper = train_ds.mapper
    n_logicals = train_ds.n_logicals
    logical_basis = build_default_observables(mapper)
    train_ds.set_logical_basis(logical_basis)
    val_ds.set_logical_basis(logical_basis)
    anchor_representatives = build_anchor_representatives(
        mapper, logical_basis, args.logical_anchor_mode
    )

    if rank == 0:
        info = mapper.mapping_info
        print(f"    num_stab={info.lm * 2}  num_t={info.num_t}  total_det={info.total_detectors}")
        print(f"    n_logicals={n_logicals}")
        print("    output convention: default BB observables")

    probed_bs = args.batch_size
    if not args.skip_oom_probe and rank == 0:
        probed_bs = probe_max_bs(args, mapper, n_logicals, device, rank,
                                 anchor_representatives)
    if dist.is_initialized():
        bs_t = torch.tensor([probed_bs], device=device, dtype=torch.long)
        dist.broadcast(bs_t, src=0)
        probed_bs = int(bs_t.item())
    args.batch_size = probed_bs
    train_ds.batch_size = args.batch_size
    val_ds.batch_size = args.batch_size

    worker_per_gpu = max(4, 32 // max(world_size, 1))
    train_loader = DataLoader(train_ds, batch_size=None, num_workers=worker_per_gpu,
                              pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=None, num_workers=2, pin_memory=True)

    model = AlphaQubitV2_BB(mapper, n_logicals=n_logicals,
                            d_model=args.d_model, n_heads=args.n_heads,
                            rope_delta_mode=args.rope_delta_mode,
                            logical_representatives=anchor_representatives).to(device)
    if rank == 0:
        n_p = sum(p.numel() for p in model.parameters())
        print(f"    model params: {n_p/1e6:.2f}M")

    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            print(f"--> Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        sd = ckpt.get('model_state', ckpt)
        sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
        model_sd = model.state_dict()
        filtered = {
            k: v for k, v in sd.items()
            if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)
            and k != "logical_readout_bias"
        }
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if rank == 0:
            print(f"    transferred {len(filtered)} tensors")
            print(f"    new-model-only keys: {len(missing)}")
            print(f"    ckpt-only keys: {len(unexpected)}")

    if args.compile:
        if rank == 0:
            print(f"    torch.compile: enabled mode={args.compile_mode}")
        model = torch.compile(model, mode=args.compile_mode)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    base_model = model.module if dist.is_initialized() else model
    optimizer = optim.AdamW(build_param_groups(base_model), lr=0.0, fused=True)
    set_optimizer_lr(optimizer, args, 0)

    ACCUM = max(1, args.target_bs // (args.batch_size * max(world_size, 1)))
    if rank == 0:
        print(f"    target_bs={args.target_bs} micro_bs={args.batch_size} world={world_size} accum={ACCUM}")

    iterator = iter(train_loader)
    update_step = 0
    best_acc = -1.0
    t_start = time.time()
    t_last = t_start
    model.train()

    while update_step < args.max_steps:
        set_optimizer_lr(optimizer, args, update_step)
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for mi in range(ACCUM):
            try:
                x, y = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                x, y = next(iterator)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            my_ctx = (model.no_sync() if (dist.is_initialized() and mi < ACCUM - 1)
                      else contextlib.nullcontext())
            with my_ctx:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    loss = F.binary_cross_entropy_with_logits(logits.float(), y.float()) / ACCUM
                loss.backward()
                running_loss += float(loss.detach()) * ACCUM

        grad_norm_val = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norm_val = float(grad_norm_val) if not isinstance(grad_norm_val, float) else grad_norm_val
        optimizer.step()

        if update_step % args.eval_every == 0:
            eval_start = time.time()
            val_stats = validate(model, val_loader, device, args.eval_samples, world_size)
            if dist.is_initialized():
                for value in val_stats.values():
                    dist.all_reduce(value, op=dist.ReduceOp.SUM)
            eval_sec = time.time() - eval_start

            total_val = val_stats["total"].clamp_min(1.0)
            per_logical = val_stats["per_logical_correct"] / total_val
            block_acc = val_stats["correct_all"] / total_val
            per_log_mean = per_logical.mean()
            per_log_worst = per_logical.min()

            if rank == 0:
                lr = optimizer.param_groups[0]["lr"]
                avg_loss = running_loss / max(ACCUM, 1)
                now = time.time()
                dt = now - t_last if update_step > 0 else 0.0
                steps_per_sec = args.eval_every / max(dt, 1e-6) if update_step > 0 else 0.0
                t_last = now
                print(f"[{time.strftime('%H:%M:%S')}] step {update_step:6d} | lr {lr:.2e} | "
                      f"loss {avg_loss:.4f} | block {block_acc*100:6.3f}% | "
                      f"per-log mean {per_log_mean*100:6.3f}% worst {per_log_worst*100:6.3f}% | "
                      f"rate {steps_per_sec:.2f} step/s | elapsed {(now-t_start)/60:.1f}min | "
                      f"eval {eval_sec:.1f}s | grad_norm {grad_norm_val:.4f}", flush=True)
                print(f"           PER_LOG_ACC % | {_format_float_list(per_logical, scale=100.0, digits=1)}", flush=True)

                if not args.no_save:
                    state = (model.module if dist.is_initialized() else model).state_dict()
                    meta = {'name': 'bb_default', 'logical_anchor_mode': args.logical_anchor_mode}
                    torch.save({'model_state': state, 'step': update_step,
                                'block_acc': float(block_acc),
                                'per_log_mean': float(per_log_mean),
                                'output_convention': meta}, args.output)
                    if args.save_every > 0 and update_step > 0 and update_step % args.save_every == 0:
                        step_path = args.output.replace(".pt", f"_step{update_step:06d}.pt")
                        torch.save({'model_state': state, 'step': update_step,
                                    'block_acc': float(block_acc),
                                    'per_log_mean': float(per_log_mean),
                                    'output_convention': meta}, step_path)
                        print(f"  -> step checkpoint saved to {step_path}", flush=True)
                    if float(block_acc) > best_acc:
                        best_acc = float(block_acc)
                        best_path = args.output.replace(".pt", "_best.pt")
                        torch.save({'model_state': state, 'step': update_step,
                                    'block_acc': float(block_acc),
                                    'per_log_mean': float(per_log_mean),
                                    'best_metric': best_acc,
                                    'output_convention': meta}, best_path)
                        print(f"  -> new best block_acc {best_acc*100:.3f}% saved to {best_path}", flush=True)

        update_step += 1

    cleanup_ddp()

def main_train_transformer():

    parser = argparse.ArgumentParser()
    # BB polynomial (defaults = [[72,12,6]])
    # NOTE: use --torus_l / --torus_m (NOT --l / --m); single-letter args
    # conflict with torchrun's argparse prefix matching (--log-dir / --master-addr etc).
    parser.add_argument("--torus_l", type=int, default=6, dest="l")
    parser.add_argument("--torus_m", type=int, default=6, dest="m")
    parser.add_argument("--A_x", type=int, nargs="+", default=[3])
    parser.add_argument("--A_y", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--B_x", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--B_y", type=int, nargs="+", default=[3])
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--p", type=float, default=0.005)
    # Model
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--rope_delta_mode", type=str, default="signed_neg_half",
                        choices=["signed_neg_half", "signed_pos_half", "raw_mod"],
                        help="Spatial RoPE pairwise delta representative.")
    parser.add_argument("--logical_anchor_mode", type=str, default="representative",
                        choices=["representative", "base_transform"],
                        help=("Which logical vectors define the static readout anchor. "
                              "representative uses T@lz+C@H_Z; base_transform uses T@lz."))
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for the SignedWrapRoPE model.")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode.")
    # Training
    parser.add_argument("--batch_size", type=int, default=64,
                        help="per-GPU micro-batch size")
    parser.add_argument("--target_bs", type=int, default=2048,
                        help="effective total batch = target_bs across all GPUs")
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--lr_schedule_steps", type=int, default=None,
                        help="Use this many steps for warmup/cosine LR decay; defaults to --max_steps.")
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_samples", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup", type=int, default=200,
                        help="LR warmup steps.")
    parser.add_argument("--skip_oom_probe", action="store_true",
                        help="Skip startup OOM probe when the requested batch size is known to fit.")
    # IO
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--no_save", action="store_true",
                        help="Disable checkpoint writes.")
    parser.add_argument("--save_every", type=int, default=0,
                        help="Also save *_stepXXXXXX.pt at validation steps divisible by this value.")
    args = parser.parse_args()
    run_training(args)



def main():
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        sys.argv.pop(1)
    main_train_transformer()


if __name__ == "__main__":
    main()
