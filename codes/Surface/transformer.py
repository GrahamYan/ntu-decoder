"""AlphaQubit V2 neural decoder for rotated surface codes.

Implements a joint X+Z stabilizer processing architecture with alternating
RNN (GRU) and Transformer blocks, trained via progressive distillation
with MWPM pseudo-labels.

Reference:
    This module corresponds to the "surface code" section of the
    associated paper (paper.pdf).
"""

import argparse
import contextlib
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import pymatching


def download_from_hf(repo_id: str, filename: str, cache_dir: str | None = None) -> str:
    """Download a checkpoint from Hugging Face Hub and return the local path.

    Args:
        repo_id: Hugging Face repository ID (e.g. ``user/model-name``).
        filename: File path within the repository (e.g. ``surface/d7.pth``).
        cache_dir: Optional custom cache directory. Defaults to
            ``~/.cache/huggingface/hub/``.

    Returns:
        Absolute path to the downloaded file on local disk.
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)


import stim
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset

# =========================================================================
# Distributed training utilities
# =========================================================================


def setup_ddp():
    """Initialize distributed data parallel training.

    Returns:
        Tuple of (rank, local_rank, world_size). If not running under
        torchrun, returns (0, 0, 1) for single-GPU operation.
    """
    if "RANK" not in os.environ:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def cleanup_ddp():
    """Destroy the distributed process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


# =========================================================================
# Physical mapping layer: joint X + Z stabilizer mapping
# =========================================================================


@dataclass
class FullMappingInfo:
    """Container for the joint X+Z stabilizer mapping tensors.

    Holds gather indices, validity masks, spatial neighbor indices,
    and cross-type hint neighbor indices for both Z-type and X-type
    stabilizers across all time steps.

    Attributes:
        gather_z: Flattened gather indices [num_t * num_z] mapping
            (time, z_idx) to stim detector indices.
        valid_z: Validity mask [num_t * num_z] for Z stabilizers.
        z_neighbors: Same-type diagonal neighbor indices [num_z, 4]
            for Z stabilizers.
        z_hint_neighbors: Cross-type hint indices [num_t * num_z, 4]
            from Z to the 4 nearest X stabilizers.
        num_z: Number of Z-type stabilizers per time step.
        gather_x: Flattened gather indices [num_t * num_x] mapping
            (time, x_idx) to stim detector indices.
        valid_x: Validity mask [num_t * num_x] for X stabilizers.
        x_neighbors: Same-type diagonal neighbor indices [num_x, 4]
            for X stabilizers.
        x_hint_neighbors: Cross-type hint indices [num_t * num_x, 4]
            from X to the 4 nearest Z stabilizers.
        num_x: Number of X-type stabilizers per time step.
        num_t: Number of time steps (unique detection times).
        rounds: Number of surface code rounds.
    """

    # Z stabilizer mapping.
    gather_z: torch.Tensor
    valid_z: torch.Tensor
    z_neighbors: torch.Tensor
    z_hint_neighbors: torch.Tensor
    num_z: int

    # X stabilizer mapping.
    gather_x: torch.Tensor
    valid_x: torch.Tensor
    x_neighbors: torch.Tensor
    x_hint_neighbors: torch.Tensor
    num_x: int

    # Shared.
    num_t: int
    rounds: int


class FullMapper(nn.Module):
    """Jointly maps X and Z stabilizers from a Stim surface code circuit.

    Replaces the original single-type (Z-only) strict mapper. Separates
    stabilizers into Z-type and X-type sets based on detector coordinates,
    and builds gather matrices, neighbor tables, and hint connections for
    both types.

    Args:
        d: Surface code distance.
        rounds: Number of syndrome extraction rounds.
    """

    def __init__(self, d: int, rounds: int):
        super().__init__()
        self.d = d
        self.rounds = rounds
        self.circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=d, rounds=rounds
        )
        self.mapping_info = self._build_full_mapping()

        # Register all buffers for device placement.
        self.register_buffer("gather_z", self.mapping_info.gather_z)
        self.register_buffer("valid_z", self.mapping_info.valid_z)
        self.register_buffer("z_neighbors", self.mapping_info.z_neighbors)
        self.register_buffer("z_hint_neighbors", self.mapping_info.z_hint_neighbors)
        self.register_buffer("gather_x", self.mapping_info.gather_x)
        self.register_buffer("valid_x", self.mapping_info.valid_x)
        self.register_buffer("x_neighbors", self.mapping_info.x_neighbors)
        self.register_buffer("x_hint_neighbors", self.mapping_info.x_hint_neighbors)

    def _build_full_mapping(self) -> FullMappingInfo:
        """Construct the complete joint X+Z mapping from detector coordinates.

        The mapping procedure:
        1. Separates detectors into Z-type (final round) and X-type (all
           others) based on Stim's coordinate convention for rotated
           surface codes.
        2. Assigns spatial grid locations via rounded coordinates.
        3. Builds gather matrices mapping (time, spatial_idx) -> stim
           detector index.
        4. Computes same-type diagonal neighbors for spatial embeddings.
        5. Computes cross-type hint neighbors (4 nearest stabilizers of
           the opposite type, sorted by Euclidean distance).

        Returns:
            A FullMappingInfo dataclass with all mapping tensors.
        """
        coords = self.circuit.get_detector_coordinates()
        if not coords:
            empty_long = torch.empty(0, dtype=torch.long)
            empty_float = torch.empty(0, dtype=torch.float32)
            return FullMappingInfo(
                empty_long,
                empty_float,
                empty_long.view(0, 4),
                empty_long.view(0, 4),
                0,
                empty_long,
                empty_float,
                empty_long.view(0, 4),
                empty_long.view(0, 4),
                0,
                0,
                self.rounds,
            )

        # 1. Separate Z and X spatial coordinates.
        # In rotated_memory_z, the final round's detectors are all Z-type.
        max_t = max(t for _, _, t in coords.values())

        z_locs_set = set()
        for idx, (x, y, t) in coords.items():
            r, c = int(round(y / 2)), int(round(x / 2))
            if math.isclose(t, max_t):
                z_locs_set.add((r, c))

        x_locs_set = set()
        for idx, (x, y, t) in coords.items():
            r, c = int(round(y / 2)), int(round(x / 2))
            if (r, c) not in z_locs_set:
                x_locs_set.add((r, c))

        z_locs = sorted(list(z_locs_set), key=lambda p: (p[0], p[1]))
        x_locs = sorted(list(x_locs_set), key=lambda p: (p[0], p[1]))

        loc_to_idx_z = {loc: i for i, loc in enumerate(z_locs)}
        loc_to_idx_x = {loc: i for i, loc in enumerate(x_locs)}

        num_z = len(z_locs)
        num_x = len(x_locs)

        # 2. Collect all time steps.
        unique_times = sorted(list(set(t for _, _, t in coords.values())))
        time_to_idx = {t: i for i, t in enumerate(unique_times)}
        num_t = len(unique_times)
        total_detectors = len(coords)

        # Build reverse lookup: (r, c, t) -> stim detector index.
        coord_time_to_idx = {}
        for idx, (x, y, t) in coords.items():
            r, c = int(round(y / 2)), int(round(x / 2))
            coord_time_to_idx[(r, c, t)] = idx

        # 3. Build Z gather matrix and validity mask.
        gather_z = torch.zeros((num_t, num_z), dtype=torch.long)
        valid_z = torch.zeros((num_t, num_z), dtype=torch.float32)

        for t_idx, t in enumerate(unique_times):
            for z_idx, (zr, zc) in enumerate(z_locs):
                if (zr, zc, t) in coord_time_to_idx:
                    gather_z[t_idx, z_idx] = coord_time_to_idx[(zr, zc, t)]
                    valid_z[t_idx, z_idx] = 1.0

        # 4. Build X gather matrix and validity mask.
        gather_x = torch.zeros((num_t, num_x), dtype=torch.long)
        valid_x = torch.zeros((num_t, num_x), dtype=torch.float32)

        for t_idx, t in enumerate(unique_times):
            for x_idx, (xr, xc) in enumerate(x_locs):
                if (xr, xc, t) in coord_time_to_idx:
                    gather_x[t_idx, x_idx] = coord_time_to_idx[(xr, xc, t)]
                    valid_x[t_idx, x_idx] = 1.0

        # 5. Z diagonal Z neighbors (for spatial embedding).
        z_neighbors = torch.full((num_z, 4), num_z, dtype=torch.long)  # padding = num_z
        for i, (r, c) in enumerate(z_locs):
            for j, (dr, dc) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
                if (r + dr, c + dc) in loc_to_idx_z:
                    z_neighbors[i, j] = loc_to_idx_z[(r + dr, c + dc)]

        # 6. X diagonal X neighbors (for spatial embedding).
        x_neighbors = torch.full((num_x, 4), num_x, dtype=torch.long)  # padding = num_x
        for i, (r, c) in enumerate(x_locs):
            for j, (dr, dc) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
                if (r + dr, c + dc) in loc_to_idx_x:
                    x_neighbors[i, j] = loc_to_idx_x[(r + dr, c + dc)]

        # 7. Z's 4 nearest X neighbors (hint, sorted by distance).
        z_hint_neighbors = torch.full(
            (num_t, num_z, 4), total_detectors, dtype=torch.long
        )
        z_to_nearest_x = {}
        for zr, zc in z_locs:
            sorted_x = sorted(
                list(x_locs_set),
                key=lambda p: (p[0] - zr) ** 2 + (p[1] - zc) ** 2,
            )
            z_to_nearest_x[(zr, zc)] = sorted_x[:4]

        for t_idx, t in enumerate(unique_times):
            for z_idx, (zr, zc) in enumerate(z_locs):
                for i, (xr, xc) in enumerate(z_to_nearest_x[(zr, zc)]):
                    if (xr, xc, t) in coord_time_to_idx:
                        z_hint_neighbors[t_idx, z_idx, i] = coord_time_to_idx[
                            (xr, xc, t)
                        ]

        # 8. X's 4 nearest Z neighbors (hint, sorted by distance).
        x_hint_neighbors = torch.full(
            (num_t, num_x, 4), total_detectors, dtype=torch.long
        )
        x_to_nearest_z = {}
        for xr, xc in x_locs:
            sorted_z = sorted(
                list(z_locs_set),
                key=lambda p: (p[0] - xr) ** 2 + (p[1] - xc) ** 2,
            )
            x_to_nearest_z[(xr, xc)] = sorted_z[:4]

        for t_idx, t in enumerate(unique_times):
            for x_idx, (xr, xc) in enumerate(x_locs):
                for i, (zr, zc) in enumerate(x_to_nearest_z[(xr, xc)]):
                    if (zr, zc, t) in coord_time_to_idx:
                        x_hint_neighbors[t_idx, x_idx, i] = coord_time_to_idx[
                            (zr, zc, t)
                        ]

        return FullMappingInfo(
            gather_z=gather_z.flatten(),
            valid_z=valid_z.flatten(),
            z_neighbors=z_neighbors,
            z_hint_neighbors=z_hint_neighbors.view(-1, 4),
            num_z=num_z,
            gather_x=gather_x.flatten(),
            valid_x=valid_x.flatten(),
            x_neighbors=x_neighbors,
            x_hint_neighbors=x_hint_neighbors.view(-1, 4),
            num_x=num_x,
            num_t=num_t,
            rounds=self.rounds,
        )

    def get_spatial_coords(self, stab_type="both"):
        """Return the spatial (x, y) coordinates of stabilizers.

        Args:
            stab_type: Which stabilizer coordinates to return.
                "z" for Z-type only, "x" for X-type only, "both" for
                concatenated [Z, X].

        Returns:
            Tensor of shape [N, 2] with (x, y) coordinates.
        """
        coords = self.circuit.get_detector_coordinates()

        def _get_coords_for_locs(gather_flat, num_spatial):
            indices = gather_flat[:num_spatial].tolist()
            xy = []
            for idx in indices:
                x_val, y_val = coords[idx][:2] if idx in coords else (0.0, 0.0)
                xy.append([float(x_val), float(y_val)])
            return torch.tensor(xy, dtype=torch.float32)

        z_coords = _get_coords_for_locs(
            self.mapping_info.gather_z, self.mapping_info.num_z
        )
        x_coords = _get_coords_for_locs(
            self.mapping_info.gather_x, self.mapping_info.num_x
        )

        if stab_type == "z":
            return z_coords
        if stab_type == "x":
            return x_coords
        return torch.cat([z_coords, x_coords], dim=0)  # [num_z + num_x, 2]


# =========================================================================
# Model components
# =========================================================================


class CoordinateRoPE(nn.Module):
    """2D Rotary Position Embedding based on spatial coordinates.

    Splits the head dimension into x- and y- halves and applies standard
    RoPE rotation to each half using coordinate-derived frequencies.

    Args:
        head_dim: Dimension per attention head. Must be divisible by 4.
    """

    def __init__(self, head_dim):
        super().__init__()
        assert head_dim % 4 == 0, "2D RoPE needs head_dim divisible by 4"
        self.head_dim = head_dim
        self.half_dim = head_dim // 2  # channels per coordinate
        self.quarter_dim = head_dim // 4  # sin/cos frequencies per coordinate

        inv_freq = 1.0 / (
            100
            ** (
                torch.arange(0, self.quarter_dim, dtype=torch.float32)
                / self.quarter_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq)

    def get_freqs(self, coords, device):
        """Compute RoPE frequencies from spatial coordinates.

        Args:
            coords: Tensor [N, 2] of (x, y) coordinates.
            device: Target device.

        Returns:
            Tuple (freqs_x, freqs_y), each of shape [N, half_dim].
        """
        x = coords[:, 0].to(device)
        y = coords[:, 1].to(device)

        fx = torch.einsum("i,j->ij", x, self.inv_freq)  # [N, quarter_dim]
        fy = torch.einsum("i,j->ij", y, self.inv_freq)  # [N, quarter_dim]

        # Duplicate each for the rotate_half sin/cos pair.
        freqs_x = torch.cat([fx, fx], dim=-1)  # [N, half_dim]
        freqs_y = torch.cat([fy, fy], dim=-1)  # [N, half_dim]
        return freqs_x, freqs_y


def apply_rope_2d(q, k, freqs_x, freqs_y):
    """Apply 2D RoPE to query and key tensors.

    The first half of the head dimension is rotated by x-coordinate
    frequencies; the second half by y-coordinate frequencies.

    Args:
        q: Query tensor [B, N, n_heads, head_dim].
        k: Key tensor [B, N, n_heads, head_dim].
        freqs_x: X-axis frequencies [N, half_dim].
        freqs_y: Y-axis frequencies [N, half_dim].

    Returns:
        Tuple of rotated (q, k) tensors.
    """
    half = q.shape[-1] // 2

    # Split into x-half and y-half.
    q_x, q_y = q[..., :half], q[..., half:]
    k_x, k_y = k[..., :half], k[..., half:]

    # Broadcast frequencies.
    fx = freqs_x.unsqueeze(0).unsqueeze(2)  # [1, N, 1, half_dim]
    fy = freqs_y.unsqueeze(0).unsqueeze(2)

    def _rotate_half(t, freqs):
        t_rot = torch.cat(
            (-t[..., t.shape[-1] // 2 :], t[..., : t.shape[-1] // 2]),
            dim=-1,
        )
        return t * freqs.cos() + t_rot * freqs.sin()

    q_x = _rotate_half(q_x, fx)
    k_x = _rotate_half(k_x, fx)
    q_y = _rotate_half(q_y, fy)
    k_y = _rotate_half(k_y, fy)

    return torch.cat([q_x, q_y], dim=-1), torch.cat([k_x, k_y], dim=-1)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Args:
        dim: Feature dimension.
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (
            self.weight
            * x.to(torch.float32).pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
            * x
        )


class SwiGLU(nn.Module):
    """SwiGLU MLP block.

    Args:
        dim: Input/output dimension.
        hidden_dim: Hidden dimension for the gated projection.
    """

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RecurrentBlock(nn.Module):
    """Lightweight per-stabilizer GRU with residual connection.

    Applies RMSNorm followed by a GRUCell independently to each
    stabilizer's hidden state, with a residual (pre-norm) skip.

    Args:
        d_model: Model dimension.
    """

    def __init__(self, d_model):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.gru = nn.GRUCell(d_model, d_model)
        self.d_model = d_model

    def forward(self, x, h):
        """Forward pass for one time step.

        Args:
            x: Current stabilizer features [B, N, D].
            h: Previous hidden state [B, N, D].

        Returns:
            Tuple (output [B, N, D], new_hidden [B, N, D]).
        """
        B, N, D = x.shape
        x_normed = self.norm(x).reshape(B * N, D)
        h_flat = h.reshape(B * N, D)
        new_h = self.gru(x_normed, h_flat).reshape(B, N, D)
        return x + new_h, new_h  # residual connection


class SpatialTransformerBlock(nn.Module):
    """Spatial self-attention block with 2D RoPE.

    Applies multi-head self-attention over stabilizers at a single time
    step, using 2D RoPE for spatial positional information. No temporal
    state is carried.

    Args:
        dim: Model dimension.
        n_heads: Number of attention heads.
    """

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

    def forward(self, x, freqs_x, freqs_y):
        """Forward pass.

        Args:
            x: Input features [B, N, D].
            freqs_x: X-axis RoPE frequencies [N, half_dim].
            freqs_y: Y-axis RoPE frequencies [N, half_dim].

        Returns:
            Output features [B, N, D].
        """
        B, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h)
        q, k, v = map(
            lambda t: t.view(B, N, self.n_heads, self.head_dim),
            qkv.chunk(3, dim=-1),
        )

        # Apply 2D RoPE.
        q, k = apply_rope_2d(q, k, freqs_x, freqs_y)

        # Scaled dot-product attention.
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
        )
        x = x + self.proj(out.transpose(1, 2).reshape(B, N, D))
        x = x + self.mlp(self.norm2(x))
        return x


class AQCrossAttentionLayer(nn.Module):
    """Cross-attention readout layer with residual MLP.

    A single query token attends to the full stabilizer sequence
    (with a padding mask for invalid positions), followed by an MLP.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, q, kv, padding_mask):
        """Forward pass.

        Args:
            q: Query tensor [B, 1, D].
            kv: Key/value sequence [B, N, D].
            padding_mask: Boolean mask [B, N], True for invalid positions.

        Returns:
            Updated query [B, 1, D].
        """
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(
            q_norm, kv_norm, kv_norm, key_padding_mask=padding_mask
        )
        q = q + attn_out
        q = q + self.mlp(self.norm_mlp(q))
        return q


# =========================================================================
# Main model: AlphaQubit V2 — X+Z joint processing, RNN+TF alternating
# =========================================================================


class AlphaQubitV2(nn.Module):
    """Neural decoder with alternating RNN-Transformer architecture.

    Processes X and Z stabilizers jointly as a single concatenated
    sequence. The architecture follows the RNN-Transformer interleaving
    pattern: RNN -> RNN -> [TF] -> RNN -> [TF] -> RNN -> [TF] -> RNN,
    with 5 RNN + 6 Transformer layers total.

    Both stabilizer types share embedding tables and participate jointly
    in spatial self-attention at each time step.

    Args:
        mapper: A FullMapper instance providing stabilizer-to-detector
            index mappings and spatial coordinates.
        d_model: Model dimension (default: 512).
        n_heads: Number of attention heads (default: 8).
    """

    def __init__(self, mapper, d_model=512, n_heads=8):
        super().__init__()
        self.d_model = d_model

        self.num_z = mapper.mapping_info.num_z
        self.num_x = mapper.mapping_info.num_x
        self.num_stab = self.num_z + self.num_x  # total stabilizers
        self.num_t = mapper.mapping_info.num_t
        self.rounds = mapper.mapping_info.rounds

        # --- Register buffers ---
        # Z mapping.
        self.register_buffer("gather_z", mapper.mapping_info.gather_z)
        self.register_buffer(
            "valid_z",
            mapper.mapping_info.valid_z.view(self.num_t, self.num_z),
        )
        self.register_buffer("z_neighbors", mapper.mapping_info.z_neighbors)
        self.register_buffer("z_hint_neighbors", mapper.mapping_info.z_hint_neighbors)
        # X mapping.
        self.register_buffer("gather_x", mapper.mapping_info.gather_x)
        self.register_buffer(
            "valid_x",
            mapper.mapping_info.valid_x.view(self.num_t, self.num_x),
        )
        self.register_buffer("x_neighbors", mapper.mapping_info.x_neighbors)
        self.register_buffer("x_hint_neighbors", mapper.mapping_info.x_hint_neighbors)
        # Spatial coordinates for RoPE.
        self.register_buffer(
            "spatial_coords",
            mapper.get_spatial_coords("both"),  # [num_z + num_x, 2]
        )

        # --- Embedding (discrete encoding, shared across X and Z) ---
        self.emb_space = nn.Embedding(32, d_model)
        self.emb_temp = nn.Embedding(4, d_model)
        self.emb_x_hints = nn.Embedding(16, d_model)

        self.stem_norm = RMSNorm(d_model)
        self.stem_resnet = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

        # --- RoPE ---
        self.rope_gen = CoordinateRoPE(d_model // n_heads)

        # --- Backbone: alternating RNN + Transformer ---
        self.n_rnn = 5
        self.rnn_layers = nn.ModuleList(
            [RecurrentBlock(d_model) for _ in range(self.n_rnn)]
        )

        self.n_tf = 6
        self.tf_layers = nn.ModuleList(
            [SpatialTransformerBlock(d_model, n_heads) for _ in range(self.n_tf)]
        )

        # --- Readout ---
        self.logical_query_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.readout_layers = nn.ModuleList(
            [AQCrossAttentionLayer(d_model, n_heads) for _ in range(2)]
        )
        self.res_dense1 = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU()
        )
        self.res_dense2 = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU()
        )
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def get_time_sinusoidal_encoding(self, num_t, d_model, device):
        """Generate sinusoidal positional encoding for time steps.

        Args:
            num_t: Number of time steps.
            d_model: Model dimension.
            device: Target device.

        Returns:
            Tensor [1, num_t, 1, d_model] of sinusoidal encodings.
        """
        position = torch.arange(num_t, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, num_t, 1, d_model, device=device)
        pe[0, :, 0, 0::2] = torch.sin(position * div_term)
        pe[0, :, 0, 1::2] = torch.cos(position * div_term)
        return pe

    def _embed_stabilizers(
        self,
        det_raw,
        gather_idx,
        valid_mask,
        neighbors_same,
        hint_neighbors,
        num_spatial,
    ):
        """Embed one stabilizer type (X or Z) using discrete encoding.

        The embedding combines:
          - Spatial embedding: C*16 + TL*8 + TR*4 + BL*2 + BR.
          - Temporal embedding: T_prev*2 + T_curr.
          - Hint embedding (cross-type): H1*8 + H2*4 + H3*2 + H4.
          - Sinusoidal time encoding.

        Args:
            det_raw: Raw detection events [B, num_detectors].
            gather_idx: Stim index mapping [num_t * num_spatial].
            valid_mask: Validity mask [num_t, num_spatial].
            neighbors_same: Same-type neighbor indices [num_spatial, 4].
            hint_neighbors: Cross-type hint indices
                [num_t * num_spatial, 4].
            num_spatial: Number of stabilizers of this type.

        Returns:
            Embedded features [B, num_t, num_spatial, d_model].
        """
        B = det_raw.shape[0]

        # 1. Gather detection events -> [B, num_t, num_spatial].
        X_3d = det_raw.gather(1, gather_idx.unsqueeze(0).expand(B, -1)).view(
            B, self.num_t, num_spatial
        )
        X_3d = (X_3d * valid_mask).long()

        # 2. Spatial embedding: C*16 + TL*8 + TR*4 + BL*2 + BR.
        # Pad with an extra spatial index for boundary handling.
        X_sp = F.pad(X_3d, (0, 1), value=0)
        N_vals = X_sp[:, :, neighbors_same]  # [B, num_t, num_spatial, 4]
        C = X_3d
        TL, TR, BL, BR = (
            N_vals[..., 0],
            N_vals[..., 1],
            N_vals[..., 2],
            N_vals[..., 3],
        )
        idx_space = C * 16 + TL * 8 + TR * 4 + BL * 2 + BR * 1

        # 3. Temporal embedding: T_prev*2 + T_curr.
        X_t = F.pad(X_3d, (0, 0, 1, 0), value=0)  # prepad in time
        T_prev = X_t[:, 0 : self.num_t, :]
        T_curr = X_3d
        idx_temp = T_prev * 2 + T_curr * 1

        # 4. Hint embedding: H1*8 + H2*4 + H3*2 + H4 (cross-type).
        det_raw_padded = F.pad(det_raw, (0, 1), value=0)
        X_hints = det_raw_padded.gather(
            1, hint_neighbors.view(-1).unsqueeze(0).expand(B, -1)
        )
        X_hints = X_hints.view(B, self.num_t, num_spatial, 4).long()
        H1, H2, H3, H4 = (
            X_hints[..., 0],
            X_hints[..., 1],
            X_hints[..., 2],
            X_hints[..., 3],
        )
        idx_hint = H1 * 8 + H2 * 4 + H3 * 2 + H4 * 1

        # 5. Time positional encoding.
        emb_time = self.get_time_sinusoidal_encoding(
            self.num_t, self.d_model, det_raw.device
        )

        # 6. Compose.
        emb = (
            self.emb_space(idx_space)
            + self.emb_temp(idx_temp)
            + self.emb_x_hints(idx_hint)
            + emb_time
        )

        emb = emb + self.stem_resnet(self.stem_norm(emb))
        return emb  # [B, num_t, num_spatial, d_model]

    def forward(self, x, mask_prob=0.8, drop_ratio=0.5):
        """Forward pass through the full AlphaQubit V2 model.

        Args:
            x: Detection events [B, num_detectors].
            mask_prob: Probability of applying stabilizer dropout.
            drop_ratio: Fraction of stabilizers to drop when dropout
                is applied.

        Returns:
            During training: predictions at all time steps [B, num_t].
            During evaluation: prediction at the final time step [B].
        """
        B = x.shape[0]
        device = x.device

        # ==================== Embedding ====================
        # Z stabilizers.
        emb_z = self._embed_stabilizers(
            x,
            gather_idx=self.gather_z,
            valid_mask=self.valid_z,
            neighbors_same=self.z_neighbors,
            hint_neighbors=self.z_hint_neighbors,
            num_spatial=self.num_z,
        )  # [B, num_t, num_z, D]

        # X stabilizers.
        emb_x = self._embed_stabilizers(
            x,
            gather_idx=self.gather_x,
            valid_mask=self.valid_x,
            neighbors_same=self.x_neighbors,
            hint_neighbors=self.x_hint_neighbors,
            num_spatial=self.num_x,
        )  # [B, num_t, num_x, D]

        # Concatenate X and Z: [B, num_t, num_z + num_x, D].
        emb = torch.cat([emb_z, emb_x], dim=2)

        # Stabilizer dropout (during training only).
        if self.training and mask_prob > 0.0 and drop_ratio > 0.0:
            if torch.rand(1).item() < mask_prob:
                dropout_mask = (
                    torch.rand(B, self.num_t, self.num_stab, 1, device=device)
                    > drop_ratio
                ).to(emb.dtype)
                emb = emb * dropout_mask * (1.0 / (1.0 - drop_ratio))

        # RoPE frequencies for the full Z+X sequence.
        freqs_x, freqs_y = self.rope_gen.get_freqs(self.spatial_coords, device)

        # ==================== Backbone: RNN + TF alternating ====================
        # Initialize RNN hidden states: [B, num_stab, D].
        rnn_states = [
            torch.zeros(B, self.num_stab, self.d_model, device=device)
            for _ in range(self.n_rnn)
        ]

        all_time_feats = []

        for t in range(self.num_t):
            curr = emb[:, t]  # [B, num_stab, D]

            # --- Block 0: RNN_0 -> RNN_1 ---
            curr, rnn_states[0] = self.rnn_layers[0](curr, rnn_states[0])
            curr, rnn_states[1] = self.rnn_layers[1](curr, rnn_states[1])

            # --- Block 1: TF_0, TF_1 ---
            curr = self.tf_layers[0](curr, freqs_x, freqs_y)
            curr = self.tf_layers[1](curr, freqs_x, freqs_y)

            # --- Block 2: RNN_2 ---
            curr, rnn_states[2] = self.rnn_layers[2](curr, rnn_states[2])

            # --- Block 3: TF_2, TF_3 ---
            curr = self.tf_layers[2](curr, freqs_x, freqs_y)
            curr = self.tf_layers[3](curr, freqs_x, freqs_y)

            # --- Block 4: RNN_3 ---
            curr, rnn_states[3] = self.rnn_layers[3](curr, rnn_states[3])

            # --- Block 5: TF_4, TF_5 ---
            curr = self.tf_layers[4](curr, freqs_x, freqs_y)
            curr = self.tf_layers[5](curr, freqs_x, freqs_y)

            # --- Block 6: RNN_4 ---
            curr, rnn_states[4] = self.rnn_layers[4](curr, rnn_states[4])

            all_time_feats.append(curr)

        # ==================== Readout ====================
        step_predictions = []

        # Produce a prediction at every time step.
        for t in range(self.num_t):

            feat_t = all_time_feats[t]  # [B, num_stab, D]

            valid_z_t = self.valid_z[t, :]
            valid_x_t = self.valid_x[t, :]
            valid_all_t = torch.cat([valid_z_t, valid_x_t], dim=0)  # [num_stab]

            padding_mask = (
                (valid_all_t == 0).unsqueeze(0).expand(B, -1)
            )  # [B, num_stab]
            mask_float = valid_all_t.view(1, -1, 1)  # [1, num_stab, 1]
            den = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)

            # Mean-pool valid nodes at step t as the query initialization.
            pooled = (feat_t * mask_float).sum(dim=1, keepdim=True) / den
            q = pooled + self.logical_query_embed.expand(B, -1, -1)

            for layer in self.readout_layers:
                q = layer(q, feat_t, padding_mask)

            q = q + self.res_dense1(q)
            q = q + self.res_dense2(q)
            out_t = self.head(self.head_norm(q)).squeeze(-1).squeeze(-1)

            step_predictions.append(out_t)

        if self.training:
            return torch.stack(step_predictions, dim=1)
        else:
            return step_predictions[-1]


# =========================================================================
# Online surface code dataset with MWPM pseudo-labeling
# =========================================================================


class OnlineSurfaceCodeDataset(IterableDataset):
    """Iterable dataset generating surface code samples on the fly.

    Samples syndrome/observable pairs from a Stim circuit and, during
    training, generates MWPM-based pseudo-labels at each intermediate
    time step for progressive distillation.

    In evaluation mode (is_eval=True), MWPM computation is skipped to
    save CPU time.

    Args:
        d: Surface code distance.
        rounds: Number of syndrome extraction rounds.
        p: Physical error rate.
        batch_size: Samples per batch.
        rank: GPU rank (for distributed training).
        world_size: Total number of GPUs.
        is_eval: If True, skip MWPM pseudo-label generation.
    """

    def __init__(self, d, rounds, p, batch_size, rank=0, world_size=1, is_eval=False):
        self.d = d
        self.rounds = rounds
        self.p = p
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.is_eval = is_eval
        self.mapper = FullMapper(d, rounds)
        self.circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=rounds,
            distance=d,
            after_clifford_depolarization=p,
            after_reset_flip_probability=p,
            before_round_data_depolarization=p,
            before_measure_flip_probability=p,
        )
        self.sampler = self.circuit.compile_detector_sampler()

        # 1. Instantiate MWPM teacher (unused in eval mode, but kept for
        #    code compatibility).
        dem = self.circuit.detector_error_model(decompose_errors=True)
        self.matcher = pymatching.Matching.from_detector_error_model(dem)

        # 2. Precompute time-step masks for FakeEnd generation.
        coords = self.circuit.get_detector_coordinates()
        self.unique_times = sorted(list(set(t for _, _, t in coords.values())))
        self.num_t = len(self.unique_times)

        # In evaluation mode, skip time_masks to save memory and time.
        self.time_masks = []
        if not self.is_eval:
            keep_idx = []
            for t_val in self.unique_times:
                current_idx = [
                    idx for idx, (_, _, t) in coords.items() if math.isclose(t, t_val)
                ]
                keep_idx.extend(current_idx)
                mask = np.zeros(len(coords), dtype=np.uint8)
                mask[keep_idx] = 1
                self.time_masks.append(mask)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        entropy = int.from_bytes(os.urandom(4), byteorder="little")
        time_ns = time.time_ns()
        final_seed = (time_ns + entropy + worker_id * 10000 + self.rank * 100000) % (
            2**32 - 1
        )
        np.random.seed(final_seed)
        torch.manual_seed(final_seed)
        sampler = self.circuit.compile_detector_sampler(seed=final_seed)

        while True:
            det, obs = sampler.sample(self.batch_size, separate_observables=True)
            x_tensor = torch.from_numpy(det).float()
            y_true_tensor = torch.from_numpy(obs).float()

            # In evaluation mode, return directly (skip MWPM computation).
            if self.is_eval:
                yield x_tensor, torch.empty(0), y_true_tensor
                continue

            # MWPM FakeEnd generation (training only).
            y_fake_list = []
            for t in range(self.num_t):
                masked_det = det * self.time_masks[t]
                predicted_obs = self.matcher.decode_batch(masked_det)
                y_fake_list.append(torch.from_numpy(predicted_obs).float())

            y_fake_tensor = torch.stack(y_fake_list, dim=1).squeeze(-1)
            yield x_tensor, y_fake_tensor, y_true_tensor

    def get_info(self):
        """Return dataset metadata.

        Returns:
            Tuple (None, None, mapper) where mapper is the FullMapper
            instance.
        """
        return None, None, self.mapper


# =========================================================================
# Training utilities
# =========================================================================


def set_freeze_mode(model, mode="freeze_backbone", rank=0):
    """Set gradient requirements for transfer learning.

    Args:
        model: The AlphaQubitV2 model.
        mode: "unfreeze_all" or "freeze_backbone". In freeze_backbone
            mode, only the readout, head, logical_query_embed, and
            res_dense parameters are trainable.
        rank: GPU rank (logging only on rank 0).
    """
    if rank == 0:
        print(f"--> [Model Status] Switching to: {mode}")
    for name, param in model.named_parameters():
        if mode == "unfreeze_all":
            param.requires_grad = True
        elif mode == "freeze_backbone":
            # Unfreeze only the final aggregation and readout layers;
            # freeze Stem, RNN, TF, and Embedding.
            if any(
                k in name
                for k in [
                    "readout",
                    "head",
                    "logical_query_embed",
                    "res_dense",
                ]
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False

    if rank == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(
            f"--> [Info] Trainable Params: {trainable / 1e6:.2f}M"
            f" / {total / 1e6:.2f}M"
        )


def log_layer_gradients(model, step, rank):
    """Log gradient norms grouped by layer type.

    Args:
        model: The model.
        step: Current training step.
        rank: GPU rank (logging only on rank 0).
    """
    if rank != 0:
        return
    norms = {"Stem": 0, "RNN": 0, "TF": 0, "Head": 0}
    counts = {"Stem": 0, "RNN": 0, "TF": 0, "Head": 0}
    for n, p in model.named_parameters():
        if p.grad is not None:
            val = p.grad.data.norm(2).item()
            if "stem" in n or "emb" in n:
                k = "Stem"
            elif "head" in n or "readout" in n:
                k = "Head"
            elif "rnn" in n or "gru" in n:
                k = "RNN"
            else:
                k = "TF"
            norms[k] += val
            counts[k] += 1
    s = f"[Grad {step}] "
    for k in norms:
        if counts[k] > 0:
            s += f"{k}: {norms[k] / counts[k]:.4f} | "
    print(s)


# =========================================================================
# Training loop
# =========================================================================


def run_training(args):
    """Run the full distributed training loop.

    Handles DDP setup, data loading, progressive distillation loss,
    validation, checkpointing, and early stopping.

    Args:
        args: Parsed command-line arguments.
    """
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    if rank == 0:
        print(
            f"\n=== ALPHA-QUBIT V2: d={args.d} | X+Z Joint |" " RNN+TF Interleaved ==="
        )

    # Build mapper from a temporary dataset.
    temp_ds = OnlineSurfaceCodeDataset(
        args.d, args.rounds, args.train_p, 1, rank=rank, world_size=world_size
    )
    mapper = temp_ds.get_info()[2]

    if rank == 0:
        print(f"    Z stabilizers: {mapper.mapping_info.num_z}")
        print(f"    X stabilizers: {mapper.mapping_info.num_x}")
        print(
            f"    Total per step:"
            f" {mapper.mapping_info.num_z + mapper.mapping_info.num_x}"
        )
        print(f"    Time steps: {mapper.mapping_info.num_t}")

    # Precompute time-progressive distillation weights w_t.
    num_t = mapper.mapping_info.num_t
    # Weights from 1/T to (T-1)/T, shape [1, num_t-1].
    w_t = torch.tensor(
        [(t + 1.0) / num_t for t in range(num_t - 1)], device=device
    ).view(1, -1)

    worker_per_gpu = max(4, 50 // world_size)
    if rank == 0:
        print(f"--> [Performance] Using {worker_per_gpu} CPU workers per GPU")

    # Training loader: requires MWPM pseudo-labels, heavily CPU-bound.
    train_loader = DataLoader(
        OnlineSurfaceCodeDataset(
            args.d,
            args.rounds,
            args.train_p,
            args.batch_size,
            rank=rank,
            world_size=world_size,
        ),
        batch_size=None,
        num_workers=worker_per_gpu,
        pin_memory=True,
        prefetch_factor=2,
    )

    # Validation loaders: is_eval=True skips MWPM, fewer workers needed.
    val_high_loader = DataLoader(
        OnlineSurfaceCodeDataset(
            args.d,
            args.rounds,
            args.train_p,
            args.batch_size,
            rank=rank,
            world_size=world_size,
            is_eval=True,
        ),
        batch_size=None,
        num_workers=4,
        pin_memory=True,
    )
    val_low_loader = DataLoader(
        OnlineSurfaceCodeDataset(
            args.d,
            args.rounds,
            args.eval_p,
            args.batch_size,
            rank=rank,
            world_size=world_size,
            is_eval=True,
        ),
        batch_size=None,
        num_workers=4,
        pin_memory=True,
    )

    model = AlphaQubitV2(mapper, d_model=512, n_heads=8).to(device)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"    Total parameters: {total_params / 1e6:.2f}M")

    start_step = 0
    # Resolve Hugging Face checkpoint if --hf_resume is provided.
    if args.hf_resume and not args.resume:
        parts = args.hf_resume.split("/")
        if len(parts) >= 3:
            repo_id = "/".join(parts[:2])
            filename = "/".join(parts[2:])
        else:
            repo_id = args.hf_resume
            filename = f"surface/d{args.d}.pth"
        if rank == 0:
            print(f"--> Downloading from HF: {repo_id}/{filename}")
        args.resume = download_from_hf(repo_id, filename)

    saved_d = -1
    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            print(f"--> Loading Checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        raw_state_dict = ckpt["model_state"]
        clean_state_dict = {}
        for k, v in raw_state_dict.items():
            new_k = k.replace("_orig_mod.", "").replace("module.", "")
            clean_state_dict[new_k] = v
        saved_d = ckpt.get("d", -1)
        saved_rounds = ckpt.get("rounds", -1)
        if saved_rounds == -1 and saved_d != -1:
            saved_rounds = saved_d
        if saved_d != args.d or saved_rounds != args.rounds:
            if rank == 0:
                print(
                    f"--> Transfer detected (d={saved_d},r={saved_rounds}"
                    f" -> d={args.d},r={args.rounds}). Dropping"
                    " Coord/Mapper buffers."
                )
            exclude_keys = [
                "coords",
                "freq",
                "mapper",
                "mask",
                "neighbors",
                "gather",
                "valid",
                "hint",
            ]
            clean_state_dict = {
                k: v
                for k, v in clean_state_dict.items()
                if not any(ex in k for ex in exclude_keys)
            }
            start_step = 0
        else:
            if rank == 0:
                print("--> Fine-tuning from base model, resetting step to 0.")
            start_step = 0

        # ==========================================================
        # Ablation: precise scaling of TF and Readout weights
        # (disabled; retained for reference)
        # ==========================================================

        # if saved_d != args.d or saved_rounds != args.rounds:
        #     if rank == 0:
        #         print("\n" + "*" * 60)
        #         print("--> [Ablation] Precisely scaling up TF and"
        #               " Readout weights using empirically determined"
        #               " factors!")
        #         print("*" * 60 + "\n")
        #
        #     scale_factor_readout = 1.055
        #     scale_factor_tf_qk = 1.042
        #
        #     for name, param in clean_state_dict.items():
        #         if not param.is_floating_point():
        #             continue
        #         if "norm" in name or "weight" not in name:
        #             continue
        #         if "tf_layers" in name:
        #             clean_state_dict[name] = param * scale_factor_tf_qk
        #         elif any(k in name for k in [
        #             "readout", "head", "logical_query_embed", "res_dense"
        #         ]):
        #             clean_state_dict[name] = param * scale_factor_readout

        model.load_state_dict(clean_state_dict, strict=False)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Enable gradients globally.
    for p in model.parameters():
        p.requires_grad = True

    def build_param_groups_for_adamw(m):
        """Separate parameters into weight-decay and no-decay groups.

        No weight decay is applied to: biases, norm parameters,
        embeddings, and the final head/query parameters.
        """
        decay, no_decay = [], []
        for n, p in m.named_parameters():
            if not p.requires_grad:
                continue
            name = n.lower()
            if (
                name.endswith("bias")
                or "norm" in name
                or "embedding" in name
                or name.startswith("emb_")
                or ".emb_" in name
            ):
                no_decay.append(p)
            elif (
                "logical_query_embed" in name
                or name.startswith("head")
                or ".head" in name
            ):
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": 1e-2},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    base_model = model.module if dist.is_initialized() else model
    param_groups = build_param_groups_for_adamw(base_model)
    optimizer = optim.AdamW(param_groups, lr=args.lr, fused=True)

    # Cosine learning rate schedule with linear warmup.
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
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        return min_lr_ratio

    last_epoch_val = start_step - 1 if start_step > 0 else -1
    if last_epoch_val >= 0:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch_val
    )
    model.train()
    ACCUM = max(1, 4096 // (args.batch_size * world_size))
    micro = 0
    iterator = iter(train_loader)
    update_step = start_step
    metrics = torch.zeros(2, device=device)

    while update_step < args.max_steps:
        try:
            X, Y_fake, Y_true = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            X, Y_fake, Y_true = next(iterator)

        # Progressive distillation: MWPM pseudo-labels + final true label.
        X = X.to(device, non_blocking=True)
        Y_fake = Y_fake.to(device, non_blocking=True)
        Y_true = Y_true.to(device, non_blocking=True)

        progress = update_step / args.max_steps

        if progress < 0.5:
            current_mask_prob = 0.8
            current_drop_ratio = 0.5
        elif progress < 0.8:
            # Linearly decay dropout over [50%, 80%] of training.
            decay_factor = 1.0 - (progress - 0.5) / 0.3
            current_mask_prob = 0.8 * decay_factor
            current_drop_ratio = 0.5 * decay_factor
        else:
            current_mask_prob = 0.0
            current_drop_ratio = 0.0

        my_context = (
            model.no_sync()
            if dist.is_initialized() and (micro + 1) % ACCUM != 0
            else contextlib.nullcontext()
        )

        with my_context:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred_all_steps = model(
                    X,
                    mask_prob=current_mask_prob,
                    drop_ratio=current_drop_ratio,
                )  # [B, num_t]

                progress = min(1.0, update_step / (args.max_steps * 0.6))
                beta = 8.0 - 7.8 * progress  # 8.0 -> 0.2

                # Split intermediate steps (1 to T-1) and final step (T).
                pred_intermediate = pred_all_steps[:, :-1]  # [B, num_t-1]
                y_mwpm_intermediate = Y_fake[:, :-1]  # [B, num_t-1]

                pred_final = pred_all_steps[:, -1]  # [B]
                y_true_final = Y_true.squeeze(-1)  # [B]

                # Process supervision (MWPM distillation).
                w_t_sum = w_t.sum().clamp_min(1e-6)
                bce_inter = F.binary_cross_entropy_with_logits(
                    pred_intermediate,
                    y_mwpm_intermediate,
                    reduction="none",
                )
                # Weight by time-progressive w_t, sum over time, mean
                # over batch.
                loss_process = (bce_inter * w_t).sum(dim=1).mean() / w_t_sum

                # Final verdict (true label supervision).
                loss_final = F.binary_cross_entropy_with_logits(
                    pred_final, y_true_final
                )

                # Composite loss.
                loss = (loss_final + beta * loss_process) / ACCUM

            loss.backward()

        micro += 1

        if micro % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if update_step % 2500 == 0:
                log_layer_gradients(model, update_step, rank)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if update_step > 0 and update_step % args.eval_freq == 0:
                acc_high = validate(
                    model,
                    val_high_loader,
                    device,
                    args.eval_samples,
                    world_size,
                )
                acc_low = validate(
                    model,
                    val_low_loader,
                    device,
                    args.eval_samples,
                    world_size,
                )
                metrics = torch.tensor([acc_high, acc_low], device=device)
                if dist.is_initialized():
                    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                    metrics /= world_size
                if rank == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"Step {update_step} | LR {lr:.2e} |"
                        f" High: {metrics[0]:.5f} |"
                        f" Low: {metrics[1]:.5f}",
                        flush=True,
                    )

                    # Write to log file for downstream plotting.
                    if args.log_file:
                        with open(args.log_file, "a") as f:
                            f.write(
                                f"{update_step},{metrics[0]:.5f}," f"{metrics[1]:.5f}\n"
                            )

                    torch.save(
                        {
                            "model_state": (
                                model.module if dist.is_initialized() else model
                            ).state_dict(),
                            "d": args.d,
                            "rounds": args.rounds,
                            "step": update_step,
                        },
                        args.output,
                    )
                if metrics[0] >= args.target_high and metrics[1] >= args.target_low:
                    if rank == 0:
                        print(
                            f"\n[SUCCESS] Target Reached at Step"
                            f" {update_step}! High: {metrics[0]:.5f},"
                            f" Low: {metrics[1]:.5f}"
                        )
                        print("--> Stopping training early to save compute.")
                    break
            update_step += 1

            # Early stop at a user-specified step.
            if args.stop_step > 0 and update_step >= args.stop_step:
                if rank == 0:
                    print(
                        f"\n[Early Stop] Reached specified stop_step:"
                        f" {args.stop_step}. Saving and exiting..."
                    )
                    torch.save(
                        {
                            "model_state": (
                                model.module if dist.is_initialized() else model
                            ).state_dict(),
                            "d": args.d,
                            "rounds": args.rounds,
                            "step": update_step,
                        },
                        args.output,
                    )
                break
    cleanup_ddp()


def validate(model, loader, device, target_samples=50000, world_size=1):
    """Evaluate model accuracy on a validation loader.

    Args:
        model: The model (in eval mode during this call).
        loader: Validation DataLoader.
        device: Target device.
        target_samples: Total number of validation samples across all
            GPUs.
        world_size: Number of GPUs.

    Returns:
        Accuracy as a float in [0, 1].
    """
    model.eval()
    correct, total = 0, 0
    target_per_rank = target_samples // world_size

    with torch.no_grad():
        for i, (X, Y_fake, Y_true) in enumerate(loader):
            X, Y_true = X.to(device), Y_true.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # In eval mode, the model returns only the final step [B].
                pred = (model(X) > 0).float().view(-1, 1)
            correct += (pred == Y_true).float().sum().item()
            total += X.size(0)

            if total >= target_per_rank:
                break

    model.train()
    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train AlphaQubit V2 neural decoder on surface codes."
    )
    parser.add_argument("--d", type=int, required=True, help="Surface code distance.")
    parser.add_argument(
        "--train_p",
        type=float,
        default=0.008,
        help="Physical error rate for training (default: 0.008).",
    )
    parser.add_argument(
        "--eval_p",
        type=float,
        default=0.005,
        help="Physical error rate for evaluation (default: 0.005).",
    )
    parser.add_argument(
        "--target_high",
        type=float,
        required=True,
        help="Target accuracy at train_p for early stopping.",
    )
    parser.add_argument(
        "--target_low",
        type=float,
        required=True,
        help="Target accuracy at eval_p for early stopping.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Per-GPU batch size (default: 128).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100000,
        help="Maximum number of optimizer steps (default: 100000).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Peak learning rate (default: 5e-5).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to a checkpoint for transfer learning.",
    )
    parser.add_argument(
        "--hf_resume",
        type=str,
        default="",
        help="Hugging Face repo+file (e.g. 'user/repo/surface/d7.pth') for transfer learning.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the trained model checkpoint.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=-1,
        help="Number of syndrome extraction rounds (default: d).",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=100000,
        help="Total validation samples across all GPUs (default: 100000).",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=500,
        help="Evaluate every N optimizer steps (default: 500).",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="",
        help="CSV path for saving evaluation metrics.",
    )
    parser.add_argument(
        "--stop_step",
        type=int,
        default=-1,
        help="Force early stop at this step (-1 disables).",
    )
    args = parser.parse_args()
    if args.rounds == -1:
        args.rounds = args.d
    run_training(args)
