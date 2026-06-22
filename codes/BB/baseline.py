#!/usr/bin/env python3
"""Classical baseline decoders for the BB [[72, 12, 6]] detector model."""
from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from functools import reduce
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.sparse
import stim
from scipy.sparse import identity, kron


DEFAULT_PS = [0.001, 0.002, 0.003, 0.004, 0.005]
BB72 = dict(
    n=72,
    k=12,
    rounds=6,
    l=6,
    m=6,
    A_x_pows=[3],
    A_y_pows=[1, 2],
    B_x_pows=[1, 2],
    B_y_pows=[3],
)


# ============================================================================
# BB [[72, 12, 6]] circuit and DEM generation
# ============================================================================

def row_echelon(mat, reduced=False):
    mat = np.asarray(mat, dtype=bool).copy()
    rows, cols = mat.shape
    transform = np.eye(rows, dtype=bool)
    pivot_row = 0
    pivot_cols = []

    for col in range(cols):
        if pivot_row >= rows:
            break
        if not mat[pivot_row, col]:
            swap = pivot_row + np.argmax(mat[pivot_row:rows, col])
            if mat[swap, col]:
                mat[[swap, pivot_row]] = mat[[pivot_row, swap]]
                transform[[swap, pivot_row]] = transform[[pivot_row, swap]]
        if mat[pivot_row, col]:
            elim = range(rows) if reduced else range(pivot_row + 1, rows)
            for r in elim:
                if r != pivot_row and mat[r, col]:
                    mat[r] ^= mat[pivot_row]
                    transform[r] ^= transform[pivot_row]
            pivot_cols.append(col)
            pivot_row += 1

    return mat.astype(np.uint8), pivot_row, transform.astype(np.uint8), pivot_cols


def kernel(mat):
    transposed = np.asarray(mat).T
    rows, _ = transposed.shape
    _, rank, transform, pivot_cols = row_echelon(transposed)
    return transform[rank:rows], rank, pivot_cols


class CSSCode:
    def __init__(self, hx, hz, check_css=False):
        self.hx = np.asarray(hx, dtype=np.uint8)
        self.hz = np.asarray(hz, dtype=np.uint8)
        if self.hx.shape[1] != self.hz.shape[1]:
            raise ValueError("hx and hz must have the same number of columns")
        if check_css and np.any((self.hx @ self.hz.T) % 2):
            raise ValueError("CSS constraint is not satisfied")

        self.N = self.hx.shape[1]
        self.hx_perp, self.rank_hx, self.pivot_hx = kernel(self.hx)
        self.hz_perp, self.rank_hz, self.pivot_hz = kernel(self.hz)
        self.hx_basis = self.hx[self.pivot_hx]
        self.hz_basis = self.hz[self.pivot_hz]
        self.K = self.N - self.rank_hx - self.rank_hz
        self.lx, self.lz = self.compute_logicals()

    def compute_logicals(self):
        def compute_lz(ker_hx, im_hz_t):
            stack = np.vstack([im_hz_t, ker_hx])
            pivots = row_echelon(stack.T)[3]
            indices = [i for i in range(im_hz_t.shape[0], stack.shape[0]) if i in pivots]
            return stack[indices].astype(np.uint8)

        lx = compute_lz(self.hz_perp, self.hx_basis)
        lz = compute_lz(self.hx_perp, self.hz_basis)
        return lx, lz


def create_circulant_matrix(size, powers):
    out = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for p in powers:
            out[(i + p) % size, i] = 1
    return out


def create_bivariate_bicycle_code(l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows):
    s_l = create_circulant_matrix(l, [-1])
    s_m = create_circulant_matrix(m, [-1])
    x = kron(s_l, identity(m, dtype=np.uint8))
    y = kron(identity(l, dtype=np.uint8), s_m)
    A_list = [x ** p for p in A_x_pows] + [y ** p for p in A_y_pows]
    B_list = [y ** p for p in B_y_pows] + [x ** p for p in B_x_pows]
    A = reduce(lambda a, b: a + b, A_list).toarray() % 2
    B = reduce(lambda a, b: a + b, B_list).toarray() % 2
    hx = np.hstack((A, B)).astype(np.uint8)
    hz = np.hstack((B.T, A.T)).astype(np.uint8)
    return CSSCode(hx, hz, check_css=True), A_list, B_list


def build_circuit(code, A_list, B_list, p, rounds, z_basis=True, use_both=True, hzh=False):
    n = code.N
    a1, a2, a3 = A_list
    b1, b2, b3 = B_list

    def nnz(mat):
        rows, cols = mat.nonzero()
        return cols[np.argsort(rows)]

    A1, A2, A3 = nnz(a1), nnz(a2), nnz(a3)
    B1, B2, B3 = nnz(b1), nnz(b2), nnz(b3)
    A1_T, A2_T, A3_T = nnz(a1.T), nnz(a2.T), nnz(a3.T)
    B1_T, B2_T, B3_T = nnz(b1.T), nnz(b2.T), nnz(b3.T)

    x_check = 0
    l_data = n // 2
    r_data = n
    z_check = 3 * n // 2

    p_clifford = p
    p_reset = p
    p_measure = p
    p_round_data = p

    def detector_initial(round_id):
        text = ""
        for i in range(n // 2):
            text += f"DETECTOR(0, 0, {round_id}) rec[{-n // 2 + i}]\n"
        return stim.Circuit(text)

    def detector_repeat(round_id):
        text = ""
        for i in range(n // 2):
            text += f"DETECTOR(0, 0, {round_id}) rec[{-n // 2 + i}] rec[{-n - n // 2 + i}]\n"
        return stim.Circuit(text)

    def append_blocks(circuit, round_id, repeat=False):
        if repeat:
            for i in range(n // 2):
                circuit.append("X_ERROR", z_check + i, p_reset)
                if hzh:
                    circuit.append("X_ERROR", x_check + i, p_reset)
                    circuit.append("H", [x_check + i])
                    circuit.append("DEPOLARIZE1", x_check + i, p_clifford)
                else:
                    circuit.append("Z_ERROR", x_check + i, p_reset)
                circuit.append("DEPOLARIZE1", r_data + i, p_round_data)
        else:
            for i in range(n // 2):
                circuit.append("H", [x_check + i])
                if hzh:
                    circuit.append("DEPOLARIZE1", x_check + i, p_clifford)

        for i in range(n // 2):
            circuit.append("CNOT", [r_data + A1_T[i], z_check + i])
            circuit.append("DEPOLARIZE2", [r_data + A1_T[i], z_check + i], p_clifford)
            circuit.append("DEPOLARIZE1", l_data + i, p_round_data)
        circuit.append("TICK")

        for i in range(n // 2):
            circuit.append("CNOT", [x_check + i, l_data + A2[i]])
            circuit.append("DEPOLARIZE2", [x_check + i, l_data + A2[i]], p_clifford)
            circuit.append("CNOT", [r_data + A3_T[i], z_check + i])
            circuit.append("DEPOLARIZE2", [r_data + A3_T[i], z_check + i], p_clifford)
        circuit.append("TICK")

        for i in range(n // 2):
            circuit.append("CNOT", [x_check + i, r_data + B2[i]])
            circuit.append("DEPOLARIZE2", [x_check + i, r_data + B2[i]], p_clifford)
            circuit.append("CNOT", [l_data + B1_T[i], z_check + i])
            circuit.append("DEPOLARIZE2", [l_data + B1_T[i], z_check + i], p_clifford)
        circuit.append("TICK")

        for i in range(n // 2):
            circuit.append("CNOT", [x_check + i, r_data + B1[i]])
            circuit.append("DEPOLARIZE2", [x_check + i, r_data + B1[i]], p_clifford)
            circuit.append("CNOT", [l_data + B2_T[i], z_check + i])
            circuit.append("DEPOLARIZE2", [l_data + B2_T[i], z_check + i], p_clifford)
        circuit.append("TICK")

        for i in range(n // 2):
            circuit.append("CNOT", [x_check + i, r_data + B3[i]])
            circuit.append("DEPOLARIZE2", [x_check + i, r_data + B3[i]], p_clifford)
            circuit.append("CNOT", [l_data + B3_T[i], z_check + i])
            circuit.append("DEPOLARIZE2", [l_data + B3_T[i], z_check + i], p_clifford)
        circuit.append("TICK")

        for i in range(n // 2):
            circuit.append("CNOT", [x_check + i, l_data + A1[i]])
            circuit.append("DEPOLARIZE2", [x_check + i, l_data + A1[i]], p_clifford)
            circuit.append("CNOT", [r_data + A2_T[i], z_check + i])
            circuit.append("DEPOLARIZE2", [r_data + A2_T[i], z_check + i], p_clifford)
        circuit.append("TICK")

        for i in range(n // 2):
            circuit.append("CNOT", [x_check + i, l_data + A3[i]])
            circuit.append("DEPOLARIZE2", [x_check + i, l_data + A3[i]], p_clifford)
            circuit.append("X_ERROR", z_check + i, p_measure)
            circuit.append("MR", [z_check + i])

        if z_basis:
            circuit += detector_repeat(round_id) if repeat else detector_initial(round_id)
        elif use_both and repeat:
            circuit += detector_repeat(round_id)
        circuit.append("TICK")

        for i in range(n // 2):
            if hzh:
                circuit.append("H", [x_check + i])
                circuit.append("DEPOLARIZE1", x_check + i, p_clifford)
                circuit.append("X_ERROR", x_check + i, p_measure)
                circuit.append("MR", [x_check + i])
            else:
                circuit.append("Z_ERROR", x_check + i, p_measure)
                circuit.append("MRX", [x_check + i])

        if not z_basis:
            circuit += detector_repeat(round_id) if repeat else detector_initial(round_id)
        elif use_both and repeat:
            circuit += detector_repeat(round_id)
        circuit.append("TICK")

    circuit = stim.Circuit()
    for i in range(n // 2):
        circuit.append("R", x_check + i)
        circuit.append("R", z_check + i)
        circuit.append("X_ERROR", x_check + i, p_reset)
        circuit.append("X_ERROR", z_check + i, p_reset)
    for i in range(n):
        circuit.append("R" if z_basis else "RX", l_data + i)
        circuit.append("X_ERROR" if z_basis else "Z_ERROR", l_data + i, p_reset)

    circuit.append("TICK")
    append_blocks(circuit, round_id=0, repeat=False)
    for round_id in range(1, rounds):
        append_blocks(circuit, round_id=round_id, repeat=True)

    for i in range(n):
        circuit.append("M" if z_basis else "MX", l_data + i)

    pcm = code.hz if z_basis else code.hx
    logical_pcm = code.lz if z_basis else code.lx

    stab_text = ""
    for i, row in enumerate(pcm):
        terms = np.nonzero(row)[0]
        line = f"DETECTOR(0, 0, {rounds})"
        for q in terms:
            line += f" rec[{-n + q}]"
        line += f" rec[{-n - n + i}]" if z_basis else f" rec[{-n - n // 2 + i}]"
        stab_text += line + "\n"
    circuit += stim.Circuit(stab_text)

    obs_text = ""
    for i, row in enumerate(logical_pcm):
        terms = np.nonzero(row)[0]
        line = f"OBSERVABLE_INCLUDE({i})"
        for q in terms:
            line += f" rec[{-n + q}]"
        obs_text += line + "\n"
    circuit += stim.Circuit(obs_text)
    return circuit


def bb72_circuit(p: float) -> stim.Circuit:
    code, A_list, B_list = create_bivariate_bicycle_code(
        BB72["l"],
        BB72["m"],
        BB72["A_x_pows"],
        BB72["A_y_pows"],
        BB72["B_x_pows"],
        BB72["B_y_pows"],
    )
    return build_circuit(code, A_list, B_list, p=p, rounds=BB72["rounds"])


def p_string(p: float) -> str:
    return f"{p:.6g}"


def dem_filename(p: float) -> str:
    return f"{BB72['n']}_{BB72['k']}_{BB72['rounds']}_{p_string(p)}.dem"


def dem_path_for(dem_dir: str | Path, p: float) -> Path:
    dem_dir = Path(dem_dir)
    paths = [
        dem_dir / dem_filename(p),
        dem_dir / f"{BB72['n']}_{BB72['k']}_{BB72['rounds']}_{p:.3f}.dem",
        dem_dir / f"{BB72['n']}_{BB72['k']}_{BB72['rounds']}_{p:.4f}.dem",
    ]
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def generate_dem(path: str | Path, p: float, force: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not force:
        print(f"[skip] {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    circuit = bb72_circuit(p)
    dem = circuit.detector_error_model(flatten_loops=True, decompose_errors=False)
    dem.to_file(str(path))
    print(f"[saved] {path} detectors={dem.num_detectors} observables={dem.num_observables}")
    return path


def ensure_dem(dem_dir: str | Path, p: float) -> Path:
    path = dem_path_for(dem_dir, p)
    if not path.exists():
        print(f"[missing] {path}; generating it now")
        generate_dem(path, p)
    return path


# ============================================================================
# DEM conversion and sampling
# ============================================================================

def load_dem_matrices(path: str | Path):
    dem = stim.DetectorErrorModel.from_file(str(path)).flattened()
    h_rows: list[int] = []
    h_cols: list[int] = []
    l_rows: list[int] = []
    l_cols: list[int] = []
    probs: list[float] = []
    var = 0

    for instr in dem:
        if instr.type != "error":
            continue
        prob = instr.args_copy()[0]
        detectors: list[int] = []
        observables: list[int] = []
        for target in instr.targets_copy():
            if target.is_logical_observable_id():
                observables.append(target.val)
            elif not target.is_separator():
                detectors.append(target.val)
        if not detectors:
            continue
        for det in detectors:
            h_rows.append(det)
            h_cols.append(var)
        for obs in observables:
            l_rows.append(obs)
            l_cols.append(var)
        probs.append(prob)
        var += 1

    H = scipy.sparse.csr_matrix(
        (np.ones(len(h_rows), dtype=np.uint8), (h_rows, h_cols)),
        shape=(dem.num_detectors, var),
    )
    L = scipy.sparse.csr_matrix(
        (np.ones(len(l_rows), dtype=np.uint8), (l_rows, l_cols)),
        shape=(dem.num_observables, var),
    )
    return H, L, np.asarray(probs, dtype=np.float64)


def sample_errors(probs, H_dense, L_T, rng, num_samples):
    errors = (rng.random((num_samples, probs.size)) < probs).astype(np.uint8)
    syndromes = (errors @ H_dense.T) % 2
    observables = (errors @ L_T) % 2
    return syndromes.astype(np.uint8), observables.astype(np.uint8)


@dataclass(frozen=True)
class EvalResult:
    total_samples: int
    correct: int
    errors: int
    acc: float
    ler: float
    time_s: float
    wall_us_per_sample: float


def evaluate_decoder(decoder, num_samples: int, chunk_size: int, seed: int) -> EvalResult:
    rng = np.random.default_rng(seed)
    correct = 0
    done = 0
    t0 = time.perf_counter()

    while done < num_samples:
        n = min(chunk_size, num_samples - done)
        syndromes, true_observables = sample_errors(
            decoder.probs, decoder.H_dense, decoder.L_T, rng, n
        )
        pred_observables = decoder.decode_observables_batch(syndromes)
        correct += int(np.all(pred_observables == true_observables, axis=1).sum())
        done += n
        if done % max(10_000, chunk_size) == 0:
            print(f"[progress] {done}/{num_samples} acc={correct / done:.6f}", flush=True)

    dt = time.perf_counter() - t0
    acc = correct / num_samples
    return EvalResult(
        total_samples=num_samples,
        correct=correct,
        errors=num_samples - correct,
        acc=acc,
        ler=1.0 - acc,
        time_s=dt,
        wall_us_per_sample=dt * 1e6 / num_samples,
    )


# ============================================================================
# BP+OSD
# ============================================================================

@dataclass(frozen=True)
class BposdConfig:
    max_iter: int = 12
    osd_order: int = 10
    bp_method: str = "product_sum"
    osd_method: str = "osd_cs"


class BposdLogicalDecoder:
    def __init__(self, dem_path: str | Path, config: BposdConfig):
        from ldpc import BpOsdDecoder

        H, L, probs = load_dem_matrices(dem_path)
        self.H = H
        self.L = L
        self.probs = probs
        self.H_dense = H.toarray().astype(np.uint8)
        self.L_T = L.toarray().T.astype(np.uint8)
        self.config = config
        self.decoder = BpOsdDecoder(
            H,
            channel_probs=probs,
            max_iter=config.max_iter,
            bp_method=config.bp_method,
            osd_method=config.osd_method,
            osd_order=config.osd_order,
        )

    def decode_observables(self, syndrome):
        correction = self.decoder.decode(np.asarray(syndrome, dtype=np.uint8))
        return ((correction @ self.L_T) % 2).astype(np.uint8)

    def decode_observables_batch(self, syndromes):
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        return np.asarray([self.decode_observables(s) for s in syndromes], dtype=np.uint8)


_BPOSD_WORKER = {}


def _init_bposd_worker(dem_path, config):
    _BPOSD_WORKER["decoder"] = BposdLogicalDecoder(dem_path, config)


def _decode_bposd_chunk(task):
    seed, num_samples = task
    decoder = _BPOSD_WORKER["decoder"]
    rng = np.random.default_rng(seed)
    syndromes, true_observables = sample_errors(
        decoder.probs, decoder.H_dense, decoder.L_T, rng, num_samples
    )
    pred_observables = decoder.decode_observables_batch(syndromes)
    correct = int(np.all(pred_observables == true_observables, axis=1).sum())
    return correct, num_samples


def build_tasks(seed: int, total_samples: int, chunk_samples: int):
    tasks = []
    done = 0
    idx = 0
    while done < total_samples:
        n = min(chunk_samples, total_samples - done)
        tasks.append((seed + idx * 1_000_003, n))
        done += n
        idx += 1
    return tasks


def run_bposd_one_p(dem_path, config, num_samples, workers, chunk_samples, seed):
    tasks = build_tasks(seed, num_samples, chunk_samples)
    correct = 0
    done = 0
    t0 = time.perf_counter()
    with Pool(processes=workers, initializer=_init_bposd_worker, initargs=(str(dem_path), config)) as pool:
        for chunk_correct, chunk_n in pool.imap_unordered(_decode_bposd_chunk, tasks):
            correct += chunk_correct
            done += chunk_n
            if done % max(10_000, chunk_samples) == 0:
                print(f"[progress] {done}/{num_samples} acc={correct / done:.6f}", flush=True)
    dt = time.perf_counter() - t0
    acc = correct / num_samples
    return EvalResult(
        total_samples=num_samples,
        correct=correct,
        errors=num_samples - correct,
        acc=acc,
        ler=1.0 - acc,
        time_s=dt,
        wall_us_per_sample=dt * 1e6 / num_samples,
    )


# ============================================================================
# RelayBP
# ============================================================================

@dataclass(frozen=True)
class RelayBpConfig:
    name: str
    alpha: float = 1.0
    gamma0: float = 0.15
    gamma_dist_interval: tuple[float, float] = (-0.226, 0.622)
    num_sets: int = 60
    pre_iter: int = 12
    set_max_iter: int = 12
    stop_nconv: int = 1


RELAYBP_PRESETS = {
    "2d-reduced": RelayBpConfig(
        name="relaybp2d_reduced",
        num_sets=60,
        pre_iter=12,
        set_max_iter=12,
        stop_nconv=1,
    ),
    "full": RelayBpConfig(
        name="relaybp_full",
        num_sets=300,
        pre_iter=80,
        set_max_iter=60,
        stop_nconv=5,
    ),
}


class RelayBpObservableDecoder:
    def __init__(self, dem_path: str | Path, config: RelayBpConfig):
        import relay_bp

        H, L, probs = load_dem_matrices(dem_path)
        self.H = H
        self.L = L
        self.probs = probs
        self.H_dense = H.toarray().astype(np.uint8)
        self.L_T = L.toarray().T.astype(np.uint8)
        self.config = config
        self.decoder = relay_bp.RelayDecoderF64(
            H,
            error_priors=probs,
            alpha=config.alpha,
            gamma0=config.gamma0,
            pre_iter=config.pre_iter,
            num_sets=config.num_sets,
            set_max_iter=config.set_max_iter,
            gamma_dist_interval=config.gamma_dist_interval,
            stop_nconv=config.stop_nconv,
        )
        self.runner = relay_bp.ObservableDecoderRunner(
            self.decoder,
            L,
            include_decode_result=False,
        )

    def decode_observables_batch(self, syndromes, parallel=True):
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        return np.asarray(
            self.runner.decode_observables_batch(syndromes, parallel=parallel),
            dtype=np.uint8,
        )

    def decode_observables(self, syndrome):
        return self.decode_observables_batch(np.asarray([syndrome], dtype=np.uint8))[0]


# ============================================================================
# Command-line interface
# ============================================================================

def write_header(path: Path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def append_row(path: Path, row: dict, fieldnames):
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)


def add_shared_sweep_args(parser):
    parser.add_argument("--dem_dir", default="data/dems")
    parser.add_argument("--ps", nargs="+", type=float, default=list(reversed(DEFAULT_PS)))
    parser.add_argument("--num_samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260528)


def main_generate_dems(args):
    for p in args.ps:
        path = Path(args.out_dir) / dem_filename(p)
        generate_dem(path, p, force=args.force)


def main_bposd(args):
    config = BposdConfig(
        max_iter=args.max_iter,
        osd_order=args.osd_order,
        bp_method=args.bp_method,
        osd_method=args.osd_method,
    )
    fieldnames = [
        "config", "p", "total_samples", "correct", "errors", "acc", "ler",
        "time_s", "wall_us_per_sample", "seed", "workers", "chunk_samples",
        "max_iter", "osd_order", "bp_method", "osd_method", "dem_path",
    ]
    out_path = Path(args.out)
    write_header(out_path, fieldnames)
    print(f"[config] bposd2d {config}")
    print(f"[workers] {args.workers}")

    for idx, p in enumerate(args.ps):
        dem_path = ensure_dem(args.dem_dir, p)
        seed = args.seed + idx * 1_000_003 + int(round(p * 1_000_000))
        print(f"[start] bposd2d p={p_string(p)} samples={args.num_samples} seed={seed}")
        result = run_bposd_one_p(
            dem_path=dem_path,
            config=config,
            num_samples=args.num_samples,
            workers=args.workers,
            chunk_samples=args.chunk_samples,
            seed=seed,
        )
        row = {
            "config": "bposd2d",
            "p": p_string(p),
            "total_samples": result.total_samples,
            "correct": result.correct,
            "errors": result.errors,
            "acc": f"{result.acc:.8f}",
            "ler": f"{result.ler:.8f}",
            "time_s": f"{result.time_s:.3f}",
            "wall_us_per_sample": f"{result.wall_us_per_sample:.3f}",
            "seed": seed,
            "workers": args.workers,
            "chunk_samples": args.chunk_samples,
            "max_iter": config.max_iter,
            "osd_order": config.osd_order,
            "bp_method": config.bp_method,
            "osd_method": config.osd_method,
            "dem_path": str(dem_path),
        }
        append_row(out_path, row, fieldnames)
        print(f"[done] p={p_string(p)} acc={row['acc']} ler={row['ler']} time_s={row['time_s']}")


def main_relaybp(args):
    config = RELAYBP_PRESETS[args.preset]
    fieldnames = [
        "config", "p", "total_samples", "correct", "errors", "acc", "ler",
        "time_s", "wall_us_per_sample", "seed", "chunk_size",
        "rayon_num_threads", "num_sets", "pre_iter", "set_max_iter",
        "stop_nconv", "alpha", "gamma0", "gamma_low", "gamma_high", "dem_path",
    ]
    out_path = Path(args.out)
    write_header(out_path, fieldnames)
    print(f"[config] {config}")
    print(f"[threads] RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS', '')}")

    for idx, p in enumerate(args.ps):
        dem_path = ensure_dem(args.dem_dir, p)
        seed = args.seed + idx * 1_000_003 + int(round(p * 1_000_000))
        print(f"[start] {config.name} p={p_string(p)} samples={args.num_samples} seed={seed}")
        decoder = RelayBpObservableDecoder(dem_path, config)
        result = evaluate_decoder(
            decoder,
            num_samples=args.num_samples,
            chunk_size=args.chunk_size,
            seed=seed,
        )
        gamma_low, gamma_high = config.gamma_dist_interval
        row = {
            "config": config.name,
            "p": p_string(p),
            "total_samples": result.total_samples,
            "correct": result.correct,
            "errors": result.errors,
            "acc": f"{result.acc:.8f}",
            "ler": f"{result.ler:.8f}",
            "time_s": f"{result.time_s:.3f}",
            "wall_us_per_sample": f"{result.wall_us_per_sample:.3f}",
            "seed": seed,
            "chunk_size": args.chunk_size,
            "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS", ""),
            "num_sets": config.num_sets,
            "pre_iter": config.pre_iter,
            "set_max_iter": config.set_max_iter,
            "stop_nconv": config.stop_nconv,
            "alpha": config.alpha,
            "gamma0": config.gamma0,
            "gamma_low": gamma_low,
            "gamma_high": gamma_high,
            "dem_path": str(dem_path),
        }
        append_row(out_path, row, fieldnames)
        print(f"[done] p={p_string(p)} acc={row['acc']} ler={row['ler']} time_s={row['time_s']}")


def build_parser():
    parser = argparse.ArgumentParser(description="BB72 baseline decoders.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-dems", help="Generate BB72 detector error models.")
    gen.add_argument("--ps", nargs="+", type=float, default=DEFAULT_PS)
    gen.add_argument("--out_dir", default="data/dems")
    gen.add_argument("--force", action="store_true")
    gen.set_defaults(func=main_generate_dems)

    bposd = sub.add_parser("bposd", help="Run the BP+OSD baseline.")
    add_shared_sweep_args(bposd)
    bposd.add_argument("--workers", type=int, default=24)
    bposd.add_argument("--chunk_samples", type=int, default=1000)
    bposd.add_argument("--max_iter", type=int, default=12)
    bposd.add_argument("--osd_order", type=int, default=10)
    bposd.add_argument("--bp_method", default="product_sum")
    bposd.add_argument("--osd_method", default="osd_cs")
    bposd.add_argument("--out", default="experiments/baselines/bposd_bb72.csv")
    bposd.set_defaults(func=main_bposd)

    relay = sub.add_parser("relaybp", help="Run the RelayBP baseline.")
    add_shared_sweep_args(relay)
    relay.add_argument("--preset", choices=sorted(RELAYBP_PRESETS), default="2d-reduced")
    relay.add_argument("--chunk_size", type=int, default=10_000)
    relay.add_argument("--out", default="experiments/baselines/relaybp_bb72.csv")
    relay.set_defaults(func=main_relaybp)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
