"""PyMatching baseline evaluation for rotated surface codes.

Provides both correlated and standard PyMatching decoding modes for
benchmarking logical error rates against neural-network-based decoders.
"""

import argparse
import time

import numpy as np
import pymatching
import stim


def evaluate_ler_pymatching(d, p, num_shots, mode="correlated"):
    """Evaluate the logical error rate (LER) using PyMatching decoding.

    Generates a rotated surface code circuit with circuit-level noise,
    extracts the detector error model, and runs batched Monte Carlo
    sampling to estimate the logical error rate.

    Args:
        d: Surface code distance.
        p: Physical error rate for circuit-level noise.
        num_shots: Total number of Monte Carlo samples.
        mode: Decoding mode, either "correlated" (enable_correlations=True)
            or "standard" (enable_correlations=False).

    Returns:
        A tuple (ler, total_errors) where ler is the logical error rate
        as a float and total_errors is the integer error count.
    """
    enable_corr = mode == "correlated"

    print(f"--- Starting {mode} evaluation for d={d}, p={p} ---")
    start_time = time.time()

    # 1. Generate the surface code circuit with circuit-level noise.
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )

    # 2. Extract the detector error model with decomposed errors.
    dem = circuit.detector_error_model(decompose_errors=True)

    # 3. Initialize PyMatching from the DEM.
    matcher = pymatching.Matching.from_detector_error_model(
        dem, enable_correlations=enable_corr
    )

    # 4. Compile the detector sampler.
    sampler = circuit.compile_detector_sampler()

    # 5. Run batched Monte Carlo simulation.
    batch_size = min(50000, num_shots)
    num_batches = num_shots // batch_size
    remainder = num_shots % batch_size

    total_errors = 0

    def process_batch(shots):
        if shots <= 0:
            return 0
        measurements, observables = sampler.sample(
            shots, separate_observables=True
        )
        predicted_observables = matcher.decode_batch(
            measurements, enable_correlations=enable_corr
        )
        errors = np.sum(np.any(predicted_observables != observables, axis=1))
        return errors

    for b in range(num_batches):
        total_errors += process_batch(batch_size)

    total_errors += process_batch(remainder)

    ler = total_errors / num_shots
    elapsed_time = time.time() - start_time

    print(
        f"Completed d={d}: {total_errors} errors / {num_shots} shots"
        f" = {ler:.6e} (Took {elapsed_time:.2f}s)\n"
    )
    return ler, total_errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate PyMatching logical error rate on surface codes."
    )
    parser.add_argument(
        "--d",
        type=int,
        nargs="+",
        required=True,
        help="Surface code distance(s) to evaluate.",
    )
    parser.add_argument(
        "--p",
        type=float,
        required=True,
        help="Physical error rate for circuit-level noise.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=5_000_000,
        help="Number of Monte Carlo samples (default: 5,000,000).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["correlated", "standard"],
        default="correlated",
        help="PyMatching decoding mode (default: correlated).",
    )
    args = parser.parse_args()

    results = {}
    for dist in args.d:
        ler, err_count = evaluate_ler_pymatching(
            d=dist, p=args.p, num_shots=args.shots, mode=args.mode
        )
        results[dist] = {"ler": ler, "errors": err_count}

    print("=== Final Results ===")
    for dist, data in results.items():
        print(
            f"d={dist}, p={args.p} -> {args.mode.capitalize()} LER:"
            f" {data['ler']:.6e} | Total Errors: {data['errors']}"
            f" / {args.shots}"
        )
