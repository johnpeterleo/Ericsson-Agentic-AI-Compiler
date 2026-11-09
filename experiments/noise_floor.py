"""Measure how much "speedup" the evaluator reports when nothing changed.

Each trial evaluates a program against an identical copy of itself, so the true
speedup is exactly 1.0. The spread of the reported values is the measurement
noise: an agent speedup inside this band is not evidence of a real improvement.

Example:

    python -m experiments.noise_floor --depths 1,4,10 --widths 32,256 --trials 10
"""

import argparse
import json
from pathlib import Path
from statistics import median

import jax
import numpy as np

from src.optimization.evaluator import evaluate
from tests.benchmark_neural_network_agent import (
    REFERENCES,
    _positive_int_list,
    make_agent_inputs,
)

RESULTS_DIR = Path("artifacts/noise-floor")
LOW_PERCENTILE = 5
HIGH_PERCENTILE = 95


def identical_copy(program):
    """A new function object with the same body, so JAX compiles it separately."""

    def copy(*inputs):
        return program(*inputs)

    return copy


def measure_case(program, inputs, trials, repeats):
    """Return the speedups reported for ``trials`` evaluations of program vs itself."""
    speedups = []
    for _ in range(trials):
        result = evaluate(program, identical_copy(program), inputs, repeats=repeats)
        if not result.correct:
            raise RuntimeError(f"Identical program failed evaluation: {result.message}")
        speedups.append(result.speedup)
    return speedups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", type=_positive_int_list, default=(1, 4, 10))
    parser.add_argument("--widths", type=_positive_int_list, default=(32, 128, 256))
    parser.add_argument("--reference", choices=tuple(REFERENCES), default="scan")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    device = jax.devices()[0]
    program = REFERENCES[args.reference]
    precision = jax.config.jax_default_matmul_precision or "default"
    print(f"Device: {device.platform} ({device.device_kind}); matmul precision: {precision}")
    print(f"Reference: {args.reference}; trials: {args.trials}; repeats: {args.repeats}")
    print("depth width median_speedup min_speedup max_speedup")

    rng = np.random.default_rng(args.seed)
    cases = []
    for depth in args.depths:
        for width in args.widths:
            inputs = make_agent_inputs(depth, width, args.samples, args.batch_size, rng)
            speedups = measure_case(program, inputs, args.trials, args.repeats)
            cases.append({"depth": depth, "width": width, "speedups": speedups})
            print(
                f"{depth:5d} {width:5d} {median(speedups):14.3f} "
                f"{min(speedups):11.3f} {max(speedups):11.3f}"
            )

    all_speedups = [value for case in cases for value in case["speedups"]]
    low, high = np.percentile(all_speedups, [LOW_PERCENTILE, HIGH_PERCENTILE])
    print(
        f"\nNoise band ({LOW_PERCENTILE}th-{HIGH_PERCENTILE}th percentile of "
        f"{len(all_speedups)} identical-program speedups): {low:.3f}x to {high:.3f}x"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"{device.platform}_{args.reference}.json"
    summary = {
        "device": f"{device.platform} ({device.device_kind})",
        "matmul_precision": precision,
        "reference": args.reference,
        "trials": args.trials,
        "repeats": args.repeats,
        "noise_band": [float(low), float(high)],
        "cases": cases,
    }
    results_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved to {results_path}")


if __name__ == "__main__":
    main()
