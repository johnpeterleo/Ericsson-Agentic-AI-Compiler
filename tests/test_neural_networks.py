"""Compare a plain NumPy MLP with the same MLP compiled by JAX/XLA.

Run the benchmark from the repository root, for example:

    python -m tests.test_neural_networks --jax-backend cpu

Each architecture processes 1,000 input examples in batches of 32 by default.
Compilation and input transfer are excluded from execution timings. NumPy runs
on the CPU; use ``--jax-backend cpu`` for a same-processor comparison, or
``--jax-backend gpu`` to include the effect of running JAX on a GPU.
"""

import argparse
from dataclasses import dataclass
from statistics import mean, median
from time import perf_counter
import unittest

import jax
import numpy as np

from src.compiler.compile import compile_program


def numpy_mlp_forward(hidden_weights, hidden_biases, output_weight, output_bias, inputs):
    """Plain NumPy inference: ReLU hidden layers and a linear output layer."""
    activations = inputs
    for layer in range(hidden_weights.shape[0]):
        activations = np.maximum(
            activations @ hidden_weights[layer] + hidden_biases[layer], 0
        )
    return activations @ output_weight + output_bias


def jax_mlp_forward(hidden_weights, hidden_biases, output_weight, output_bias, inputs):
    """JAX version of the same forward pass, compiled by the project wrapper."""
    activations = inputs
    for layer in range(hidden_weights.shape[0]):
        activations = jax.nn.relu(
            activations @ hidden_weights[layer] + hidden_biases[layer]
        )
    return activations @ output_weight + output_bias


def make_case(depth, width, samples, batch_size, rng, output_size=10):
    """Create one seeded MLP and split its input examples into batches."""
    if min(depth, width, samples, batch_size, output_size) < 1:
        raise ValueError("depth, width, samples, batch_size, and output_size must be positive")

    hidden_weights = rng.standard_normal((depth, width, width), dtype=np.float32)
    hidden_weights /= np.float32(np.sqrt(width))
    hidden_biases = (
        rng.standard_normal((depth, width), dtype=np.float32) * np.float32(0.01)
    )
    output_weight = rng.standard_normal((width, output_size), dtype=np.float32)
    output_weight /= np.float32(np.sqrt(width))
    output_bias = rng.standard_normal((output_size,), dtype=np.float32) * np.float32(0.01)
    input_examples = rng.standard_normal((samples, width), dtype=np.float32)
    batches = tuple(
        input_examples[start : start + batch_size]
        for start in range(0, samples, batch_size)
    )
    parameters = (hidden_weights, hidden_biases, output_weight, output_bias)
    return parameters, batches


@dataclass(frozen=True)
class BenchmarkResult:
    depth: int
    width: int
    samples: int
    numpy_ms: float
    jax_ms: float
    jax_compile_ms: float

    @property
    def speedup(self):
        """Ratio above 1 means compiled JAX was faster than the NumPy baseline."""
        return self.numpy_ms / self.jax_ms


def benchmark_case(*, depth, width, parameters, batches, device, repeats):
    """Compile once per batch shape, check parity, then time both implementations."""
    if repeats < 1:
        raise ValueError("repeats must be positive")

    device_parameters = tuple(jax.device_put(value, device) for value in parameters)
    device_batches = tuple(jax.device_put(batch, device) for batch in batches)
    executables = {}
    compile_ms = 0.0

    # The last batch may be smaller, so compile once for each distinct batch shape.
    for batch, device_batch in zip(batches, device_batches):
        batch_size = batch.shape[0]
        if batch_size in executables:
            continue
        start = perf_counter()
        compilation = compile_program(
            jax_mlp_forward, device_parameters + (device_batch,)
        )
        compile_ms += (perf_counter() - start) * 1000
        executables[batch_size] = compilation.executable

    def run_numpy():
        return tuple(
            numpy_mlp_forward(*parameters, batch)
            for batch in batches
        )

    def run_jax():
        outputs = [
            executables[batch.shape[0]](*device_parameters, batch)
            for batch in device_batches
        ]
        return jax.block_until_ready(outputs)

    # Warm both paths and verify the full 1,000-example workload before timing.
    expected = np.concatenate(run_numpy(), axis=0)
    actual = np.concatenate([np.asarray(value) for value in run_jax()], axis=0)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    numpy_samples = []
    jax_samples = []
    # Alternate order to reduce bias from device warm-up or clock changes.
    for sample_index in range(repeats):
        timed_functions = (
            (("numpy", run_numpy), ("jax", run_jax))
            if sample_index % 2 == 0
            else (("jax", run_jax), ("numpy", run_numpy))
        )
        for implementation, function in timed_functions:
            start = perf_counter()
            function()
            elapsed_ms = (perf_counter() - start) * 1000
            if implementation == "numpy":
                numpy_samples.append(elapsed_ms)
            else:
                jax_samples.append(elapsed_ms)

    return BenchmarkResult(
        depth=depth,
        width=width,
        samples=sum(batch.shape[0] for batch in batches),
        numpy_ms=median(numpy_samples),
        jax_ms=median(jax_samples),
        jax_compile_ms=compile_ms,
    )


def _positive_int_list(value):
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive integers")
    return values


def run_benchmark(
    *,
    depths=tuple(range(1, 11)),
    widths=(32, 128, 256),
    samples=1000,
    batch_size=32,
    output_size=10,
    repeats=20,
    seed=2026,
    jax_backend="default",
):
    """Print per-architecture JAX/NumPy speedups and aggregate statistics."""
    if min(samples, batch_size, output_size, repeats) < 1:
        raise ValueError("samples, batch_size, output_size, and repeats must be positive")

    try:
        device = jax.devices()[0] if jax_backend == "default" else jax.devices(jax_backend)[0]
    except (RuntimeError, IndexError) as error:
        raise ValueError(f"JAX backend {jax_backend!r} is unavailable") from error

    print(f"JAX device: {device.platform} ({device.device_kind})")
    print("Non-JAX baseline: NumPy on CPU")
    if device.platform != "cpu":
        print("This comparison includes both the JAX compiler and the JAX GPU device.")
        print("Use --jax-backend cpu to compare on the same processor.")
    print(
        f"Input examples per architecture: {samples}; batch size: {batch_size}; "
        f"output size: {output_size}; timing repeats: {repeats}"
    )
    print("hidden_layers hidden_width numpy_ms jax_ms compile_ms speedup")

    rng = np.random.default_rng(seed)
    results = []
    for depth in depths:
        for width in widths:
            parameters, batches = make_case(
                depth, width, samples, batch_size, rng, output_size
            )
            result = benchmark_case(
                depth=depth,
                width=width,
                parameters=parameters,
                batches=batches,
                device=device,
                repeats=repeats,
            )
            results.append(result)
            print(
                f"{result.depth:12d} {result.width:12d} "
                f"{result.numpy_ms:8.3f} {result.jax_ms:7.3f} "
                f"{result.jax_compile_ms:10.1f} {result.speedup:7.3f}x"
            )

    speedups = [result.speedup for result in results]
    if speedups:
        print(f"Mean speedup across architectures: {mean(speedups):.3f}x")
        print(f"Median speedup across architectures: {median(speedups):.3f}x")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--depths", type=_positive_int_list, default=tuple(range(1, 11)),
        help="comma-separated hidden-layer counts (default: 1 through 10)",
    )
    parser.add_argument(
        "--widths", type=_positive_int_list, default=(32, 128, 256),
        help="comma-separated neurons per layer (default: 32,128,256)",
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-size", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--jax-backend", choices=("default", "cpu", "gpu"), default="default")
    args = parser.parse_args()
    run_benchmark(
        depths=args.depths,
        widths=args.widths,
        samples=args.samples,
        batch_size=args.batch_size,
        output_size=args.output_size,
        repeats=args.repeats,
        seed=args.seed,
        jax_backend=args.jax_backend,
    )


class NeuralNetworkComparisonTests(unittest.TestCase):
    def test_compiled_jax_matches_plain_numpy_for_all_examples(self):
        rng = np.random.default_rng(7)
        parameters, batches = make_case(
            depth=2, width=4, samples=17, batch_size=8, rng=rng
        )
        result = benchmark_case(
            depth=2,
            width=4,
            parameters=parameters,
            batches=batches,
            device=jax.devices("cpu")[0],
            repeats=1,
        )

        self.assertEqual(result.samples, 17)
        self.assertGreater(result.numpy_ms, 0)
        self.assertGreater(result.jax_ms, 0)
        self.assertGreater(result.jax_compile_ms, 0)
        self.assertGreater(result.speedup, 0)


if __name__ == "__main__":
    main()
