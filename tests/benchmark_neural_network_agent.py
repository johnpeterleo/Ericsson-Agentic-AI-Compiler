"""Run the LLM-driven optimizer on seeded ReLU MLP forward passes.

This is the agentic benchmark. It uses the project's ``OptimizationAgent`` and
``GeminiPolicy``; it makes one LLM request per architecture per iteration. Set
``GEMINI_API_KEY`` in the environment or repository ``.env`` before running.

Example (one small architecture to start):

    python -m tests.benchmark_neural_network_agent --depths 2 --widths 32 --jax-backend cpu

The default sweep covers 1-10 hidden layers and widths 32, 128, and 256. Each
architecture gets 1,000 random input examples, padded to batches of 32.
"""

import argparse
import json
from datetime import datetime
from math import ceil
from pathlib import Path
from statistics import mean, median

import jax
import jax.numpy as jnp
import numpy as np

from src.agent.agent import AgentConfig, OptimizationAgent, format_step_summary
from src.agent.llm_policy import (
    DEFAULT_GEMINI_MODEL,
    GeminiPolicy,
    LLMUnavailableError,
    load_project_env,
)
from src.agent.tools import AgentToolkit, JaxProfiler

RESULTS_DIR = Path("artifacts/agent-runs")


def _default_results_path():
    return RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.jsonl"


def _append_record(results_path, record):
    """Write one JSON line right away, so a crash keeps the earlier results."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a") as results_file:
        results_file.write(json.dumps(record) + "\n")


def _step_record(step):
    evaluation = step.evaluation
    return {
        "status": "evaluated",
        "step": step.index,
        "rationale": step.proposal.rationale,
        "candidate_source": step.proposal.metadata.get("candidate_source"),
        "correct": evaluation.correct,
        "failure_stage": evaluation.failure_stage,
        "message": evaluation.message,
        "reference_ms": evaluation.reference_ms,
        "candidate_ms": evaluation.candidate_ms,
        "speedup": evaluation.speedup,
    }


def mlp_dataset_reference(
    hidden_weights, hidden_biases, output_weight, output_bias, input_batches
):
    """Run each batch through ReLU hidden layers and a linear output layer."""

    def apply_batch(_, batch):
        activations = batch
        for layer in range(hidden_weights.shape[0]):
            activations = jax.nn.relu(
                activations @ hidden_weights[layer] + hidden_biases[layer]
            )
        output = activations @ output_weight + output_bias
        return None, output

    _, batch_outputs = jax.lax.scan(apply_batch, None, input_batches)
    return batch_outputs.reshape((-1, output_weight.shape[-1]))


def mlp_vectorized_reference(
    hidden_weights, hidden_biases, output_weight, output_bias, input_batches
):
    """The same MLP written the efficient way: every batch at once, no scan.

    This is the fair baseline. Any speedup here is beyond a plain rewrite.
    """
    activations = input_batches.reshape((-1, input_batches.shape[-1]))
    for layer in range(hidden_weights.shape[0]):
        activations = jax.nn.relu(
            activations @ hidden_weights[layer] + hidden_biases[layer]
        )
    return activations @ output_weight + output_bias


REFERENCES = {"scan": mlp_dataset_reference, "vectorized": mlp_vectorized_reference}


def make_agent_inputs(depth, width, samples, batch_size, rng, output_size=10):
    """Build one MLP and a padded stack of fixed-size input batches."""
    if min(depth, width, samples, batch_size, output_size) < 1:
        raise ValueError(
            "depth, width, samples, batch_size, and output_size must be positive"
        )

    hidden_weights = rng.standard_normal((depth, width, width), dtype=np.float32)
    hidden_weights /= np.float32(np.sqrt(width))
    hidden_biases = (
        rng.standard_normal((depth, width), dtype=np.float32) * np.float32(0.01)
    )
    output_weight = rng.standard_normal((width, output_size), dtype=np.float32)
    output_weight /= np.float32(np.sqrt(width))
    output_bias = rng.standard_normal((output_size,), dtype=np.float32) * np.float32(0.01)

    input_examples = rng.standard_normal((samples, width), dtype=np.float32)
    padded_samples = ceil(samples / batch_size) * batch_size
    padded_examples = np.zeros((padded_samples, width), dtype=np.float32)
    padded_examples[:samples] = input_examples
    input_batches = padded_examples.reshape(-1, batch_size, width)
    return hidden_weights, hidden_biases, output_weight, output_bias, input_batches


def _positive_int_list(value):
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive integers")
    return values


def run_agent_benchmark(
    *,
    depths,
    widths,
    samples=1000,
    batch_size=32,
    repeats=10,
    iterations=1,
    seed=2026,
    jax_backend="default",
    model=DEFAULT_GEMINI_MODEL,
    results_path=None,
    reference="scan",
):
    """Ask Gemini for JAX rewrites and evaluate them through the project agent."""
    results_path = results_path or _default_results_path()
    reference_fn = REFERENCES[reference]
    if min(samples, batch_size, repeats, iterations) < 1:
        raise ValueError("samples, batch_size, repeats, and iterations must be positive")

    load_project_env()
    policy = GeminiPolicy(model=model)
    try:
        device = (
            jax.devices()[0]
            if jax_backend == "default"
            else jax.devices(jax_backend)[0]
        )
    except (RuntimeError, IndexError) as error:
        raise ValueError(f"JAX backend {jax_backend!r} is unavailable") from error

    hardware_note = f"Target JAX device: {device.platform} ({device.device_kind})."
    agent = OptimizationAgent(
        toolkit=AgentToolkit(
            profiler=JaxProfiler(repeats=repeats), profile_repeats=repeats
        ),
        policy=policy,
        config=AgentConfig(
            max_iterations=iterations,
            benchmark_repeats=repeats,
            hardware_note=hardware_note,
        ),
    )

    print(f"LLM policy: Gemini ({model})")
    print(f"JAX device: {device.platform} ({device.device_kind})")
    print(
        f"Input examples per architecture: {samples}; batch size: {batch_size}; "
        f"hidden layers: {depths}; widths: {widths}; iterations: {iterations}"
    )
    padded_count = ceil(samples / batch_size) * batch_size
    print(f"Batch padding: {padded_count - samples} zero-filled examples")
    print(f"Saving results to: {results_path}")
    matmul_precision = jax.config.jax_default_matmul_precision or "default"
    print(f"Matmul precision: {matmul_precision}")
    print(f"Reference program: {reference}")

    rng = np.random.default_rng(seed)
    successful_speedups = []
    for depth in depths:
        for width in widths:
            host_inputs = make_agent_inputs(depth, width, samples, batch_size, rng)
            device_inputs = tuple(jax.device_put(value, device) for value in host_inputs)
            run_details = {
                "depth": depth,
                "width": width,
                "model": model,
                "device": f"{device.platform} ({device.device_kind})",
                "matmul_precision": matmul_precision,
                "reference": reference,
            }
            try:
                with jax.default_device(device):
                    result = agent.run(reference_fn, device_inputs)
            except (LLMUnavailableError, ValueError) as error:
                print(f"\nArchitecture {depth} x {width} skipped: {error}")
                _append_record(
                    results_path,
                    {**run_details, "status": "skipped", "message": str(error)},
                )
                continue

            baseline_ms = result.context.baseline_profile.total_ms
            baseline_text = (
                f"{baseline_ms:.4f} ms" if baseline_ms is not None else "unavailable"
            )
            print(
                f"\nArchitecture: {depth} hidden layers x {width} neurons; "
                f"baseline={baseline_text} per pass"
            )
            if not result.context.history:
                print(f"Agent stopped: {result.message}")
                _append_record(
                    results_path,
                    {**run_details, "status": "stopped", "message": result.message},
                )
                continue

            case_speedups = []
            for step in result.context.history:
                print(format_step_summary(step))
                _append_record(results_path, {**run_details, **_step_record(step)})
                evaluation = step.evaluation
                if evaluation and evaluation.correct and evaluation.speedup is not None:
                    case_speedups.append(evaluation.speedup)
            if case_speedups:
                best_speedup = max(case_speedups)
                successful_speedups.append(best_speedup)
                print(f"Best agent candidate for this architecture: {best_speedup:.3f}x")

    total_architectures = len(depths) * len(widths)
    print(
        f"\nSuccessful architectures: {len(successful_speedups)}/{total_architectures}"
    )
    if successful_speedups:
        print(f"Mean best-candidate speedup: {mean(successful_speedups):.3f}x")
        print(f"Median best-candidate speedup: {median(successful_speedups):.3f}x")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", type=_positive_int_list, default=tuple(range(1, 11)))
    parser.add_argument("--widths", type=_positive_int_list, default=(32, 128, 256))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--jax-backend", choices=("default", "cpu", "gpu"), default="default")
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--results-file", type=Path, default=None)
    parser.add_argument("--reference", choices=tuple(REFERENCES), default="scan")
    args = parser.parse_args()
    run_agent_benchmark(
        depths=args.depths,
        widths=args.widths,
        samples=args.samples,
        batch_size=args.batch_size,
        repeats=args.repeats,
        iterations=args.iterations,
        seed=args.seed,
        jax_backend=args.jax_backend,
        model=args.model,
        results_path=args.results_file,
        reference=args.reference,
    )


if __name__ == "__main__":
    main()
