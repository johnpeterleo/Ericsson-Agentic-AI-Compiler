"""Compare correctness and latency of JAX functions on one input case.

Functions must be pure, accept array inputs, and return one finite
floating-point array. Evaluation does not measure peak memory.
"""

from dataclasses import dataclass
from math import isfinite
from operator import index
from statistics import median
from time import perf_counter
from typing import Literal

import jax
import numpy as np


@dataclass
class EvaluationResult:
    correct: bool
    reference_ms: float | None = None
    candidate_ms: float | None = None
    failure_stage: Literal["compile", "execute", "correctness", "benchmark"] | None = None
    message: str | None = None

    @property
    def speedup(self) -> float | None:
        """A ratio above 1 means the candidate was faster in this measurement."""
        if self.reference_ms is None or self.candidate_ms is None:
            return None
        return self.reference_ms / self.candidate_ms


def _validate_repeats(repeats):
    repeats = index(repeats)
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    return repeats


def _validate_tolerances(rtol, atol):
    for name, value in (("rtol", rtol), ("atol", atol)):
        if not isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")


def _output_mismatch_reason(reference_output, candidate_output, *, rtol=1e-5, atol=1e-6):
    """Return the first mismatch description, or None when the outputs match."""
    _validate_tolerances(rtol, atol)
    reference = np.asarray(reference_output)
    candidate = np.asarray(candidate_output)

    if reference.shape != candidate.shape:
        return f"Shape mismatch: expected {reference.shape}, got {candidate.shape}."
    if reference.dtype != candidate.dtype:
        return f"Dtype mismatch: expected {reference.dtype}, got {candidate.dtype}."

    nonfinite_outputs = []
    if not np.all(np.isfinite(reference)):
        nonfinite_outputs.append("reference")
    if not np.all(np.isfinite(candidate)):
        nonfinite_outputs.append("candidate")
    if nonfinite_outputs:
        return f"Nonfinite values in {' and '.join(nonfinite_outputs)} output(s)."

    if not np.allclose(candidate, reference, rtol=rtol, atol=atol):
        return f"Numerical mismatch: values differ beyond rtol={rtol}, atol={atol}."
    return None


def outputs_match(reference_output, candidate_output, *, rtol=1e-5, atol=1e-6):
    """Return a Python bool: same shape/dtype, finite values, close numbers."""
    return _output_mismatch_reason(
        reference_output, candidate_output, rtol=rtol, atol=atol
    ) is None


def measure_latency(compiled_fn, inputs, *, repeats=20):
    """Return median execution latency in milliseconds, excluding compilation.
    `compiled_fn` is already compiled; `inputs` are already on its device.
    """

    repeats = _validate_repeats(repeats)

    jax.block_until_ready(compiled_fn(*inputs))
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        result = compiled_fn(*inputs)
        result.block_until_ready()
        samples.append((perf_counter() - start) * 1000)

    return median(samples)


def evaluate(reference_fn, candidate_fn, inputs, *, repeats=20, rtol=1e-5, atol=1e-6):
    """Compile, check correctness, and measure both functions on identical inputs.

    Candidate failures return diagnostics without timings. Invalid arguments,
    input-placement errors, and reference-function failures propagate.
    """
    repeats = _validate_repeats(repeats)
    _validate_tolerances(rtol, atol)

    # Finish input transfers before starting any execution measurements.
    device_inputs = jax.block_until_ready(jax.device_put(inputs))
    reference = jax.jit(reference_fn).lower(*device_inputs).compile()
    expected = jax.block_until_ready(reference(*device_inputs))

    try:
        candidate = jax.jit(candidate_fn).lower(*device_inputs).compile()
    except Exception as error:
        return EvaluationResult(
            correct=False, failure_stage="compile",
            message=f"{type(error).__name__}: {error}",
        )

    try:
        actual = jax.block_until_ready(candidate(*device_inputs))
    except Exception as error:
        return EvaluationResult(
            correct=False, failure_stage="execute",
            message=f"{type(error).__name__}: {error}",
        )

    reason = _output_mismatch_reason(expected, actual, rtol=rtol, atol=atol)
    if reason is not None:
        return EvaluationResult(correct=False, failure_stage="correctness", message=reason)

    # A fast wrong answer must never receive a speedup score.
    reference_ms = measure_latency(reference, device_inputs, repeats=repeats)
    try:
        candidate_ms = measure_latency(candidate, device_inputs, repeats=repeats)
    except Exception as error:
        return EvaluationResult(
            correct=False, failure_stage="benchmark",
            message=f"{type(error).__name__}: {error}",
        )

    return EvaluationResult(
        correct=True,
        reference_ms=reference_ms,
        candidate_ms=candidate_ms,
    )
