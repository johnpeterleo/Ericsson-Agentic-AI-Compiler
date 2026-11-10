"""Compare correctness and latency of a reference and a candidate JAX function.

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

from src.compiler.compile import compile_program
from src.compiler.compile_time_limit import check_compile_time


@dataclass
class EvaluationResult:
    """`failure_stage` "load" is set by the agent, not here: the proposed source
    could not be turned into a Python function, so there was nothing to compile.
    """

    correct: bool
    reference_ms: float | None = None
    candidate_ms: float | None = None
    failure_stage: Literal["load", "compile", "execute", "correctness", "benchmark"] | None = None
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


class CandidateTimingError(RuntimeError):
    """The candidate failed while it was being timed. The reference did not."""


def _time_call_ms(compiled_fn, inputs):
    start = perf_counter()
    compiled_fn(*inputs).block_until_ready()
    return (perf_counter() - start) * 1000


def _time_candidate_ms(candidate, inputs):
    try:
        return _time_call_ms(candidate, inputs)
    except Exception as error:
        raise CandidateTimingError(f"{type(error).__name__}: {error}") from error


def measure_latency(compiled_fn, inputs, *, repeats=20):
    """Return median execution latency in milliseconds, excluding compilation.
    `compiled_fn` is already compiled; `inputs` are already on its device.
    """
    repeats = _validate_repeats(repeats)

    jax.block_until_ready(compiled_fn(*inputs))
    return median(_time_call_ms(compiled_fn, inputs) for _ in range(repeats))


def measure_latency_pair(reference, candidate, inputs, *, repeats=20):
    """Return the median latency of both compiled programs, in milliseconds.

    The programs take turns, and the one that goes first swaps every round.
    Timing one program completely before the other favours the second one:
    two identical programs then measured 1% to 3% apart.
    """
    repeats = _validate_repeats(repeats)

    _time_call_ms(reference, inputs)  # Warm-up calls are not recorded.
    _time_candidate_ms(candidate, inputs)
    reference_samples, candidate_samples = [], []
    for round_index in range(repeats):
        reference_goes_first = round_index % 2 == 0
        if reference_goes_first:
            reference_samples.append(_time_call_ms(reference, inputs))
            candidate_samples.append(_time_candidate_ms(candidate, inputs))
        else:
            candidate_samples.append(_time_candidate_ms(candidate, inputs))
            reference_samples.append(_time_call_ms(reference, inputs))
    return median(reference_samples), median(candidate_samples)


def _candidate_failure(candidate, device_inputs, expected, *, rtol, atol):
    """Run the candidate on one input case. Return its failure, or None if it matches."""
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
    return None


def _holdout_failure(reference, candidate, holdout_inputs, *, rtol, atol):
    """Check the candidate on a second input case that the agent never saw."""
    device_inputs = jax.block_until_ready(jax.device_put(holdout_inputs))
    expected = jax.block_until_ready(reference(*device_inputs))
    failure = _candidate_failure(candidate, device_inputs, expected, rtol=rtol, atol=atol)
    if failure is not None:
        failure.message = f"On the held-out input: {failure.message}"
    return failure


def evaluate(
    reference_fn,
    candidate_fn,
    inputs,
    *,
    repeats=20,
    rtol=1e-5,
    atol=1e-6,
    holdout_inputs=None,
    compile_time_limit=None,
):
    """Compile, check correctness, and measure both functions on identical inputs.

    `holdout_inputs` is an optional second input case with the same shapes and
    dtypes. The candidate must match the reference there too, so a rewrite that
    is only right for the values in `inputs` is rejected. Timing uses `inputs`.

    `compile_time_limit` is in seconds. A candidate that takes longer to compile
    is rejected, and a reference that does raises CompileTimeoutError. It is off
    when None, because each check starts a separate process (see
    `src.compiler.compile_time_limit`).

    Candidate failures return diagnostics without timings. Invalid arguments,
    input-placement errors, and reference-function failures propagate.
    """
    repeats = _validate_repeats(repeats)
    _validate_tolerances(rtol, atol)

    # Finish input transfers before starting any execution measurements.
    device_inputs = jax.block_until_ready(jax.device_put(inputs))
    if compile_time_limit is not None:
        check_compile_time(reference_fn, device_inputs, compile_time_limit)
    reference = compile_program(reference_fn, device_inputs).executable
    expected = jax.block_until_ready(reference(*device_inputs))

    try:
        if compile_time_limit is not None:
            check_compile_time(candidate_fn, device_inputs, compile_time_limit)
        candidate = compile_program(candidate_fn, device_inputs).executable
    except Exception as error:
        return EvaluationResult(
            correct=False, failure_stage="compile",
            message=f"{type(error).__name__}: {error}",
        )

    failure = _candidate_failure(candidate, device_inputs, expected, rtol=rtol, atol=atol)
    if failure is None and holdout_inputs is not None:
        failure = _holdout_failure(reference, candidate, holdout_inputs, rtol=rtol, atol=atol)
    if failure is not None:
        return failure

    # A fast wrong answer must never receive a speedup score.
    try:
        reference_ms, candidate_ms = measure_latency_pair(
            reference, candidate, device_inputs, repeats=repeats
        )
    except CandidateTimingError as error:
        return EvaluationResult(correct=False, failure_stage="benchmark", message=str(error))

    return EvaluationResult(
        correct=True,
        reference_ms=reference_ms,
        candidate_ms=candidate_ms,
    )
