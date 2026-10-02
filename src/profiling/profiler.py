"""Structured runtime profiling for compiled JAX functions.

This module profiles one function at a time.  It complements, rather than
replaces, ``src.optimization.evaluator``: the evaluator compares a reference
and candidate, while this module supplies richer feedback about one program.
"""

from dataclasses import dataclass
from math import isfinite
from operator import index
from pathlib import Path
from statistics import mean, median, pstdev
from time import perf_counter
from typing import Callable

import jax

#container for the results of profiling a single function. 
#It is frozen to make it hashable and immutable, such that it can be used as a key in a dictionary
@dataclass(frozen=True)
class ProfileResult:
    """Measurements and environment information for one compiled function."""

    samples_ms: tuple[float, ...]
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    stddev_ms: float
    compile_ms: float
    backend: str
    device_platform: str
    device_kind: str
    device_id: int
    stablehlo_text: str | None = None
    stablehlo_path: Path | None = None

#internal helper functions to validate inputs/outputs of profiling functions.
def _validate_repeats(repeats: int) -> int:
    repeats = index(repeats)
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    return repeats

#fail if no samples, or if any sample is not finite or negative.
def _validate_samples(samples: tuple[float, ...]) -> None:
    if not samples or any(not isfinite(sample) or sample < 0 for sample in samples):
        raise ValueError("samples must contain finite, nonnegative timings")


#main public function to profile a JAX function.
def profile_function(
    fn: Callable,
    inputs: tuple,
    *,
    repeats: int = 20,
    capture_stablehlo: bool = False,
    stablehlo_path: str | Path | None = None,
) -> ProfileResult:
    """Compile, warm up, and profile ``fn`` on ``inputs``.

    Compilation and input transfer happen before the latency samples. Each
    sample waits for device completion, which is essential for asynchronous GPU
    execution. Set ``capture_stablehlo`` to retain the lowered program, or set
    ``stablehlo_path`` to also save it to a file.
    """
    repeats = _validate_repeats(repeats)
    if stablehlo_path is not None:
        capture_stablehlo = True

    device_inputs = jax.block_until_ready(jax.device_put(inputs))
    jitted = jax.jit(fn) #ask jax to compile function fn

    compile_start = perf_counter()
    lowered = jitted.lower(*device_inputs)
    stablehlo_text = lowered.as_text() if capture_stablehlo else None
    executable = lowered.compile()
    compile_ms = (perf_counter() - compile_start) * 1000

    # Warm-up is deliberately outside the measured samples.
    jax.block_until_ready(executable(*device_inputs))
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        result = executable(*device_inputs)
        jax.block_until_ready(result)
        samples.append((perf_counter() - start) * 1000)

    sample_values = tuple(samples)
    _validate_samples(sample_values)
    output_path = None
    if stablehlo_path is not None:
        output_path = Path(stablehlo_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(stablehlo_text, encoding="utf-8")

    device = jax.devices()[0]
    return ProfileResult(
        samples_ms=sample_values,
        median_ms=median(sample_values),
        mean_ms=mean(sample_values),
        min_ms=min(sample_values),
        max_ms=max(sample_values),
        stddev_ms=pstdev(sample_values),
        compile_ms=compile_ms,
        backend=jax.default_backend(),
        device_platform=device.platform,
        device_kind=device.device_kind,
        device_id=device.id,
        stablehlo_text=stablehlo_text,
        stablehlo_path=output_path,
    )
