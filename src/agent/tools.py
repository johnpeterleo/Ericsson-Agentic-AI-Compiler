"""Agent tools wired to compiler, profiler, and evaluator modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax

from src.agent.types import HLOSnapshot, InputTuple, JaxFn, ProfileReport
from src.compiler.compile import compile_program
from src.optimization.evaluator import EvaluationResult, evaluate
from src.profiling.profiler import ProfileResult, profile_function


class Profiler(Protocol):
    def profile(self, fn: JaxFn, inputs: InputTuple) -> ProfileReport:
        ...


def profile_result_to_report(result: ProfileResult) -> ProfileReport:
    """Map profiler module output into agent ``ProfileReport``."""
    return ProfileReport(
        backend=result.backend,
        total_ms=result.median_ms,
        raw={
            "median_ms": result.median_ms,
            "mean_ms": result.mean_ms,
            "min_ms": result.min_ms,
            "max_ms": result.max_ms,
            "stddev_ms": result.stddev_ms,
            "compile_ms": result.compile_ms,
            "device_platform": result.device_platform,
            "device_kind": result.device_kind,
            "device_id": result.device_id,
            "samples_ms": result.samples_ms,
            "stablehlo_text": result.stablehlo_text,
        },
    )


class JAXProfiler:
    """Adapter around ``src.profiling.profiler.profile_function``."""

    def __init__(self, *, repeats: int = 20, capture_stablehlo: bool = False):
        self.repeats = repeats
        self.capture_stablehlo = capture_stablehlo

    def profile(self, fn: JaxFn, inputs: InputTuple) -> ProfileReport:
        result = profile_function(
            fn,
            inputs,
            repeats=self.repeats,
            capture_stablehlo=self.capture_stablehlo,
        )
        return profile_result_to_report(result)


class StubProfiler:
    """Minimal profiler for tests; prefer ``JAXProfiler`` in real runs."""

    def __init__(self, *, repeats: int = 5):
        self.repeats = repeats

    def profile(self, fn: JaxFn, inputs: InputTuple) -> ProfileReport:
        wrapped = JAXProfiler(repeats=self.repeats)
        return wrapped.profile(fn, inputs)


def lower_to_hlo_text(fn: JaxFn, inputs: InputTuple) -> HLOSnapshot:
    """Lower via ``src.compiler.compile.compile_program``."""
    backend = jax.default_backend()
    device_inputs = jax.block_until_ready(jax.device_put(inputs))
    compilation = compile_program(fn, device_inputs)
    return HLOSnapshot(
        text=compilation.stablehlo,
        backend=backend,
        optimized_hlo=compilation.optimized_hlo,
    )


def evaluate_candidate(
    reference_fn: JaxFn,
    candidate_fn: JaxFn,
    inputs: InputTuple,
    *,
    repeats: int = 20,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> EvaluationResult:
    """Correctness + timing vs reference (``src.optimization.evaluator``)."""
    return evaluate(reference_fn, candidate_fn, inputs, repeats=repeats, rtol=rtol, atol=atol)


@dataclass
class AgentToolkit:
    profiler: Profiler
    profile_repeats: int = 20

    def inspect_compilation(self, fn: JaxFn, inputs: InputTuple) -> HLOSnapshot:
        return lower_to_hlo_text(fn, inputs)

    def inspect_profile(self, fn: JaxFn, inputs: InputTuple) -> ProfileReport:
        return self.profiler.profile(fn, inputs)

    def measure_against_reference(
        self,
        reference_fn: JaxFn,
        candidate_fn: JaxFn,
        inputs: InputTuple,
        *,
        repeats: int | None = None,
    ) -> EvaluationResult:
        return evaluate_candidate(
            reference_fn,
            candidate_fn,
            inputs,
            repeats=repeats if repeats is not None else self.profile_repeats,
        )
