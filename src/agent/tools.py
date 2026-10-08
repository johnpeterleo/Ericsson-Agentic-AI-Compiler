from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import jax

from src.agent.types import HLOSnapshot, InputTuple, JaxFn, ProfileReport
from src.optimization.evaluator import EvaluationResult, evaluate, measure_latency
from src.profiling.profiler import profile_function


class Profiler(Protocol):
    # To be replaced StubProfiler when the real module exists

    def profile(self, fn: JaxFn, inputs: InputTuple) -> ProfileReport:
        ...


class StubProfiler:
    """Small compatibility profiler used only when explicitly requested."""

    def profile(self, fn: JaxFn, inputs: InputTuple) -> ProfileReport:
        backend = jax.default_backend()
        device_inputs = jax.block_until_ready(jax.device_put(inputs))
        compiled = jax.jit(fn).lower(*device_inputs).compile()
        try:
            total_ms = measure_latency(compiled, device_inputs, repeats=5)
        except Exception as error:
            return ProfileReport(
                backend=backend,
                raw={"error": f"{type(error).__name__}: {error}"},
            )
        return ProfileReport(
            backend=backend,
            total_ms=total_ms,
            raw={"source": "stub_profiler_median_ms", "repeats": 5},
        )


@dataclass(frozen=True)
class JaxProfiler:
    """Adapt the structured profiler for use in the optimization agent.

    The underlying profiler compiles, warms up, synchronizes device execution,
    and records repeated samples.  This makes the value shown to the agent a
    real end-to-end device latency rather than asynchronous dispatch time.
    """

    repeats: int = 20

    def profile(self, fn: JaxFn, inputs: InputTuple) -> ProfileReport:
        result = profile_function(fn, inputs, repeats=self.repeats)
        return ProfileReport(
            backend=result.backend,
            total_ms=result.median_ms,
            raw={
                "source": "profile_function",
                "repeats": self.repeats,
                "samples_ms": result.samples_ms,
                "mean_ms": result.mean_ms,
                "min_ms": result.min_ms,
                "max_ms": result.max_ms,
                "stddev_ms": result.stddev_ms,
                "compile_ms": result.compile_ms,
                "device_platform": result.device_platform,
                "device_kind": result.device_kind,
                "device_id": result.device_id,
            },
        )


def lower_to_hlo_text(fn: JaxFn, inputs: InputTuple) -> HLOSnapshot:
    # Lower a JAX function and return StableHLO/HLO text (compiler IR)
    backend = jax.default_backend()
    device_inputs = jax.block_until_ready(jax.device_put(inputs))
    lowered = jax.jit(fn).lower(*device_inputs)
    return HLOSnapshot(text=lowered.as_text(), backend=backend)

# To be adjusted when the evaluator is written
def evaluate_candidate(reference_fn: JaxFn, candidate_fn: JaxFn, inputs: InputTuple, *, repeats: int = 10, rtol: float = 1e-5, atol: float = 1e-6) -> EvaluationResult:
    # Correctness + timing vs reference 
    return evaluate(reference_fn, candidate_fn, inputs, repeats=repeats, rtol=rtol, atol=atol)


@dataclass
class AgentToolkit:
    # Bundle of tools passed into the agent loop 

    profiler: Profiler

    def inspect_compilation(self, fn: JaxFn, inputs: InputTuple) -> HLOSnapshot:
        return lower_to_hlo_text(fn, inputs)

    def inspect_profile(self, fn: JaxFn, inputs: InputTuple) -> ProfileReport:
        return self.profiler.profile(fn, inputs)

    def measure_against_reference(self, reference_fn: JaxFn, candidate_fn: JaxFn, inputs: InputTuple, *, repeats: int = 10) -> EvaluationResult:
        return evaluate_candidate(reference_fn, candidate_fn, inputs, repeats=repeats)
