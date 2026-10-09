#Shared data structures for the optimization agent

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence
import numpy as np
from src.optimization.evaluator import EvaluationResult
JaxFn = Callable[..., Any]
InputTuple = tuple[np.ndarray, ...]


@dataclass(frozen=True)
class HLOSnapshot:
    """Compiler IR snapshots from ``src.compiler.compile.compile_program``."""

    text: str  # StableHLO (pre-optimization lowering)
    backend: str
    optimized_hlo: str | None = None  # post-XLA optimized HLO when available


@dataclass(frozen=True)
class ProfileReport:
    """Agent-facing view of ``src.profiling.profiler.ProfileResult``."""

    backend: str
    total_ms: float | None = None
    kernel_stats: tuple[Mapping[str, Any], ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [f"backend={self.backend}"]
        if self.total_ms is not None:
            parts.append(f"median_ms={self.total_ms:.3f}")
        device = self.raw.get("device_kind")
        if device:
            parts.append(f"device={self.raw.get('device_platform')}:{device}")
        if self.kernel_stats:
            parts.append(f"kernels={len(self.kernel_stats)}")
        return "; ".join(parts)


@dataclass
class RewriteProposal:
    #What the agent (LLM or human) suggests for the next experiment
    rationale: str
    candidate_fn: JaxFn | None = None
    stop: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationStep:
    #Record of one observe -> propose -> measure iteration.
    index: int
    proposal: RewriteProposal
    evaluation: EvaluationResult | None = None
    hlo: HLOSnapshot | None = None
    profile: ProfileReport | None = None
    notes: str | None = None


@dataclass
class AgentContext:
    #Everything the policy sees when choosing the next action

    reference_fn: JaxFn  #the Jax function to improve
    inputs: InputTuple
    backend: str
    baseline_hlo: HLOSnapshot
    baseline_profile: ProfileReport
    baseline_evaluation: EvaluationResult | None
    history: list[OptimizationStep] = field(default_factory=list)
    hardware_note: str = "Target: NVIDIA GPU via JAX cuda backend."

    def last_step(self) -> OptimizationStep | None:
        return self.history[-1] if self.history else None

    def best_speedup(self) -> float | None:
        best: float | None = None
        for step in self.history:
            if step.evaluation and step.evaluation.correct and step.evaluation.speedup is not None:
                if best is None or step.evaluation.speedup > best:
                    best = step.evaluation.speedup
        return best


@dataclass
class AgentRunResult:
    status: Literal["stopped_by_policy", "max_iterations", "no_improvement"]
    context: AgentContext
    message: str
