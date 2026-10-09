from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence
import jax
from src.agent.tools import AgentToolkit, JaxProfiler, Profiler
from src.agent.types import (
    AgentContext,
    AgentRunResult,
    InputTuple,
    JaxFn,
    OptimizationStep,
    RewriteProposal,
)


class OptimizationPolicy(Protocol):
    # Brain of the agent, LLM
    def propose(self, context: AgentContext) -> RewriteProposal:
        ...


@dataclass
class AgentConfig:
    max_iterations: int = 5
    benchmark_repeats: int = 10
    min_speedup_to_continue: float = 1.0
    hardware_note: str = "Target: NVIDIA GPU via JAX cuda backend."


class DemoPolicy:
    """To be replaced with an LLM policy that reads `prompts.SYSTEM_PROMPT` and calls tools."""

    def __init__(self, candidates: Sequence[JaxFn] | None = None):
        self._candidates = list(candidates or [])
        self._tried = 0

    def propose(self, context: AgentContext) -> RewriteProposal:
        if self._tried >= len(self._candidates):
            return RewriteProposal(
                rationale="No more scripted candidates; connect LLM policy or add candidates.",
                stop=True,
            )
        candidate = self._candidates[self._tried]
        self._tried += 1
        return RewriteProposal(
            rationale=f"Scripted candidate #{self._tried} for pipeline smoke test.",
            candidate_fn=candidate,
        )


class OptimizationAgent:
    # Runs the observe -> propose -> measure loop
    def __init__(self, toolkit: AgentToolkit, policy: OptimizationPolicy, config: AgentConfig | None = None):
        self.toolkit = toolkit
        self.policy = policy
        self.config = config or AgentConfig()

    def build_context(self, reference_fn: JaxFn, inputs: InputTuple) -> AgentContext:
        backend = jax.default_backend()
        baseline_hlo = self.toolkit.inspect_compilation(reference_fn, inputs)
        baseline_profile = self.toolkit.inspect_profile(reference_fn, inputs)
        return AgentContext(
            reference_fn=reference_fn,
            inputs=inputs,
            backend=backend,
            baseline_hlo=baseline_hlo,
            baseline_profile=baseline_profile,
            baseline_evaluation=None,
            hardware_note=self.config.hardware_note,
        )

    def run(self, reference_fn: JaxFn, inputs: InputTuple) -> AgentRunResult:
        context = self.build_context(reference_fn, inputs)
        for step_index in range(self.config.max_iterations):
            proposal = self.policy.propose(context)
            if proposal.stop or proposal.candidate_fn is None:
                return AgentRunResult(
                    status="stopped_by_policy",
                    context=context,
                    message=proposal.rationale,
                )
            candidate_fn = proposal.candidate_fn
            evaluation = self.toolkit.measure_against_reference(
                reference_fn,
                candidate_fn,
                inputs,
                repeats=self.config.benchmark_repeats,
            )
            hlo = self.toolkit.inspect_compilation(candidate_fn, inputs)
            profile = self.toolkit.inspect_profile(candidate_fn, inputs)
            step = OptimizationStep(
                index=step_index,
                proposal=proposal,
                evaluation=evaluation,
                hlo=hlo,
                profile=profile,
            )
            context.history.append(step)
            if not evaluation.correct:
                step.notes = f"Rejected: {evaluation.failure_stage} — {evaluation.message}"
                continue
            speedup = evaluation.speedup
            if speedup is not None and speedup < self.config.min_speedup_to_continue:
                step.notes = f"No meaningful speedup ({speedup:.3f}x); policy may try another idea."
        return AgentRunResult(
            status="max_iterations",
            context=context,
            message=f"Reached max_iterations={self.config.max_iterations}.",
        )


def default_agent(
    policy: OptimizationPolicy | None = None,
    *,
    profiler: Profiler | None = None,
    config: AgentConfig | None = None,
) -> OptimizationAgent:
    """Factory with structured profiling, compiler IR, and an optional demo policy."""
    cfg = config or AgentConfig()
    active_profiler = profiler or JaxProfiler(repeats=cfg.benchmark_repeats)
    toolkit = AgentToolkit(profiler=active_profiler, profile_repeats=cfg.benchmark_repeats)
    return OptimizationAgent(toolkit=toolkit, policy=policy or DemoPolicy(), config=cfg)


def format_step_summary(step: OptimizationStep) -> str:
    parts = [f"step={step.index}", step.proposal.rationale]
    ev = step.evaluation
    if ev is None:
        parts.append("no evaluation")
    elif not ev.correct:
        parts.append(f"FAIL@{ev.failure_stage}: {ev.message}")
    else:
        parts.append(
            f"ok speedup={ev.speedup:.3f}x "
            f"(ref={ev.reference_ms:.3f}ms cand={ev.candidate_ms:.3f}ms)"
        )
    if step.profile:
        parts.append(f"profile[{step.profile.summary()}]")
    if step.notes:
        parts.append(step.notes)
    return " | ".join(parts)
