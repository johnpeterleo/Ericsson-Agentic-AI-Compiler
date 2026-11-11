"""The existing agent as baselines for the tool agent, at a given number of LLM calls.

`single` is one call, `best_of_n` is N independent single calls, and `chain` is
the existing agent with N attempts. Their candidates go through the same final
re-measurement as the tool agent's (see `search.remeasure_top`).

The baselines send the existing user prompt (`format_context_for_llm`) with
the task's visible source, so that a `data/programs/` reference comes with the
helpers and constants it uses. The system prompt is a setting: the fair one
(`FAIR_BASELINE_PROMPT`) or the November one (`SYSTEM_PROMPT`). With the
November one, a suite program gets exactly the prompt the November runs sent,
because its visible source is `inspect.getsource(reference)`.
"""

from dataclasses import dataclass
from time import perf_counter

from src.agent.agent import AgentConfig, OptimizationAgent
from src.agent.feedback import summarize_compiled
from src.agent.llm_policy import format_context_for_llm, proposal_from_reply
from src.agent.search import (
    Candidate,
    EvaluationSettings,
    Remeasurement,
    RemeasureSettings,
    remeasure_top,
    search_record,
)
from src.agent.tasks import Task
from src.agent.text_model import RecordingModel
from src.agent.tool_agent import device_name
from src.agent.tools import AgentToolkit, JaxProfiler
from src.agent.types import AgentContext, RewriteProposal


class TaskSourcePolicy:
    """The existing policy's prompt and reply format, sent through a text model."""

    def __init__(self, model, source_text: str, system_prompt: str):
        self.model = model
        self.source_text = source_text
        self.system_prompt = system_prompt

    def propose(self, context: AgentContext) -> RewriteProposal:
        prompt = format_context_for_llm(context, reference_source=self.source_text)
        return proposal_from_reply(self.model.ask(self.system_prompt, prompt))


@dataclass(frozen=True)
class ArmSettings:
    evaluation: EvaluationSettings
    remeasure: RemeasureSettings
    system_prompt: str


@dataclass
class ArmResult:
    task_name: str
    stop_reason: str
    llm_calls: int
    candidates: list[Candidate]
    remeasurements: list[Remeasurement]
    wall_seconds: float
    transcript: list[dict]

    def record(self) -> dict:
        return {
            "task": self.task_name,
            "device": device_name(),
            "stop_reason": self.stop_reason,
            "llm_calls": self.llm_calls,
            "wall_seconds": self.wall_seconds,
            **search_record(self.candidates, self.remeasurements),
            "transcript": self.transcript,
        }


def _candidates_from_history(history, prefix):
    return [
        Candidate(
            id=f"{prefix}{step.index}",
            idea=step.proposal.rationale,
            source=step.proposal.metadata.get("candidate_source", ""),
            result=step.evaluation,
            function=step.proposal.candidate_fn,
        )
        for step in history
    ]


class BaselineRunner:
    """Runs the existing agent on one task and collects its candidates."""

    def __init__(self, model, task: Task, settings: ArmSettings):
        self.model = RecordingModel(model)
        self.task = task
        self.settings = settings
        self.statuses: list[str] = []

    def _agent(self, attempts):
        repeats = self.settings.evaluation.repeats
        return OptimizationAgent(
            toolkit=AgentToolkit(
                profiler=JaxProfiler(repeats=repeats),
                profile_repeats=repeats,
                rtol=self.task.rtol,
                atol=self.task.atol,
            ),
            policy=TaskSourcePolicy(self.model, self.task.source_text, self.settings.system_prompt),
            config=AgentConfig(
                max_iterations=attempts,
                benchmark_repeats=repeats,
                hardware_note=f"Target JAX device: {device_name()}.",
                compile_time_limit_seconds=self.settings.evaluation.compile_time_limit,
            ),
        )

    def candidates(self, attempts: int, prefix: str) -> list[Candidate]:
        run = self._agent(attempts).run(self.task.reference, self.task.inputs, self.task.holdout_inputs)
        self.statuses.append(run.status)
        return _candidates_from_history(run.context.history, prefix)

    def finish(self, candidates: list[Candidate], started: float) -> ArmResult:
        """Re-measure the top candidates and describe the arm's result."""
        remeasurements = remeasure_top(self.task, candidates, self.settings.remeasure)
        for remeasurement in remeasurements:
            candidate = remeasurement.candidate
            candidate.compiled = summarize_compiled(candidate.function, self.task.inputs)
        return ArmResult(
            task_name=self.task.name,
            stop_reason=", ".join(sorted(set(self.statuses))),
            llm_calls=self.model.calls,
            candidates=candidates,
            remeasurements=remeasurements,
            wall_seconds=perf_counter() - started,
            transcript=self.model.exchanges,
        )


def run_single(runner: BaselineRunner) -> ArmResult:
    started = perf_counter()
    return runner.finish(runner.candidates(1, "a"), started)


def run_best_of_n(runner: BaselineRunner, calls: int) -> ArmResult:
    started = perf_counter()
    candidates = [c for sample in range(calls) for c in runner.candidates(1, f"s{sample}-a")]
    return runner.finish(candidates, started)


def run_chain(runner: BaselineRunner, calls: int) -> ArmResult:
    started = perf_counter()
    return runner.finish(runner.candidates(calls, "a"), started)
