"""An optimization agent that chooses its own actions.

Each turn the model sends one action (inspect, time, evaluate or finish), the
harness runs it, and the observation goes back into the transcript. The model
is sent the whole transcript every turn. When the run ends, the top correct
candidates are re-measured on a fresh input (see `search`).
"""

import logging
from dataclasses import dataclass, field
from time import perf_counter

import jax

from src.agent.actions import (
    MAX_CANDIDATES_PER_TURN,
    ActionError,
    Workspace,
    parse_action,
    run_evaluate,
    run_inspect,
    run_time,
)
from src.agent.feedback import format_summary, summarize_compiled
from src.agent.search import (
    Candidate,
    EvaluationSettings,
    Remeasurement,
    RemeasureSettings,
    SearchState,
    best_remeasured,
    format_leaderboard,
    remeasure_top,
    search_record,
)
from src.agent.tasks import Task
from src.agent.text_model import TextModel
from src.agent.tool_prompts import NEXT_ACTION_REQUEST, TOOL_SYSTEM_PROMPT
from src.compiler.compile import compile_program
from src.optimization.evaluator import measure_latency

MAX_TURNS = 12
MAX_EVALUATIONS = 16
# The spec said 2. In the first real run the model sent one candidate per turn,
# none beat the margin, and the run stopped after 2 of its 12 turns.
PATIENCE = 4
IMPROVEMENT_MARGIN = 1.05
MAX_OBSERVATION_CHARS = 8000

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolAgentConfig:
    max_turns: int = MAX_TURNS
    max_evaluations: int = MAX_EVALUATIONS
    patience: int = PATIENCE
    improvement_margin: float = IMPROVEMENT_MARGIN
    evaluation: EvaluationSettings = EvaluationSettings()
    remeasure: RemeasureSettings = RemeasureSettings()


@dataclass(frozen=True)
class Turn:
    reply: str
    action: str | None
    observation: str


@dataclass
class Progress:
    """`stale_evaluate_turns` counts evaluate turns in a row that brought no improvement."""

    turns: list[Turn] = field(default_factory=list)
    turns_used: int = 0
    stale_evaluate_turns: int = 0
    finished: bool = False


@dataclass
class ToolAgentRun:
    task_name: str
    device: str
    stop_reason: str
    opening: str
    turns: list[Turn]
    candidates: list[Candidate]
    remeasurements: list[Remeasurement]
    wall_seconds: float

    @property
    def best(self) -> Remeasurement | None:
        return best_remeasured(self.remeasurements)

    def record(self) -> dict:
        return {
            "task": self.task_name,
            "device": self.device,
            "stop_reason": self.stop_reason,
            "llm_calls": len(self.turns),
            "wall_seconds": self.wall_seconds,
            **search_record(self.candidates, self.remeasurements),
            "transcript": {
                "opening": self.opening,
                "turns": [vars(turn) for turn in self.turns],
            },
        }


def device_name() -> str:
    device = jax.devices()[0]
    return f"{device.platform} ({device.device_kind})"


def _reference_ms(task, repeats):
    device_inputs = jax.block_until_ready(jax.device_put(task.inputs))
    executable = compile_program(task.reference, device_inputs).executable
    return measure_latency(executable, device_inputs, repeats=repeats)


def _budget_text(config):
    return (
        f"Budget: {config.max_turns} turns (one reply each), {config.max_evaluations} candidate "
        f"evaluations, and the run stops after {config.patience} evaluate turns in a row without "
        f"a candidate that beats the best by {config.improvement_margin:.2f}x. Turns are "
        f"scarcer than evaluations: one evaluate turn can test up to {MAX_CANDIDATES_PER_TURN} candidates."
    )


def opening_message(workspace: Workspace, lessons_text: str, config: ToolAgentConfig) -> str:
    task = workspace.task
    lines = [
        f"Task: {task.name}",
        f"Target device: {device_name()}",
        f"Inputs (dtype and shape): {task.input_summary()}",
        f"Tolerances: rtol={task.rtol}, atol={task.atol}",
        "Reference source (the function to rewrite is `reference`):",
        task.source_text.strip(),
        f"Reference time: {_reference_ms(task, workspace.settings.repeats):.3f} ms",
        "Compiled form of the reference:",
        format_summary(workspace.reference_summary),
    ]
    if lessons_text:
        lines.extend(["Lessons from earlier runs on other programs:", lessons_text])
    lines.extend([_budget_text(config), NEXT_ACTION_REQUEST])
    return "\n".join(lines)


def render_transcript(opening: str, turns: list[Turn]) -> str:
    parts = [opening]
    for number, turn in enumerate(turns, start=1):
        parts.extend([
            f"## Turn {number}: your reply",
            turn.reply,
            f"## Turn {number}: observation",
            turn.observation,
        ])
    if turns:
        parts.append(NEXT_ACTION_REQUEST)
    return "\n\n".join(parts)


def _cap(text):
    if len(text) <= MAX_OBSERVATION_CHARS:
        return text
    return text[:MAX_OBSERVATION_CHARS] + f"\n... (observation cut at {MAX_OBSERVATION_CHARS} characters)"


class ToolAgent:
    def __init__(self, model: TextModel, config: ToolAgentConfig | None = None):
        self.model = model
        self.config = config or ToolAgentConfig()

    def run(self, task: Task, lessons_text: str = "") -> ToolAgentRun:
        started = perf_counter()
        workspace = Workspace(
            task=task,
            settings=self.config.evaluation,
            reference_summary=summarize_compiled(task.reference, task.inputs),
            search=SearchState(self.config.improvement_margin),
        )
        opening = opening_message(workspace, lessons_text, self.config)
        progress = Progress()
        stop_reason = None
        while stop_reason is None:
            reply = self.model.ask(TOOL_SYSTEM_PROMPT, render_transcript(opening, progress.turns))
            progress.turns_used += 1
            action_name, observation = self._take_turn(reply, workspace, progress)
            stop_reason = self._stop_reason(workspace, progress)
            footer = self._footer(workspace, progress, stop_reason)
            progress.turns.append(Turn(reply, action_name, f"{_cap(observation)}\n\n{footer}"))
            log.info(
                "turn %d: %s; %d evaluations; best counted speedup %.2fx",
                progress.turns_used, action_name or "invalid reply",
                len(workspace.search.candidates), workspace.search.best_counted_speedup,
            )
        candidates = workspace.search.candidates
        return ToolAgentRun(
            task_name=task.name,
            device=device_name(),
            stop_reason=stop_reason,
            opening=opening,
            turns=progress.turns,
            candidates=candidates,
            remeasurements=remeasure_top(task, candidates, self.config.remeasure),
            wall_seconds=perf_counter() - started,
        )

    def _take_turn(self, reply, workspace, progress):
        """Run the reply's action. Return the action's name (None if invalid) and the observation."""
        try:
            action = parse_action(reply)
            if action.name == "finish":
                progress.finished = True
                return action.name, "Run finished."
            if action.name == "evaluate":
                return action.name, self._evaluate(action, workspace, progress)
            run = run_inspect if action.name == "inspect" else run_time
            return action.name, run(action, workspace)
        except ActionError as error:
            return None, f"Your reply could not be used: {error}"

    def _evaluate(self, action, workspace, progress):
        """Evaluate as many candidates as the budget allows and update the patience count."""
        search = workspace.search
        evaluations_left = self.config.max_evaluations - len(search.candidates)
        specs = action.fields["candidates"]
        best_before = search.best_counted_speedup
        observation = run_evaluate(specs[:evaluations_left], workspace)
        if len(specs) > evaluations_left:
            observation += f"\nOnly the first {evaluations_left} were evaluated: the evaluation budget is used up."
        improved = search.best_counted_speedup > best_before
        progress.stale_evaluate_turns = 0 if improved else progress.stale_evaluate_turns + 1
        return observation

    def _stop_reason(self, workspace, progress):
        config = self.config
        if progress.finished:
            return "finished"
        if len(workspace.search.candidates) >= config.max_evaluations:
            return "max_evaluations"
        if progress.stale_evaluate_turns >= config.patience:
            return "patience"
        if progress.turns_used >= config.max_turns:
            return "max_turns"
        return None

    def _footer(self, workspace, progress, stop_reason):
        config = self.config
        lines = [
            format_leaderboard(workspace.search.candidates),
            f"Budget left: {config.max_turns - progress.turns_used} turns, "
            f"{config.max_evaluations - len(workspace.search.candidates)} evaluations. "
            f"Evaluate turns in a row without improvement: "
            f"{progress.stale_evaluate_turns} of {config.patience}.",
        ]
        if stop_reason is not None:
            lines.append(f"The run stops here ({stop_reason}).")
        return "\n".join(lines)
