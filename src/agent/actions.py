"""Turn a model reply into an action, run it, and describe the result.

Model-written code runs in this process with `exec`, in a namespace that
holds only `jax` and `jnp`, as the existing agent already does.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from src.agent.feedback import CompiledSummary, format_op_diff, format_summary, summarize_compiled
from src.agent.llm_policy import MAX_FAILURE_MESSAGE_CHARS, _extract_json_object
from src.agent.search import Candidate, EvaluationSettings, SearchState, evaluate_on
from src.agent.tasks import Task
from src.compiler.compile import compile_program
from src.compiler.compile_time_limit import check_compile_time
from src.optimization.evaluator import EvaluationResult, measure_latency_pair

ACTIONS = ("inspect", "time", "evaluate", "finish")
MAX_CANDIDATES_PER_TURN = 4
REFERENCE_ID = "reference"
HOLDOUT_PREFIX = "On the held-out input:"


class ActionError(ValueError):
    """The reply cannot be run; the message tells the model what was wrong."""


@dataclass(frozen=True)
class Action:
    name: str
    fields: dict[str, Any]
    thought: str = ""


@dataclass
class Workspace:
    """What the actions work on during one run."""

    task: Task
    settings: EvaluationSettings
    reference_summary: CompiledSummary
    search: SearchState


def _require_text(fields, name):
    value = fields.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ActionError(f"The field `{name}` must be a non-empty string.")
    return value


def _check_inspect(fields):
    if ("code" in fields) == ("id" in fields):
        raise ActionError("`inspect` needs exactly one of `code` or `id`.")
    _require_text(fields, "code" if "code" in fields else "id")


def _check_candidates(fields):
    candidates = fields.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= MAX_CANDIDATES_PER_TURN:
        raise ActionError(f"`candidates` must be a list of 1 to {MAX_CANDIDATES_PER_TURN} objects.")
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ActionError('Each candidate must be an object {"idea": ..., "code": ...}.')
        _require_text(candidate, "idea")
        _require_text(candidate, "code")


_FIELD_CHECKS = {
    "inspect": _check_inspect,
    "time": lambda fields: _require_text(fields, "code"),
    "evaluate": _check_candidates,
    "finish": lambda fields: _require_text(fields, "summary"),
}


def parse_action(reply: str) -> Action:
    try:
        payload = _extract_json_object(reply)
    except ValueError as error:
        raise ActionError(f"The reply was not one JSON object: {error}") from error
    name = payload.get("action")
    if name not in ACTIONS:
        raise ActionError(f"Unknown action {name!r}; choose one of {', '.join(ACTIONS)}.")
    _FIELD_CHECKS[name](payload)
    return Action(name=name, fields=payload, thought=str(payload.get("thought", "")))


def load_function(source: str, name: str):
    """Run model-written source and return the function it defines under `name`."""
    namespace: dict[str, Any] = {"jax": jax, "jnp": jnp}
    exec(source, namespace)  # noqa: S102 — the same trust model as the existing agent
    function = namespace.get(name)
    if not callable(function):
        raise ValueError(f"The code must define a function named `{name}`.")
    return function


def _error_text(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"[:MAX_FAILURE_MESSAGE_CHARS]


def _inspect_target(action, workspace):
    """Return a label and the function to inspect."""
    if "code" in action.fields:
        return "your code", load_function(action.fields["code"], "candidate")
    candidate_id = action.fields["id"]
    if candidate_id == REFERENCE_ID:
        return REFERENCE_ID, workspace.task.reference
    candidate = workspace.search.find(candidate_id)
    if candidate is None or candidate.function is None:
        raise ActionError(f"There is no loadable candidate with id {candidate_id!r}.")
    return candidate_id, candidate.function


def run_inspect(action: Action, workspace: Workspace) -> str:
    try:
        label, function = _inspect_target(action, workspace)
        summary = summarize_compiled(function, workspace.task.inputs)
    except ActionError:
        raise
    except Exception as error:
        return f"Could not compile it: {_error_text(error)}"
    lines = [f"Compiled form of {label}:", format_summary(summary)]
    if label != REFERENCE_ID:
        lines.append(f"Compared with the reference: {format_op_diff(workspace.reference_summary, summary)}")
    return "\n".join(lines)


def _time_probe(probe, workspace):
    device_inputs = jax.block_until_ready(jax.device_put(workspace.task.inputs))
    if workspace.settings.compile_time_limit is not None:
        check_compile_time(probe, device_inputs, workspace.settings.compile_time_limit)
    reference = compile_program(workspace.task.reference, device_inputs).executable
    compiled_probe = compile_program(probe, device_inputs).executable
    return measure_latency_pair(reference, compiled_probe, device_inputs, repeats=workspace.settings.repeats)


def run_time(action: Action, workspace: Workspace) -> str:
    try:
        probe = load_function(action.fields["code"], "probe")
        reference_ms, probe_ms = _time_probe(probe, workspace)
    except Exception as error:
        return f"Could not time the probe (it must return one array): {_error_text(error)}"
    return (
        f"probe: {probe_ms:.3f} ms; the whole reference, timed alongside: {reference_ms:.3f} ms. "
        f"The probe takes {probe_ms / reference_ms:.0%} of the reference's time. "
        "This has no correctness check."
    )


def _evaluate_spec(spec, workspace):
    candidate_id = workspace.search.next_id()
    try:
        function = load_function(spec["code"], "candidate")
    except Exception as error:
        failure = EvaluationResult(correct=False, failure_stage="load", message=_error_text(error))
        return Candidate(candidate_id, spec["idea"], spec["code"], failure)
    result = evaluate_on(workspace.task, function, workspace.settings)
    compiled = summarize_compiled(function, workspace.task.inputs) if result.correct else None
    return Candidate(candidate_id, spec["idea"], spec["code"], result, function, compiled)


def _failure_text(result):
    message = (result.message or "")[:MAX_FAILURE_MESSAGE_CHARS]
    if result.failure_stage in ("execute", "correctness") and not message.startswith(HOLDOUT_PREFIX):
        message = f"On the first input: {message}"
    return f"rejected at {result.failure_stage}. {message}"


def _success_text(candidate, improved, workspace):
    result = candidate.result
    search = workspace.search
    verdict = (
        "new best" if improved
        else f"not better than the best ({search.best_counted_speedup:.2f}x) "
        f"by the required margin of {search.margin:.2f}x"
    )
    return (
        f"correct. reference {result.reference_ms:.3f} ms, candidate {result.candidate_ms:.3f} ms, "
        f"speedup {result.speedup:.2f}x: {verdict}. "
        f"Compiled operations against the reference: "
        f"{format_op_diff(workspace.reference_summary, candidate.compiled)}"
    )


def run_evaluate(specs: list[dict], workspace: Workspace) -> str:
    """Evaluate each spec, store it in the search state, and describe every outcome."""
    lines = []
    for spec in specs:
        candidate = _evaluate_spec(spec, workspace)
        improved = workspace.search.add(candidate)
        outcome = (
            _success_text(candidate, improved, workspace) if candidate.result.correct
            else _failure_text(candidate.result)
        )
        lines.append(f"{candidate.id} ({candidate.idea}): {outcome}")
    return "\n".join(lines)
