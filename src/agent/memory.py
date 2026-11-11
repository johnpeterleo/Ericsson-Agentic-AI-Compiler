"""Lessons from finished runs, stored in `data/rewrite_memory.jsonl` and shown on later runs.

The file is tracked in git so the team shares it. A lesson is never shown to
the task it came from, so a run cannot be helped by its own earlier answer.
Version one shows every stored lesson, newest first, up to a cap.
"""

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from src.agent.llm_policy import _extract_json_object
from src.agent.text_model import TextModel
from src.agent.tool_agent import ToolAgentRun
from src.benchmarks.registry import REPO_ROOT

MEMORY_PATH = REPO_ROOT / "data" / "rewrite_memory.jsonl"
MEMORY_MODES = ("off", "read", "readwrite")
MAX_LESSONS_SHOWN = 20
MAX_LESSONS_PER_RUN = 3
# A rewrite only counts as having worked when its re-measured speedup reaches this.
WORKED_SPEEDUP = 1.2
KINDS = ("worked", "did_not_help")

LESSON_SYSTEM_PROMPT = """You review a finished attempt to make a JAX program faster than what XLA compiles it into. Write down what a performance engineer should remember for other programs.

Reply with one JSON object and nothing else:
{"lessons": [{"rule": "<one or two general sentences>", "when": "<what to look for in the source or the compiled form>", "kind": "worked" or "did_not_help"}]}

Give at most three lessons. A rule must be general: name the pattern, not this program's variable names. Use "worked" only for an idea whose re-measured speedup was at least 1.2x. Use "did_not_help" for an idea that was correct but not faster; that is useful to know too. Give an empty list if nothing general was learned."""


@dataclass(frozen=True)
class Lesson:
    """`evidence` holds the task, device, re-measured speedup and date it came from."""

    rule: str
    when: str
    kind: str
    evidence: dict


def load_lessons(path: Path = MEMORY_PATH) -> list[Lesson]:
    """Every stored lesson, oldest first."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Lesson(**json.loads(line)) for line in lines if line.strip()]


def append_lessons(lessons: list[Lesson], path: Path = MEMORY_PATH) -> None:
    with path.open("a", encoding="utf-8") as memory_file:
        for lesson in lessons:
            memory_file.write(json.dumps(asdict(lesson)) + "\n")


def lessons_for_task(lessons: list[Lesson], task_name: str) -> list[Lesson]:
    """Lessons from other tasks, newest first, capped."""
    others = [lesson for lesson in reversed(lessons) if lesson.evidence.get("task") != task_name]
    return others[:MAX_LESSONS_SHOWN]


def format_lessons(lessons: list[Lesson]) -> str:
    def line(lesson):
        evidence = lesson.evidence
        source = f"{evidence.get('task')}, {evidence.get('speedup', 0):.2f}x on {evidence.get('device')}"
        return f"- [{lesson.kind}; {source}] {lesson.rule} Look for: {lesson.when}"

    return "\n".join(line(lesson) for lesson in lessons)


def _outcome(candidate):
    if candidate.speedup is not None:
        return f"correct, {candidate.speedup:.2f}x in the search"
    return f"rejected at {candidate.result.failure_stage}"


def lesson_request(run: ToolAgentRun, source_text: str) -> str:
    lines = [f"Task: {run.task_name} on {run.device}", "Reference source:", source_text.strip(), "Candidates:"]
    lines.extend(f"- {c.id} ({c.idea}): {_outcome(c)}" for c in run.candidates)
    lines.append("Re-measured on a fresh input:")
    lines.extend(f"- {r.candidate.id}: {r.speedup:.2f}x" for r in run.remeasurements if r.speedup is not None)
    best = run.best
    if best is not None:
        lines.extend([f"Source of the best candidate, {best.candidate.id}:", best.candidate.source.strip()])
    return "\n".join(lines)


def parse_lessons(reply: str, evidence: dict) -> list[Lesson]:
    """Keep well-formed lessons; `worked` ones only when the run reached `WORKED_SPEEDUP`."""
    try:
        entries = _extract_json_object(reply).get("lessons")
    except ValueError:
        return []
    if not isinstance(entries, list):
        return []
    allowed = KINDS if evidence["speedup"] >= WORKED_SPEEDUP else ("did_not_help",)
    lessons = [
        Lesson(entry["rule"], entry["when"], entry["kind"], evidence)
        for entry in entries
        if _is_well_formed(entry) and entry["kind"] in allowed
    ]
    return lessons[:MAX_LESSONS_PER_RUN]


def _is_well_formed(entry):
    return isinstance(entry, dict) and all(isinstance(entry.get(name), str) for name in ("rule", "when", "kind"))


def learn_from_run(model: TextModel, run: ToolAgentRun, source_text: str) -> list[Lesson]:
    """One extra LLM call, made only when the run had a correct, re-measured candidate."""
    best = run.best
    if best is None:
        return []
    evidence = {
        "task": run.task_name,
        "device": run.device,
        "speedup": best.speedup,
        "date": date.today().isoformat(),
    }
    reply = model.ask(LESSON_SYSTEM_PROMPT, lesson_request(run, source_text))
    return parse_lessons(reply, evidence)


def lessons_text(mode: str, task_name: str, path: Path = MEMORY_PATH) -> str:
    """What the opening message shows: nothing when memory is off."""
    if mode not in MEMORY_MODES:
        raise ValueError(f"Unknown memory mode {mode!r}; choose from {MEMORY_MODES}.")
    if mode == "off":
        return ""
    return format_lessons(lessons_for_task(load_lessons(path), task_name))
