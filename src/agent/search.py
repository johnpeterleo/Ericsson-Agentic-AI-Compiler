"""Every evaluated candidate, the leaderboard, and the final re-measurement.

Picking the fastest of many noisy measurements overstates the winner. So at
the end the top candidates are evaluated again, with more repeats, on an input
case the search never saw, and the result is the re-measured number.
"""

from dataclasses import dataclass, field

from src.agent.feedback import CompiledSummary
from src.agent.tasks import Task
from src.agent.types import JaxFn
from src.compiler.compile_time_limit import COMPILE_TIME_LIMIT_SECONDS
from src.optimization.evaluator import EvaluationResult, evaluate

LEADERBOARD_SIZE = 3
REMEASURE_COUNT = 3
REMEASURE_REPEATS = 50
# A run starts with the reference as its best program.
REFERENCE_SPEEDUP = 1.0


@dataclass(frozen=True)
class EvaluationSettings:
    """`compile_time_limit` is in seconds; None turns the check off (each check starts a process)."""

    repeats: int = 20
    compile_time_limit: float | None = COMPILE_TIME_LIMIT_SECONDS


@dataclass(frozen=True)
class RemeasureSettings:
    count: int = REMEASURE_COUNT
    repeats: int = REMEASURE_REPEATS


@dataclass
class Candidate:
    id: str
    idea: str
    source: str
    result: EvaluationResult
    function: JaxFn | None = field(default=None, repr=False)
    compiled: CompiledSummary | None = None

    @property
    def speedup(self) -> float | None:
        return self.result.speedup if self.result.correct else None


@dataclass
class Remeasurement:
    candidate: Candidate
    result: EvaluationResult

    @property
    def speedup(self) -> float | None:
        return self.result.speedup if self.result.correct else None


class SearchState:
    """The candidates so far, and the best speedup that counted as an improvement.

    Timing is noisy, so a candidate only counts as better when it beats that
    best by `margin`; a candidate that beats it by less does not move it.
    """

    def __init__(self, margin: float):
        self.margin = margin
        self.candidates: list[Candidate] = []
        self.best_counted_speedup = REFERENCE_SPEEDUP

    def next_id(self) -> str:
        return f"c{len(self.candidates) + 1}"

    def find(self, candidate_id: str) -> Candidate | None:
        return next((c for c in self.candidates if c.id == candidate_id), None)

    def add(self, candidate: Candidate) -> bool:
        """Store the candidate; return whether it counts as an improvement."""
        self.candidates.append(candidate)
        speedup = candidate.speedup
        if speedup is None or speedup <= self.best_counted_speedup * self.margin:
            return False
        self.best_counted_speedup = speedup
        return True


def ranked(candidates: list[Candidate]) -> list[Candidate]:
    """Correct candidates, fastest first."""
    correct = [candidate for candidate in candidates if candidate.speedup is not None]
    return sorted(correct, key=lambda candidate: candidate.speedup, reverse=True)


def format_leaderboard(candidates: list[Candidate]) -> str:
    leaders = ranked(candidates)[:LEADERBOARD_SIZE]
    if not leaders:
        return "Leaderboard: no correct candidate yet."
    lines = ["Leaderboard (correct candidates, fastest first):"]
    lines.extend(f"  {c.id}  {c.speedup:.2f}x  {c.idea}" for c in leaders)
    lines.extend([f"Source of the best candidate, {leaders[0].id}:", leaders[0].source.strip()])
    return "\n".join(lines)


def evaluate_on(task: Task, function: JaxFn, settings: EvaluationSettings) -> EvaluationResult:
    """The full check: both input cases, paired timing, and the compile time limit."""
    return evaluate(
        task.reference,
        function,
        task.inputs,
        repeats=settings.repeats,
        rtol=task.rtol,
        atol=task.atol,
        holdout_inputs=task.holdout_inputs,
        compile_time_limit=settings.compile_time_limit,
    )


def remeasure(task: Task, candidate: Candidate, repeats: int) -> Remeasurement:
    """Evaluate again on the fresh input case. The candidate already passed the compile limit."""
    result = evaluate(
        task.reference,
        candidate.function,
        task.remeasure_inputs,
        repeats=repeats,
        rtol=task.rtol,
        atol=task.atol,
    )
    return Remeasurement(candidate, result)


def remeasure_top(task: Task, candidates: list[Candidate], settings: RemeasureSettings) -> list[Remeasurement]:
    """Re-measure the top candidates and order them by the new speedup, correct ones first."""
    remeasured = [remeasure(task, c, settings.repeats) for c in ranked(candidates)[: settings.count]]
    return sorted(remeasured, key=lambda r: (r.speedup is not None, r.speedup or 0.0), reverse=True)


def best_remeasured(remeasurements: list[Remeasurement]) -> Remeasurement | None:
    return next((r for r in remeasurements if r.speedup is not None), None)


def candidate_record(candidate: Candidate) -> dict:
    result = candidate.result
    return {
        "id": candidate.id,
        "idea": candidate.idea,
        "source": candidate.source,
        "correct": result.correct,
        "failure_stage": result.failure_stage,
        "message": result.message,
        "reference_ms": result.reference_ms,
        "candidate_ms": result.candidate_ms,
        "speedup": result.speedup,
        "float_dtypes": list(candidate.compiled.float_dtypes) if candidate.compiled else None,
    }


def remeasurement_record(remeasurement: Remeasurement) -> dict:
    result = remeasurement.result
    return {
        "id": remeasurement.candidate.id,
        "search_speedup": remeasurement.candidate.speedup,
        "remeasured_speedup": remeasurement.speedup,
        "reference_ms": result.reference_ms,
        "candidate_ms": result.candidate_ms,
        "message": result.message,
    }
