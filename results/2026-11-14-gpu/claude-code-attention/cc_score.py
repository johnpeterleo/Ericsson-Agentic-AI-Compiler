"""Score Claude Code's candidate.py the way compare_agents scores its arms; print one JSON line.

    python cc_score.py <task> <folder>

The full check (two input cases, compile time limit), then the re-measurement with 50
repeats on the input case the search never saw.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "frozen-6a612ed"))
import jax
from src.agent.search import REMEASURE_REPEATS, EvaluationSettings, evaluate_on
from src.agent.tasks import load_task
from src.optimization.evaluator import evaluate

LOWER_PRECISION_WORDS = ("precision", "tf32", "bfloat16", "float16", "bf16", "half")

if __name__ == "__main__":
    task_name, folder = sys.argv[1], Path(sys.argv[2])
    task = load_task(task_name, "default")
    source = (folder / "candidate.py").read_text()
    namespace = {"__name__": "candidate"}
    exec(source, namespace)
    candidate = namespace["candidate"]
    check = evaluate_on(task, candidate, EvaluationSettings())
    remeasured = None
    if check.correct:
        remeasured = evaluate(task.reference, candidate, task.remeasure_inputs, repeats=REMEASURE_REPEATS,
                              rtol=task.rtol, atol=task.atol)
    print(json.dumps({
        "task": task_name,
        "arm": "claude_code",
        "matmul_precision": jax.config.jax_default_matmul_precision or "default",
        "check_correct": check.correct,
        "check_message": check.message,
        "search_speedup": check.speedup,
        "remeasured_correct": None if remeasured is None else remeasured.correct,
        "remeasured_speedup": None if remeasured is None else remeasured.speedup,
        "precision_words_in_source": [w for w in LOWER_PRECISION_WORDS if w in source.lower()],
        "source": source,
    }))
