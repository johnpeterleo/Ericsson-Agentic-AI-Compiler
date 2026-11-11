"""Run the tool agent once on one task and save the full run.

The model chooses its own actions each turn (inspect, time, evaluate, finish);
see `src.agent.tool_agent`. Every turn is one LLM request. `--model claude`
uses the logged-in Claude Code CLI; `--model gemini` needs GEMINI_API_KEY.

    python -m experiments.run_tool_agent --task transformer_block --size small --model claude
    python -m experiments.run_tool_agent --task suite:prefix_sums --memory readwrite

Memory is off unless asked for. `readwrite` adds lessons to
data/rewrite_memory.jsonl, which is tracked in git.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import jax

from src.agent import memory
from src.agent.search import EvaluationSettings, RemeasureSettings
from src.agent.tasks import SIZES, load_task
from src.agent.text_model import PROVIDERS, create_text_model
from src.agent.tool_agent import ToolAgent, ToolAgentConfig
from src.compiler.compile_time_limit import COMPILE_TIME_LIMIT_SECONDS

RESULTS_DIR = Path("artifacts/tool-agent-runs")


def add_common_arguments(parser):
    """Arguments shared with `experiments.compare_agents`."""
    defaults = ToolAgentConfig()
    parser.add_argument("--size", choices=SIZES, default="small")
    parser.add_argument("--model", choices=PROVIDERS, default="claude", help="LLM provider")
    parser.add_argument("--model-name", help="model name (default depends on --model)")
    parser.add_argument("--repeats", type=int, default=defaults.evaluation.repeats)
    parser.add_argument("--remeasure-repeats", type=int, default=defaults.remeasure.repeats)
    parser.add_argument("--remeasure-count", type=int, default=defaults.remeasure.count)
    parser.add_argument("--compile-time-limit", type=float, default=COMPILE_TIME_LIMIT_SECONDS,
                        help="seconds; 0 turns the check off")
    parser.add_argument("--max-evaluations", type=int, default=defaults.max_evaluations)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--margin", type=float, default=defaults.improvement_margin)


def evaluation_settings(args):
    return EvaluationSettings(repeats=args.repeats, compile_time_limit=args.compile_time_limit or None)


def remeasure_settings(args):
    return RemeasureSettings(count=args.remeasure_count, repeats=args.remeasure_repeats)


def tool_agent_config(args, max_turns):
    return ToolAgentConfig(
        max_turns=max_turns,
        max_evaluations=args.max_evaluations,
        patience=args.patience,
        improvement_margin=args.margin,
        evaluation=evaluation_settings(args),
        remeasure=remeasure_settings(args),
    )


def run_details(args):
    """What every saved record says about how it was produced."""
    device = jax.devices()[0]
    return {
        "device": f"{device.platform} ({device.device_kind})",
        "matmul_precision": jax.config.jax_default_matmul_precision or "default",
        "size": args.size,
        "provider": args.model,
        "model_name": args.model_name,
        "repeats": args.repeats,
        "remeasure_repeats": args.remeasure_repeats,
        "started": datetime.now().isoformat(timespec="seconds"),
    }


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="e.g. transformer_block or suite:prefix_sums")
    parser.add_argument("--memory", choices=memory.MEMORY_MODES, default="off")
    parser.add_argument("--max-turns", type=int, default=ToolAgentConfig().max_turns)
    add_common_arguments(parser)
    return parser.parse_args(argv)


def print_result(run):
    print(f"Stopped: {run.stop_reason} after {len(run.turns)} turns and {len(run.candidates)} evaluations.")
    for remeasurement in run.remeasurements:
        candidate = remeasurement.candidate
        remeasured = f"{remeasurement.speedup:.2f}x" if remeasurement.speedup is not None else "failed"
        print(f"  {candidate.id}: search {candidate.speedup:.2f}x, re-measured {remeasured}  ({candidate.idea})")
    if run.best is None:
        print("No correct candidate.")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_arguments(argv)
    task = load_task(args.task, args.size)
    model = create_text_model(args.model, args.model_name)
    details = run_details(args)
    print(f"Device: {details['device']}; matmul precision: {details['matmul_precision']}")

    agent = ToolAgent(model, tool_agent_config(args, args.max_turns))
    run = agent.run(task, memory.lessons_text(args.memory, task.name))
    print_result(run)

    learned = []
    if args.memory == "readwrite":
        learned = memory.learn_from_run(model, run, task.source_text)
        memory.append_lessons(learned)
        print(f"Stored {len(learned)} lessons in {memory.MEMORY_PATH}")

    path = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{task.name.replace(':', '-')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {**details, "memory": args.memory, **run.record(), "lessons_learned": [vars(l) for l in learned]}
    path.write_text(json.dumps(record, indent=2))
    print(f"Saved the run to {path}")


if __name__ == "__main__":
    main()
