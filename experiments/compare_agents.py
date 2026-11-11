"""Compare the tool agent with the existing agent at the same number of LLM calls.

For each task and each repetition, with N calls (`--calls`, default 12):

    single          one call to the existing agent
    best_of_n       N independent single calls
    chain           the existing agent with N attempts
    chain_original  the same with the November system prompt, which names tools it lacks
    tool_agent      the tool agent with N turns, memory off

The first three use a fair system prompt that describes what they really get
(`FAIR_BASELINE_PROMPT`). Every arm's top candidates are re-measured on a
fresh input, and that is the speedup reported. One JSON line per task, arm and
repetition goes to artifacts/agent-comparisons/, with the model name and the
full transcript. A full run with the defaults is up to
10 tasks x (1 + 4 x 12) = 490 LLM calls per repetition.

    python -m experiments.compare_agents --tasks softmax,suite:prefix_sums --calls 3
    JAX_DEFAULT_MATMUL_PRECISION=highest python -m experiments.compare_agents --size default
"""

import argparse
from datetime import datetime
from pathlib import Path

from experiments.run_tool_agent import (
    add_common_arguments,
    evaluation_settings,
    remeasure_settings,
    run_details,
    tool_agent_config,
)
from src.agent.baselines import ArmSettings, BaselineRunner, run_best_of_n, run_chain, run_single
from src.agent.llm_policy import LLMQuotaExhaustedError, LLMUnavailableError
from src.agent.tasks import load_task, task_names
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.text_model import create_text_model
from src.agent.tool_agent import ToolAgent
from src.agent.tool_prompts import FAIR_BASELINE_PROMPT
from src.compiler.compile_time_limit import CompileTimeoutError
from tests.benchmark_neural_network_agent import _append_record

RESULTS_DIR = Path("artifacts/agent-comparisons")
ARMS = ("single", "best_of_n", "chain", "chain_original", "tool_agent")
DEFAULT_TASKS = (
    "softmax", "fused_gemm_gelu_layernorm", "attention", "transformer_block", "gpt",
    "planted:causal_self_attention", "planted:embed_and_transform", "planted:local_standardize",
    "suite:matrix_chain_times_vector", "suite:prefix_sums",
)
DEFAULT_CALLS = 12


def make_arms(model, args):
    """Map each arm's name to a function that runs it once on a task and returns its record."""

    def runner(task, system_prompt):
        settings = ArmSettings(evaluation_settings(args), remeasure_settings(args), system_prompt)
        return BaselineRunner(model, task, settings)

    return {
        "single": lambda task: run_single(runner(task, FAIR_BASELINE_PROMPT)).record(),
        "best_of_n": lambda task: run_best_of_n(runner(task, FAIR_BASELINE_PROMPT), args.calls).record(),
        "chain": lambda task: run_chain(runner(task, FAIR_BASELINE_PROMPT), args.calls).record(),
        "chain_original": lambda task: run_chain(runner(task, SYSTEM_PROMPT), args.calls).record(),
        "tool_agent": lambda task: ToolAgent(model, tool_agent_config(args, args.calls)).run(task).record(),
    }


def _speedup_text(speedup):
    return f"{speedup:.2f}x" if speedup is not None else "-"


def print_row(task_name, arm, record):
    print(
        f"{task_name:34s} {arm:11s} re-measured {_speedup_text(record['remeasured_speedup']):>7s} "
        f"(search {_speedup_text(record['search_speedup']):>7s}), "
        f"{record['llm_calls']} calls, {record['evaluations']} evaluations, {record['wall_seconds']:.0f} s"
    )


def _name_list(choices):
    def parse(text):
        names = tuple(text.split(","))
        unknown = [name for name in names if name not in choices]
        if unknown:
            raise argparse.ArgumentTypeError(f"unknown {unknown}; choose from: {', '.join(choices)}")
        return names

    return parse


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=_name_list(task_names()), default=DEFAULT_TASKS)
    parser.add_argument("--arms", type=_name_list(ARMS), default=ARMS)
    parser.add_argument("--calls", type=int, default=DEFAULT_CALLS, help="LLM calls per arm (N)")
    parser.add_argument("--repetitions", type=int, default=1)
    add_common_arguments(parser)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    model = create_text_model(args.model, args.model_name)
    arms = make_arms(model, args)
    details = {**run_details(args, model), "calls": args.calls}
    path = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    print(f"Device: {details['device']}; matmul precision: {details['matmul_precision']}")
    print(f"Model: {details['model_name']}")
    print(f"Saving results to: {path}")

    for task_name in args.tasks:
        task = load_task(task_name, args.size)
        for repetition in range(args.repetitions):
            for arm in args.arms:
                row = {**details, "task": task_name, "arm": arm, "repetition": repetition}
                try:
                    record = arms[arm](task)
                except LLMQuotaExhaustedError as error:
                    print(f"Stopping: {error}")
                    _append_record(path, {**row, "status": "quota", "message": str(error)})
                    return
                except (LLMUnavailableError, CompileTimeoutError) as error:
                    print(f"{task_name} {arm}: skipped: {error}")
                    _append_record(path, {**row, "status": "skipped", "message": str(error)})
                    continue
                print_row(task_name, arm, record)
                _append_record(path, {**row, "status": "done", **record})


if __name__ == "__main__":
    main()
