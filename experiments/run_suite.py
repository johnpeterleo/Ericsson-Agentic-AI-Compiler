"""Run the benchmark suite: how much of the expert speedup does the agent reach?

Without --agent no LLM request is made. Each program's expert rewrite is
evaluated against its reference, which shows the speedup available on this
device and that the expert passes the correctness check here:

    python -m experiments.run_suite

With --agent, an LLM also gets `--iterations` attempts at every program. Each
attempt is one request, so 13 programs x 3 attempts is 39 requests. The LLM is
Gemini by default; `--policy claude` uses Claude through the logged-in Claude
Code CLI instead of an API key:

    python -m experiments.run_suite --agent
    python -m experiments.run_suite --agent --programs attention,prefix_sums
    python -m experiments.run_suite --agent --policy claude --programs prefix_sums
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import jax
import numpy as np

from src.agent.agent import AgentConfig, OptimizationAgent, format_step_summary
from src.agent.llm_policy import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_GEMINI_MODEL,
    ClaudeCLIPolicy,
    GeminiPolicy,
    LLMQuotaExhaustedError,
    LLMUnavailableError,
    load_project_env,
)
from src.agent.tools import AgentToolkit, JaxProfiler
from src.benchmarks.programs import PROGRAMS, PROGRAMS_BY_NAME
from src.compiler.compile_time_limit import COMPILE_TIME_LIMIT_SECONDS, CompileTimeoutError
from src.optimization.evaluator import EvaluationResult, evaluate
from tests.benchmark_neural_network_agent import _append_record, _step_record

RESULTS_DIR = Path("artifacts/suite-runs")
INPUT_SEED = 2026
HOLDOUT_SEED = 2027
# A program counts as solved once a correct attempt reaches this share of the
# expert speedup. The bar sits below the timing spread: on a laptop CPU the same
# rewrite measured 0.8x to 1.3x of its median speedup between evaluations.
SOLVED_FRACTION = 0.7


@dataclass
class ProgramResult:
    """One row of the final table."""

    name: str
    expert: EvaluationResult
    agent_best_speedup: float | None = None
    solved_at_step: int | None = None


class RunLog:
    """Appends one JSON line per result, each tagged with the details of this run."""

    def __init__(self, path, run_details):
        self.path = path
        self.run_details = run_details

    def save(self, program_name, record):
        _append_record(self.path, {**self.run_details, "program": program_name, **record})


def make_input_cases(program):
    """The inputs the agent works with, and a held-out case only the evaluator sees."""
    inputs = program.make_inputs(np.random.default_rng(INPUT_SEED), program.benchmark_size)
    holdout = program.make_inputs(np.random.default_rng(HOLDOUT_SEED), program.benchmark_size)
    return inputs, holdout


def first_solving_step(history, expert_speedup):
    """Index of the first correct attempt that reached most of the expert speedup."""
    target = SOLVED_FRACTION * expert_speedup
    for step in history:
        if step.evaluation.correct and step.evaluation.speedup >= target:
            return step.index
    return None


def make_policy(args):
    if args.policy == "claude":
        return ClaudeCLIPolicy(model=args.model)
    load_project_env()
    return GeminiPolicy(model=args.model)


def make_agent(args, device):
    return OptimizationAgent(
        toolkit=AgentToolkit(
            profiler=JaxProfiler(repeats=args.repeats), profile_repeats=args.repeats
        ),
        policy=make_policy(args),
        config=AgentConfig(
            max_iterations=args.iterations,
            benchmark_repeats=args.repeats,
            hardware_note=f"Target JAX device: {device.platform} ({device.device_kind}).",
        ),
    )


def add_agent_result(result, run, log):
    """Record every attempt of one agent run and summarize it in the table row."""
    for step in run.context.history:
        print(f"  {format_step_summary(step)}")
        log.save(result.name, {"kind": "agent", **_step_record(step)})
    if not run.context.history:
        log.save(result.name, {"kind": "agent", "status": "stopped", "message": run.message})
    result.agent_best_speedup = run.context.best_speedup()
    if result.expert.correct:
        result.solved_at_step = first_solving_step(run.context.history, result.expert.speedup)


def _speedup_text(speedup):
    return f"{speedup:.2f}x" if speedup is not None else "-"


def _share_of_expert_text(result):
    if result.agent_best_speedup is None or not result.expert.correct:
        return "-"
    return f"{result.agent_best_speedup / result.expert.speedup:.0%}"


def print_expert_result(result):
    expert = result.expert
    if not expert.correct:
        print(f"{result.name}: expert rejected at {expert.failure_stage}: {expert.message}")
        return
    print(
        f"{result.name}: expert {expert.speedup:.2f}x "
        f"(reference {expert.reference_ms:.3f} ms, expert {expert.candidate_ms:.3f} ms)"
    )


def print_table(results):
    print(f"\n{'program':32s} {'expert':>9s} {'agent':>9s} {'% of expert':>12s} {'solved at':>10s}")
    for result in results:
        solved_at = "-" if result.solved_at_step is None else f"step {result.solved_at_step}"
        print(
            f"{result.name:32s} {_speedup_text(result.expert.speedup):>9s} "
            f"{_speedup_text(result.agent_best_speedup):>9s} "
            f"{_share_of_expert_text(result):>12s} {solved_at:>10s}"
        )


def _program_list(text):
    """Parse `name1,name2` from the command line."""
    try:
        return tuple(PROGRAMS_BY_NAME[name] for name in text.split(","))
    except KeyError as error:
        known = ", ".join(PROGRAMS_BY_NAME)
        raise argparse.ArgumentTypeError(f"unknown program {error}; choose from: {known}")


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--programs", type=_program_list, default=PROGRAMS)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--agent", action="store_true", help="also let an LLM try each program")
    parser.add_argument("--policy", choices=("gemini", "claude"), default="gemini")
    parser.add_argument("--iterations", type=int, default=3, help="agent attempts per program")
    parser.add_argument("--model", help="model name (default depends on --policy)")
    args = parser.parse_args(argv)
    if args.model is None:
        args.model = DEFAULT_CLAUDE_MODEL if args.policy == "claude" else DEFAULT_GEMINI_MODEL
    return args


def main(argv=None):
    args = parse_arguments(argv)
    device = jax.devices()[0]
    run_details = {
        "device": f"{device.platform} ({device.device_kind})",
        "matmul_precision": jax.config.jax_default_matmul_precision or "default",
        "repeats": args.repeats,
        "policy": args.policy if args.agent else None,
        "model": args.model if args.agent else None,
    }
    log = RunLog(RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.jsonl", run_details)
    agent = make_agent(args, device) if args.agent else None
    print(f"Device: {run_details['device']}; matmul precision: {run_details['matmul_precision']}")
    print(f"Saving results to: {log.path}")

    results = []
    for program in args.programs:
        inputs, holdout = make_input_cases(program)
        try:
            expert = evaluate(
                program.reference,
                program.expert,
                inputs,
                repeats=args.repeats,
                holdout_inputs=holdout,
                compile_time_limit=COMPILE_TIME_LIMIT_SECONDS,
            )
        except CompileTimeoutError as error:
            print(f"{program.name}: skipped, the reference compiles too slowly: {error}")
            log.save(program.name, {"kind": "expert", "status": "compile_timeout", "message": str(error)})
            continue
        result = ProgramResult(program.name, expert)
        results.append(result)
        print_expert_result(result)
        log.save(program.name, {"kind": "expert", **asdict(expert), "speedup": expert.speedup})
        if agent is None:
            continue

        try:
            run = agent.run(program.reference, inputs, holdout_inputs=holdout)
        except LLMQuotaExhaustedError as error:
            print(f"Stopping before {program.name} is finished: {error}")
            log.save(program.name, {"kind": "agent", "status": "quota", "message": str(error)})
            break
        except LLMUnavailableError as error:
            print(f"  skipped: {error}")
            log.save(program.name, {"kind": "agent", "status": "skipped", "message": str(error)})
            continue
        add_agent_result(result, run, log)

    if agent is not None:
        print_table(results)


if __name__ == "__main__":
    main()
