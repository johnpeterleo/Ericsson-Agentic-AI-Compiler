"""Run the optimization agent on one benchmark program and save the result.

Run from the repository root:

    python -m src.benchmarks.run_agent --program softmax --size small
    python -m src.benchmarks.run_agent --program attention --max-iterations 8

Uses Gemini when GEMINI_API_KEY is set (see src/agent/llm_policy.py); without
a key the scripted DemoPolicy has no candidates and stops immediately, which
still checks that the program works with the whole pipeline. The program's
own RTOL/ATOL are used for the correctness check. Results are written to
``artifacts/agent_runs/``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import jax

from src.agent.agent import AgentConfig, OptimizationAgent, format_step_summary
from src.agent.llm_policy import create_default_policy
from src.agent.tools import AgentToolkit, JaxProfiler
from src.benchmarks.baseline import environment_info
from src.benchmarks.registry import REPO_ROOT, get_program

DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "agent_runs"


def build_agent(program, *, max_iterations: int, repeats: int, demo: bool) -> OptimizationAgent:
    toolkit = AgentToolkit(
        profiler=JaxProfiler(repeats=repeats),
        profile_repeats=repeats,
        rtol=program.rtol,
        atol=program.atol,
    )
    policy = create_default_policy(prefer_llm=not demo)
    config = AgentConfig(max_iterations=max_iterations, benchmark_repeats=repeats)
    return OptimizationAgent(toolkit=toolkit, policy=policy, config=config)


def run(program_name: str, *, size: str = "default", seed: int = 0, max_iterations: int = 5,
        repeats: int = 20, demo: bool = False, out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    program = get_program(program_name)
    inputs = program.make_inputs(seed=seed, size=size)
    agent = build_agent(program, max_iterations=max_iterations, repeats=repeats, demo=demo)
    print(f"Agent on {program.name} (size={size}) with {type(agent.policy).__name__} "
          f"on {jax.default_backend()}")

    result = agent.run(program.reference, inputs)
    steps = []
    for step in result.context.history:
        ev = step.evaluation
        steps.append({
            "index": step.index,
            "rationale": step.proposal.rationale,
            "correct": ev.correct if ev else None,
            "failure_stage": ev.failure_stage if ev else None,
            "message": ev.message if ev else None,
            "reference_ms": ev.reference_ms if ev else None,
            "candidate_ms": ev.candidate_ms if ev else None,
            "speedup": ev.speedup if ev else None,
            "candidate_source": step.proposal.metadata.get("candidate_source"),
            "notes": step.notes,
        })
        print(format_step_summary(step))

    record = {
        "environment": environment_info(),
        "program": program.name,
        "size": size,
        "seed": seed,
        "policy": type(agent.policy).__name__,
        "status": result.status,
        "message": result.message,
        "baseline_profile": result.context.baseline_profile.summary(),
        "best_speedup": result.context.best_speedup(),
        "steps": steps,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{program.name}_{size}_{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(f"{result.status}: {result.message}\nBest speedup: {record['best_speedup']}\nWrote {path}")
    return path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the agent on a benchmark program.")
    parser.add_argument("--program", required=True)
    parser.add_argument("--size", default="default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--demo", action="store_true", help="force the scripted policy (no API calls)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    run(args.program, size=args.size, seed=args.seed, max_iterations=args.max_iterations,
        repeats=args.repeats, demo=args.demo, out_dir=args.out)


if __name__ == "__main__":
    main()
