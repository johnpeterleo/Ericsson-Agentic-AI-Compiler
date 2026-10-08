"""Run from repository root: python -m src.agent.demo_agent"""

import numpy as np

from src.agent.agent import AgentConfig, DemoPolicy, OptimizationAgent, format_step_summary
from src.agent.tools import AgentToolkit, JaxProfiler
from src.compiler.compile import simple_program
import jax.numpy as jnp


def candidate_v1(x, y):
    """Same math as simple_program; different surface syntax (demo only)."""
    return jnp.sin(x) + (y + y)


def main():
    rng = np.random.default_rng(0)
    inputs = tuple(rng.normal(size=(256, 256)).astype(np.float32) for _ in range(2))

    policy = DemoPolicy(candidates=[candidate_v1])
    agent = OptimizationAgent(
        toolkit=AgentToolkit(profiler=JaxProfiler(repeats=5)),
        policy=policy,
        config=AgentConfig(max_iterations=3, benchmark_repeats=5),
    )

    print(f"JAX backend: {agent.build_context(simple_program, inputs).backend}")
    result = agent.run(simple_program, inputs)
    print(f"Run finished: {result.status} — {result.message}")
    for step in result.context.history:
        print(format_step_summary(step))
    best = result.context.best_speedup()
    if best is not None:
        print(f"Best observed speedup in history: {best:.3f}x")
    else:
        print("No successful candidate speedups recorded.")


if __name__ == "__main__":
    main()
