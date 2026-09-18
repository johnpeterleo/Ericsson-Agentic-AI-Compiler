"""Run from the repository root: python -m src.optimization.demo_evaluator."""

import jax
import jax.numpy as jnp
import numpy as np

from src.compiler.compile import simple_program
from src.optimization.evaluator import evaluate


def candidate(x, y):
    # Same computation, written differently. No speedup is assumed.
    return jnp.sin(x) + (y + y)


def main():
    # Placeholder inputs only; the benchmark suite can be supplied separately.
    rng = np.random.default_rng(42)
    inputs = tuple(rng.normal(size=(128, 128)).astype(np.float32) for _ in range(2))
    backend = jax.default_backend()
    print(f"Backend: {backend}")
    if backend != "gpu":
        print("Use the target GPU before drawing conclusions about GPU performance.")
    result = evaluate(simple_program, candidate, inputs)
    if not result.correct:
        print(f"Evaluation failed at {result.failure_stage}: {result.message}")
    else:
        print(result)
        print(f"Observed speedup: {result.speedup:.3f}x (remeasure before claiming a win)")


if __name__ == "__main__":
    main()
