"""Shared JAX/XLA compilation and compiler representation inspection."""

import time
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class CompilationResult:
    """An executable specialized to input signatures and its diagnostic text."""

    executable: jax.stages.Compiled
    stablehlo: str
    optimized_hlo: str | None


def compile_program(program: Callable[..., jax.Array], inputs: tuple) -> CompilationResult:
    """Lower and compile a pure function for positional array inputs.

    Callers own device placement and execution. Compilation and inspection
    errors propagate; unavailable optimized HLO is represented by None.
    The text representations are for inspection, not executable serialization.
    """
    lowered = jax.jit(program).lower(*inputs)
    stablehlo = lowered.as_text(dialect="stablehlo")
    executable = lowered.compile()
    return CompilationResult(
        executable=executable,
        stablehlo=stablehlo,
        optimized_hlo=executable.as_text(),
    )


def simple_program(x, y):
    return jnp.sin(x) + y * 2


def benchmark():
    x = jnp.ones((1000, 1000))
    y = jnp.ones((1000, 1000))
    inputs = jax.block_until_ready((x, y))

    compilation = compile_program(simple_program, inputs)
    compiled_program = compilation.executable

    print("\n=== StableHLO ===")
    print(compilation.stablehlo)
    print("\n=== Optimized HLO ===")
    if compilation.optimized_hlo is None:
        print("Optimized HLO is unavailable for this backend.")
    else:
        print(compilation.optimized_hlo)

    # Warm up the executable before timing execution.
    result = compiled_program(*inputs)
    result.block_until_ready()

    # Measure execution after compilation.
    start = time.perf_counter()

    result = compiled_program(*inputs)
    result.block_until_ready()

    elapsed = time.perf_counter() - start

    print(f"Result shape: {result.shape}")
    print(f"Execution time: {elapsed * 1000:.3f} ms")


if __name__ == "__main__":
    benchmark()
