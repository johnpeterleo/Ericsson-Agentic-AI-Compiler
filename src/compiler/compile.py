import time

import jax
import jax.numpy as jnp


def simple_program(x, y):
    return jnp.sin(x) + y * 2


def benchmark():
    x = jnp.ones((1000, 1000))
    y = jnp.ones((1000, 1000))

    compiled_program = jax.jit(simple_program)

    # Show what JAX lowers the program to.
    lowered = compiled_program.lower(x, y)
    print("\n=== Lowered program ===")
    print(lowered.as_text())

    # First call compiles the program.
    result = compiled_program(x, y)
    result.block_until_ready()

    # Measure execution after compilation.
    start = time.perf_counter()

    result = compiled_program(x, y)
    result.block_until_ready()

    elapsed = time.perf_counter() - start

    print(f"Result shape: {result.shape}")
    print(f"Execution time: {elapsed * 1000:.3f} ms")


if __name__ == "__main__":
    benchmark()