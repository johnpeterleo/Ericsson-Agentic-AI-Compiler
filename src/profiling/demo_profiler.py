"""Run from the repository root: python -m src.profiling.demo_profiler."""

import jax.numpy as jnp
import numpy as np

from src.compiler.compile import simple_program
from src.profiling.profiler import profile_function


def main():
    print("size     median (ms)    spread (ms)    device")
    for size in (128, 512, 1024):
        rng = np.random.default_rng(42)

        # Generate random inputs for the program. (x, y) are both square matrices of the given size.
        inputs = tuple(
            rng.normal(size=(size, size)).astype(np.float32) for _ in range(2)
        )
        result = profile_function(
            simple_program,
            inputs,
            repeats=10,
            capture_stablehlo=(size == 128),
            stablehlo_path="artifacts/simple_program.stablehlo" if size == 128 else None,
        )
        print(
            f"{size:4d}     {result.median_ms:9.3f}"
            f"    {result.min_ms:.3f}–{result.max_ms:.3f}"
            f"    {result.device_platform}: {result.device_kind}"
        )
    print("StableHLO for the 128x128 case was saved under artifacts/.")


if __name__ == "__main__":
    main()
