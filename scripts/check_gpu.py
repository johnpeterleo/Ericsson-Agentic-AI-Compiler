"""Run from the repository root: python -m scripts.check_gpu."""

import sys

import jax
import jax.numpy as jnp

from src.compiler.compile import compile_program, simple_program


def main():
    backend = jax.default_backend()
    device = jax.devices()[0]
    print(f"Backend: {backend}")
    print(f"Device: {device.platform} ({device.device_kind})")
    if backend != "gpu":
        print("FAILED: JAX is not using a GPU.")
        return 1

    inputs = jax.block_until_ready((jnp.ones((32, 32)), jnp.ones((32, 32))))
    compilation = compile_program(simple_program, inputs)
    result = jax.block_until_ready(compilation.executable(*inputs))
    if compilation.optimized_hlo is None:
        print("FAILED: optimized HLO is unavailable.")
        return 1

    print(f"Compiled and ran on GPU; result shape: {result.shape}")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
