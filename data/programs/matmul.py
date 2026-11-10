"""Square matrix multiplication, C = A @ B.

KernelBench Level 1 style (single operator). Compute-bound: on GPU, XLA hands
this to cuBLAS, which is already close to peak. This is the control case;
a large reported speedup here is more likely a measurement error than a real win.
"""

import jax.numpy as jnp
import numpy as np

NAME = "matmul"
LEVEL = 1
CHARACTERISTIC = "compute-bound"
DESCRIPTION = "Square float32 matrix multiplication C = A @ B."
SOURCE = "KernelBench Level 1: square matrix multiplication"

# n x n matrices. "small" is for tests/CPU, "default" and "large" for the GPU.
SIZES = {"small": {"n": 256}, "default": {"n": 4096}, "large": {"n": 8192}}

# float32 matmuls on NVIDIA GPUs may use TF32 by default, so rewrites that
# change the reduction order can differ by roughly 1e-3 relative.
RTOL = 1e-2
ATOL = 1e-2


def reference(a, b):
    return jnp.matmul(a, b)


def numpy_reference(a, b):
    return np.asarray(a, np.float64) @ np.asarray(b, np.float64)


def make_inputs(seed=0, size="default"):
    n = SIZES[size]["n"]
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n), dtype=np.float32)
    # Scale B so the entries of C are roughly N(0, 1) for every n.
    b = (rng.standard_normal((n, n), dtype=np.float32) / np.sqrt(n)).astype(np.float32)
    return a, b
