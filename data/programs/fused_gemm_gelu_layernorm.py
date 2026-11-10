"""GEMM -> bias -> GELU -> LayerNorm.

KernelBench Level 2 style (a chain of operators). Mixes a compute-bound GEMM
with memory-bound elementwise and reduction ops; this is the typical
transformer pattern. Tests how much the agent can gain from fusion beyond
what XLA already fuses (epilogue fusion into the GEMM, a fused LayerNorm).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

NAME = "fused_gemm_gelu_layernorm"
LEVEL = 2
CHARACTERISTIC = "mixed: GEMM + elementwise + reduction"
DESCRIPTION = "y = LayerNorm(GELU(x @ W + b)) * gamma + beta over the feature axis."
SOURCE = "KernelBench Level 2 style: Gemm + BiasAdd + GELU + LayerNorm"

SIZES = {
    "small": {"batch": 128, "d_in": 256, "d_out": 256},
    "default": {"batch": 4096, "d_in": 4096, "d_out": 4096},
    "large": {"batch": 8192, "d_in": 8192, "d_out": 8192},
}

RTOL = 1e-2  # GEMM may use TF32 on GPU, see matmul.py
ATOL = 1e-2
EPS = 1e-5


def reference(x, w, b, gamma, beta):
    h = x @ w + b
    h = jax.nn.gelu(h, approximate=False)
    mean = jnp.mean(h, axis=-1, keepdims=True)
    var = jnp.mean((h - mean) ** 2, axis=-1, keepdims=True)
    return (h - mean) / jnp.sqrt(var + EPS) * gamma + beta


_erf = np.vectorize(math.erf)


def numpy_reference(x, w, b, gamma, beta):
    x, w, b, gamma, beta = (np.asarray(a, np.float64) for a in (x, w, b, gamma, beta))
    h = x @ w + b
    h = 0.5 * h * (1.0 + _erf(h / math.sqrt(2.0)))
    mean = h.mean(axis=-1, keepdims=True)
    var = ((h - mean) ** 2).mean(axis=-1, keepdims=True)
    return (h - mean) / np.sqrt(var + EPS) * gamma + beta


def make_inputs(seed=0, size="default"):
    s = SIZES[size]
    rng = np.random.default_rng(seed)
    f32 = np.float32
    x = rng.standard_normal((s["batch"], s["d_in"]), dtype=f32)
    w = (rng.standard_normal((s["d_in"], s["d_out"]), dtype=f32) / np.sqrt(s["d_in"])).astype(f32)
    b = (0.1 * rng.standard_normal((s["d_out"],), dtype=f32)).astype(f32)
    gamma = (1.0 + 0.1 * rng.standard_normal((s["d_out"],), dtype=f32)).astype(f32)
    beta = (0.1 * rng.standard_normal((s["d_out"],), dtype=f32)).astype(f32)
    return x, w, b, gamma, beta
