"""Row-wise softmax over the last axis.

KernelBench Level 1 style (single operator). Memory-bound reduction: the
naive form reads the data several times (max, exp/sum, divide). Room for the
agent is in reducing passes over memory, e.g. an online/streaming softmax or
a fused Pallas kernel.
"""

import jax.numpy as jnp
import numpy as np

NAME = "softmax"
LEVEL = 1
CHARACTERISTIC = "memory-bound reduction"
DESCRIPTION = "Numerically stable softmax over the last axis of a (rows, cols) matrix."
SOURCE = "KernelBench Level 1: softmax"

SIZES = {
    "small": {"rows": 64, "cols": 1024},
    "default": {"rows": 4096, "cols": 8192},
    "large": {"rows": 8192, "cols": 16384},
}

RTOL = 1e-4
ATOL = 1e-6


def reference(x):
    # Written out explicitly (rather than jax.nn.softmax) so the agent sees
    # each pass over the data in the source and the HLO.
    x_max = jnp.max(x, axis=-1, keepdims=True)
    e = jnp.exp(x - x_max)
    return e / jnp.sum(e, axis=-1, keepdims=True)


def numpy_reference(x):
    x = np.asarray(x, np.float64)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def make_inputs(seed=0, size="default"):
    s = SIZES[size]
    rng = np.random.default_rng(seed)
    # Scale 3 gives a peaked but not degenerate distribution.
    x = (3.0 * rng.standard_normal((s["rows"], s["cols"]), dtype=np.float32)).astype(np.float32)
    return (x,)
