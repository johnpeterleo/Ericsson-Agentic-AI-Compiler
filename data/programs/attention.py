"""Causal multi-head scaled dot-product attention.

KernelBench Level 3 style (a model building block). The naive form
materialises the full (seq x seq) score matrix per head, so memory traffic
grows quadratically. XLA cannot restructure this on its own; the target
rewrite is a FlashAttention-style tiled, streaming-softmax version (in JAX or
as a Pallas kernel), which can also skip fully masked blocks.
"""

import jax.numpy as jnp
import numpy as np

NAME = "attention"
LEVEL = 3
CHARACTERISTIC = "memory-bound at long sequences (quadratic score matrix)"
DESCRIPTION = "softmax(mask(Q K^T / sqrt(d))) V with a causal mask, shape (batch, heads, seq, head_dim)."
SOURCE = "KernelBench Level 3 style: scaled dot-product attention"

SIZES = {
    "small": {"batch": 1, "heads": 2, "seq": 128, "head_dim": 64},
    "default": {"batch": 2, "heads": 16, "seq": 2048, "head_dim": 64},
    "large": {"batch": 2, "heads": 16, "seq": 4096, "head_dim": 64},
}

RTOL = 1e-2  # matmuls may use TF32 on GPU, see matmul.py
ATOL = 1e-2


def reference(q, k, v):
    seq, head_dim = q.shape[-2], q.shape[-1]
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(jnp.float32(head_dim))
    causal = jnp.tril(jnp.ones((seq, seq), dtype=bool))
    scores = jnp.where(causal, scores, -jnp.inf)
    scores = scores - jnp.max(scores, axis=-1, keepdims=True)
    weights = jnp.exp(scores)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    return jnp.einsum("bhqk,bhkd->bhqd", weights, v)


def numpy_reference(q, k, v):
    q, k, v = (np.asarray(a, np.float64) for a in (q, k, v))
    seq, head_dim = q.shape[-2], q.shape[-1]
    scores = np.einsum("bhqd,bhkd->bhqk", q, k) / np.sqrt(head_dim)
    scores = np.where(np.tril(np.ones((seq, seq), dtype=bool)), scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.einsum("bhqk,bhkd->bhqd", weights, v)


def make_inputs(seed=0, size="default"):
    s = SIZES[size]
    shape = (s["batch"], s["heads"], s["seq"], s["head_dim"])
    rng = np.random.default_rng(seed)
    return tuple(rng.standard_normal(shape, dtype=np.float32) for _ in range(3))
