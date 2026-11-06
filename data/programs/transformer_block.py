"""Pre-LayerNorm transformer block (attention + MLP), optional fifth program.

KernelBench Level 3 style (a full model block). Combines everything in the
other programs: LayerNorm, QKV projection, causal attention, output
projection, GELU MLP, and residuals. Tests whether the agent can find wins
across operator boundaries in a realistic model. Skipped by default; run it
with --include-optional.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

NAME = "transformer_block"
LEVEL = 3
OPTIONAL = True
CHARACTERISTIC = "mixed: full model block"
DESCRIPTION = "x + Attn(LN(x)), then + MLP(LN(.)); causal multi-head attention, GELU MLP with 4x expansion."
SOURCE = "KernelBench Level 3 style: GPT-2 style transformer block"

SIZES = {
    "small": {"batch": 1, "seq": 64, "d_model": 128, "heads": 2},
    "default": {"batch": 8, "seq": 1024, "d_model": 1024, "heads": 16},
    "large": {"batch": 8, "seq": 2048, "d_model": 2048, "heads": 32},
}

RTOL = 1e-2
ATOL = 1e-2
EPS = 1e-5
HEAD_DIM = 64  # heads = d_model // HEAD_DIM; the head count is not an input


def _layer_norm(x, gamma, beta):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + EPS) * gamma + beta


def reference(x, ln1_g, ln1_b, w_qkv, w_o, ln2_g, ln2_b, w_1, w_2):
    batch, seq, d_model = x.shape
    head_dim = HEAD_DIM
    heads = d_model // head_dim

    h = _layer_norm(x, ln1_g, ln1_b)
    qkv = h @ w_qkv  # (batch, seq, 3 * d_model)
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q, k, v = (t.reshape(batch, seq, heads, head_dim).transpose(0, 2, 1, 3) for t in (q, k, v))
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(jnp.float32(head_dim))
    scores = jnp.where(jnp.tril(jnp.ones((seq, seq), dtype=bool)), scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)
    attn = jnp.einsum("bhqk,bhkd->bhqd", weights, v).transpose(0, 2, 1, 3).reshape(batch, seq, d_model)
    x = x + attn @ w_o

    h = _layer_norm(x, ln2_g, ln2_b)
    return x + jax.nn.gelu(h @ w_1, approximate=False) @ w_2


_erf = np.vectorize(math.erf)


def numpy_reference(x, ln1_g, ln1_b, w_qkv, w_o, ln2_g, ln2_b, w_1, w_2):
    x, ln1_g, ln1_b, w_qkv, w_o, ln2_g, ln2_b, w_1, w_2 = (
        np.asarray(a, np.float64) for a in (x, ln1_g, ln1_b, w_qkv, w_o, ln2_g, ln2_b, w_1, w_2)
    )
    batch, seq, d_model = x.shape
    heads = d_model // HEAD_DIM

    def ln(t, g, b):
        m = t.mean(-1, keepdims=True)
        return (t - m) / np.sqrt(((t - m) ** 2).mean(-1, keepdims=True) + EPS) * g + b

    q, k, v = np.split(ln(x, ln1_g, ln1_b) @ w_qkv, 3, axis=-1)
    q, k, v = (t.reshape(batch, seq, heads, HEAD_DIM).transpose(0, 2, 1, 3) for t in (q, k, v))
    s = np.einsum("bhqd,bhkd->bhqk", q, k) / np.sqrt(HEAD_DIM)
    s = np.where(np.tril(np.ones((seq, seq), dtype=bool)), s, -np.inf)
    w = np.exp(s - s.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    attn = np.einsum("bhqk,bhkd->bhqd", w, v).transpose(0, 2, 1, 3).reshape(batch, seq, d_model)
    x = x + attn @ w_o
    h = ln(x, ln2_g, ln2_b) @ w_1
    return x + (0.5 * h * (1.0 + _erf(h / math.sqrt(2.0)))) @ w_2


def make_inputs(seed=0, size="default"):
    s = SIZES[size]
    d, f32 = s["d_model"], np.float32
    if d // s["heads"] != HEAD_DIM:
        raise ValueError(f"d_model / heads must equal HEAD_DIM={HEAD_DIM}")
    rng = np.random.default_rng(seed)

    def mat(rows, cols):
        return (rng.standard_normal((rows, cols), dtype=f32) / np.sqrt(rows)).astype(f32)

    def vec(center):
        return (center + 0.1 * rng.standard_normal((d,), dtype=f32)).astype(f32)

    x = rng.standard_normal((s["batch"], s["seq"], d), dtype=f32)
    return (
        x, vec(1.0), vec(0.0), mat(d, 3 * d), mat(d, d),
        vec(1.0), vec(0.0), mat(d, 4 * d), mat(4 * d, d),
    )
