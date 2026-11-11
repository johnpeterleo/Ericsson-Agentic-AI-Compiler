"""GPT-2 style decoder-only language model forward pass, optional program.

KernelBench Level 3 style (a full model). A stack of pre-LayerNorm
transformer blocks (causal multi-head attention + GELU MLP), a final
LayerNorm, and a language-model head tied to the token embedding. The
"default" size has GPT-2 small shapes (124M parameters) and "large" has
GPT-2 medium shapes (350M). Weights are random with GPT-2 initialisation:
speed depends on shapes and operations, not on trained values. Per-layer
weights are stacked along a leading layer axis. Token ids are looked up in
make_inputs, so the program receives float32 token embeddings. Skipped by
default; run it with --include-optional.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

NAME = "gpt"
LEVEL = 3
OPTIONAL = True
CHARACTERISTIC = "mixed: full model (stacked transformer blocks + LM head)"
DESCRIPTION = (
    "logits = LN_f(blocks(tok_emb + pos_emb)) @ wte.T, where each block is "
    "x + Attn(LN(x)) then + MLP(LN(.)) with causal attention and a 4x GELU MLP."
)
SOURCE = "KernelBench Level 3 style: GPT-2 forward pass (nanoGPT layout)"

SIZES = {
    "small": {"batch": 1, "seq": 32, "d_model": 128, "heads": 2, "layers": 2, "vocab": 512},
    "default": {"batch": 4, "seq": 1024, "d_model": 768, "heads": 12, "layers": 12, "vocab": 50257},
    "large": {"batch": 4, "seq": 1024, "d_model": 1024, "heads": 16, "layers": 24, "vocab": 50257},
}

RTOL = 1e-2  # matmuls may use TF32 on GPU, see matmul.py
ATOL = 1e-2
EPS = 1e-5
HEAD_DIM = 64  # heads = d_model // HEAD_DIM; the head count is not an input


def _layer_norm(x, gamma, beta):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + EPS) * gamma + beta


def reference(
    tok_emb, wpe,
    ln1_g, ln1_b, w_qkv, b_qkv, w_o, b_o,
    ln2_g, ln2_b, w_1, b_1, w_2, b_2,
    lnf_g, lnf_b, wte,
):
    batch, seq, d_model = tok_emb.shape
    heads = d_model // HEAD_DIM
    causal = jnp.tril(jnp.ones((seq, seq), dtype=bool))

    x = tok_emb + wpe
    for layer in range(w_qkv.shape[0]):
        h = _layer_norm(x, ln1_g[layer], ln1_b[layer])
        qkv = h @ w_qkv[layer] + b_qkv[layer]
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q, k, v = (t.reshape(batch, seq, heads, HEAD_DIM).transpose(0, 2, 1, 3) for t in (q, k, v))
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(jnp.float32(HEAD_DIM))
        scores = jnp.where(causal, scores, -jnp.inf)
        weights = jax.nn.softmax(scores, axis=-1)
        attn = jnp.einsum("bhqk,bhkd->bhqd", weights, v).transpose(0, 2, 1, 3).reshape(batch, seq, d_model)
        x = x + attn @ w_o[layer] + b_o[layer]

        h = _layer_norm(x, ln2_g[layer], ln2_b[layer])
        h = jax.nn.gelu(h @ w_1[layer] + b_1[layer], approximate=False)
        x = x + h @ w_2[layer] + b_2[layer]

    x = _layer_norm(x, lnf_g, lnf_b)
    return x @ wte.T


_erf = np.vectorize(math.erf)


def numpy_reference(
    tok_emb, wpe,
    ln1_g, ln1_b, w_qkv, b_qkv, w_o, b_o,
    ln2_g, ln2_b, w_1, b_1, w_2, b_2,
    lnf_g, lnf_b, wte,
):
    (tok_emb, wpe, ln1_g, ln1_b, w_qkv, b_qkv, w_o, b_o,
     ln2_g, ln2_b, w_1, b_1, w_2, b_2, lnf_g, lnf_b, wte) = (
        np.asarray(a, np.float64) for a in (
            tok_emb, wpe, ln1_g, ln1_b, w_qkv, b_qkv, w_o, b_o,
            ln2_g, ln2_b, w_1, b_1, w_2, b_2, lnf_g, lnf_b, wte,
        )
    )
    batch, seq, d_model = tok_emb.shape
    heads = d_model // HEAD_DIM
    causal = np.tril(np.ones((seq, seq), dtype=bool))

    def ln(t, g, b):
        m = t.mean(-1, keepdims=True)
        return (t - m) / np.sqrt(((t - m) ** 2).mean(-1, keepdims=True) + EPS) * g + b

    x = tok_emb + wpe
    for i in range(w_qkv.shape[0]):
        q, k, v = np.split(ln(x, ln1_g[i], ln1_b[i]) @ w_qkv[i] + b_qkv[i], 3, axis=-1)
        q, k, v = (t.reshape(batch, seq, heads, HEAD_DIM).transpose(0, 2, 1, 3) for t in (q, k, v))
        s = np.einsum("bhqd,bhkd->bhqk", q, k) / np.sqrt(HEAD_DIM)
        s = np.where(causal, s, -np.inf)
        w = np.exp(s - s.max(-1, keepdims=True))
        w /= w.sum(-1, keepdims=True)
        attn = np.einsum("bhqk,bhkd->bhqd", w, v).transpose(0, 2, 1, 3).reshape(batch, seq, d_model)
        x = x + attn @ w_o[i] + b_o[i]
        h = ln(x, ln2_g[i], ln2_b[i]) @ w_1[i] + b_1[i]
        x = x + (0.5 * h * (1.0 + _erf(h / math.sqrt(2.0)))) @ w_2[i] + b_2[i]
    return ln(x, lnf_g, lnf_b) @ wte.T


def make_inputs(seed=0, size="default"):
    s = SIZES[size]
    d, n_layers, f32 = s["d_model"], s["layers"], np.float32
    if d // s["heads"] != HEAD_DIM:
        raise ValueError(f"d_model / heads must equal HEAD_DIM={HEAD_DIM}")
    rng = np.random.default_rng(seed)
    proj_std = 0.02 / math.sqrt(2 * n_layers)  # GPT-2 scaling for residual projections

    def normal(shape, std):
        return (std * rng.standard_normal(shape, dtype=f32)).astype(f32)

    def around(center, shape):
        return (center + normal(shape, 0.02)).astype(f32)

    wte = normal((s["vocab"], d), 0.02)
    tokens = rng.integers(0, s["vocab"], size=(s["batch"], s["seq"]))
    return (
        wte[tokens], normal((s["seq"], d), 0.01),
        around(1.0, (n_layers, d)), around(0.0, (n_layers, d)),
        normal((n_layers, d, 3 * d), 0.02), around(0.0, (n_layers, 3 * d)),
        normal((n_layers, d, d), proj_std), around(0.0, (n_layers, d)),
        around(1.0, (n_layers, d)), around(0.0, (n_layers, d)),
        normal((n_layers, d, 4 * d), 0.02), around(0.0, (n_layers, 4 * d)),
        normal((n_layers, 4 * d, d), proj_std), around(0.0, (n_layers, d)),
        around(1.0, (d,)), around(0.0, (d,)),
        wte,
    )
