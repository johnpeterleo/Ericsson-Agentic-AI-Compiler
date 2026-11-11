"""Realistic programs with several planted inefficiencies, and their clean versions.

The suite in `programs.py` has one known trick per program, and a single LLM
call finds it. These programs each hide three problems of different
visibility: an obvious one (a Python loop over heads), one that looks like
ordinary code but costs a full matrix product (a mask or an embedding built
with one), and one that only shows up when the parts are timed (a sort used
to find a maximum, a sequential map, a linear solve for a division). The clean
version is the expert, so the headroom is known.

The same rules as the suite apply: the agent sees only the source of
`reference`, which uses only `jax`, `jnp` and its arguments, and its name does
not give the rewrite away. `BenchmarkProgram` is the suite's class.

The sizes are a first guess for the NVIDIA L4 and have not been measured there.
"""

import jax
import jax.numpy as jnp
import numpy as np

from src.benchmarks.programs import BenchmarkProgram

ATTENTION_BATCH = 4
ATTENTION_WIDTH = 256
VOCABULARY = 8192
TOKEN_WIDTH = 256
TOKEN_HIDDEN = 1024
SIGNAL_COUNT = 64
SIGNAL_WINDOW = 33


def causal_self_attention(x, w_qkv, w_out):
    batch, seq, width = x.shape
    heads = 8
    head_width = width // heads
    q, k, v = jnp.split(x @ w_qkv, 3, axis=-1)
    future = jnp.eye(seq, dtype=x.dtype) @ jnp.triu(jnp.ones((seq, seq), x.dtype), k=1) > 0.5
    head_outputs = []
    for head in range(heads):
        part = slice(head * head_width, (head + 1) * head_width)
        scores = q[..., part] @ jnp.swapaxes(k[..., part], -1, -2) / jnp.sqrt(jnp.float32(head_width))
        scores = jnp.where(future, -jnp.inf, scores)
        largest = jnp.sort(scores, axis=-1)[..., -1:]
        weights = jnp.exp(scores - largest)
        weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
        head_outputs.append(weights @ v[..., part])
    return jnp.concatenate(head_outputs, axis=-1) @ w_out


def causal_self_attention_expert(x, w_qkv, w_out):
    batch, seq, width = x.shape
    heads = 8
    head_width = width // heads
    q, k, v = (
        part.reshape(batch, seq, heads, head_width) for part in jnp.split(x @ w_qkv, 3, axis=-1)
    )
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(jnp.float32(head_width))
    scores = jnp.where(jnp.tril(jnp.ones((seq, seq), dtype=bool)), scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, v).reshape(batch, seq, width) @ w_out


def _causal_self_attention_inputs(rng, size):
    x = rng.standard_normal((ATTENTION_BATCH, size, ATTENTION_WIDTH), dtype=np.float32)

    def weight(columns):
        scale = np.float32(1 / np.sqrt(ATTENTION_WIDTH))
        return rng.standard_normal((ATTENTION_WIDTH, columns), dtype=np.float32) * scale

    return x, weight(3 * ATTENTION_WIDTH), weight(ATTENTION_WIDTH)


def embed_and_transform(token_ids, table, gamma, beta, w_in, w_out):
    vocabulary = table.shape[0]
    tokens = jax.nn.one_hot(token_ids, vocabulary, dtype=table.dtype) @ table
    centered = tokens - jnp.mean(tokens, axis=-1, keepdims=True)
    spread = jnp.sqrt(jnp.mean(centered**2, axis=-1) + 1e-5)
    normalized = jnp.linalg.solve(jnp.diag(spread), centered) * gamma + beta
    hidden = jax.lax.map(lambda row: jax.nn.gelu(row @ w_in) @ w_out, normalized)
    return tokens + hidden


def embed_and_transform_expert(token_ids, table, gamma, beta, w_in, w_out):
    vocabulary = table.shape[0]
    # A one-hot row is all zeros for an id outside the table; a plain gather would wrap or clip it.
    in_table = (token_ids >= 0) & (token_ids < vocabulary)
    tokens = jnp.where(in_table[:, None], table[jnp.clip(token_ids, 0, vocabulary - 1)], 0.0)
    centered = tokens - jnp.mean(tokens, axis=-1, keepdims=True)
    spread = jnp.sqrt(jnp.mean(centered**2, axis=-1, keepdims=True) + 1e-5)
    normalized = centered / spread * gamma + beta
    return tokens + jax.nn.gelu(normalized @ w_in) @ w_out


def _embed_and_transform_inputs(rng, size):
    def weight(rows, columns):
        return rng.standard_normal((rows, columns), dtype=np.float32) / np.float32(np.sqrt(rows))

    token_ids = rng.integers(0, VOCABULARY, size, dtype=np.int32)
    table = rng.standard_normal((VOCABULARY, TOKEN_WIDTH), dtype=np.float32)
    gamma = rng.uniform(0.5, 1.5, TOKEN_WIDTH).astype(np.float32)
    beta = (0.1 * rng.standard_normal(TOKEN_WIDTH)).astype(np.float32)
    return (
        token_ids, table, gamma, beta,
        weight(TOKEN_WIDTH, TOKEN_HIDDEN), weight(TOKEN_HIDDEN, TOKEN_WIDTH),
    )


def local_standardize(signals):
    length = signals.shape[-1]
    window = 33
    index = jnp.arange(length)
    band = (jnp.abs(index[:, None] - index[None, :]) <= window // 2).astype(signals.dtype)
    counts = jnp.sum(band, axis=0)
    local_mean = signals @ band / counts
    local_square = (signals**2) @ band / counts
    local_std = jnp.sqrt(jnp.maximum(local_square - local_mean**2, 0.0) + 1e-6)
    standardized = (signals - local_mean) / local_std
    peak = jnp.sort(jnp.abs(standardized), axis=-1)[..., -1:]
    return standardized / peak


def _window_sums(values, window):
    half = window // 2
    return jax.lax.reduce_window(
        values, 0.0, jax.lax.add, (1, window), (1, 1), ((0, 0), (half, half))
    )


def local_standardize_expert(signals):
    counts = _window_sums(jnp.ones_like(signals[:1]), SIGNAL_WINDOW)
    local_mean = _window_sums(signals, SIGNAL_WINDOW) / counts
    local_square = _window_sums(signals**2, SIGNAL_WINDOW) / counts
    local_std = jnp.sqrt(jnp.maximum(local_square - local_mean**2, 0.0) + 1e-6)
    standardized = (signals - local_mean) / local_std
    return standardized / jnp.max(jnp.abs(standardized), axis=-1, keepdims=True)


def _local_standardize_inputs(rng, size):
    # Positive signals with a slow trend, so local statistics differ along the row.
    trend = np.linspace(1.0, 3.0, size, dtype=np.float32)
    noise = rng.uniform(0.5, 1.5, (SIGNAL_COUNT, size)).astype(np.float32)
    return (trend * noise,)


PLANTED_PROGRAMS = (
    BenchmarkProgram(
        reference=causal_self_attention,
        expert=causal_self_attention_expert,
        rewrite="batched heads, a triangular mask, a max reduction instead of a sort",
        make_inputs=_causal_self_attention_inputs,
        benchmark_size=1024,
        test_size=8,
    ),
    BenchmarkProgram(
        reference=embed_and_transform,
        expert=embed_and_transform_expert,
        rewrite="a gather instead of a one-hot product, a division instead of a solve, one batched MLP",
        make_inputs=_embed_and_transform_inputs,
        benchmark_size=4096,
        test_size=8,
    ),
    BenchmarkProgram(
        reference=local_standardize,
        expert=local_standardize_expert,
        rewrite="window sums with reduce_window instead of band-matrix products, a max instead of a sort",
        make_inputs=_local_standardize_inputs,
        benchmark_size=8192,
        test_size=40,
    ),
)

PLANTED_BY_NAME = {program.name: program for program in PLANTED_PROGRAMS}
