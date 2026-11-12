"""Realistic programs with several planted inefficiencies, and their clean versions.

The suite in `programs.py` has one known trick per program, and a single LLM
call finds it. Each program here hides two or three problems, and the clean
version is the expert, so the headroom is known. On the NVIDIA L4 no single
fix gets far, but all of them together do (2026-11-12, matmul precision
highest, benchmark sizes, speedup over the reference):

    program                 expert   one fix alone
    causal_self_attention    5.6x    sort 1.6x, scale 1.4x, row sums 1.3x
    embed_and_transform     85x      lax.map 2.9x, solve 1.4x, one-hot 1.0x
    local_standardize       75x      loop 6.1x, band matrix 3.2x, sort 1.2x

Some problems were tried and dropped because they cost nothing on the L4: a
Python loop over attention heads (it was faster than one batched product, so
the attention expert keeps it), and a mask built with a matrix product.

The same rules as the suite apply: the agent sees only the source of
`reference`, which uses only `jax`, `jnp` and its arguments, and its name does
not give the rewrite away. `BenchmarkProgram` is the suite's class.
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
    future = jnp.triu(jnp.ones((seq, seq), dtype=bool), k=1)
    scale = jnp.eye(seq, dtype=x.dtype) / jnp.sqrt(jnp.float32(head_width))
    head_outputs = []
    for head in range(heads):
        part = slice(head * head_width, (head + 1) * head_width)
        scores = scale @ (q[..., part] @ jnp.swapaxes(k[..., part], -1, -2))
        scores = jnp.where(future, -jnp.inf, scores)
        largest = jnp.sort(scores, axis=-1)[..., -1:]
        weights = jnp.exp(scores - largest)
        weights = weights / (weights @ jnp.ones((seq, seq), x.dtype))
        head_outputs.append(weights @ v[..., part])
    return jnp.concatenate(head_outputs, axis=-1) @ w_out


def causal_self_attention_expert(x, w_qkv, w_out):
    # On the L4 a loop over heads was faster than one batched product, so the expert keeps it.
    batch, seq, width = x.shape
    heads = 8
    head_width = width // heads
    q, k, v = jnp.split(x @ w_qkv, 3, axis=-1)
    future = jnp.triu(jnp.ones((seq, seq), dtype=bool), k=1)
    head_outputs = []
    for head in range(heads):
        part = slice(head * head_width, (head + 1) * head_width)
        scores = q[..., part] @ jnp.swapaxes(k[..., part], -1, -2) / jnp.sqrt(jnp.float32(head_width))
        scores = jnp.where(future, -jnp.inf, scores)
        weights = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))
        weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
        head_outputs.append(weights @ v[..., part])
    return jnp.concatenate(head_outputs, axis=-1) @ w_out


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
    rows = []
    for signal in signals:
        local_mean = signal @ band / counts
        local_square = (signal**2) @ band / counts
        local_std = jnp.sqrt(jnp.maximum(local_square - local_mean**2, 0.0) + 1e-6)
        standardized = (signal - local_mean) / local_std
        peak = jnp.sort(jnp.abs(standardized))[-1]
        rows.append(standardized / peak)
    return jnp.stack(rows)


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
        rewrite="a max instead of a sort, a division instead of a product with a scaled identity, "
        "row sums instead of a product with ones",
        make_inputs=_causal_self_attention_inputs,
        # At 1024 the reference took 86 s in the compile check, over the 60 s limit; 41 s at 512.
        benchmark_size=512,
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
        rewrite="all signals at once, window sums with reduce_window instead of band-matrix products, "
        "a max instead of a sort",
        make_inputs=_local_standardize_inputs,
        benchmark_size=8192,
        test_size=40,
    ),
)

PLANTED_BY_NAME = {program.name: program for program in PLANTED_PROGRAMS}
