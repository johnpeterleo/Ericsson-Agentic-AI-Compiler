"""A benchmark suite of small JAX programs that have a known faster rewrite.

Every program is a pair. The `reference` is a natural way to write the
computation. The `expert` is the fastest version we know that returns the same
values. The rewrites change the algorithm (for example the order of matrix
products), which XLA does not do by itself; the measured expert speedup shows
how much it leaves unused.

The agent is shown the source of `reference` and nothing else from this file.
So a reference must only use `jax`, `jnp` and its own arguments, and its name
and body must not give the rewrite away.

Inputs that get summed are positive, so the terms cannot cancel. Two correct
programs then agree to rounding error whatever order they add in.

Pairwise distances (broadcast difference versus one matrix product) was left
out: on an NVIDIA L4, XLA's autotuner took over 15 minutes to compile the
broadcast version for 1024 points.
"""

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

FEATURES = 64
POLYNOMIAL_TERMS = 24
EMBEDDING_WIDTH = 128
WINDOW = 64
LOW_RANK = 16
VECTOR_COUNT = 256


@dataclass(frozen=True)
class BenchmarkProgram:
    """One reference program, its expert rewrite, and how to build inputs for them.

    `make_inputs(rng, size)` returns one tuple of inputs. `size` is the
    program's main dimension: large when timing, small in the unit tests.
    `rewrite` says what the expert changed. It is for result tables only.
    """

    reference: Callable
    expert: Callable
    rewrite: str
    make_inputs: Callable[[np.random.Generator, int], tuple]
    benchmark_size: int
    test_size: int

    @property
    def name(self) -> str:
        return self.reference.__name__


def _normal(rng, *shape):
    return rng.standard_normal(shape, dtype=np.float32)


def _positive(rng, *shape):
    return rng.uniform(0.5, 1.5, shape).astype(np.float32)


def matrix_chain_times_vector(first, second, third, vector):
    return first @ second @ third @ vector


def matrix_chain_times_vector_expert(first, second, third, vector):
    return first @ (second @ (third @ vector))


def _matrix_chain_times_vector_inputs(rng, size):
    matrices = [_positive(rng, size, size) / size for _ in range(3)]
    return *matrices, _positive(rng, size)


def scale_rows_and_columns(row_scale, matrix, column_scale):
    return jnp.diag(row_scale) @ matrix @ jnp.diag(column_scale)


def scale_rows_and_columns_expert(row_scale, matrix, column_scale):
    return row_scale[:, None] * matrix * column_scale[None, :]


def _scale_rows_and_columns_inputs(rng, size):
    return _normal(rng, size), _normal(rng, size, size), _normal(rng, size)


def trace_of_product(left, right):
    return jnp.trace(left @ right)


def trace_of_product_expert(left, right):
    return jnp.sum(left * right.T)


def _trace_of_product_inputs(rng, size):
    return _positive(rng, size, size), _positive(rng, size, size)


def kronecker_product_times_vector(left, right, vector):
    return jnp.kron(left, right) @ vector


def kronecker_product_times_vector_expert(left, right, vector):
    as_matrix = vector.reshape(left.shape[1], right.shape[1])
    return (left @ as_matrix @ right.T).reshape(-1)


def _kronecker_product_times_vector_inputs(rng, size):
    left, right = _positive(rng, size, size) / size, _positive(rng, size, size) / size
    return left, right, _positive(rng, size * size)


def evaluate_polynomial(coefficients, x):
    exponents = jnp.arange(coefficients.shape[0])
    terms = coefficients[:, None] * x[None, :] ** exponents[:, None]
    return jnp.sum(terms, axis=0)


def evaluate_polynomial_expert(coefficients, x):
    result = jnp.zeros_like(x)
    for coefficient in coefficients[::-1]:
        result = result * x + coefficient
    return result


def _evaluate_polynomial_inputs(rng, size):
    return _positive(rng, POLYNOMIAL_TERMS), rng.random(size, dtype=np.float32)


def attention(queries, keys, values):
    def attend(query):
        scores = keys @ query / jnp.sqrt(query.shape[0])
        return jax.nn.softmax(scores) @ values

    return jax.lax.map(attend, queries)


def attention_expert(queries, keys, values):
    scores = queries @ keys.T / jnp.sqrt(queries.shape[-1])
    return jax.nn.softmax(scores, axis=-1) @ values


def _attention_inputs(rng, size):
    queries, keys = _normal(rng, size, FEATURES), _normal(rng, size, FEATURES)
    return queries, keys, _positive(rng, size, FEATURES)


def matrix_power(transition):
    power = 16
    result = transition
    for _ in range(power - 1):
        result = result @ transition
    return result


def matrix_power_expert(transition):
    result = transition
    for _ in range(4):  # Squaring four times gives the 16th power.
        result = result @ result
    return result


def _matrix_power_inputs(rng, size):
    # Rows that sum to 1, so the powers neither grow nor vanish.
    weights = _positive(rng, size, size)
    return (weights / weights.sum(axis=1, keepdims=True),)


def embed_tokens(table, token_ids):
    one_hot = jax.nn.one_hot(token_ids, table.shape[0], dtype=table.dtype)
    return one_hot @ table


def embed_tokens_expert(table, token_ids):
    return table[token_ids]


def _embed_tokens_inputs(rng, size):
    token_ids = rng.integers(0, size, size // 2, dtype=np.int32)
    return _normal(rng, size, EMBEDDING_WIDTH), token_ids


def prefix_sums(sequences):
    length = sequences.shape[-1]
    includes = jnp.triu(jnp.ones((length, length), dtype=sequences.dtype))
    return sequences @ includes


def prefix_sums_expert(sequences):
    return jnp.cumsum(sequences, axis=-1)


def _prefix_sums_inputs(rng, size):
    return (_positive(rng, size, size),)


# The programs below are harder: a correctness trap, a hidden structure, and
# two that need two rewrites each.


def moving_window_sums(sequences):
    window = 64
    positions = jnp.arange(sequences.shape[-1])
    distance = positions[None, :] - positions[:, None]
    in_window = (distance >= 0) & (distance < window)
    return sequences @ in_window.astype(sequences.dtype)


def moving_window_sums_expert(sequences):
    # Sums each window directly. Differences of a running total are faster to
    # write, but on long rows they lose too much float32 precision to pass.
    return jax.lax.reduce_window(
        sequences, 0.0, jax.lax.add, (1, WINDOW), (1, 1), ((0, 0), (WINDOW - 1, 0))
    )


def _moving_window_sums_inputs(rng, size):
    return (_positive(rng, size // 4, size),)


def circulant_times_vectors(kernel, vectors):
    length = kernel.shape[0]
    positions = jnp.arange(length)
    matrix = kernel[(positions[:, None] - positions[None, :]) % length]
    return vectors @ matrix.T


def circulant_times_vectors_expert(kernel, vectors):
    length = kernel.shape[0]
    spectrum = jnp.fft.rfft(vectors, axis=-1) * jnp.fft.rfft(kernel)
    return jnp.fft.irfft(spectrum, n=length, axis=-1)


def _circulant_times_vectors_inputs(rng, size):
    return _positive(rng, size), _positive(rng, size, size)


def low_rank_plus_diagonal_times_vectors(left, right, diagonal, vectors):
    matrix = left @ right.T + jnp.diag(diagonal)
    return matrix @ vectors


def low_rank_plus_diagonal_times_vectors_expert(left, right, diagonal, vectors):
    return left @ (right.T @ vectors) + diagonal[:, None] * vectors


def _low_rank_plus_diagonal_times_vectors_inputs(rng, size):
    return (
        _positive(rng, size, LOW_RANK),
        _positive(rng, size, LOW_RANK),
        _positive(rng, size),
        _positive(rng, size, VECTOR_COUNT),
    )


def matrix_polynomial_times_vector(matrix, vector):
    square = matrix @ matrix
    cube = square @ matrix
    return (cube + 2.0 * square + 3.0 * matrix) @ vector


def matrix_polynomial_times_vector_expert(matrix, vector):
    # Horner's rule: A(A(Av + 2v) + 3v) = (A^3 + 2A^2 + 3A)v.
    return matrix @ (matrix @ (matrix @ vector + 2.0 * vector) + 3.0 * vector)


def _matrix_polynomial_times_vector_inputs(rng, size):
    weights = _positive(rng, size, size)
    return weights / weights.sum(axis=1, keepdims=True), _positive(rng, size)


# Benchmark sizes were chosen on an NVIDIA L4 (results/2026-11-10-gpu/ and
# results/2026-11-11-gpu/): the smallest size tried at which the expert takes at
# least 0.7 ms, five times the GPU's fixed cost per call (about 0.14 ms). Below
# that, the measured speedup mostly reflects that fixed cost. The experts of
# kronecker_product_times_vector, embed_tokens and
# low_rank_plus_diagonal_times_vectors stay near 0.2 ms at every size tried,
# so those use the largest size tried.
PROGRAMS = (
    BenchmarkProgram(
        reference=matrix_chain_times_vector,
        expert=matrix_chain_times_vector_expert,
        rewrite="multiply right to left: only matrix-vector products",
        make_inputs=_matrix_chain_times_vector_inputs,
        benchmark_size=4096,
        test_size=6,
    ),
    BenchmarkProgram(
        reference=scale_rows_and_columns,
        expert=scale_rows_and_columns_expert,
        rewrite="broadcast multiply, no products with diagonal matrices",
        make_inputs=_scale_rows_and_columns_inputs,
        benchmark_size=4096,
        test_size=6,
    ),
    BenchmarkProgram(
        reference=trace_of_product,
        expert=trace_of_product_expert,
        rewrite="sum of an elementwise product, no matrix product",
        make_inputs=_trace_of_product_inputs,
        benchmark_size=4096,
        test_size=6,
    ),
    BenchmarkProgram(
        reference=kronecker_product_times_vector,
        expert=kronecker_product_times_vector_expert,
        rewrite="two small matrix products, no Kronecker product",
        make_inputs=_kronecker_product_times_vector_inputs,
        benchmark_size=192,
        test_size=4,
    ),
    BenchmarkProgram(
        reference=evaluate_polynomial,
        expert=evaluate_polynomial_expert,
        rewrite="Horner's rule, no powers",
        make_inputs=_evaluate_polynomial_inputs,
        benchmark_size=16_000_000,
        test_size=50,
    ),
    BenchmarkProgram(
        reference=attention,
        expert=attention_expert,
        rewrite="one batched matrix product, no loop over the queries",
        make_inputs=_attention_inputs,
        benchmark_size=4096,
        test_size=6,
    ),
    BenchmarkProgram(
        reference=matrix_power,
        expert=matrix_power_expert,
        rewrite="repeated squaring: 4 matrix products, not 15",
        make_inputs=_matrix_power_inputs,
        benchmark_size=1024,
        test_size=6,
    ),
    BenchmarkProgram(
        reference=embed_tokens,
        expert=embed_tokens_expert,
        rewrite="index the table, no one-hot matrix product",
        make_inputs=_embed_tokens_inputs,
        benchmark_size=32768,
        test_size=10,
    ),
    BenchmarkProgram(
        reference=prefix_sums,
        expert=prefix_sums_expert,
        rewrite="cumulative sum, no product with a triangular matrix",
        make_inputs=_prefix_sums_inputs,
        benchmark_size=4096,
        test_size=6,
    ),
    BenchmarkProgram(
        reference=moving_window_sums,
        expert=moving_window_sums_expert,
        rewrite="sum each window directly, no product with a banded matrix",
        make_inputs=_moving_window_sums_inputs,
        benchmark_size=8192,
        test_size=128,
    ),
    BenchmarkProgram(
        reference=circulant_times_vectors,
        expert=circulant_times_vectors_expert,
        rewrite="the matrix is circulant: multiply with FFTs",
        make_inputs=_circulant_times_vectors_inputs,
        benchmark_size=4096,
        test_size=8,
    ),
    BenchmarkProgram(
        reference=low_rank_plus_diagonal_times_vectors,
        expert=low_rank_plus_diagonal_times_vectors_expert,
        rewrite="never build the matrix; multiply the thin factors first",
        make_inputs=_low_rank_plus_diagonal_times_vectors_inputs,
        benchmark_size=8192,
        test_size=8,
    ),
    BenchmarkProgram(
        reference=matrix_polynomial_times_vector,
        expert=matrix_polynomial_times_vector_expert,
        rewrite="Horner's rule with matrix-vector products only",
        make_inputs=_matrix_polynomial_times_vector_inputs,
        benchmark_size=4096,
        test_size=8,
    ),
)

PROGRAMS_BY_NAME = {program.name: program for program in PROGRAMS}
