import jax
import jax.numpy as jnp

BLOCK = 128      # size of the masked diagonal tiles
GUARD = 60.0     # max allowed gap between the shift and a true score per row


def _exact(q, k, v):
    # Fallback: exact row max as the softmax shift (always safe).
    seq = q.shape[-2]
    sc = jnp.einsum("bhqd,bhkd->bhqk", q, k)
    row = jax.lax.broadcasted_iota(jnp.int32, sc.shape, 2)
    col = jax.lax.broadcasted_iota(jnp.int32, sc.shape, 3)
    sc = jnp.where(col <= row, sc, -jnp.inf)
    w = jnp.exp(sc - jnp.max(sc, axis=-1, keepdims=True))
    return jnp.einsum("bhqk,bhkd->bhqd", w, v)


def _fast(q, k, v, shift):
    # Softmax is shift invariant, so any per-row shift that avoids overflow /
    # underflow gives the same result; with a shift known up front every tile
    # of the causal triangle is independent and contributions simply add.
    b, h, seq, d = q.shape
    dv = v.shape[-1]
    qs = jnp.concatenate([q, -shift], axis=-1)                    # (b,h,N,d+1)
    ks = jnp.concatenate([k, jnp.ones_like(k[..., :1])], axis=-1)  # q.k - shift in one gemm
    acc = jnp.zeros((b, h, seq, dv), v.dtype)

    # strictly-below-diagonal squares, recursive halving: no mask needed
    n = seq // 2
    while n >= BLOCK:
        g = seq // (2 * n)
        qb = qs.reshape(b, h, g, 2, n, d + 1)[:, :, :, 1]
        kb = ks.reshape(b, h, g, 2, n, d + 1)[:, :, :, 0]
        vb = v.reshape(b, h, g, 2, n, dv)[:, :, :, 0]
        w = jnp.exp(jnp.einsum("bhgqd,bhgkd->bhgqk", qb, kb))
        o = jnp.einsum("bhgqk,bhgkd->bhgqd", w, vb)
        o = jnp.stack([jnp.zeros_like(o), o], axis=3).reshape(b, h, seq, dv)
        acc = acc + o
        n //= 2

    # diagonal tiles with the causal mask
    n = BLOCK
    g = seq // n
    qb = qs.reshape(b, h, g, n, d + 1)
    kb = ks.reshape(b, h, g, n, d + 1)
    vb = v.reshape(b, h, g, n, dv)
    sc = jnp.einsum("bhgqd,bhgkd->bhgqk", qb, kb)
    row = jax.lax.broadcasted_iota(jnp.int32, sc.shape, 3)
    col = jax.lax.broadcasted_iota(jnp.int32, sc.shape, 4)
    w = jnp.where(col <= row, jnp.exp(sc), 0.0)
    o = jnp.einsum("bhgqk,bhgkd->bhgqd", w, vb).reshape(b, h, seq, dv)
    return acc + o


def candidate(q, k, v):
    b, h, seq, d = q.shape
    q = q * (1.0 / jnp.sqrt(jnp.float32(d)))
    v1 = jnp.concatenate([v, jnp.ones_like(v[..., :1])], axis=-1)  # ones column -> denominator
    if seq % BLOCK or (seq // BLOCK) & (seq // BLOCK - 1):
        o = _exact(q, k, v1)
        return o[..., :d] / o[..., d:]
    # Cauchy-Schwarz upper bound on every score of a row: |q_i| * max_j |k_j|
    qn = jnp.sqrt(jnp.sum(q * q, axis=-1, keepdims=True))
    kn = jnp.sqrt(jnp.max(jnp.sum(k * k, axis=-1, keepdims=True), axis=-2, keepdims=True))
    shift = qn * kn
    # the diagonal score is always unmasked; if the bound is within GUARD of it
    # the shifted exponentials can neither overflow nor underflow.
    diag = jnp.sum(q * k, axis=-1, keepdims=True)
    ok = jnp.all(shift - diag <= GUARD)
    o = _fast(q, k, v1, shift)
    o = jax.lax.cond(ok, lambda: o, lambda: _exact(q, k, v1))
    return o[..., :d] / o[..., d:]
