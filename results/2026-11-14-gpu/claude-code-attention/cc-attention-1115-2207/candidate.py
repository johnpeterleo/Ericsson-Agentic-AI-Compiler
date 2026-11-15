import jax
import jax.numpy as jnp

_BOUNDS = (0, 512, 768, 1024, 1280, 1536, 1792, 2048)  # in units of seq/2048

def candidate(q, k, v):
    seq, d = q.shape[-2], q.shape[-1]
    q = q * (1.0 / jnp.sqrt(jnp.float32(d)))
    v1 = jnp.concatenate([v, jnp.ones(v.shape[:-1] + (1,), v.dtype)], axis=-1)
    outs = []
    # Few blocks on purpose: every distinct matmul shape costs seconds of XLA autotuning at compile time.
    bs = sorted({(b * seq) // 2048 for b in _BOUNDS})
    for s, e in zip(bs[:-1], bs[1:]):
        qb = q[..., s:e, :]
        sc = jnp.einsum("bhqd,bhkd->bhqk", qb, k[..., :e, :])
        mask = jnp.arange(e)[None, :] <= jnp.arange(s, e)[:, None]
        sc = jnp.where(mask, sc, -jnp.inf)
        w = jnp.exp(sc - jnp.max(sc, axis=-1, keepdims=True))
        o = jnp.einsum("bhqk,bhkd->bhqd", w, v1[..., :e, :])
        outs.append(o[..., :d] / o[..., d:])
    return jnp.concatenate(outs, axis=-2)
