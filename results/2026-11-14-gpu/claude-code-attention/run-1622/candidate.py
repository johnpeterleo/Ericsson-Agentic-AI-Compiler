import jax
import jax.numpy as jnp

_TILE = 512  # query rows per tile; more tiles run faster but compile too slowly


def candidate(q, k, v):
    seq, d = q.shape[-2], q.shape[-1]
    q = q * jnp.float32(1.0 / (d ** 0.5))  # scale the small input, not the scores
    outs = []
    for s in range(0, seq, _TILE):
        e = min(s + _TILE, seq)
        # causal mask: rows [s, e) only see keys [0, e), so skip the rest entirely
        sc = jnp.einsum("bhqd,bhkd->bhqk", q[:, :, s:e], k[:, :, :e])
        mask = jnp.arange(e)[None, :] <= jnp.arange(s, e)[:, None]
        sc = jnp.where(mask, sc, -jnp.inf)
        sc = sc - jnp.max(sc, axis=-1, keepdims=True)
        w = jnp.exp(sc)
        o = jnp.einsum("bhqk,bhkd->bhqd", w, v[:, :, :e])
        # normalise the small output instead of the big weight matrix
        outs.append(o / jnp.sum(w, axis=-1, keepdims=True))
    return jnp.concatenate(outs, axis=2)
