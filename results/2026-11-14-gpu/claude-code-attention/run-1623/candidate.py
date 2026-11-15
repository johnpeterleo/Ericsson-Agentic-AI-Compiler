import jax
import jax.numpy as jnp

NB = 4  # query blocks; block i only reads keys [0, end of block i)


def candidate(q, k, v):
    seq, head_dim = q.shape[-2], q.shape[-1]
    q = q * (1.0 / jnp.sqrt(jnp.float32(head_dim)))
    # extra ones column: the second matmul then also yields the row sums
    v1 = jnp.concatenate([v, jnp.ones(v.shape[:-1] + (1,), v.dtype)], axis=-1)
    kt = jnp.swapaxes(k, -1, -2)
    nb = NB if seq % NB == 0 and seq >= NB else 1
    bs = seq // nb
    outs = []
    for i in range(nb):
        lo, hi = i * bs, (i + 1) * bs
        s = jnp.matmul(q[..., lo:hi, :], kt[..., :hi])
        mask = jnp.arange(hi)[None, :] <= jnp.arange(lo, hi)[:, None]
        s = jnp.where(mask, s, -jnp.inf)
        w = jnp.exp(s - jnp.max(s, axis=-1, keepdims=True))
        o = jnp.matmul(w, v1[..., :hi, :])
        outs.append(o[..., :head_dim] / o[..., head_dim:])
    return jnp.concatenate(outs, axis=-2)
