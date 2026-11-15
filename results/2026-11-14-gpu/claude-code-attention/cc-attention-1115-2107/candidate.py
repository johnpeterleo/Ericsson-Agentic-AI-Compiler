import jax
import jax.numpy as jnp

# Number of query chunks. Each chunk needs two gemm shapes, and XLA spends ~2 s
# compiling every distinct gemm shape, so this is kept small.
NCHUNK = 8


def candidate(q, k, v):
    b, h, seq, d = q.shape
    q = q * (1.0 / jnp.sqrt(jnp.float32(d)))

    # Softmax is invariant to a per-row shift, so the exact row max (a full extra
    # pass over the quadratic score matrix) is not needed for stability: any shift
    # between a lower and an upper bound of the row max is exact as well.
    #   upper bound: |q_i| * max_{j<=i} |k_j|   (Cauchy-Schwarz)
    #   lower bound: q_i . k_i                  (the diagonal is never masked)
    qn = jnp.sqrt(jnp.sum(q * q, -1))
    kn = jax.lax.cummax(jnp.sqrt(jnp.sum(k * k, -1)), axis=2)
    m = (0.5 * (qn * kn + jnp.sum(q * k, -1)))[..., None]

    n = min(NCHUNK, seq)
    bounds = [(seq * (i + 1)) // n for i in range(n)]
    outs = []
    lo = 0
    for hi in bounds:
        c = hi - lo
        # Queries in [lo, hi) only attend to keys [0, hi): the fully masked
        # blocks to the right are never computed.
        s = jnp.einsum("bhqd,bhkd->bhqk", q[:, :, lo:hi], k[:, :, :hi])
        mask = jnp.arange(hi)[None, :] <= jnp.arange(lo, hi)[:, None]
        w = jnp.where(mask, jnp.exp(s - m[:, :, lo:hi]), 0.0)
        num = jnp.einsum("bhqk,bhkd->bhqd", w, v[:, :, :hi])
        outs.append(num / jnp.sum(w, axis=-1, keepdims=True))
        lo = hi
    return jnp.concatenate(outs, axis=2)
