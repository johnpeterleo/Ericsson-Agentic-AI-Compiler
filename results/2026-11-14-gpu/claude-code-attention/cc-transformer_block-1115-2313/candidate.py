import jax
import jax.numpy as jnp

EPS = 1e-5
HEAD_DIM = 64
NCHUNK = 4


def _layer_norm(x, gamma, beta):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    xc = x - mean
    var = jnp.mean(xc * xc, axis=-1, keepdims=True)
    return xc * jax.lax.rsqrt(var + EPS) * gamma + beta


def _bounds(seq, n):
    n = max(1, min(n, seq))
    return [(i * seq // n, (i + 1) * seq // n) for i in range(n)]


def candidate(x, ln1_g, ln1_b, w_qkv, w_o, ln2_g, ln2_b, w_1, w_2):
    batch, seq, d_model = x.shape
    head_dim = HEAD_DIM
    heads = d_model // head_dim

    h = _layer_norm(x, ln1_g, ln1_b)
    qkv = h @ w_qkv
    qkv = qkv.reshape(batch, seq, 3, heads, head_dim)
    q = qkv[:, :, 0].transpose(0, 2, 1, 3) * jnp.float32(1.0 / (head_dim ** 0.5))
    k = qkv[:, :, 1].transpose(0, 2, 3, 1)  # b h d s
    v = qkv[:, :, 2].transpose(0, 2, 1, 3)

    outs = []
    # causal blocking: queries in [lo, hi) only ever see keys in [0, hi)
    for lo, hi in _bounds(seq, NCHUNK):
        s = q[:, :, lo:hi] @ k[:, :, :, :hi]  # b h c hi
        mask = jnp.arange(hi)[None, :] <= jnp.arange(lo, hi)[:, None]
        s = jnp.where(mask, s, -jnp.inf)
        m = jnp.max(s, axis=-1, keepdims=True)
        e = jnp.exp(s - m)
        w = e / jnp.sum(e, axis=-1, keepdims=True)
        outs.append(w @ v[:, :, :hi])
    attn = jnp.concatenate(outs, axis=2).transpose(0, 2, 1, 3).reshape(batch, seq, d_model)
    x = x + attn @ w_o

    h = _layer_norm(x, ln2_g, ln2_b)
    return x + jax.nn.gelu(h @ w_1, approximate=False) @ w_2
