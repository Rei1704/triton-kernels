import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


def naive_attention(q, k, v):
    """Reference attention that materializes the full N x N score matrix."""
    scale = q.shape[-1] ** -0.5
    s = (q @ k.transpose(-2, -1)) * scale
    p = torch.softmax(s, dim=-1)
    return p @ v


@triton.jit
def flash_attn_kernel(Q, K, V, O, stride_h, stride_n, N, qk_scale,
                      HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    head = tl.program_id(1)
    base = head * stride_h

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # this program's block of queries, loaded once and reused for every key block
    q = tl.load(Q + base + offs_m[:, None] * stride_n + offs_d[None, :],
                mask=offs_m[:, None] < N, other=0.0)

    m = tl.full([BLOCK_M], float("-inf"), tl.float32)   # running max (log2 domain)
    l = tl.zeros([BLOCK_M], tl.float32)                 # running sum of weights
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)     # running weighted sum of V

    for start_n in range(0, N, BLOCK_N):
        cols = start_n + offs_n
        # K loaded transposed as (HEAD_DIM, BLOCK_N) so q @ k gives scores directly
        k = tl.load(K + base + cols[None, :] * stride_n + offs_d[:, None],
                    mask=cols[None, :] < N, other=0.0)
        s = tl.dot(q, k) * qk_scale
        s = tl.where(cols[None, :] < N, s, float("-inf"))

        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp2(m - m_new)
        p = tl.exp2(s - m_new[:, None])
        l = l * alpha + tl.sum(p, axis=1)

        v = tl.load(V + base + cols[:, None] * stride_n + offs_d[None, :],
                    mask=cols[:, None] < N, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m = m_new

    out = acc / l[:, None]
    tl.store(O + base + offs_m[:, None] * stride_n + offs_d[None, :], out,
             mask=offs_m[:, None] < N)


def flash_attention(q, k, v, BLOCK_M=32, BLOCK_N=128, num_warps=4, num_stages=1):
    """FlashAttention forward pass. q, k, v: (batch, heads, N, head_dim)."""
    assert q.shape == k.shape == v.shape and q.ndim == 4
    B, H, N, d = q.shape
    q3 = q.reshape(B * H, N, d).contiguous()
    k3 = k.reshape(B * H, N, d).contiguous()
    v3 = v.reshape(B * H, N, d).contiguous()
    o = torch.empty_like(q3)
    qk_scale = (d ** -0.5) * LOG2E
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    flash_attn_kernel[grid](q3, k3, v3, o, q3.stride(0), q3.stride(1), N, qk_scale,
                            HEAD_DIM=d, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                            num_warps=num_warps, num_stages=num_stages)
    return o.reshape(B, H, N, d)
