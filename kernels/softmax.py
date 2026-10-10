import torch
import triton
import triton.language as tl


def naive_softmax(x):
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    num = torch.exp(z)
    return num / num.sum(dim=1)[:, None]


@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_stride, out_stride, n_cols,
                   BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(in_ptr + row * in_stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)

    tl.store(out_ptr + row * out_stride + cols, y, mask=mask)


def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 1 if BLOCK_SIZE <= 128 else min(16, BLOCK_SIZE // 64)
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_cols,
                              BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return y



@triton.jit
def online_softmax_kernel(out_ptr, in_ptr, in_stride, out_stride, n_cols,
                          BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    in_row = in_ptr + row * in_stride
    out_row = out_ptr + row * out_stride

    # pass 1: per lane running max and running sum
    m = tl.full([BLOCK_SIZE], -1e30, tl.float32)
    d = tl.zeros([BLOCK_SIZE], tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        x = tl.load(in_row + cols, mask=cols < n_cols, other=-float("inf"))
        m_new = tl.maximum(m, x)
        d = d * tl.exp(m - m_new) + tl.exp(x - m_new)
        m = m_new

    # combine lanes into one max and one sum for the row
    m_row = tl.max(m, axis=0)
    d_row = tl.sum(d * tl.exp(m - m_row), axis=0)

    # pass 2: read again, normalize, write
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(in_row + cols, mask=mask, other=-float("inf"))
        tl.store(out_row + cols, tl.exp(x - m_row) / d_row, mask=mask)


def online_softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 1024)
    y = torch.empty_like(x)
    online_softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_cols,
                                     BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return y


def softmax_auto(x):
    """Fused single pass kernel when the row fits, online two pass kernel otherwise."""
    if x.shape[1] <= 32768:
        return softmax(x)
    return online_softmax(x)




    
