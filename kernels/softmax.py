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
