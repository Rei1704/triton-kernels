import pytest
import torch
from kernels.attention import flash_attention, naive_attention

# (batch, heads, N, head_dim)
SHAPES = [
    (1, 1, 1, 64),      # a single token
    (1, 2, 1000, 64),   # N not a multiple of any block size
    (2, 4, 128, 64),    # multiple batches and heads
    (1, 1, 4097, 64),   # one past a power of two
    (1, 2, 333, 32),    # smaller head dimension
]


@pytest.mark.parametrize("shape", SHAPES)
def test_matches_naive(shape):
    torch.manual_seed(0)
    q = torch.randn(*shape, device="cuda")
    k = torch.randn(*shape, device="cuda")
    v = torch.randn(*shape, device="cuda")
    torch.testing.assert_close(flash_attention(q, k, v), naive_attention(q, k, v),
                               atol=1e-4, rtol=1e-4)


def test_large_scores_stay_finite():
    # large inputs give large scores; without the running max trick exp would overflow
    torch.manual_seed(0)
    q = torch.randn(1, 1, 256, 64, device="cuda") * 20
    k = torch.randn(1, 1, 256, 64, device="cuda") * 20
    v = torch.randn(1, 1, 256, 64, device="cuda")
    out = flash_attention(q, k, v)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, naive_attention(q, k, v), atol=1e-3, rtol=1e-3)
