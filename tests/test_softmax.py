import pytest
import torch
from kernels.softmax import softmax, online_softmax, softmax_auto

SHAPES = [(1, 1), (7, 13), (1823, 781), (128, 1024), (64, 4097), (16, 32768)]
WIDE_SHAPES = [(4, 65536), (2, 131072), (3, 100003)]


@pytest.mark.parametrize("shape", SHAPES)
def test_fused_matches_torch(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda")
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=1), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", SHAPES + WIDE_SHAPES)
def test_online_matches_torch(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda")
    torch.testing.assert_close(online_softmax(x), torch.softmax(x, dim=1), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", SHAPES + WIDE_SHAPES)
def test_auto_matches_torch(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda")
    torch.testing.assert_close(softmax_auto(x), torch.softmax(x, dim=1), atol=1e-5, rtol=1e-5)
