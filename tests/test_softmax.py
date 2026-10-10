import pytest
import torch
from kernels.softmax import softmax


@pytest.mark.parametrize("shape", [(1, 1), (7, 13), (1823, 781), (128, 1024), (64, 4097), (16, 32768)])
def test_matches_torch(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda")
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=1), atol=1e-5, rtol=1e-5)
