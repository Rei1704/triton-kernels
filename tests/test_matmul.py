import pytest
import torch
from kernels.matmul import matmul

SHAPES = [
    (1, 1, 1),         # smallest possible
    (333, 517, 271),   # nothing divides evenly into any block size
    (128, 128, 128),   # exactly one 128x128 tile
    (1000, 17, 64),    # K smaller than BLOCK_K
    (512, 2048, 256),  # long K loop
]


@pytest.mark.parametrize("M,K,N", SHAPES)
def test_fp32_matches_torch(M, K, N):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(matmul(a, b), a @ b, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("M,K,N", SHAPES)
def test_fp16_matches_torch(M, K, N):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(matmul(a, b), a @ b, atol=0.2, rtol=1e-2)
