# triton-kernels

Fused GPU kernels written in [Triton](https://github.com/triton-lang/triton), benchmarked against PyTorch on an NVIDIA T4.

## Fused softmax

`kernels/softmax.py` computes a row wise softmax in a single kernel. Each program loads one row into on chip memory once, computes the max, exp, sum and division there, and writes the result once. The naive PyTorch version launches five separate kernels and moves about 4x more data through global memory.

The base kernel follows the official Triton fused softmax tutorial. The benchmarking, the analysis of register spilling, and the `num_warps` tuning rule are my additions.

### Results (NVIDIA T4, 4096 rows, fp32)

Effective bandwidth in GB/s, computed as the minimum required traffic (read input once, write output once) divided by runtime. A plain elementwise copy reaches about 239 GB/s on this GPU, so that's the practical ceiling.

| Columns | Naive | torch.softmax | Triton (this repo) |
|--------:|------:|--------------:|-------------------:|
| 128 | 38.5 | 187.4 | 195.9 |
| 256 | 54.0 | 215.3 | 219.2 |
| 512 | 55.5 | 218.5 | 231.3 |
| 1024 | 57.8 | 212.5 | 237.4 |
| 2048 | 59.3 | 213.2 | 239.2 |
| 4096 | 60.1 | 242.5 | 232.6 |
| 8192 | 61.1 | 237.1 | 230.6 |
| 16384 | 61.3 | 169.6 | 229.7 |
| 32768 | 61.4 | 127.1 | 231.7 |

The Triton kernel matches or beats `torch.softmax` at most widths and is 1.8x faster at 32,768 columns, where PyTorch's kernel slows down.

### Tuning num_warps

Each thread holds `BLOCK_SIZE / (32 * num_warps)` values in registers. Sweeping every width against 1 to 16 warps showed that 256 or more values per thread consistently spills to local memory and collapses throughput (down to about 45 GB/s), while 128 or fewer stays fast. With too many warps on narrow rows, threads sit idle. The wrapper uses `num_warps = min(16, BLOCK_SIZE // 64)`, with 1 warp for rows of 128 or fewer. These thresholds were measured on a T4 and may differ on other GPUs.

### Limitations

The whole row must fit in one block, so very wide rows will eventually spill again. Only fp32 is supported for now.

## Running

Requires an NVIDIA GPU (Google Colab's free T4 works).

```
pip install -r requirements.txt
pytest tests/ -v
```
