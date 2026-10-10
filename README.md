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

## Online softmax for wide rows

The fused kernel holds an entire row in registers, so past about 32,768 columns it starts spilling and slows down sharply. `online_softmax` instead streams across the row in fixed blocks of 1024. Each lane keeps a running max and a running sum, rescaling the sum by `exp(m_old - m_new)` whenever the max grows, so the true max and sum are known after a single pass without revisiting earlier data. A second pass normalizes and writes the output. This is the same technique FlashAttention uses.

It reads the input twice, so it moves 3MN bytes instead of 2MN. That puts its ceiling at about two thirds of the copy bandwidth, roughly 159 GB/s on a T4, and it runs right at that ceiling regardless of width.

### Results (NVIDIA T4, 1024 rows, fp32, effective GB/s)

| Columns | torch.softmax | Fused | Online |
|--------:|--------------:|------:|-------:|
| 1024 | 205.1 | 174.1 | 206.5 |
| 8192 | 225.3 | 227.1 | 157.1 |
| 32768 | 126.8 | 229.3 | 158.3 |
| 65536 | 125.4 | 154.4 | 158.5 |
| 131072 | 124.8 | 53.2 | 157.7 |

At 131,072 columns the online kernel is 3x faster than the fused one and 1.26x faster than PyTorch. At 1024 columns it beats its own two pass ceiling, most likely because each 4 KB row is still in the T4's 4 MB L2 cache when the second pass rereads it.

`softmax_auto` uses the fused kernel up to 32,768 columns and the online kernel above that.

### Limitations

Only fp32 is supported for now. Thresholds were measured on a T4 and may differ on other GPUs.
## Running

Requires an NVIDIA GPU (Google Colab's free T4 works).

```
pip install -r requirements.txt
pytest tests/ -v
```
