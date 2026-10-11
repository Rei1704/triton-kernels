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

## Tiled matmul

`kernels/matmul.py` computes C = A @ B. Each program computes one `BLOCK_M x BLOCK_N` tile of C, walking along K in chunks of `BLOCK_K` and accumulating `tl.dot` results in fp32. Each loaded element of A is reused `BLOCK_N` times and each element of B `BLOCK_M` times, which is what makes the kernel compute bound rather than memory bound. Block sizes, `num_warps` and `num_stages` are chosen by `@triton.autotune`.

### Results (NVIDIA T4, 4096 x 4096 x 4096, fp32)

| Kernel | TFLOPS |
|--------|-------:|
| cuBLAS (`torch.matmul`) | 3.76 |
| Triton, 32x32 tiles | 1.62 |
| Triton, 64x64 tiles | 2.62 |
| Triton, 128x128 tiles | 3.33 |
| Triton, autotuned | 3.66 |

The autotuned kernel reaches about 97% of cuBLAS. The winning config was 128x128x32 tiles with 8 warps and 3 pipeline stages. Across the configs tried, 8 warps beat 4 by about 9% at the same tile (64 accumulator values per thread instead of 128), and a `BLOCK_K` of 16 was the slowest since each loop iteration does too little work to cover its fixed overhead.

### Note on fp16 and tensor cores on the T4

In fp16, cuBLAS reaches 21.7 TFLOPS on the T4 using tensor cores, while this kernel only reaches about 1 TFLOPS. Inspecting the compiled PTX shows 0 `mma` (tensor core) instructions and 1024 `fma` instructions: with Triton 3.6 on a T4 (compute capability 7.5), `tl.dot` compiles to ordinary multiply and add rather than tensor core instructions. The fp32 comparison above is therefore the fair one on this GPU. On Ampere or newer GPUs (compute capability 8.0+) the same kernel should use tensor cores, and fp16 would become the fast path.

## FlashAttention (forward)

`kernels/attention.py` computes softmax(Q Kᵀ / √d) V without ever building the N x N score matrix. Each program takes a block of 32 query rows for one head and streams across the keys in blocks of 128. For each key block it computes that block's scores on chip, then updates a running max, a running sum, and a running weighted sum of V, rescaling the old values by `exp(m_old - m_new)` whenever the max grows. This is the online softmax from the wide row kernel above, extended to carry the output along with it. After the last block it divides by the running sum.

Exponentials use `exp2` with log2(e) folded into the score scaling, as production FlashAttention kernels do. On the T4 this made no measurable difference, since the kernel is limited by the two matmuls rather than by exp.

### Results (NVIDIA T4, fp32, 8 heads, head_dim 64)

Compared against naive attention and PyTorch's memory efficient attention backend (`SDPBackend.EFFICIENT_ATTENTION`).

| N | Naive | Mem efficient | Triton (this repo) |
|------:|------:|------:|------:|
| 8192 | 87.9 ms, 4.42 GB | 49.2 ms, 0.13 GB | 82.6 ms, 0.13 GB |
| 16384 | out of memory | 197.1 ms, 0.20 GB | 333.2 ms, 0.20 GB |
| 32768 | out of memory | 843.7 ms, 0.33 GB | 1358.7 ms, 0.33 GB |

Memory matches PyTorch's memory efficient kernel exactly and grows linearly with N, while naive attention needs about 34x more at N = 8192 and cannot run at all beyond that on a 16 GB GPU. Runtime still grows with N² since the amount of computation is unchanged. Speed is about 60% of PyTorch's memory efficient kernel across all lengths.

### What limits speed on the T4

A sweep over block sizes, warps and pipeline stages showed every config with `BLOCK_M x BLOCK_N` of 8192 or more fails with `OutOfResources`. In fp32, a 64x128 P tile plus the V and K blocks it is multiplied with already exceed the T4's 64 KB of shared memory per SM. That caps the tiles at 32x128, so each K and V block is reused by only 32 query rows before being discarded, which limits math per byte loaded. Production FlashAttention runs in fp16 or bf16 on tensor cores, where tiles are half the size and the matmuls are far faster. Pipeline stages made no measurable difference for this kernel.

Also worth noting: PyTorch's fused attention backends require 4D (batch, heads, N, d) inputs. With 3D inputs `scaled_dot_product_attention` silently falls back to its unfused math path, which on this GPU used as much memory as the naive version.

### Limitations

Forward pass only. fp32 only. Tested with head_dim 32 and 64; larger head dimensions would likely need smaller blocks to fit in shared memory.

## Running

Requires an NVIDIA GPU (Google Colab's free T4 works).

```
pip install -r requirements.txt
pytest tests/ -v
```
