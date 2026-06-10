# Examples

Curated kernels from [KernelBench](https://github.com/ScalingIntelligence/KernelBench) Level 2 and the [Intel XPU Triton benchmarks](https://github.com/intel/intel-xpu-backend-for-triton/tree/main/benchmarks/triton_kernels_benchmark), organized by optimization pattern. Each example includes a Triton kernel (`.py`) and a spec (`.yaml`). Source files live in `test_kernels/`; the `examples/` directory provides a categorized view via symlinks.

## Running an example

```bash
xe-forge -i examples/gemm/14_Gemm_Divide_Sum_Scaling.py \
         -s examples/gemm/14_Gemm_Divide_Sum_Scaling.yaml \
         -o optimized.py
```

## Categories

### GEMM

GEMM with post-matmul elementwise or reduction operations.

| Kernel | Operations |
|--------|-----------|
| [14_Gemm_Divide_Sum_Scaling](examples/gemm/14_Gemm_Divide_Sum_Scaling.py) | GEMM + divide + column sum + scaling |
| [39_Gemm_Scale_BatchNorm](examples/gemm/39_Gemm_Scale_BatchNorm.py) | GEMM + scaling + batch normalization |
| [45_Gemm_Sigmoid_LogSumExp](examples/gemm/45_Gemm_Sigmoid_LogSumExp.py) | GEMM + sigmoid + log-sum-exp reduction |

### Fused

Long activation chains fused into a single kernel.

| Kernel | Operations |
|--------|-----------|
| [81_Gemm_Swish_Divide_Clamp_Tanh_Clamp](examples/fused/81_Gemm_Swish_Divide_Clamp_Tanh_Clamp.py) | GEMM + swish + divide + clamp + tanh + clamp |
| [95_Matmul_Add_Swish_Tanh_GELU_Hardtanh](examples/fused/95_Matmul_Add_Swish_Tanh_GELU_Hardtanh.py) | Matmul + add + swish + tanh + GELU + hardtanh |
| [99_Matmul_GELU_Softmax](examples/fused/99_Matmul_GELU_Softmax.py) | Matmul + GELU + softmax |

### Reduction / Normalization

Kernels with reduction passes (batch norm, softmax).

| Kernel | Operations |
|--------|-----------|
| [84_Gemm_BatchNorm_Scaling_Softmax](examples/reduction/84_Gemm_BatchNorm_Scaling_Softmax.py) | GEMM + batch norm + scaling + softmax |

### Attention

| Kernel | Operations |
|--------|-----------|
| [1_FlashAttention_Fwd](examples/attention/1_FlashAttention_Fwd.py) | Flash Attention forward (Q @ K, softmax, @ V) |

### Mixed Ops

Matmul combined with pooling, min/max, or other non-standard operations.

| Kernel | Operations |
|--------|-----------|
| [55_Matmul_MaxPool_Sum_Scale](examples/mixed_ops/55_Matmul_MaxPool_Sum_Scale.py) | Matmul + max pool + sum + scaling |
| [68_Matmul_Min_Subtract](examples/mixed_ops/68_Matmul_Min_Subtract.py) | Matmul + row min + subtract |

## Adding a new example

1. Add kernel `.py` and spec `.yaml` to `test_kernels/`
2. Symlink into the appropriate `examples/` category:
   ```bash
   cd examples/gemm
   ln -s ../../test_kernels/MyKernel.py .
   ln -s ../../test_kernels/MyKernel.yaml .
   ```
3. Update this file
