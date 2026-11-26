# Examples

Kernels from [KernelBench](https://github.com/ScalingIntelligence/KernelBench) Level 2 and
the [Intel XPU Triton benchmarks](https://github.com/intel/intel-xpu-backend-for-triton/tree/main/benchmarks/triton_kernels_benchmark),
grouped by pattern. Each is a Triton kernel (`.py`, with a `Model` class) and its spec
(`.yaml`); [KERNEL.md](KERNEL.md) describes both. The files live in `test_kernels/`, and
the category directories under `examples/` link to them.

## Running an example

```bash
x-forge -i examples/gemm/14_Gemm_Divide_Sum_Scaling.py \
         -s examples/gemm/14_Gemm_Divide_Sum_Scaling.yaml \
         -o optimized.py
```

## Categories

### GEMM

| Kernel | Operations |
|---|---|
| [14_Gemm_Divide_Sum_Scaling](../examples/gemm/14_Gemm_Divide_Sum_Scaling.py) | GEMM, divide, column sum, scaling |
| [39_Gemm_Scale_BatchNorm](../examples/gemm/39_Gemm_Scale_BatchNorm.py) | GEMM, scaling, batch normalization |
| [45_Gemm_Sigmoid_LogSumExp](../examples/gemm/45_Gemm_Sigmoid_LogSumExp.py) | GEMM, sigmoid, log-sum-exp |

### Fused activation chains

| Kernel | Operations |
|---|---|
| [81_Gemm_Swish_Divide_Clamp_Tanh_Clamp](../examples/fused/81_Gemm_Swish_Divide_Clamp_Tanh_Clamp.py) | GEMM, swish, divide, clamp, tanh, clamp |
| [95_Matmul_Add_Swish_Tanh_GELU_Hardtanh](../examples/fused/95_Matmul_Add_Swish_Tanh_GELU_Hardtanh.py) | matmul, add, swish, tanh, GELU, hardtanh |
| [99_Matmul_GELU_Softmax](../examples/fused/99_Matmul_GELU_Softmax.py) | matmul, GELU, softmax |

### Reduction and normalization

| Kernel | Operations |
|---|---|
| [84_Gemm_BatchNorm_Scaling_Softmax](../examples/reduction/84_Gemm_BatchNorm_Scaling_Softmax.py) | GEMM, batch norm, scaling, softmax |

### Attention

| Kernel | Operations |
|---|---|
| [1_FlashAttention_Fwd](../examples/attention/1_FlashAttention_Fwd.py) | Flash Attention forward |

### Mixed ops

| Kernel | Operations |
|---|---|
| [55_Matmul_MaxPool_Sum_Scale](../examples/mixed_ops/55_Matmul_MaxPool_Sum_Scale.py) | matmul, max pool, sum, scaling |
| [68_Matmul_Min_Subtract](../examples/mixed_ops/68_Matmul_Min_Subtract.py) | matmul, row min, subtract |

### Serving kernels

[examples/vllm/](../examples/vllm/README.md): Triton MoE and paged-attention kernels taken
from an LLM serving framework, with specs; its README names the source files.

## Adding an example

1. Add the kernel `.py` and spec `.yaml` to `test_kernels/`.
2. Link them into a category:
   ```bash
   cd examples/gemm
   ln -s ../../test_kernels/MyKernel.py .
   ln -s ../../test_kernels/MyKernel.yaml .
   ```
3. Add a row here.
