# vLLM Triton kernels

Triton kernels from [vLLM](https://github.com/vllm-project/vllm) (commit
`ff712f6447093d07747c88680b9d006b119f5890`), with the XPU tensor-descriptor patches of the
[Intel XPU Triton benchmarks](https://github.com/intel/intel-xpu-backend-for-triton/tree/main/benchmarks/triton_kernels_benchmark/vllm)
applied. Each `.py` holds the kernel and a `Model` class; each `.yaml` is its kernel spec
(inputs, inits and `ci` / `bench-gpu` variants; see [docs/KERNEL.md](../../docs/KERNEL.md)).

| File | Kernel | vLLM source |
|---|---|---|
| `1_BatchedMoE.py`, `.yaml` | Batched mixture-of-experts GEMM | `vllm/model_executor/layers/fused_moe/fused_batched_moe.py` |
| `2_FusedMoE.py`, `.yaml` | Fused mixture-of-experts GEMM | `vllm/model_executor/layers/fused_moe/fused_moe.py` |
| `3_UnifiedAttention.py`, `.yaml` | Unified paged attention: alibi, softcap, sinks, quantized KV | `vllm/v1/attention/ops/triton_unified_attention.py` |
| `4_UnifiedAttention_MixedDtype.yaml` | A spec for `3_UnifiedAttention.py` with the KV cache in its stored dtype (`float8_e4m3fn`, `int8`) and the query in `bfloat16` | |

```bash
xe-forge -i examples/vllm/3_UnifiedAttention.py -s examples/vllm/3_UnifiedAttention.yaml \
    -n unified_attention -o unified_attention_optimized.py
```
