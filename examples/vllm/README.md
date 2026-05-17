# vLLM Kernel Benchmarks

Triton kernels extracted from the [vllm](https://github.com/vllm-project/vllm) repository at commit `ff712f6447093d07747c88680b9d006b119f5890`.

| Kernel | Source |
|--------|--------|
| `1_FlashAttention_Fwd` | Flash Attention forward pass |
| `2_BatchedMoE` | Batched Mixture-of-Experts GEMM |
| `3_FusedMoE` | Fused Mixture-of-Experts GEMM |
| `4_UnifiedAttention` | Unified paged attention with alibi, softcap, KV quant |

Each kernel has a corresponding `.yaml` config for tile tuning via Xe-Forge.
