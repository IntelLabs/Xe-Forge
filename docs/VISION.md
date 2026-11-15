# Vision workloads: what was measured

Ten diffusion models captured, named, analyzed, optimized and proved under vLLM-Omni on an
Arc Pro B70. Three had no pipeline in the framework at all; three needed a missing XPU op
first. Each kernel was kept only if it was faster and `torch.allclose` against the baseline at
the spec's tolerance.

| model (type) | fits B70 | Omni | diffusers | missing | speedup vs reference (diffusers / pytorch) | kernels optimized | kernel opt speedup | overall speedup vs reference |
|---|---|---|---|---|---|---|---|---|
| FLUX.2-klein-9B (image, DiT 9B) | V (fp8) | V | V | — | — | `mha_varlen_fwd` (1.25x) | 1.019x | 1.019x |
| FLUX.2-klein-4B-edit (image+input, DiT) | V (fp8) | V | V | — | — | `mha_varlen_fwd` (1.12x) | 1.009x | 1.009x |
| SDXL-base (image, UNet/CNN) | V | V | V | — | — | GEMM (1.03x), `mha_varlen_fwd` (1.07x) | 1.016x | 1.016x |
| LTX-2.3-Distilled (video, DiT) | layerwise offload | V | V | — | — | `mha_varlen_fwd` (1.21x) | 1.013x | 1.013x |
| Z-Image-Turbo (image, DiT) | V | V | V | ops: `norm`, `rope` (XPU) | 1.419x | `mha_varlen_fwd` (1.11x), `apply_rotary_emb` (1.04x), GEMM (1.00x) | 1.007x | 1.43x |
| Qwen-Image (image, DiT 20B) | layerwise offload | V | V | op: `rope` (XPU) | 1.043x | `mha_varlen_fwd` (1.20x) | 1.007x | 1.05x |
| Wan2.2-TI2V-5B (video, DiT 5B) | V | V | V | ops: `norm`, `rope` (XPU) | 1.125x | `mha_varlen_fwd` (1.11x), `rms_norm_kernel` (1.08x), GEMM (1.02x) | 1.033x | 1.16x |
| PixArt-Sigma (image, DiT) | V | X | V | pipeline (6 files) | 1.046x | `mha_varlen_fwd` (1.94x), GEMM (1.00x), `_sdpa_attention` (1.00x) | 1.286x | 1.35x |
| CogVideoX-2B (video, DiT) | V | X | V | pipeline (7 files) | 1.042x | `mha_varlen_fwd` (1.24x) | 1.115x | 1.16x |
| HunyuanVideo (video, DiT 13B) | model offload | X | V | pipeline (7 files) | 1.155x | `mha_varlen_fwd` (1.15x), GEMM (1.06x), `_move_params` (6.77x) | 1.096x | 1.27x |

Notes:

- fits B70 = Arc Pro B70, 31.9 GiB. V = one card at the served precision, no offload;
  otherwise the technique that made it fit.
- Reference depends on what was missing. A missing pipeline is measured against diffusers =
  warm generation, model loaded, median of requests 2-4. A missing op is measured against the
  framework's own plain-torch path (`custom_ops ["none"]`). Where nothing was missing the
  unpatched framework is the reference, so the overall figure is the kernel work alone.
- overall = the two speedups multiplied.
