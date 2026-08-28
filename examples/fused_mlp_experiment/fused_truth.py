"""Which path is closer to fp64 truth? Decides rounding-vs-defect (§17.5 discipline)."""
import sys

import torch
import vllm_xpu_kernels._C  # noqa: F401

torch.ops.load_library(sys.argv[1])
dev, eps = "xpu", 1e-6
torch.manual_seed(0)
m, k, two_i = 16, 896, 9728
hidden = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
residual = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
gamma = torch.randn(k, device=dev, dtype=torch.bfloat16)
w = torch.randn(two_i, k, device=dev, dtype=torch.bfloat16) * 0.05

# fp64 ground truth of the mathematical chain.
x64 = (hidden.double() + residual.double())
s64 = torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + eps)
h64 = x64 * s64 * gamma.double()[None, :]
gu64 = h64 @ w.double().t()
g64, u64 = gu64[:, : two_i // 2], gu64[:, two_i // 2 :]
truth = (g64 * torch.sigmoid(g64)) * u64

# Unfused vLLM path.
r1, i1 = residual.clone(), hidden.clone()
torch.ops._C.fused_add_rms_norm(i1, r1, gamma, eps)
gu = torch.nn.functional.linear(i1, w)
ref_act = torch.empty(m, two_i // 2, device=dev, dtype=torch.bfloat16)
torch.ops._C.silu_and_mul(ref_act, gu)

# Fused path.
r2, i2 = residual.clone(), hidden.clone()
scale = torch.ops.orbit_fused.add_rms_scale(i2, r2, eps)
folded = (w.float() * gamma.float()[None, :]).to(torch.bfloat16)
packed = torch.empty(k, two_i, device=dev, dtype=torch.bfloat16)
packed[:, 0::2] = folded[: two_i // 2].t()
packed[:, 1::2] = folded[two_i // 2 :].t()
d = torch.ops.orbit_fused.gate_up_swiglu(r2.contiguous(), packed.contiguous(), scale)
fused_act = d[:, 0::2]

def dist(a):
    rel = (a.double() - truth).abs() / truth.abs().clamp(min=1e-2)
    return rel.median().item(), rel.quantile(0.99).item(), rel.max().item()

for name, act in (("unfused(vLLM)", ref_act), ("fused(orbit) ", fused_act)):
    med, p99, mx = dist(act)
    print(f"{name} vs fp64: median {med:.5f}  p99 {p99:.4f}  max {mx:.3f}")

# Fold-aware truth: same math in fp64 but through the SAME quantization points the
# fused path uses (bf16 gamma-fold, bf16 residual, its own fp32 scale). If fused
# matches THIS tightly, the tail above is the fold's quantization — a precision
# design choice to be judged at L3 (token match) — not a kernel defect.
truth2_in = r2.double()  # the bf16 residual the kernel actually read
gu2 = truth2_in @ packed.double()
gu2 = gu2 * scale.double()[:, None]
g2, u2 = gu2[:, 0::2], gu2[:, 1::2]
truth2 = (g2 * torch.sigmoid(g2)) * u2
rel2 = (fused_act.double() - truth2).abs() / truth2.abs().clamp(min=1e-2)
print(f"fused vs fold-aware fp64: median {rel2.median().item():.6f}  p99 {rel2.quantile(0.99).item():.5f}  max {rel2.max().item():.4f}")
