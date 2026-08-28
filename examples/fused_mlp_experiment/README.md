# The fused-MLP experiment: the first full end-to-end loop, and its honest REJECT

The complete §25 chain, run 2026-08-27 on Wildcat Lake against live vLLM
(Qwen2.5-0.5B, batch-16 decode, temperature 0.8 seeded):

region detection (r0/r1, 67% of GPU time) → Xe-Fuse k2 executor (kernel-level
**+3.1%**, CI [0.24%, 5.94%], numerics verified against fp64 truth — the fused path
is *closer* to truth than vLLM's unfused chain) → `orbit_fused.cpp` torch extension
(two tile instantiations, M-dispatched; `add_rms_scale` eliminating the normalize
write, gamma folded into the packed weight) → journalled in-place patch of
`Qwen2DecoderLayer.forward` (`apply_patch.py`, guarded: `ORBIT_FUSED_MLP=1`, M ≤ 32)
→ two-arm e2e A/B, six in-process replicates per arm:

    pristine: 612.7 612.1 611.8 612.1 610.7 612.1  tok/s
    fused:    599.7 596.9 597.7 595.8 594.8 594.3  tok/s
    verdict:  REJECT  -2.52%  95% CI [-2.81%, -2.22%]  MDE 0.19%

**Why the kernel win did not convert:** the extension launches on cutlass-sycl's
compat queue, not torch's stream, so each op ends in a `wait()` — two serializations
per layer, 24 layers per decode step. That overhead exceeds the fused kernel's
~1-5 µs margin. The declared risk in `orbit_fused.cpp`'s header was the cause.

**The named follow-up with real headroom:** launch on torch's current XPU stream
(share the in-order queue, drop both waits). The kernel-level margin is real and
verified; the integration cost is what rejected it. The revert restored vllm-src
byte-for-byte (journal empty, `git status` clean).

This directory is the reproduction kit, not shipped code: compile per the flags in
plan.md §13.4's measured notes (icpx, sycl-tla + Xe-Fuse includes,
`--spirv-ext=+SPV_INTEL_split_barrier`).
