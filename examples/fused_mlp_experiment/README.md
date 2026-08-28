# The fused-MLP experiment: the first full end-to-end loop — REJECT, then the fix, then ACCEPT

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


## Round 2: the fix, and the ACCEPT (same day)

The REJECT's instrumented cause pointed at one line: the cutlass adapter's `run()`
accepts a `sycl::queue*` (`gemm_universal_adapter.h:551`), so both ops now launch on
torch's current in-order XPU stream and **both waits are gone**. Numerics unchanged
(identical fp64-truth distances). Fresh two-arm A/B, six replicates each:

    pristine: 615.0 611.2 610.6 611.3 611.3 611.6  tok/s
    fused:    619.0 617.3 616.1 613.1 613.0 613.3  tok/s
    verdict:  ACCEPT  +0.56%  95% CI [0.21%, 0.91%]  MDE 0.46%

Plan §25's primary criterion, first met: a validated candidate, patched into live
serving through the journalled mechanism, with an end-to-end throughput improvement
whose 95% CI excludes zero. Honest scope: one session, one serving configuration
(Qwen2.5-0.5B bf16, batch 16, temperature 0.8 seeded, enforce_eager, pinned KV);
§25 asks for reproducibility across three independent sessions before the full
claim, and §14.3 for the profile matrix. The tree is reverted; redeploy with
`apply_patch.py apply` + `ORBIT_FUSED_MLP=1 ORBIT_FUSED_LIB=<liborbit_fused.so>`.
