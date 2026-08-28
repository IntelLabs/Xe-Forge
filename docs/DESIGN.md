# Xe-Orbit design reference

The durable design decisions behind `src/xe_forge/orbit/`, in one place. Code
comments state local constraints only; the reasoning lives here. `plan.md` is the
historical audit log — section numbers cited there (§N) are its; this file is the
curated distillation and the one the code may point at.

## 1. The deterministic/agent boundary

Never call an LLM for a task with a deterministic answer. Parsing a trace,
classifying a failure, matching an anchor, choosing a preset, applying a patch,
timing a kernel, deciding accept/reject — all deterministic, all code. The agent
is asked exactly two kinds of question: *what to try* (optimization proposals,
rewrites) and *what to make of structure that exact matching could not decide*
(a decoder that deviates from the known idiom, an unclassified failure). AMD's
Hyperloom states the same rule independently ("no LLM turn is consumed" on its
kernel path), which is evidence the boundary is an engineering constraint, not a
style choice. Residue that deterministic passes cannot decide is handed to the
agent explicitly — never approximated with a looser regex.

## 2. Measurement discipline

- Thresholds are declared **before** measuring; no optional stopping; a declared
  gate is never relaxed post hoc. An underpowered result is reported as exactly
  that.
- Interleaved ABBA arms (not ABAB); paired statistics when arms pair; CI95 and a
  minimum detectable effect on every comparison.
- Four verdicts: ACCEPT / REJECT / **INCONCLUSIVE** (CI straddles zero — a real
  outcome, reported with the MDE) / **INVALID** (too few samples, unstable
  clocks, broken baseline).
- The GPU must be quiet: on shared-TDP iGPUs a busy CPU throttles the GPU (~35%
  measured), so leases (see §9) make collisions impossible rather than
  disciplined-against. Kernel timing on Intel needs ≥100 warmup passes.
- Gain accounting refuses two temptations: an unmeasured stack reports
  NOT ESTABLISHED (never a projection), and drift is its own figure, never
  absorbed into a total. Per-entry method is MEASURED / LOCAL_ONLY / MISSING;
  local deltas are marked unsummable.
- The only stopping bar for the system as a whole is end-to-end tokens/second on
  a real serving framework (vLLM or SGLang). Kernel-level wins are inputs, not
  conclusions.

## 3. Discovery and ranking

Traces (torch profiler, unitrace when present) become a kernel catalog with
GPU-time shares, providers (onednn / triton / sycl / custom / runtime), dispatch
chains, and derived ranking fields. The Amdahl ceiling gates ambition: a kernel
at 0.25% of GPU time cannot justify an agent's attention, and the gate says so
before tokens are spent. Source resolution records its tier
(build_graph / symbol_index / name_match / agent / unresolved); deterministic
tiers report confidence "exact", agent tiers carry a float and the previous
answer for review. A memcpy-style runtime provider has no kernel body to
rewrite — that is a different fact from "source not found", and the catalog says
which one it is.

## 4. Extraction (E1–E4) and why it exists at all

Hyperloom does not extract kernels: it edits framework source in place, which
works on ROCm where the installed tree is commonly the built-from-source tree.
On Intel, kernel wheels ship compiled `.so` files whose sources live in other
repositories (vllm-xpu-kernels, sgl-kernel-xpu, torch-xpu-ops) — so Orbit clones
those trees, **pins them by URL + revision**, and extracts:

- E1/E2: closure-resolved standalone bundles (Triton via AST closure; SYCL via
  compile_commands + `-MM`), verified — isolated import/compile, launch-record
  match, mutation check, template-instantiation match. Proving the extracted
  kernel is the kernel that ran, rather than assuming it.
- E3: in-situ (the installed tree is patched where it stands).
- E4: opaque (vendor library); its lever is launch/config knobs, not rewrites.

Version skew between an installed wheel and its source tree is classified and
surfaced; a wheel version with no matching git tag is pinned to the inferred
commit with the inference recorded.

## 5. Patching

A ladder from least to most invasive: P1 operator override (a
`TORCH_LIBRARY_IMPL` registration on the dispatch key — the framework tree is
never touched), through scoped source patches, to in-place edits of installed
source. In-place edits go through a journalled patcher: the original is copied
and the journal fsynced *before* the target changes, so a crash between steps
leaves a recoverable record; reverts are digest-verified; a no-op patch is
refused rather than journalled as a change. Dispatch assertions confirm a patch
actually took effect (new kernel present *and* old kernel absent). Journals are
per-model: a shared journal's `revert_all` once restored another model's kept
patch. Fused-path patches are guarded (env opt-in + shape guard + library
present) so the unpatched path stays byte-identical — one tree serves both arms
of an A/B.

## 6. Enablement

Before a workload can be optimized it must run. The ladder: rung 0 diagnose
(deterministic classification of a failed launch into named capability gaps —
oom, missing_device, missing_op, missing_package, backend_codegen, config,
quant_capability, kernel_capability, build_resolution, honest unknown), then
climbs: rung 1 serve-flag, rung 2 source patch, rung 3 attempt-scoped venv,
rung 4 source localization, rung 5 off-loop compiled build. Two rules:

- **Runnable gate**: a fix earns KEEP only when the workload boots *and*
  re-passes its quality eval. Boot alone is never KEEP; artifact verification
  alone is never KEEP.
- **Cheapest rung first**: phi-2's missing head-size bucket was served the same
  hour by a backend flag (rung 1) while the kernel rebuild (rung 5) ran as the
  bracket's alternative — the flag answer is not the *final* answer, but it is
  the first one.

An unclassified failure is a finding, not a license to guess: the unknown class
exists so everything in it gets a second look.

## 7. Fusion (Xe-Fuse) and the region path

Adjacent-kernel regions (gemm+activation, gemm+rmsnorm, gemm+geglu) route to
Xe-Fuse presets by pattern lookup. The executor generates, compiles (sourcing
the oneAPI environment that owns the resolved compiler — subprocesses do not
inherit the caller's shell luck), runs with warmup, and sweeps tiles; every
result is kept, the winner named. Timing tables are never acceptance:
correctness is gated by a host-side high-precision referee (fp64) comparing the
fused path against the mathematical chain — "closer to truth than the unfused
path" is the bar that distinguishes rounding from defect. Kernel-level margin
decides whether a candidate even reaches e2e: a 3% region win projects ~1% e2e
and is unprovable on a noisy box; a 9–10% win proved out at +0.62% tokens/s.
The generated kernels are ordinary SYCL source and remain extraction/rewrite
targets themselves.

## 8. The optimization loop

Gate order is cheapest-first: novelty (free) → sandbox (free) → critic (one
call) → apply → correctness → measure. This deliberately inverts Hyperloom
(their eval is the expensive half; ours is the engine load). Verdicts: KEPT,
REVERTED_WRONG, REVERTED_SLOWER, UNPROVEN (≠ wrong — the distinction is kept),
REFUSED. A novelty ledger treats a repeat of the same attempt (component, ref,
arch, command) as a stall; timeouts get one retry. Failed directions feed back
into the proposer as session memory — the agent finds out what happened, in
process and across invocations.

## 9. Policy: allowlist, single-writer, leases

An agent-proposed action is untrusted input: contexts grant an explicit action
allowlist and everything else is refused by default. Path invariants are the
patcher's own checks, wrapped — one implementation, one exception type. An
advisory single-writer lock refuses a second concurrent writer by name; a stale
lock (dead holder) is broken with a logged note, never silently. The per-device
resource lease uses the same idiom: every GPU-touching command holds it, a
second claimant is refused with holder/reason/since, and acquisition can run a
quiet-machine probe so taking the device and validating the measurement
precondition are one gesture. The full phase machine waits until agent-proposed
writes exist outside the loop's gate — invariants with no writers to constrain
would be a name for a component that does not exist.

## 10. The build lane

Long compiles never block the loop: a single-slot, journalled, resumable queue.
Crash recovery is honest — a builder that dies leaves "builder pid died before
finishing", not a stale RUNNING. Operational hard-won facts: run the drain under
a systemd user unit (a runner tied to a terminal dies with it, taking the
compiler along); exempt it from systemd-oomd and cap `MemoryHigh` (single
cutlass TUs exceed 30 GB virtual with `MAX_JOBS=1`; oomd kills at swap pressure
long before true exhaustion); ≥30 GB swap on ≤16 GB boxes. A successful build is
not KEPT until the runnable gate passes.

## 11. The candidate bracket

For one op, several implementations compete: the framework's own SYCL kernel,
its native fallback, oneDNN, a Xe-Fuse preset, a Triton or agent rewrite.
Granularity is part of the candidate space — a serve flag flips a whole kernel
class, a source patch routes per phase, the dispatch override swaps one op, a
runtime guard dispatches per shape. Free coarse candidates enter first; a mixed
candidate is admitted only when measurement justifies its patch cost, and the
Amdahl ceiling is stated before any candidate spends one. The winner is recorded
per serving profile **with the losers' numbers** — a recorded choice without
the losers' numbers is an opinion.

## 12. The day-1 contract

An unknown model arrives; the serving stack may not run it. The job, in order:
**enable** it on the framework's own path (SYCL or Triton) via the ladder, then
**optimize the kernels the framework actually ships**, in place or by override.
Replacement paths (fusion, custom kernels) enter later as bracket candidates,
never as the definition. The measured proof-of-shape: five small models through
identical commands with zero per-model code; the blocked one diagnosed, served
through a backend flag the same hour, and profiled on the stock path.

## 13. Relationship to AMD Hyperloom

Aligned on mechanisms (closed loop, enablement ladder + runnable gate, in-place
operation on framework source, novelty stall gate, deterministic kernel path,
component decomposition with JSON contracts). Deliberately divergent where
Intel's reality or measurement honesty demands it: extraction with verification
(compiled wheels vs built-from-source trees), INCONCLUSIVE with declared MDE,
NOT-ESTABLISHED instead of projected gains, correctness gated before throughput.
Canonical Intel stacks are the targets: vllm-xpu-kernels, sgl-kernel-xpu,
torch-xpu-ops, in-wheel Triton source — aligned with Intel, not far from AMD.
