# Xe-Orbit

Workload-level performance optimization inside Xe-Forge: from inference back to the kernel.

Orbit is the **control plane**. It decides *what to work on* and *whether it worked*. It never
contains kernel transformation logic — kernel rewrites go to Xe-Forge, region fusion to Xe-Fuse,
and everything with a deterministic answer (trace parsing, ranking, dependency closure,
accept/reject arithmetic) is computed here without an LLM.

Design document: [`plan.md`](../../../plan.md) at the repository root. Section references below
point into it.

## Status

**The loop closes.** Every stage of the §24 milestone order is implemented: Orbit can trace a
workload, decide whether it is even worth optimizing, rank and attribute its kernels, extract one
and prove the bundle is the kernel that actually ran, emit the candidate Xe-Forge consumes, patch
an optimized kernel back through the dispatcher, and decide acceptance with intervals and
per-profile gates.

What it cannot do is *measure* on this machine: everything requiring a live Intel GPU runtime is
implemented but unvalidated here (see "What runs without a GPU" below).

| Stage | Command | State |
| --- | --- | --- |
| Package skeleton, typed models, schemas, `LocalExecutor` | — | done (PR 1) |
| Measurement backbone | `orbit-bench` (library) | done (PR 1b) |
| Baseline run | `xe-orbit run` | done (PR 2) |
| Trace ingest, unitrace, launch interception | `xe-orbit trace` | done (PR 3) |
| Kernel catalog, gating, ranking | `xe-orbit kernels` | done (PR 4) |
| Replay on every stage, T0 CI | `--replay <run-id>` | done (PR 4a) |
| Reference workload, stub optimizer, selftest | `xe-orbit selftest` | done (PR 4a2) |
| Adapter protocol + conformance suite | `xe-orbit conformance` | done (PR 4b) |
| Tier 0 adapter | `--framework generic_torch` | done (PR 4c) |
| Provenance resolvers | `xe-orbit inspect` | done (PR 5) |
| `LanguageBackend` + Triton and SYCL backends | — | done (PR 5b, 5c) |
| Real input capture | `xe-orbit capture` | done (PR 6) |
| Extraction E1–E4 with closure resolution | `xe-orbit extract` | done (PR 7, 8) |
| Bundle test rig | `xe-orbit bundle test` | done (PR 7b) |
| Spec/Model emission | `xe-orbit emit` | done (PR 9) |
| `optimize_kernel_dir` wrapper | — | done (PR 10, §9.9) |
| Patch-back ladder P1–P5 + dispatch assertion | `xe-orbit apply` | done (PR 11) |
| Serving profiles and shape matrix | `xe-orbit matrix` | done (PR 11b) |
| Correctness ladder L0–L5, matrix acceptance | `xe-orbit compare` | done (PR 12) |
| vLLM Tier 1 adapter | `--framework vllm` | done (PR 12b, unvalidated without vLLM) |
| Full loop orchestration | `xe-orbit pipeline` | done (PR 13) |
| Regions + Xe-Fuse routing | `xe-orbit regions` | done (PR 14) |
| Roofline headroom in ranking | — | done (PR 15) |
| Batch extraction + coverage report | `xe-orbit extract --all` | done |
| SYCL operator override (out-of-tree extension) | `xe-orbit apply` | done (§11.8, needs icpx to build) |
| Reinsertion proven end to end on CPU | `pytest tests/orbit/test_replacement_e2e.py` | done |

## Quick start

```bash
xe-orbit frameworks                        # adapters, tiers, declared capabilities
xe-orbit run -- python bench.py            # baseline with repetitions and a 95% CI
xe-orbit trace --from-trace trace.json     # ingest an existing Chrome trace
xe-orbit kernels                           # catalog, gating verdict, ranking
xe-orbit inspect k0                        # provenance, headroom, extraction level
xe-orbit extract k0                        # build a KernelBundle, downgrading if needed
xe-orbit extract --all                     # every kernel + coverage weighted by GPU time
xe-orbit bundle test k0                    # prove it is the kernel that actually ran
xe-orbit emit k0                           # write the Model + spec Xe-Forge consumes
xe-orbit regions                           # fusable multi-kernel regions -> Xe-Fuse
xe-orbit apply k0                          # patch back, highest rung that works
xe-orbit compare k0                        # the L0-L5 correctness ladder
xe-orbit matrix                            # serving profiles and their weights
xe-orbit pipeline                          # the whole loop, stopping where it should
xe-orbit selftest --chaos                  # full-loop invariants: no GPU, no LLM
```

Everything after `run` accepts `--replay <run-id>` to re-run from stored artifacts instead of
live hardware.

## Validated on real hardware

`tests/orbit/test_xpu_hardware.py` runs against an actual Intel GPU and skips cleanly without one.
What it establishes that no fixture can:

- **A SYCL operator override really works** (§11.8). A generated `.cpp` compiles with icpx into a
  shared object, `torch.ops.load_library` registers it on the XPU dispatch key, and the op's result
  changes — with nothing in PyTorch, vLLM or SGLang forked.
- **The host-bound gate discriminates on real device timing** (§18). A warm saturating loop reads
  99.9% GPU busy and returns `KERNEL_REWRITE`; the same workload with allocation inside the measured
  region reads a few percent and returns `HOST_OPTIMIZATION`.
- **A Triton override replaces a kernel on the device and the gain is real.** Measured with
  interleaved ABBA runs on an Intel iGPU: `ACCEPT`, +71.9% with a 95% CI of [+70.4, +73.3] against
  an MDE of 0.58%, dispatch assertion satisfied, output equal within tolerance. The same P1 rung as
  the SYCL path, a different language — which is the point of §11.3.
- **Provenance attributes genuine Intel symbols.** Real oneDNN GEMMs resolve to E4 with fusion and
  backend actions; real `at::native::xpu::` kernels resolve to SYCL at E3, and a templated symbol
  like `VectorizedElementwiseKernel<4, GeluErfFunctor<float>, ...>` is graded down in confidence
  rather than pinned to one instantiation (§11.4).

## What runs without a GPU

By design, almost all of it. The stages most likely to produce a plausible-looking wrong number —
provenance, gating, ranking, the accept/reject arithmetic — need no silicon to test, which is why
they are the ones covered by CPU-only CI (§16.3).

Genuinely needs an Intel GPU with a working runtime: live profiling with device activity, unitrace
GPU-busy and launch-gap numbers, Triton launch interception against real kernels, SYCL builds
(needs `icpx`), and anything §17 calls a measurement.

### Host packages the hardware path needs

```
sudo pacman -S intel-compute-runtime level-zero-loader level-zero-headers intel-oneapi-dpcpp-cpp
```

`intel-compute-runtime` + `level-zero-loader` make `torch.xpu` work at all; `level-zero-headers` is
additionally required before Triton can JIT for XPU; `intel-oneapi-dpcpp-cpp` provides `icpx` for
SYCL overrides (the compiler lives under `/opt` and is found without sourcing `setvars.sh`).

## Layout

```
orbit/
├── models.py         typed artifacts (§7, §10, §12, §14) — every one schema-versioned
├── artifacts.py      run store, persistence, replay (§16.3, §23)
├── schemas/          generated JSON Schema, committed so changes are reviewable (§16.2)
├── stats.py          intervals, MDE, the decision rule (§17) — stdlib only
├── executor.py       Executor protocol + LocalExecutor (§20)
├── bench/            the measurement backbone (§5.4)
├── runtime/          environment, device and version capture (§12.9)
├── profiling/        trace ingest, unitrace, launch interception (§12.4)
├── analysis/         catalog, gating, ranking, regions, roofline (§18, §7.3)
├── provenance/       kernel name → provider → source → actions (§12.5)
├── languages/        LanguageBackend: Triton and SYCL as peers (§11)
├── adapters/         framework protocol, Tier 0, conformance suite (§10)
├── capture/          real tensors with strides preserved (§7.5)
├── extract/          bundles, the E0-E4 ladder, the bundle test rig (§12)
├── emit/             the Model + spec contract Xe-Forge consumes (§8)
├── patch/            the P1-P5 ladder and its dispatch assertion (§13)
├── compare/          the L0-L5 ladder and matrix acceptance (§19, §14.3)
├── optimize.py       optimize_kernel_dir over Xe-Forge's pipeline (§9.9)
├── pipeline.py       the full loop and its stop conditions (§24)
├── selftest.py       StubOptimizer and the pipeline invariants (§15)
└── cli.py            the xe-orbit command line (§21)
```

## Invariants worth knowing before you change anything

These are enforced by tests, not convention. Each one exists because breaking it produces a
confident wrong answer rather than an error.

- **No point-value measurements.** `stats.compare` refuses fewer than five repetitions and
  returns `INVALID`. `INCONCLUSIVE` is a real outcome, never a soft `REJECT`.
- **Comparisons are counterbalanced (ABBA), not merely alternated.** Under strict `A,B,A,B` the
  baseline always runs first within a pair and absorbs every first-position effect, which paired
  statistics then report as a real difference. This was observed making the null test flaky.
- **Capabilities are declared, never assumed.** An adapter that has not declared a capability
  raises `AdapterError` instead of silently no-opping, and never reports a metric it did not
  declare.
- **Tractability breaks ties; it does not decide.** The term is bounded to a 1.43× span so a
  cheap-to-extract Triton kernel cannot outrank a much larger opaque GEMM (§11.10). Widening that
  band re-introduces the bias.
- **Unknown provenance never gets an optimization action.** It gets `PROFILE_MORE` and appears in
  the "considered but not attempted" list.
- **An opaque kernel is not an unactionable one.** A oneDNN GEMM has no editable source but still
  supports fusion, backend, layout and library-config actions.
- **A closure follows every in-package import, not just the calls it can see.** A module cannot be
  imported unless its module-level imports resolve, so a call-graph-only closure produces a bundle
  that looks complete and fails on import. Relative imports (`from .helpers import f`) must be
  resolved against the importing file, or real packaged kernels all wrongly fall back to E3.
- **An unverified bundle is never optimized.** `emit` refuses one unless `bundle test` passed.
  The mutation check is what makes that meaningful: it proves the bundle executes its own
  extracted source rather than the installed package.
- **Verification "passed" is not the same claim as "proven".** An E4 opaque bundle passes because
  every identity check was *skipped*; the pipeline says so rather than implying a proof it does
  not have.
- **"Can we extract all the kernels" has no yes/no answer.** Every kernel reaches *a* level
  (E3 and E4 are always available), so a bare "100% extracted" is true and useless. Coverage is
  reported per level and weighted by GPU time, because 90% of kernels covered means nothing if
  the other 10% own most of the runtime.
- **A patch is not applied until re-profiling proves it.** The new kernel must appear in the trace
  AND the old one must be gone. Checking only for the new one passes a workload now running both,
  which is a regression wearing a success's clothes.
- **The correctness ladder short-circuits.** A numerically wrong candidate must never reach the
  timing gate, because a timing number for a wrong kernel looks like evidence.
- **A trade is not an improvement.** A per-profile regression rejects the candidate even when the
  weighted average is positive (§14.3).
- **A null test gated on a 95% interval fails 5% of the time by construction.** The conformance
  suite retries rather than treating that as a broken adapter — a genuinely broken one fails every
  attempt.
- **The core imports no serving framework.** Checked by scanning imports in `selftest`, not by
  convention.

## Testing

```bash
pytest tests/orbit -q                       # 300+ tests, CPU only, ~6s
PYTHONPATH=src python -m xe_forge.orbit.cli selftest --chaos
```

The reference workload lives in [`examples/orbit_mini/`](../../../examples/orbit_mini/) and is
deliberately adversarial for extraction — a test workload that extracts cleanly tests nothing
(§15.2).

[`examples/kernel_replacement/`](../../../examples/kernel_replacement/) is the reinsertion
proof: a real dispatcher op, a real operator override, and a dispatch assertion checked against
what actually executed. It runs on CPU, so the one thing that cannot be faked — did the
optimized kernel really replace the original — is verified on every run of the suite.
