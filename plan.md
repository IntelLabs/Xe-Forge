# Xe-Orbit — Revised Implementation Plan

**Workload-level performance optimization inside Xe-Forge: from inference back to the kernel.**

Status: draft v10. Supersedes `Xe-Orbit_AI_Agent_Implementation_Plan.md`.
Changes from v1 are marked **[CHANGED]** where they reverse a prior decision.
v3 added §12, kernel extraction and dependency closure.
v4 adds §10, the framework support model — one adapter per framework, with a
conformance suite, so vLLM, SGLang and everything after them are peers rather
than a first-class case and a series of retrofits.
v5 adds multi-file closure resolution and a bundle test rig (§12.6, §12.12),
rewrites patch-back as a mechanism ladder with operator override at the top
(§13), adds serving profiles and the shape matrix (§14), and adds the reference
micro-workload that lets the whole loop run in seconds in CI (§15).
v6 reads AMD's Hyperloom structurally (§5): split measurement into a standalone
`orbit-bench` component, add an agent arena outside the loop, publish a support
matrix — and turns "every step must be testable" into a contract with a
per-stage test matrix, record/replay, failure injection and CI tiers (§16).
v7 corrects a Triton bias: §11 makes SYCL a first-class language path — where
Intel's hand-written kernels actually live (torch-xpu-ops, IPEX, vLLM-XPU csrc,
sgl-kernel-xpu, sycl-tla) — with build-graph closure, compiler-option actions,
and operator-override patch-back that needs no framework fork.
v8 aligns the document with the Xe-Forge repository at commit `4dcb508` (branch
`xeorbit`): corrects every factual claim about existing paths, CLI flags, schema
keys, defaults and module names; renumbers §9's work items, which were previously
mislabeled 8.1–8.8; marks Xe-Fuse as an external project rather than something
present here; and fixes several broken cross-references plus a duplicated table
header in §13. No proposal is redesigned — only the claims about what already
exists.
v9 records what building the thing against real hardware and real frameworks
corrected. §11.2 gains the finding that the obstacle to SYCL extraction is
packaging rather than language — every Intel kernel wheel ships a compiled `.so`
with its sources in another repository — and the source registry that answers it.
§10.6 gains `kernel_sources`, which makes "where a framework keeps its kernels"
knowledge rather than code, so adding SGLang's kernel tree became a YAML file.
§11.4 records why symbol resolution is split the way §3 asks: the length-prefixed
part of Itanium mangling is parsed exactly, and only the ambiguous residue goes
to an agent, after a regular expression silently destroyed identifiers containing
an `I`; it also gains the namespace rule, after `vllm::rms_norm_kernel` resolved
to torch-xpu-ops because that tree came first in a list. §12.12 gains the C++
forms of the bundle checks — the Triton ones had been running against SYCL
bundles, reporting `ModuleNotFoundError` on a `.cpp` file as a broken closure —
along with two rules that outlived their origin: a skipped check is not a passed
check, and a failure has to say which failure it is. §12.10 gains version skew,
which is how a SYCL bundle most often turns out to be a different kernel than the
one that ran. §12.5 gains runtime memory operations, which were being reported as
unattributed kernels awaiting further profiling that could never resolve them.
Nothing here changes a proposal; each entry is a claim that measurement made more
specific, and most of them are places where the design was right and the first
implementation quietly answered a different question.
v9 also checks §5 against the `AMD-AGI/Hyperloom` source at tag 1.0.0 rather than
the announcement it was written from. That comparison added §5.6 (what the
implementation shows, including one rule both designs reached independently),
§17.6 (accepted gains do not add up — the headline is measured, never summed),
and §20.4 (the stall gate). It also names the largest thing Xe-Orbit lacks:
enablement, the work of making a workload run at all before optimizing it — and
establishes that Hyperloom does not extract kernels at all, operating entirely
at E3 against the installed source tree. §11.4 gains the recorded resolution
tier and the rule that a deterministic tier reports no confidence, because it is
either right or silent. §13.2 answers whether Orbit should edit an installed tree
in place: not by default, since P1 never touches it — but where it is
unavoidable, the mechanism is judged by its recovery path rather than its success
path, and is journalled, atomic and crash-recoverable accordingly. §17.5 records
what a real vLLM run on an iGPU forced: pin derived capacity, refuse to measure
under contention, declare the warmup rather than discarding an outlier after the
fact, and know what the chosen harness includes — the framework's own benchmark
and a hand-rolled one disagreed 4x, because one times a cold pass and the other a
warm one. §13.5 adds the agentic loop that drives §13.2's in-place edits: the
agent plans and implements, Orbit re-verifies in a fresh process and owns every
verdict, and measurement is framework-native throughout (`vllm bench throughput`,
`torch.utils.benchmark.Timer`, `FlopCounterMode`) rather than through ai_bench —
which, with correctness done by importing the patched kernel, is why the in-place
path needs neither ai_bench nor DSPy.
v10 audits the document against the working tree rather than against the state at
`4dcb508`: the implementation now exists — `src/xe_forge/orbit/` (~21k lines),
`tests/orbit/` (~680 CPU-only tests), `orbit_mini`, three framework knowledge files and
a T0 CI workflow — so §4.1 becomes a three-state audit and most of §10–§13, §15–§17 and
§20.4 move from proposal to status. Two v9 claims did not survive the check: the
recorded resolution tier stops at `SourceLocation` and never reaches the catalog, and
`kernel_sources` carries no revision pin — both are reopened as work items rather than
restated as done. §24's PR ladder is replaced by the remaining delta, whose first entry
is the one that matters: the agentic loop (§13.5, renumbered from §13.7) is a tested
library that no CLI command reaches. §13's subsections are renumbered into reading
order (13.6→13.2, 13.6.1→13.3, 13.8→13.4, 13.7→13.5), with cross-references updated
throughout, including in the older changelog entries above. As in v8, no proposal is
redesigned. The four Tier A gaps §24 opens were closed in-tree immediately after this
audit — §24 records each closure against its original statement — along with one
finding the audit itself surfaced: the T0 workflow's dependency list was missing
`python-dotenv`, without which every test fails at collection, so the committed CI had
never actually been able to run. Tier B followed: the §9.1 weighted objective, the
§9.9 synchronous Claude path, the §9.7 dispatcher-op harness, the §9.5 loader guard
and the §16.6 CI items are all closed, §24 recording each — plus two defects the CI
work surfaced (an eager ai_bench import blocking the root tests, and a private-dspy
import broken by upstream 3.3.1). Tier C's bounded halves closed next — the
enablement ladder's diagnosis rung and runnable gate, the minimal policy gate, and
`orbit-bench` as a real standalone — and the whole stack was then validated against
real hardware on this branch's own Wildcat Lake machine: the source registry reports
`ok` against the actually-pinned torch-xpu-ops checkout, the XPU hardware tier passes
10/10, `orbit_mini` runs on the device (after a real-Triton constexpr fix §15
records), its real trace reproduces §11.2's taxonomy and §18's host-bound gate stops
honestly, `orbit-bench` produces a decision-grade measurement (MDE 1.29%), and the
§22 primary workload — vLLM Qwen2.5-0.5B — loads in 93s and decodes at 85.6 tok/s
warm, with §17.5's memory-floor and pinned-capacity disciplines exercised on the way.
The §13.5 loop then ran closed end-to-end on that workload: a real batch-16 decode
trace (GPU-bound, 82.2% busy) reproduced §13.4's regions almost exactly (gemm+activation
39.5% + gemm+rmsnorm 27.5% = 67.0%, against 66.9% measured in v9), and `xe-orbit
optimize k4 --apply` drove two rounds over `_gumbel_sample_kernel` — the CLI now feeds
the previous round's verdicts and the device's facts to the proposer, closing §13.5's
"the agent never found out" seam — with all four candidates applied through the
journalled patcher into the live vllm-src tree, gated, measured, and rejected with
CIs excluding zero (+4x block size: −127%; −2x: −11.5%; a warps sweep; and an
unrecognized launch knob correctly recorded UNPROVEN, not WRONG). The tree finished
pristine and the journal empty. Measured conclusion, per §18's own arithmetic: the
sampling kernel sits at a device-local optimum under a 0.28% ceiling, and this
workload's entire headroom is the vendor GEMM — reachable through §13.4's
vendor-GEMM-dominant fusion (Xe-Fuse territory) or E4 config actions, not through
any rewritable kernel. Refusing four plausible candidates with intervals attached is
§25's secondary success criterion, live. The SYCL half of the loop was then validated
on the same trace: Orbit generated, compiled (icpx, seconds — §13.3's cost class,
confirmed) and shadow-registered a P1 override for `_C::fused_add_rms_norm` (k3, the
op §13.3 first proved), the §13 dispatch assertion passed — override kernel present
in the profile, original absent — correctness was 7/7 against both the torch
reference and the original op, and the measurement came back **ACCEPT: +26.9% at
kernel level, 95% CI [11.3%, 42.6%] excluding zero** (five fresh-process samples per
arm, ≥100-launch declared warmup; the wide interval reflects a noisy candidate arm,
per §17.5.1). Stated with §18's arithmetic attached: at k3's 1.2% GPU share the e2e
ceiling of that win is ~0.3%, below anything resolvable end-to-end — the finding is
the *mechanism* at its measured cost, and that the vendor kernel is beatable at
skinny decode shapes, which is the same evidence §13.4's GEMM-region question turns
on.

---

## 1. Goal

Extend Xe-Forge from a *kernel* optimizer into a *workload* optimizer, so that a user can point it at a running inference workload and get back a validated, end-to-end-measured improvement.

The system must answer:

- Is this workload even GPU-bound? (asked **first**, before anything else)
- Which kernels actually execute, and what fraction of wall time do they own?
- Which operators, providers, and source files produced them?
- **Can the kernel be pulled out of the framework at all, and at what fidelity?**
- What is the *ceiling* on end-to-end gain for each candidate action?
- Which action applies: kernel rewrite, region fusion, backend swap, runtime config, host-side fix, or "nothing worth doing"?
- Did the candidate improve both the kernel **and** the workload, beyond measurement noise?

The last clause is the whole point. Xe-Forge already produces large isolated kernel speedups (2.8x geomean on vLLM attention variants, up to 13.3x on FlashAttention forward). Xe-Orbit exists to establish whether those numbers convert into tokens/s — and to make the conversion automatic when they do.

---

## 2. Where this lives **[CHANGED]**

**v1 said:** create a separate `Xe-Orbit/` repository that depends on Xe-Forge.

**v2 says:** build it inside the Xe-Forge repository as `src/xe_forge/orbit/`, shipped in the same distribution, exposed as a second console script.

Rationale:

- v1 put "stabilize the Xe-Forge Python API" on the critical path between the repo skeleton and the first demo. That is a cross-repo serialization hazard for a single-digit-person team.
- The knowledge base, trial tree, VTune wrapper, roofline scripts, and benchmark harness already exist in Xe-Forge. A separate repo forks all of them.
- Leadership is asking for *one* entry point. Two repos and two names is the opposite of that.

```
Xe-Forge/
+-- src/xe_forge/
|   +-- ...                  # existing: agents/, engines/, claude/, core/, knowledge/,
|   |                        #   prompts/, skills/, utils/, cli.py, pipeline.py
|   |                        #   (stages are the OptimizationStage enum in models.py,
|   |                        #    gated per-DSL by dsl_registry.py — not a package)
|   +-- orbit/               # as built (v10) — differs from the v5 sketch in the ways noted
|       +-- cli.py           # 22 subcommands (§21)
|       +-- models.py        # single pydantic module, not a model/ package (§7)
|       +-- artifacts.py     # RunStore, --replay, schema-major refusal
|       +-- stats.py         # §17 statistics, stdlib-only
|       +-- pipeline.py, executor.py, selftest.py, novelty.py
|       +-- device.py, knowledge.py, support.py
|       +-- runtime/         # environment capture
|       +-- provenance/      # per-provider resolver chain
|       +-- bench/           # measurement backbone (§5.4) — with its own `orbit-bench` CLI
|       +-- schemas/         # versioned artifact schemas (§16.2) — 11 committed
|       +-- adapters/        # one per framework (§10): base, conformance,
|       |                    #   generic_torch, vllm   (sglang: v0.2)
|       +-- capture/         # real input capture (§7.5)
|       +-- extract/         # kernel extraction + closure + verification (§12)
|       +-- profiling/       # trace, unitrace, vtune, launch interception
|       +-- analysis/        # catalog, regions, roofline, xe_fuse handoff
|       +-- languages/       # LanguageBackend: triton, sycl, source registry (§11)
|       +-- emit/            # generates spec.yaml + Model wrapper
|       +-- patch/           # patch ladder + journalled in-place edits (§13)
|       +-- optimize/        # loop, proposer, harness, session, kernel_dir (§13.5)
|       +-- compare/         # gates, accuracy, cumulative (§17.6, §19)
|       +-- agents/          # RepoAgent protocol + Claude
+-- knowledge_base/          # existing (scoped: common/ + <dsl>/<device>/, §9.5);
                             #   extended, not duplicated
```

Differences from the v5 sketch, so nobody goes looking for what moved: there is no
`discovery/` package — discovery is `profiling/trace.py` plus `analysis/catalog.py`;
`model/` is the single module `models.py`; `optimizers/` landed as `optimize/`; and
`arena/` does not exist (post-v0.1, §5.4). `languages/`, `compare/` and the top-level
modules above were not in the sketch and are where several sections' machinery lives.

Console scripts: `xe-forge` (unchanged), `xe-forge-skill` (unchanged), `xe-orbit` (registered, `xe_forge.orbit.cli:main`), `orbit-bench` (registered, `xe_forge.orbit.bench.cli:main`, §5.4), plus the `xe_orbit.frameworks` entry-point group (§10.3).

Split into its own repository later if and only if the layering survives contact with vLLM. The internal boundary below is what matters; the repository boundary is not.

---

## 3. Architectural boundary

Orbit is the **control plane**. It decides *what to work on* and *whether it worked*. It never contains kernel transformation logic.

```text
                         xe-orbit
                            |
                    Workload Analyzer
                            |
        +-------------------+-------------------+
        |                   |                   |
   Kernel action      Region action      Runtime/host action
        |                   |                   |
        v                   v                   v
    Xe-Forge            Xe-Fuse           Deterministic
  (Triton/SYCL         (sycl-tla            executor
   rewrite, tile        epilogue           (config sweep,
   search, autotune)    fusion)             backend swap)
        |                   |
     DSPy | Claude       codegen
```

Xe-Fuse is an external sibling project: it is not a dependency, submodule or import of this repository today, so routing `REGION_FUSION` to it is new integration work (§9.6).

Rules:

- Orbit never calls an LLM for a task that has a deterministic answer (trace parsing, ranking, shape aggregation, git state, accept/reject arithmetic, dependency closure).
- Orbit never imports DSPy or Claude tooling directly. Kernel optimization goes through Xe-Forge's existing engine seam — `engines.base.BaseEngine` instances built by `create_engine()`, with `dspy` and `claude` as today's choices — and repository-level agent tasks go through a new `RepoAgent` protocol added in Orbit, provider selected by config.
- Hardware knowledge stays in the Xe-Forge knowledge base. Orbit reads it; it does not restate it.

---

## 4. What is actually new **[CHANGED]**

v1 specified a lot of components that already exist. The genuinely new surface is smaller than it looked:

| Capability | Status |
| --- | --- |
| Kernel analysis, staged optimization, CoVeR | exists (stages are the `OptimizationStage` enum in `models.py`; `agents/cover.py`) |
| Trial tree, candidate tracking | exists (`core/trial_manager.py`); the pipeline keeps a single best candidate per run |
| VTune profiling of a kernel | exists (`xe-forge-skill profile` → `core/profiler.py`; off by default) |
| Roofline model + plots | exists (`scripts/roofline.py` + the two CSV converters; presets B580, B70, Max-1550/1100, Flex-170) |
| Correctness + benchmark harness for a kernel | exists (kernel `.py` + sibling YAML spec + duck-typed `Model`); timing is a single scalar mean — no statistics, see §17 |
| Knowledge base with stage-scoped delivery | exists (`knowledge_base/`, scoped `common/` + `<dsl>/<device>/`; disabled by default) |
| SYCL/CUTLASS tile search | exists (`--tile-tune`, `core/tile_search/`, `docs/TILE.md`) |
| **Runtime kernel discovery from a live workload** | built — `orbit/profiling/` (trace, unitrace, interception) + `orbit/analysis/catalog.py` |
| **Kernel → operator → provider → source provenance** | built — `orbit/provenance/resolvers.py`; the resolution tier persists through the catalog (G2, closed) |
| **Framework adapter layer + conformance suite** | built (§10) — `orbit/adapters/`, incl. null test + positive control, run in CI |
| **Multi-file closure resolution + bundle test rig** | built (§12.6, §12.12) — `orbit/extract/`, Python **and** C++ check forms |
| **Operator-override patch-back through the dispatcher** | built (§13) — `orbit/patch/` |
| **Serving profiles / shape matrix across models and regimes** | partial (§14) — models + `compare/gates.decide_matrix` + `xe-orbit matrix`; breadth unexercised |
| **Reference micro-workload + deterministic pipeline test rig** | built and wired (§15) — `orbit_mini`'s CPU-viable traps armed in T0 by `tests/orbit/test_orbit_mini.py` (G4, closed); `selftest` remains synthetic |
| **Standalone measurement backbone (`orbit-bench`)** | built (§5.4) — library plus the `orbit-bench` console script and JSON contract; runs without torch |
| **Agent arena for A/B-ing engines and models** | absent (§5.4, post-v0.1 as scheduled) |
| **Per-stage schemas, record/replay, CI tiers** | built (§16) — 11 committed schemas, `--replay` on 11 stages, T0 CI; T1–T3 open |
| **`LanguageBackend` layer: SYCL, sycl-tla and C++ as first-class as Triton** | built (§11) — `orbit/languages/` |
| **Build-graph closure and compiler-option action space for SYCL** | built (§11.5, §11.7) — `orbit/languages/sycl_backend.py` |
| **Multi-source kernel extraction with dependency closure** | built (§12) — incl. version-skew classification |
| **Real input capture from a running workload** | built — `orbit/capture/` (strides preserved, round-trip verified) |
| **Automatic generation of `Model` + `spec.yaml` from a runtime kernel** | built — `orbit/emit/` |
| **Patch-back of an optimized kernel into the framework** | built (§13) — incl. journalled in-place edits and recovery |
| **Measurement statistics, kernel-level and end-to-end** | built (§17) — `orbit/stats.py` + `orbit/bench/`; the §9.1 weighted objective is closed (`core/weighted.py`), though each per-variant timing inside it is still core's single scalar mean rather than an interval |
| **Amdahl / host-bound gating before optimization** | built — `stats.amdahl_ceiling` + catalog gating |
| **Region (multi-kernel) actions routed to Xe-Fuse** | partial — detection + handoff shim only (`orbit/analysis/xe_fuse.py`); no executor (§9.6 open) |

Most of the former "new" column is now built, inside `src/xe_forge/orbit/` and under test in `tests/orbit/`. What remains is the delta in §24 — Tier A of which, headed by wiring the §13.5 loop into the CLI, is now closed. Do not rebuild the left column, and do not rebuild the built rows either.

### 4.1 Corrected status **[NEW in v8, re-audited in v10]**

The v8 audit was against `4dcb508`. The v10 audit is against the working tree, with the untracked orbit implementation present, and splits every claim three ways: still absent, present in the tracked core, or present in `orbit/`. The v8 bullets, re-checked:

| v8 claim | v10 state |
| --- | --- |
| `TARGET_SPEEDUP=2.0` is inert | Still inert in core (`config.py:60`, zero readers). The caller-supplied threshold now exists — in Orbit: `orbit/optimize/kernel_dir.py` enforces `required_speedup` itself, with a comment citing this very finding. §9.2 is done Orbit-side; the core config field remains decorative. |
| Per-variant `rtol`/`atol` exist, spec-over-config | Unchanged, and slightly stronger: resolution is caller-arg > spec > config (`pipeline.py:191-204`). |
| `get_example_inputs` has zero in-tree users | Two users now: `examples/orbit_mini/__init__.py` (deliberately non-contiguous inputs) and the harness `orbit/extract/bundle.py` emits, which loads captured tensors from disk. §9.4 is done. |
| Engines are exactly `dspy` and `claude`; `ClaudeEngine` is fire-and-forget | Still exactly two engines, and the fire-and-forget default is unchanged. The §9.9 synchronous path now exists behind `EngineConfig.synchronous`: blocking run, edited kernel read back from the workspace, returned with `success=False` and an explicit unmeasured note (§19). Orbit's wrapper still refuses to launder the async path's unconditional success. |
| No unitrace or Level Zero support of any kind | Now false: `orbit/profiling/unitrace.py` and `trace.py` implement it, gated by `orbit/support.py`. Entirely inside `orbit/`; core is untouched. |
| No `AGENTS.md`/`CLAUDE.md`; CI runs no tests at all | CI is resolved: T0 (`.github/workflows/tests.yaml`) runs the orbit suite, `selftest --chaos`, two-adapter conformance, the replay loop and a schema-drift check under a deliberately minimal install, and a second `core` job installs the full project and runs every root test file; `ruff check` joined the format check. `AGENTS.md`/`CLAUDE.md` still do not exist (§26). |

Two v9 claims also failed this audit and were reopened rather than restated — the resolution tier was designed but never persisted past `SourceLocation` (§5.6, gap G2), and `kernel_sources` carried no revision pin (§12.10, gap G3). Both have since been closed in-tree; §24 records the closures.

---

## 5. Positioning: what to take from Hyperloom **[NEW]**

AMD shipped ROCm Hyperloom in July 2026 as an open-source agentic system for end-to-end inference optimization. It is the closest public analogue to what this plan describes, and it is worth reading structurally rather than competitively. Sources: the ROCm blog announcement (`rocm.blogs.amd.com/software-tools-optimization/hyperloom/`), the ROCm docs, and the component repositories under `github.com/AMD-AGI`.

**v9 note.** §5.1–§5.5 were written from the announcement and the docs site. They have now been checked against the `AMD-AGI/Hyperloom` source at tag 1.0.0 (`7b50bee`, 2026-08-26): ~1100 Python files, MI300X/308X/325X/355X, SGLang / vLLM / Atom / xDiT / `custom`, kernel languages HIP / Triton / FlyDSL, Claude as the LLM backend. The mapping below survives that check; §5.6 records what reading the implementation added, including one place where the two designs independently reached the same rule and several where Hyperloom is ahead.

### 5.1 What it is

Hyperloom runs a closed loop — profile, analyse, plan, optimize, validate — over a user-supplied workload and configuration. It orchestrates five components that are each independently released and independently useful:

- **TraceLens** — a Python SDK that turns GPU trace files into hierarchical performance breakdowns, roofline estimates and a ranked list of optimization opportunities. A TraceLens-Agent layer proposes a plan and checks that accepted fixes actually removed the bottleneck without regressions.
- **Magpie** — the profiling and benchmarking backbone. It runs the workload, produces traces, and emits structured JSON for everything downstream; it evaluates kernel correctness and performance across environments.
- **IntelliKit** — a toolbox of low-level profiling and validation utilities with clean Python APIs and MCP server support, consumed by the other components rather than by the orchestrator.
- **GEAK** — the kernel optimization agent, spanning several kernel languages, able to work against a live SGLang or vLLM stack, with a hierarchical agent structure and cross-session memory.
- **Arbor** — a tree-structured search layer for long-horizon campaigns over a curated knowledge base.

Separately, and deliberately **outside** the optimization loop, **AgentKernelArena** provides an isolated, reproducible harness for A/B-testing different agents on the same kernel tasks under one shared scoring pipeline.

Delivery is as an installable skill driven from Claude Code, Cursor or Codex, invoked in natural language, with an explicit support matrix: named GPUs, a pinned ROCm version, minimum SGLang and vLLM versions, and a fixed set of kernel languages.

### 5.2 Mapping to our stack

| Hyperloom component | Intel equivalent | State |
| --- | --- | --- |
| GEAK (kernel agent) | **Xe-Forge** | Exists, published, competitive |
| Arbor (campaign search) | Xe-Forge trial tree | Partial — no long-horizon campaign layer |
| — (kernel library / fusion) | **Xe-Fuse** | External sibling project, not in this repo; no direct AMD analogue in the loop |
| TraceLens (trace → ranked opportunities) | **Orbit §18 analysis** | Built (v10) — `orbit/analysis/` catalog, regions, roofline |
| Magpie (measurement + benchmarking backbone) | `orbit/bench/` | Partial (v10) — a library, not standalone; see §5.4 and gap in §24 Tier C |
| IntelliKit (low-level profiling toolbox) | `orbit/profiling/` (unitrace, VTune, interception) over `core/profiler.py` | Built (v10), packaged inside Orbit |
| Orchestrator skill | Orbit CLI + agent layer | Built (v10) — 22 subcommands; `optimize --apply` runs the closed loop (G1, closed §24) |
| AgentKernelArena (agent A/B) | `orbit/arena.py` + `xe-orbit arena` | Built (v10) — isolated per-pair workspaces, resumable, held-out gap honest about unmeasured |

### 5.3 Three structural lessons worth adopting

**1. Decompose into separately installable, separately useful components with JSON contracts between them.** This is the important one, and it is what makes each step testable in isolation (§16). A monolith with an internal call graph cannot be validated stage by stage; five components exchanging structured artifacts can. Our plan already produces per-stage artifacts (§22) — the change is to treat the boundaries as *product* boundaries, each with a published schema, its own CLI and its own tests, not merely as files on disk.

**2. Keep the agent-comparison harness outside the optimization loop.** Mixing "which agent is better" into "make this workload faster" corrupts both. A separate arena lets you A/B a model, prompt, knowledge-base revision or engine under one scoring pipeline, and it is what turns "Claude vs DSPy" — and any engine added after them — from an opinion into a number — directly relevant to the cost-effectiveness case this work has to make internally.

**3. Publish an explicit support matrix from day one.** Named GPUs, driver and oneAPI versions, minimum vLLM/SGLang versions, supported kernel languages, Python version. It is a credibility artifact, it sets expectations before someone files an issue, and it forces the version pinning that §12.9 requires anyway.

### 5.4 The two gaps this creates in our plan

**A measurement backbone as a standalone component.** Our plan currently spreads measurement across the adapters. Split it out: `orbit-bench`, a component that runs a workload or a kernel, produces traces, and emits structured JSON — usable on its own by someone who wants nothing else from the project. Everything downstream consumes its output rather than re-deriving it. This is also what makes replay testing possible (§16.3). AI-bench is the natural seed rather than a from-scratch build — it is already a git dependency of Xe-Forge and already supplies the kernel-side timing the executor uses.

**Status (v10, closed since):** `orbit/bench/` is the library and `orbit-bench` is the console script — `run` emits the §17-grade JSON document (samples, CI95, MDE, declared discarded warmup, `valid`/`decision_grade` flags), `compare` decides with `stats.compare` and maps the four verdicts to exit codes. It imports only the stdlib and Orbit's own stats/models, and runs without torch installed — usable by someone who wants nothing else from the project, as specified.

**An agent arena.** `orbit-arena`, explicitly outside the loop: same task set, same scoring, isolated workspace per run, resumable, with held-out shapes to measure the generalization gap. Scheduled after v0.1 — but designed for now, because the task-definition format it needs is the same `KernelBundle` + `spec.yaml` the pipeline already produces (§8, §12.2). Getting that format right once serves both.

### 5.5 Where we should hold ourselves to a higher standard

Not as criticism of anyone else's system — as the specific ground on which this work should be defensible:

- **Statistical honesty.** Time-boxed autonomous runs that report an expected gain are easy to produce and hard to trust. Interleaved runs, confidence intervals, a declared minimum detectable effect, and `INCONCLUSIVE` as a real outcome (§17) are cheap to implement and are what makes a number survive scrutiny.
- **Extraction verification.** Proving the extracted kernel is the kernel that ran — mutation check, isolated import, specialization match (§12.12) — rather than assuming it.
- **Multi-profile acceptance.** A gain at one serving configuration is not a gain (§14).
- **Negative results as output.** `NO_ACTION` and `INCONCLUSIVE` reported with the reasoning intact.

The differentiation is not "more autonomous." It is "the number is true."

### 5.6 What reading the 1.0.0 source changed (v9)

**One rule arrived at independently, which is the most reassuring finding.** Hyperloom's kernel path states it outright: *"Kernel work in Hyperloom is not handled by an LLM agent. Every kernel `REQUEST` emitted by orchestration is intercepted inline by the Coordinator and routed to a registered Python handler. No LLM turn is consumed."* That is §3's rule — never call an LLM for a task with a deterministic answer — reached from the other direction, by a team with production pressure and a different accelerator. It is the strongest available evidence that the deterministic/agent boundary is a real engineering constraint rather than this plan's stylistic preference, and it should make us less willing to relax it under schedule pressure, not more.

The same convergence produced §17.6: Hyperloom separates a summable baseline-referenced `gain_pct` from an explicitly unsummable `local_gain_pct`, flags `chain_continuous` where a step's finishing throughput was never recorded, and headlines a `cumulative_gain_pct_validated` taken from a measured end state. Xe-Orbit had rigorous per-change statistics and nothing at all on composition; that gap is now closed.

**Three places where Hyperloom is genuinely ahead, in descending order of how much it matters here:**

1. **Enablement — making the workload run at all, before optimizing it.** Hyperloom has a six-rung escalation ladder (§ *Enablement escalation ladder*): diagnose the capability gap, then climb from serve-flag wire-up, to an in-tree source patch, to an attempt-scoped runtime in an isolated venv, to source localization, to an off-loop compiled build on a dedicated single-slot lane so a long compile never blocks the tick loop. Crucially, a build does not earn KEEP on artifact verification — a **runnable gate** boots the model with the built runtime and re-runs the accuracy eval before anything counts. Xe-Orbit has no analogue and demonstrably needs one: on Wildcat Lake, `GRAPH_CAPTURE` was unavailable (`No valid triton configs`, `Internal Triton ZEBIN codegen error`) and the pipeline correctly reported `INSUFFICIENT SAMPLES` — honest, and the end of the road. A diagnosis-then-climb ladder is the difference between reporting that a lever is unavailable and making it available. *Status (v10): the bottom of the ladder exists — `orbit/enablement.py` diagnoses deterministically, enforces the runnable gate (boot alone is never KEEP), and is wired into `BenchRunner.measure`; the climb (rungs 3–5) is the v0.2 headline (§24 Tier C).*

2. **A phase machine with a policy gate.** `PRELUDE → FRAMEWORK_AGENT → EXPLORE → KERNEL_AGENT → SWEEP → CLOSE`, with a per-phase action allowlist and a `PolicyGate` enforcing path sandboxing, resource leases, phase ordering, data dependencies and single-writer rules *before* an intent mutates state. Xe-Orbit's stages are a pipeline, not a machine with invariants; every write we make is trusted because the code path is short, which stops being true as soon as an agent is proposing the writes. *Status (v10): the minimal gate exists — `orbit/policy.py` enforces the action allowlist, the sandbox invariants and single-writer before the loop mutates state (§24 Tier C); phase ordering, resource leases and data dependencies remain the deferred phase-machine half.*

3. **A stall gate on repeated identical work.** A novelty ledger treats a repeat of the same attempt — same component, ref, arch and command — as a stall and reverts, while a genuinely novel attempt or a timeout advances. Cheap, and the difference between a loop that spends its budget and one that burns it.

**How Hyperloom extracts kernels — it doesn't, and that is the sharpest difference between the two systems.** Its kernel path is: resolve the symbol to a file and line, classify the source type, decide *dispatchability*, then hand the **path** to GEAK or Forge, which edits the source **in place in the installed framework tree**; `integrate` then patches, re-baselines, and KEEPs or REVERTs. There is no standalone bundle, no closure, and no build-in-isolation anywhere in it. In Xe-Orbit's terms the whole system operates at **E3**, and §12's E1/E2 rungs have no counterpart. That is a defensible choice on ROCm, where the framework is commonly installed from source and the tree that ran *is* the tree on disk — and it is the reason their resolver "self-heals across file moves/renames and version drift", because it resolves against the installed source rather than a separate clone. It does not transfer to Intel, where every kernel wheel ships a compiled `.so` with its sources in another repository (§11.2), which is exactly why we clone and exactly why we hit version skew (§12.10).

So on extraction we do something they do not attempt, and the E2 verification of §12.12 has no analogue there. Where they are well ahead is **triage** — deciding what is not worth attempting at all. Their patchability gate is a long list of specific rejections: vendor binary, vendor dispatch wrapper, torch dispatch shim with no kernel body, runtime-generated Inductor cache entry, `aten::*` backed by a vendor library, source not under a reusable framework root, and a host launcher whose device code lives in a sibling file. One of those rejections states our own §12.5 rule almost word for word — *"A bare launch API has no kernel body to rewrite, which is a different situation from a kernel whose source we merely failed to locate. Say so rather than sending a reader looking for a file that cannot exist."* That is the memcpy finding, reached independently. Their **vendor operator playbook** goes one step further than our E4: a closed-source operator with no rewritable source can still carry a curated, validated set of launch-config knobs worth tuning, which is `LIBRARY_CONFIG` with an actual playbook behind it rather than a category name.

**Two details of their resolver worth adopting outright — implemented at one layer, and v10 found the claim stopped there.** First, their source resolution is a *recorded* tier — `build_graph` / `symbol_index` / `name_match` / `agent` / `unresolved` — persisted per kernel, because a path alone cannot be reviewed: "we found `Indexing.cpp`" is a different claim depending on which tier said so. Second, and sharper: **confidence is `None` for the deterministic tiers, "which are either right or silent"**. Xe-Orbit was attaching 0.85 to an exact symbol-index hit, which made it sort alongside an agent's self-reported 0.85 — a category error, since the first is a lookup that matched and the second is a model's opinion about its own reliability. A float now means "someone estimated this", and an exact hit reports `exact`. Where an agent overrides a deterministic tier, the tier and path it replaced are kept, so the override stays reviewable and reversible.

**v10 correction, since closed.** At audit time all of that was true of `SourceLocation` (`orbit/models.py`: `ResolutionMethod` with exactly the five tiers, `confidence: float | None`, `describe_confidence() → "exact"`, `previous_file`/`previous_method`) — and none of it reached the artifact a reader actually sees: `analysis/catalog.py` copied only a plain-float `provenance_confidence` onto `KernelRecord`, so `catalog.json` and `xe-orbit inspect` carried no tier, and the float-vs-exact category error this paragraph describes as fixed was still live at the catalog layer. A claim of "implemented" that stops one layer short of the persisted artifact is precisely the wrong-question-answered-confidently failure this document keeps finding elsewhere, which is why the miss is recorded here rather than quietly patched. Gap G2 (§24) closed it: the record now carries `resolution_method` and the nullable confidence, the pattern resolvers stamp `NAME_MATCH`, ranking goes through `confidence_factor` (deterministic → 1.0), and every rendering says `exact`, a float, or `—` — never a number nobody estimated.

**Two places where the plan should hold its position rather than converge.** Hyperloom's `cumulative_gain_pct` is a running total that explicitly includes unattributed drift, and its `gain_method` admits a `local_gain_projected` case where a missing measurement is projected onto the chain. §17.6 refuses both: an unmeasured stack reports `NOT ESTABLISHED` rather than a projection, and drift is surfaced as its own figure rather than absorbed into a total. Similarly, `INCONCLUSIVE` as a first-class outcome with a declared MDE (§17) has no counterpart there. These are the specific grounds on which "the number is true" is a real differentiator rather than a slogan — and they are cheap, which is the argument for keeping them.

---

## 6. v0.1 scope

In scope:

- Linux, Intel XPU
- PyTorch eager and `torch.compile`
- Framework adapter layer with two adapters shipping together: `GenericTorchAdapter` (Tier 0) and `VLLMAdapter` (Tier 1) — see §10
- vLLM-XPU as the **primary** target workload (see §22); SGLang follows in v0.2 as the portability test, not as an afterthought
- `torch.profiler` + unitrace (Level Zero) from the start
- Kernel catalog, provenance, shape distribution, input capture
- Kernel extraction levels E1–E3 for Triton; E3/E4 for SYCL and library kernels (§12)
- Spec/Model emission → Xe-Forge → patch-back → e2e compare
- Language backends for Triton and SYCL, both exercised in CI (§11); sycl-tla via the existing tile search
- Serving profile matrix: several configurations per model (Qwen-class first), with per-profile acceptance (§14)
- `orbit_mini` reference workload and `xe-orbit selftest` (§15)
- Versioned artifact schemas, `--replay` on every stage, T0/T1 CI tiers (§16)
- A published support matrix: device SKUs, driver/oneAPI versions, torch/IPEX/Triton ranges, minimum vLLM version, supported kernel languages (§5.3)
- Local execution only
- Claude Code as the repository agent for the hard-to-automate steps — the `RepoAgent` protocol and a Claude implementation now exist in `orbit/agents/` (v10); Xe-Forge's own Claude engine still only generates a workspace and optionally spawns `claude -p` without waiting for a result, and there is no Codex integration today (§9.9)

Explicitly out of scope for v0.1:

- Slurm integration **[CHANGED]** — running inside an interactive Slurm allocation is just local execution. Keep the executor interface thin enough that a batch backend can be added later, but build only `LocalExecutor`. No job arrays, no queue polling, no distributed campaign.
- Autonomous planner and campaign search
- MCP server, web UI
- Cross-run learned memory
- AMD/NVIDIA backends
- SGLang
- Automatic framework-wide rewrites

---

## 7. Data model

**Status (v10): implemented in `orbit/models.py`, with one systematic difference.** The models are pydantic `BaseModel` artifacts — every artifact carries `schema_version` (currently `"1.1"`), which is what §16.2's committed JSON schemas and compatibility checks are generated from — not the stdlib dataclasses sketched below; the sketches keep dataclass syntax for brevity. Beyond the sketches, `KernelRecord` carries the derived ranking fields (`gpu_time_share`, `max_e2e_gain`, `roofline_headroom`, `extraction_tractability`, `priority`, `skip_reason`), `Provider.RUNTIME` exists as its own provider (the §12.5 memcpy finding, landed), and `BuildRecipe.instantiation` records concrete template arguments (§11.5 item 4). Since the audit, `KernelRecord` also carries `resolution_method` and a nullable `provenance_confidence` mirroring `SourceLocation` — the schema 1.1 change that closed gap G2 (§5.6, §24).

### 7.1 Workload

```python
@dataclass
class WorkloadSpec:
    command: list[str]
    cwd: Path
    env: dict[str, str]
    framework: str | None  # "pytorch" | "vllm"
    warmup_iterations: int
    repetitions: int  # >= 5 for any accept/reject decision
```

### 7.2 Kernel

```python
@dataclass
class KernelRecord:
    id: str
    runtime_name: str
    demangled_name: str | None
    framework_op: str | None
    graph_node: str | None
    provider: str  # inductor | triton | onednn | sycl | ipex | custom | unknown
    language: str | None  # triton | sycl | sycl_tla | cpp | opaque
    build_system: str | None  # cmake | setuptools | jit | prebuilt
    aot: bool | None  # AOT device build vs SPIR-V JIT
    source_file: Path | None
    source_symbol: str | None
    dispatch_chain: list[str]  # why this kernel ran                [new]
    calls: int
    total_time_us: float
    avg_time_us: float
    shapes: list[ShapeObservation]
    actions_available: list[ActionType]  # [CHANGED] replaces `editable: bool`
    provenance_confidence: float
    extraction_level: str | None  # E0..E4, see §12              [new]
    bundle: Path | None  # extracted KernelBundle      [new]
    captured_inputs: Path | None
```

**[CHANGED] `editable: bool` is gone.** A binary flag makes the framework declare the hottest kernel in a transformer (the oneDNN GEMM at ~40% of GPU time) permanently unactionable, which is exactly the kernel Xe-Fuse targets. Replace it with the set of actions that apply. An opaque library GEMM has `[REGION_FUSION, BACKEND_CHANGE, LAYOUT_CHANGE, LIBRARY_CONFIG]` even though it has no editable source.

### 7.3 Region **[CHANGED — new]**

The optimization unit is not always a kernel. The largest wins in inference come from eliminating kernels, not speeding them up.

```python
@dataclass
class RegionRecord:
    id: str
    kernel_ids: list[str]
    aten_ops: list[str]
    producer_consumer_edges: list[tuple[str, str]]
    intermediate_tensors: list[TensorInfo]  # candidates for elimination
    combined_time_us: float
    fusion_pattern: str | None  # e.g. "gemm+rmsnorm+swiglu"
    actions_available: list[ActionType]
```

`RegionRecord` is what gets handed to Xe-Fuse: "these three kernels become one sycl-tla kernel with a fused epilogue." A `KernelRecord`-only model cannot express a many-to-one replacement, which is why v1's `KERNEL_FUSION` action had no executor behind it.

### 7.4 Measurement **[CHANGED]**

```python
@dataclass
class MetricEstimate:
    mean: float
    stdev: float
    n: int
    ci95_low: float
    ci95_high: float


@dataclass
class WorkloadMeasurement:
    wall_time: MetricEstimate
    throughput: MetricEstimate | None
    ttft_ms: MetricEstimate | None
    tpot_ms: MetricEstimate | None
    gpu_busy_percent: float
    launch_gap_total_us: float  # from unitrace
    host_bound_fraction: float
    kernels: list[KernelRecord]
    regions: list[RegionRecord]
    minimum_detectable_effect: float  # derived from observed variance
    frequency_locked: bool
    clock_samples: list[float]
```

Single-value measurements are not accepted anywhere in the decision path. See §17.

### 7.5 Captured inputs **[CHANGED — new, high priority]**

```python
@dataclass
class CapturedInvocation:
    kernel_id: str
    call_index: int
    tensors: list[Path]  # saved real tensors (.pt), strides and layout preserved
    scalars: dict[str, Any]
    dtype_map: dict[str, str]
    output_reference: Path  # baseline output for correctness comparison
```

v1 buried "reconstruct representative inputs" inside PR 8 and implicitly assigned it to an LLM. That is the highest-risk decision in the plan. Randomly generated inputs silently break masked attention, KV-cache layouts, paged block tables, quantization scales, and any non-contiguous stride pattern. The result is a kernel that benchmarks faster and is wrong, or that is correct on synthetic data and wrong in the model.

**Capture real tensors from the running workload.** Xe-Forge's executor already probes the model for a `get_example_inputs(input_shapes, device)` hook before synthesizing inputs from shape and dtype — though no in-tree kernel implements it yet, so §9.4 supplies the first disk-loading implementation. This turns input construction from an AI reasoning problem into a file-read.

### 7.6 Actions

```text
KERNEL_REWRITE        -> Xe-Forge
KERNEL_AUTOTUNE       -> Xe-Forge (autotuning stage)
KERNEL_TILE_SEARCH    -> Xe-Forge (--tile-tune, SYCL/CUTLASS)
REGION_FUSION         -> Xe-Fuse                        [new]
LAYOUT_CHANGE         -> deterministic executor         [new]
BACKEND_CHANGE        -> deterministic executor
LIBRARY_CONFIG        -> deterministic executor
CONFIG_CHANGE         -> deterministic executor
COMPILER_OPTION       -> deterministic executor
HOST_OPTIMIZATION     -> repo agent                     [new priority]
GRAPH_CAPTURE         -> deterministic executor         [new]
PROFILE_MORE          -> profiling subsystem
NO_ACTION             -> terminal, with justification   [new]
```

`NO_ACTION` is a first-class result. A framework that cannot credibly say "there is no headroom here" is not a measurement instrument.

---

## 8. Handoff to Xe-Forge **[CHANGED]**

**v1 said:** define `KernelOptimizationTask` / `KernelOptimizationResult` and a new `KernelOptimizer` library API.

**v2 says, corrected at v8 and again at v10:** the *file* contract already exists — a kernel `.py`, a YAML spec, and a `*_pytorch.py` reference implementation — and the *directory-based library entry point*, new thin-wrapper work at v8, has since landed in Orbit (see the status note at the end of this section). Orbit emits the layout below.

```text
.orbit/runs/<run-id>/candidates/<kernel-id>/
    kernel.py            # extracted kernel (or in-situ harness) + Model wrapper
    kernel_pytorch.py    # reference implementation from the aten op
    spec.yaml            # inputs, inits, weighted bench variants
    bundle/              # extraction closure, see §12
    inputs/              # captured real tensors
    reference_out.pt     # baseline output for correctness
```

Naming, as the repository actually resolves it: the CLI takes the kernel file plus an explicit `--spec <path>` (and `XeForgePipeline.optimize()` / `optimize_file()` both accept `spec_path=`), so a file named `spec.yaml` works when passed explicitly — the in-tree convention is instead a sibling `<KernelName>.yaml`, as in `test_kernels/1_FlashAttention_Fwd.py` plus `1_FlashAttention_Fwd.yaml`. The PyTorch reference, by contrast, *is* resolved by name substitution — `<stem>_pytorch.py` — so `kernel_pytorch.py` sitting next to `kernel.py` is load-bearing rather than illustrative. No `*_pytorch.py` file exists in the repository today; Orbit's emitted candidates would be the first.

Shape distribution maps onto Xe-Forge's variant mechanism, which already supports arbitrary `bench-gpu-N` families; the `weight:` field is the one addition (§9.1), and today the spec loader silently drops unknown keys, so it would neither work nor raise an error:

```yaml
inputs:
  hidden:   { shape: [M, H], dtype: bfloat16 }

bench-gpu:                       # weight 0.61
  - params: [hidden]
    dims: { M: 4096, H: 8192 }
    weight: 0.61
    flop: "..."
bench-gpu-1:                     # weight 0.23
  - params: [hidden]
    dims: { M: 2048, H: 8192 }
    weight: 0.23
bench-gpu-2:                     # weight 0.09
  - params: [hidden]
    dims: { M: 1024, H: 8192 }
    weight: 0.09
```

The thin library API is **new, but genuinely thin** — a wrapper to add over the existing `XeForgePipeline.optimize()` / `optimize_file()` and `create_engine()`:

```python
from xe_forge import (
    optimize_kernel_dir,
)  # today: from xe_forge.orbit.optimize — see the v10 status note below

result = optimize_kernel_dir(
    path=candidate_dir,
    engine="claude",  # or "dspy" — the only two engines today
    device="xpu",
    objective="weighted_latency",  # new; §9.1
    budget=Budget(trials=10),  # new; today TrialConfig.max_trials
)
```

One caveat the wrapper has to resolve: today `engine="claude"` is fire-and-forget. `ClaudeEngine` generates a Claude Code workspace, optionally spawns `claude -p`, and returns success immediately with no measured speedup. `optimize_kernel_dir(engine="claude")` therefore needs a synchronous result path — block on the run and collect the trial result — which is part of §9.9, not existing behaviour.

**Status (v10): the wrapper exists — as `xe_forge.orbit.optimize.optimize_kernel_dir`, not the top-level `from xe_forge import` shown above** (promotion to the core namespace is deliberate future work, once the weighted objective gives core something to promote). What landed: candidate resolution with honest refusals (a missing or still-stub `kernel_pytorch.py` is named, not ignored), `required_speedup` enforced caller-side (§9.2), a `dry_run` path for CPU-only CI, and a Claude path that reports `success=False, synchronous=False` with instructions to run `xe-orbit compare` — refusing to launder the fire-and-forget engine's unconditional success. `objective="weighted_latency"` now threads through to the pipeline's weighted objective (§9.1, closed): the emitted `weight:` keys are parsed, the family is scored with the per-variant no-regression constraint, and the pipeline enforces `required_speedup` itself.

This still deletes most of v1's PR 7 and reduces the extraction PR from "reconstruct a standalone kernel" to "fill a template whose shape is already defined."

---

## 9. Changes required inside Xe-Forge

These are the Xe-Forge-side work items implied by the above. They are small and can proceed in parallel with Orbit PRs 1–3.

**9.1 Weighted multi-variant objective.**
Today `--variant` selects one benchmark configuration. Orbit needs `score(C) = Σ wᵢ · latency(C, variantᵢ)` across all variants, plus a hard **no-regression constraint on every variant**. A candidate tuned for the dominant shape that collapses on the tail distribution will lose end-to-end while winning the microbenchmark. Add `weight:` to the variant schema and a `--objective weighted` mode.
*Status (v10, closed since): implemented. `VariantSpec.weight` parses (malformed weights raise), `KernelSpec.weighted_family` walks the family, `core/weighted.py` scores it with the hard per-variant no-regression constraint, and `--objective weighted` / `pipeline.optimize(objective="weighted")` gates acceptance on the result — the per-variant table persists on `OptimizationResult.weighted`, never a single number (§14.3). Tested in `tests/test_weighted.py` and the spec-loader suite.*

**9.2 Accept threshold must be caller-supplied.**
`TARGET_SPEEDUP=2.0` is a reasonable default for standalone kernel research and wrong for e2e work. In fact it is currently loaded into config and never read by anything — so this item is "wire an accept threshold up, caller-supplied", not "change a default". A kernel owning 30% of GPU time needs only ~1.15x to be worth accepting; a kernel owning 3% needs an implausible speedup to matter at all. Orbit computes the required speedup from the Amdahl ceiling (§18) and passes it in.
*Status (v10): done, Orbit-side — `orbit/optimize/kernel_dir.py` enforces `required_speedup` at the call site; `target_speedup` in core config remains inert.*

**9.3 Tolerance tightening.**
Defaults are `rtol=0.01, atol=1e-5`. Per-kernel that is fine. Across 32 decoder layers it compounds into token divergence. Orbit emits tighter per-kernel tolerances derived from the e2e budget, and **never** passes `--no-correctness`. The hook already exists: `VariantSpec` carries per-variant `rtol`/`atol` and the pipeline resolves spec values over config defaults, so Orbit only has to emit them in the spec it generates.
*Status (v10): the hook is unchanged (resolution is caller-arg > spec > config, `pipeline.py:191-204`); nothing further was needed on the core side.*

**9.4 `get_example_inputs` from disk.**
Support (or document) loading captured tensors, including preserved strides and non-contiguous layouts, rather than regenerating from shape+dtype. The executor-side probe already exists — it calls `get_example_inputs(input_shapes, device)` when the model defines it — but no in-tree kernel implements the hook, so this item supplies the first real implementation.
*Status (v10): done — two real implementations exist: `orbit_mini`'s deliberately non-contiguous generator, and the emitted E3 harness that loads captured tensors from disk (`orbit/extract/bundle.py`, loader in `orbit/capture/capture.py`).*

**9.5 Knowledge base extension, not fork.** **[CHANGED]**
v1 created `knowledge/frameworks/pytorch/*.md` in a new repo while leaving hardware knowledge in Xe-Forge. Those diverge within a quarter. Extend the existing `knowledge_base/` YAML with `common/framework_pytorch.yaml`, `common/framework_vllm.yaml`, `common/provenance_patterns.yaml` and `common/extraction_patterns.yaml`, delivered through the same stage-scoped mechanism and consumed by both layers. One KB, versioned, reusable across projects and stakeholders.

Two properties of the existing loader constrain where these files go. It collects `common/` → `<dsl>/common/` → `<dsl>/<device>/`, so a flat file dropped at the knowledge-base root is **silently not loaded** — framework knowledge is DSL-agnostic and belongs under `common/`. The stage-scoped delivery (`format_for_stage()`) is real and is exactly what these files plug into. The knowledge base is also disabled by default (`KnowledgeConfig.enabled = False`); Orbit enables it for its own runs.

*Status (v10): partially done, and amended.* The three framework files exist and are consumed — by Orbit's own reader (`orbit/languages/sources.py`), not the legacy loader; their schema is the adapter/`kernel_sources` shape of §10.6, not the legacy `patterns:`/`constraints:` shape. `provenance_patterns.yaml` and `extraction_patterns.yaml` were never created, and **v10 drops them from this item**: provenance resolution is logic, and lives in `orbit/provenance/resolvers.py`; `orbit/knowledge.py` already writes measured facts into `common/`. One trap found in passing: with `KnowledgeConfig.enabled=True`, the legacy loader globs `common/*.yaml` and loads the framework files as **silent no-ops** — no `patterns:`/`constraints:` keys, absent even from `kb.skipped`. Exactly the shape of failure this document keeps finding — and closed since: the loader now records every framework file in `kb.skipped` with a reason naming `xe_forge.orbit` as its consumer, so the skip is visible instead of indistinguishable from a clean load.

**9.6 Xe-Fuse as a second optimizer backend.**
Xe-Fuse is an **external sibling project** — it appears nowhere in this repository today: not a dependency, not a submodule, not an import. Register it behind the engine seam Xe-Forge already uses (`engines.base.BaseEngine`, selected through `create_engine`) so `REGION_FUSION` has an executor, which includes adding it as an optional dependency. Xe-Fuse's model presets (LLaMA 2/3, Gemma 2, Mistral, Qwen 2.5, Phi-3), as published in that project, already encode H, H_kv, FFN dim, activation, RoPE — Orbit's job is to detect which preset the observed region matches and hand it over. Its tile auto-selector and roofline data supply the headroom estimates in §18.
*Status (v10, closed since): the executor is real. `orbit/optimize/xe_fuse_executor.py` drives Xe-Fuse as it actually ships — a checkout with a per-shape kernel generator compiled by icpx, not a pip package — locating Xe-Fuse and sycl-tla under the §11.2 source roots (env overrides authoritative in both directions), mapping region patterns to presets as data, and running generate → compile → benchmark with every stage failure named. `xe_fuse_available()` now recognizes the checkout, so `default_executor` prefers Xe-Fuse on machines that have it and degrades to authoring elsewhere. Measured live on the traced Qwen decode regions: see §13.4.*

**9.7 A SYCL kernel contract.**
The `Model` contract assumes a Python kernel launch. SYCL needs a parallel one: source translation unit, build recipe, and a thin `torch.library`/pybind harness driven by the same spec, so correctness and weighted benchmarking work identically across languages. Standalone SYCL sources already compile and run through `core/sycl_executor.py` and ai_bench's `SYCLCompiler`; what is missing is the harness that lets a *dispatcher-registered* SYCL op be driven from a spec. Extend the existing `knowledge_base/sycl/xpu/` coverage — which already carries `cutlass_sycl_framework.yaml`, `xetla_patterns.yaml` and the model/workload shape tables — with the remaining Xe SYCL depth: sub-group sizes per generation, DPAS/XMX constraints, 2D block loads and prefetch, SLM sizing, GRF modes, work-group shaping (§11.9) — and make compiler flags a proposable candidate axis, not just code.
*Status (v10, closed since): `orbit/patch/sycl_harness.py` supplies the missing half — `render_dispatcher_model` emits a self-contained `Model` driving `torch.ops.<ns>.<op>` (op resolved eagerly at construction, with every load attempt named on failure), and `emit_dispatcher_candidate` writes the candidate layout `optimize_kernel_dir` resolves. The knowledge-depth extension for `knowledge_base/sycl/xpu/` remains the open remainder of this item.*

**9.8 Accept an in-situ `Model`.**
Xe-Forge assumes `Model.forward` launches the kernel directly. For extraction level E3 (§12.3), `forward` instead calls into the installed framework. Confirm nothing in the analyzer or stage pipeline breaks when the kernel source and the `Model` live in different files, and when the kernel is reached through a framework dispatch rather than a direct launch. The executor resolves `Model` by duck typing — a module-level attribute, constructed from spec `init_args`, `get_init_inputs()`, or no-arg, in that order — so nothing structurally requires the kernel source and the `Model` to share a file. This item is a verification pass, not a redesign.
*Status (v10): verified by construction — `orbit/extract/bundle.py` emits exactly such in-situ `Model`s at E3, and the bundle rig runs them.*

**9.9 A directory-level `optimize_kernel_dir` wrapper.**
The entry point §8 hands work to did not exist when this item was written. Add it as a thin wrapper over `XeForgePipeline.optimize()` / `optimize_file()` plus `create_engine()`: resolve `kernel.py`, the sibling `kernel_pytorch.py` reference and the spec path; thread through the weighted objective (§9.1) and the caller-supplied accept threshold (§9.2); and give `engine="claude"` a synchronous result path, since `ClaudeEngine` today generates a workspace and returns without measuring anything.
*Status (v10, closed since): the wrapper exists (§8), and `ClaudeEngine` now has the synchronous path — `EngineConfig.synchronous` blocks on the session, reads the edited kernel back from the workspace's documented output location, and returns it with `success=False` and an explicit unmeasured note, because §19 forbids success on generated reasoning alone. The caller measures; the async default is unchanged.*

---

## 10. Framework support model — one adapter per framework **[NEW]**

### 10.1 The requirement

vLLM is the first target, not a special one. The same pipeline has to work for SGLang, TGI, IPEX-LLM, OpenVINO GenAI, internal serving harnesses, and whatever the next one is. The way that fails is subtle: vLLM is built first, its assumptions leak into the analyzer, the measurement layer and the patch logic, and the second framework needs a core rewrite.

Two hard rules:

- **The core imports no framework.** Nothing under `orbit/model/`, `orbit/analysis/`, `orbit/extract/` or `orbit/patch/` may import vllm, sglang, or any serving package. Enforced by a test that scans imports, not by convention.
- **Every framework is reached through the same protocol**, one adapter per framework, and every adapter passes the same conformance suite (§10.7). That is the one-to-one guarantee: adding a framework is a bounded, testable unit of work, not a negotiation with the core.

### 10.2 Two tiers, so unknown frameworks still work

| Tier | What it is | What you get |
| --- | --- | --- |
| **Tier 0** — `GenericTorchAdapter` | No framework-specific code at all. Works on any torch-based workload. | Kernel discovery, provenance, input capture, extraction E2/E3, wall-clock end-to-end, config actions limited to environment variables |
| **Tier 1** — named adapter | ~one class + one knowledge file | Adds serving metrics (TTFT/TPOT/throughput), determinism control, in-situ harness construction, framework-aware patch points, a real config action space, quality gates |

An unfamiliar framework **degrades to Tier 0 rather than failing**. That matters for adoption: someone can point Orbit at an internal harness on day one and still get a kernel catalog and a wall-clock delta. The adapter buys precision, not basic function.

### 10.3 The protocol

```python
@dataclass
class FrameworkCapabilities:
    metrics: set[str]  # {"wall_time", "throughput", "ttft", "tpot"}
    can_reset_state: bool
    can_pin_batching: bool
    can_disable_prefix_cache: bool
    can_construct_single_layer: bool  # E3 harness
    patchable_layers: set[str]


class FrameworkAdapter(Protocol):
    name: str
    capabilities: FrameworkCapabilities

    # identity and lifecycle
    def detect(self, spec: WorkloadSpec) -> bool: ...
    def versions(self) -> dict[str, str]: ...
    def prepare(self, spec: WorkloadSpec) -> PreparedWorkload: ...
    def launch(self, spec: WorkloadSpec, executor: Executor) -> Handle: ...
    def warmup(self, handle: Handle) -> None: ...
    def teardown(self, handle: Handle) -> None: ...

    # measurement
    def benchmark(self, handle: Handle, load: LoadSpec) -> WorkloadMeasurement: ...
    def metrics_schema(self) -> list[MetricSpec]: ...

    # reproducibility
    def determinism_profile(self) -> DeterminismProfile: ...
    def reset_state(self, handle: Handle) -> None: ...

    # discovery and provenance
    def dispatch_roots(self) -> list[str]: ...
    def provenance_hints(self) -> list[ProvenanceRule]: ...

    # extraction (§12)
    def build_in_situ_harness(self, kernel: KernelRecord, inputs: CapturedInvocation) -> Path: ...
    def patch_points(self, kernel: KernelRecord) -> list[PatchPoint]: ...

    # action space
    def config_axes(self) -> list[ConfigAxis]: ...
    def apply_config(self, spec: WorkloadSpec, config: dict) -> WorkloadSpec: ...

    # correctness
    def quality_gate(self, handle: Handle, prompts: list[str]) -> QualityResult: ...
```

Adapters are resolved by `detect()` and registered through the Python entry-point group `xe_orbit.frameworks`, so internal or proprietary frameworks can plug in **without forking the repository**.

**Status (v10): implemented.** The protocol and `BaseAdapter` live in `orbit/adapters/base.py` (the capabilities model in `orbit/models.py`, with two fields beyond the sketch: `profiles_in_process` and `profile_hook`); `GenericTorchAdapter` and `VLLMAdapter` are registered through `xe_orbit.frameworks` in pyproject and loaded in `orbit/adapters/__init__.py` — a broken third-party adapter is logged, not fatal. The vLLM adapter reads its knowledge from `framework_vllm.yaml` (§10.6) and implements metrics parsing, TTFT/TPOT extraction, `reset_state`, patch points and a token-exactness quality gate. SGLang: knowledge file present, adapter absent — v0.2, as §10.9 schedules.

### 10.4 Capabilities are declared, never assumed

The decision layer reads `capabilities` and adapts. If an adapter cannot report TTFT, the analysis falls back to throughput or wall-clock **and says so in the report**. It never substitutes one metric for another silently, and it never assumes a metric exists because vLLM has it.

Where a capability is missing, the affected action types are removed from the space rather than attempted and failed. An adapter that cannot construct a single layer has no E3 extraction available, so its tangled kernels rank lower on `extraction_tractability` (§18) — the ranking degrades correctly instead of producing candidates that cannot be built.

### 10.5 Determinism is per-framework, and it is what breaks measurement

Each serving framework has its own sources of run-to-run nondeterminism, and they are exactly the ones that make an A/B comparison meaningless:

- prefix / radix / KV-cache reuse across requests
- continuous batching order and scheduler decisions
- chunked-prefill boundary placement
- speculative decoding accept rates
- graph capture and compile warmup state
- request arrival jitter under a load generator

Each adapter declares, in `DeterminismProfile`, which of these it can pin and how. The measurement layer then refuses to emit `ACCEPT` when a non-pinnable source is active and observed variance exceeds the MDE — it emits `INCONCLUSIVE` with the reason named. This is the difference between a framework-portable measurement instrument and a vLLM benchmark script with extra steps.

### 10.6 Per-framework knowledge is data, not code

One file per framework, identical schema, in the existing knowledge base:

```text
knowledge_base/common/framework_pytorch.yaml
knowledge_base/common/framework_vllm.yaml
knowledge_base/common/framework_sglang.yaml
```

Under `common/` because the loader collects `common/` → `<dsl>/common/` → `<dsl>/<device>/`; a flat file at the knowledge-base root is silently ignored.

Each declares: dispatch roots and kernel module globs, backend-selection environment variables, config axes and their legal values, metrics parsing rules, determinism knobs, known opaque providers, candidate patch points, version compatibility ranges, and — added in v9 — `kernel_sources`: where that framework's hand-written kernels live, as a repository URL, the subdirectory holding them, and the dispatcher namespace they register into (§11.2).

`kernel_sources` belongs here rather than in code for the reason the whole section exists: where SGLang keeps its kernels is a fact about SGLang, not logic. Putting it in a Python table would have made "support SGLang's kernels" a code change. It is the part of §10.6 that is now literally true rather than aspirational — the source registry that extraction resolves against is built by reading these files, and adding a framework's kernel tree is adding a YAML file.

The target shape is that **most of a new framework is the YAML file**, with the adapter class supplying only what genuinely needs code — process lifecycle, harness construction, and quality gating.

### 10.7 Conformance suite — the one-to-one guarantee

Every adapter, Tier 0 and Tier 1, passes the same framework-agnostic test set before it ships:

1. `detect()` and `versions()` round-trip on a real installation.
2. Full lifecycle — `prepare → launch → warmup → benchmark → teardown` — on a small model.
3. Reported metrics exactly match declared `capabilities`; no extras, no missing.
4. `reset_state()` demonstrably changes cache-hit behaviour where the capability is claimed.
5. **Null test:** benchmark an unchanged workload against itself; the resulting CI must contain zero. An adapter that reports a difference where none exists cannot be trusted to report one where it does.
6. **Positive control:** inject an artificial slowdown of known magnitude into a hot kernel; the adapter must detect it, and the measured delta must be consistent with the injected one. This validates the entire measurement chain — trace, attribution, statistics — per framework.
7. In-situ harness reproduces the reference output for one known kernel (Tier 1 with `can_construct_single_layer`).
8. At least one patch point round-trips: apply, verify in re-profile, revert.

Tests 5 and 6 are what make a new framework trustworthy rather than hopeful. They are also cheap to run and belong in CI for every adapter.

**Status (v10): implemented and in CI.** `orbit/adapters/conformance.py` runs the suite — including the null test (with a retry for the 5%-of-the-time-by-construction CI flake) and the positive control — against a hardware-free workload; `xe-orbit conformance <adapter>` exposes it, T0 CI runs it for both shipped adapters, and `tests/orbit/test_conformance_rules.py` tests the suite's own asymmetry rules.

### 10.8 Cost target, and what a miss means

A new Tier-1 adapter should be one knowledge file plus one class implementing the protocol, passing conformance, within a bounded effort — roughly a week for someone who knows the framework.

**If SGLang requires a change to the core, the abstraction is wrong.** Treat that as the signal to fix the boundary, not to special-case the framework. Record it, because the second and third adapters are the only real evidence the layering works.

**Status (v10): the test ran, and the boundary held — zero core lines changed.** The
`SGLangAdapter` (Tier 1, 74 tests, runs without SGLang installed like its vLLM
sibling) cost: 0 lines in orbit core, +1 line of pyproject registration, the
knowledge file extended as data (§10.6 working as designed), and the adapter itself.
Two framework-shaped differences stayed inside the adapter where they belong
(store_true config flags needing a presence mode; a smaller honest
`patchable_layers` set with no CustomOp registry), and every fact that could not be
verified against a live install ships marked `confidence: unverified` as
documentation the parser never consumes — the wrong-question failure class, refused
at the schema level.

### 10.9 Roadmap

- **v0.1** — `GenericTorchAdapter` (Tier 0) and `VLLMAdapter` (Tier 1). Both ship together deliberately: building the generic path alongside the specific one is what keeps vLLM assumptions out of the core.
- **v0.2** — `SGLangAdapter`, scheduled **before** any planner work, specifically as the portability test. Its cost in lines changed outside `adapters/` is a reported metric.
- **later** — TGI, IPEX-LLM, OpenVINO GenAI, internal harnesses, out-of-tree adapters via entry points.

---

## 11. Kernel languages: SYCL is not the exception **[NEW]**

### 11.1 The observation

This plan has been reading Triton-heavy, and on Intel that is backwards. Triton is what Inductor *generates*; it is not what most of the hand-written Intel kernel surface is made of. In a real vLLM-XPU or SGLang-XPU decode run, a large share of GPU time sits in SYCL C++ — ATen XPU operators, extension ops, and templated sycl-tla GEMM/attention — with oneDNN and oneMKL taking most of the rest. A Triton-only pipeline optimizes the tail and reports it as a win.

SYCL therefore gets the same first-class treatment as Triton throughout: identity, closure, build, harness, patch-back and verification. Where the two differ, they differ in mechanism, not in status.

### 11.2 Where the kernels actually live

| Source | Language | Registration | Build | Default level |
| --- | --- | --- | --- | --- |
| TorchInductor codegen | Triton | generated, cache-resident | JIT | E2 (with torch pin) |
| `torch-xpu-ops` ATen XPU kernels | **SYCL C++** | `aten::*` on the XPU key | CMake, in `libtorch_xpu` | E2 via build graph, else E3 |
| IPEX custom ops | **SYCL C++** | `torch.ops.torch_ipex.*` | CMake / setuptools ext | E2/E3 |
| vLLM-XPU `csrc` | **SYCL C++** | `torch.ops._C.*` | setuptools ext | E2/E3 |
| vLLM Python kernels | Triton | called from layer code | JIT | E2 |
| `sgl-kernel-xpu` | **SYCL C++** + some Triton | `torch.ops.sgl_kernel.*` | CMake ext | E2/E3 |
| Xe-Fuse (external) / sycl-tla | **templated SYCL C++** | direct or via ext | CMake + template instantiation | E2 (instantiation harness) |
| oneDNN / oneMKL | opaque | library dispatch | prebuilt | E4 |

The middle five rows are the ones v1 had no story for, and they are where the time is. All of them are **editable** — real source, in a known repository, with a known build. They are not oneDNN.

**The obstacle is packaging, not language (v9).** Every one of those rows installs as a wheel containing a compiled shared object, with the sources in a *separate* repository. On a normal machine there is therefore no build database and no source tree, and the "Default level" column above is unreachable — not because the kernel is tangled, but because nobody checked the code out. A resolver that requires a local tree reports E3 forever and calls it a property of the kernel.

So the trees are named as knowledge (`kernel_sources` in the framework YAML, §10.6) and indexed wherever they are checked out, searched under `ORBIT_SYCL_SOURCES` then a small default list. Measured on this repository's stack, that is 1480 symbols from `torch-xpu-ops`, 172 from `sgl-kernel-xpu` and 156 from `vllm-xpu-kernels` — 1808 kernels that resolve to a file and would otherwise have resolved to nothing.

A tree that is absent is reported as **absent**, never folded into the coverage numbers. "We could not find the source" and "this kernel has no source" are different findings, and only the second one is about the kernel; conflating them turns a missing `git clone` into a permanent E4 verdict.

### 11.3 `LanguageBackend`, mirroring the adapter design

Language is a dimension, not a special case, and it gets the same treatment as framework (§10): one backend per language, same protocol, same conformance obligations.

```python
class LanguageBackend(Protocol):
    name: str  # triton | sycl | sycl_tla | cpp | opaque
    cost_profile: CostProfile  # build latency, iteration cost

    def identify(self, event: RuntimeKernelEvent) -> float: ...  # confidence
    def resolve_source(self, event, launch) -> SourceLocation: ...
    def closure(self, source: SourceLocation) -> list[Path]: ...
    def build(self, bundle: KernelBundle) -> BuildResult: ...
    def harness(self, bundle: KernelBundle) -> Path: ...  # Model wrapper
    def patch_points(self, kernel: KernelRecord) -> list[PatchPoint]: ...
    def verify(self, bundle: KernelBundle) -> ExtractionCheck: ...
    def option_axes(self) -> list[CompilerAxis]: ...
```

Xe-Forge already documents its own per-language seam: `docs/DSL.md` is an eleven-step guide for adding a kernel DSL, with Triton as the reference path and `gluon`, `sycl` and `cuda` registered alongside it in `dsl_registry.py`. `LanguageBackend` should extend that seam rather than build a parallel one.

**Status (v10): implemented** in `orbit/languages/` — the protocol plus `CostProfile`/`CompilerAxis`/`BuildResult` in `base.py`, `TritonBackend` (AST closure, import resolution across modules, autotune capture, verify) and `SyclBackend` (demangling with exact length-prefix parsing, template-argument splitting, `compile_commands.json` closure, AOT-target detection, GRF flag axes, device diagnostics), with the §11.2 source registry in `sources.py` (`kernel_sources` from the framework YAMLs, `ORBIT_SYCL_SOURCES`, the §11.4 namespace rule, and `resolve_with_agent` with the answer verified against the indexed trees).

`cost_profile` is not decoration. A Triton iteration is a JIT compile measured in seconds; a SYCL iteration is a rebuild measured in minutes. That difference propagates into budget accounting and into ranking (§11.10), and pretending the two are interchangeable is how an eight-hour budget disappears into three SYCL trials.

### 11.4 SYCL identity and provenance

Triton gives you a Python function; SYCL gives you a mangled symbol. The chain is:

1. Level Zero / unitrace reports the kernel's mangled name at launch.
2. Demangle it. SYCL kernel names encode the functor or lambda type, which carries the enclosing namespace and often the source location.
3. Map the type back to a translation unit and line — via device-code dump, the build's debug info, or the symbol table of the shared object it came from. Where none of those are present (the wheel case, §11.2), the fallback is a symbol index over the checked-out tree: scan the sources once, and look the identifier up.

   **Step 3 is where the deterministic/agent split (§3) gets decided in practice, and it decided against the obvious answer.** Recovering the identifier from a mangled SYCL name looks like string handling, and was written that way. A greedy pattern for Itanium template mangling recovered `GeluErfFunctor` correctly and reduced `IgammaFunctor` to the empty string — because the identifier contains the `I` the pattern keyed on. The regular expression was not parsing C++; it was guessing, and the guess failed silently, which is the exact failure mode §19 exists to refuse.

   The fix has two parts, in this order. The genuinely deterministic piece — Itanium's length-prefixed encoding, where `_ZTS13IgammaFunctor` names its own length — is parsed as the specification defines it, not pattern-matched. That alone took symbol coverage on `torch-xpu-ops` from 83.1% to 100%. What remains after an exact parse is genuinely ambiguous, and goes to a `RepoAgent` (§3, §6) rather than to a cleverer regular expression: the deterministic path keeps its exactness and costs nothing, and the residue gets something that can read the code. The agent's answer is verified before it is believed — the file must exist and must sit inside an indexed tree — and it is off by default, because §15.3 requires that CI never call an LLM.
4. Record whether the kernel was **AOT-compiled** for a specific device or **JIT-compiled from SPIR-V at runtime**. These perform differently and rebuild differently; a bundle that AOT-builds a kernel the workload JITs is not the same kernel.
5. Capture the compiler's device-side diagnostics where available — register pressure, spills, SLM usage — which is the SYCL counterpart to the Triton compiled-kernel metadata in §12.4 and feeds the same verification.
6. **When more than one tree defines the identifier, the namespace decides which.** Identifiers collide across frameworks: `rms_norm_kernel` is defined by both `torch-xpu-ops` and `vllm-xpu-kernels`. A registry that scans trees in declaration order resolved `vllm::rms_norm_kernel` to torch-xpu-ops' `LayerNormKernels.cpp` — the wrong framework's kernel, chosen by list position, reported at full confidence. The demangled name carries the namespace, so this is decidable rather than ambiguous: each tree declares the namespaces it owns in its knowledge file (§10.6), and the collision is only real when no tree claims the namespace. Note how it surfaced — the wrongly chosen file happened not to compile. A collision between two files that both build would have been silent, which is the argument for running §12.10 on every bundle rather than on the ones that look suspicious.

Confidence is graded, not binary. A demangled name that resolves to a unique TU is high confidence; a heavily templated lambda that resolves to several instantiations is not, and the resolver says so rather than picking one.

### 11.5 Closure via the build graph, not the AST

For Triton, closure is an AST walk (§12.6). For SYCL, the build system already knows the answer: `compile_commands.json` gives the exact compile line for the translation unit — every include path, every define, every flag. Closure is:

1. The translation unit containing the kernel.
2. Its transitive header closure, resolved by the preprocessor rather than by guessing.
3. The exact compile command, verbatim.
4. For templated code, the **instantiation**: the concrete template arguments the workload actually used, recovered from the demangled name. A sycl-tla kernel without its tile shape and layout parameters is not a kernel.
5. Link dependencies needed to build a standalone harness.

This is more reliable than the Triton path, not less — the build system is authoritative where AST resolution is inferential. The catch is that a project without a compile-commands database forces E3.

### 11.6 Rebuild economics

SYCL iteration is dominated by compile time, so the plan has to design for it rather than discover it:

- **Single-TU isolation.** Build one translation unit against the installed headers, not the whole repository. This is the difference between a two-minute loop and a forty-minute one.
- **Compiler cache.** ccache or equivalent, warmed once per session, with the cache state recorded in the manifest (a cold-cache trial and a warm-cache trial are not comparable on wall-clock cost).
- **Budget in trials, not hours.** A SYCL candidate costs roughly an order of magnitude more than a Triton candidate; the budget model (later work) must price them separately.
- **Parallel candidate builds** across worktrees, since builds are CPU-bound and the GPU is idle during them — worth noting even though the executor stays local.

### 11.7 Compiler and runtime options are a first-class action space

For SYCL there is a whole class of wins that require no code change at all, and they should be swept deterministically **before** any agent is invoked:

- GRF mode (large vs default register file) — often the single largest lever on Xe, and interacting with occupancy
- required sub-group size, and the generation-dependent choice between 16 and 32
- AOT device target versus SPIR-V JIT
- optimization level and the floating-point contract (fast-math changes numerics, so it is gated by §19, not free)
- SLM allocation and work-group shaping
- for sycl-tla, tile shape and layout — which Xe-Forge's existing tile search already handles

These are `COMPILER_OPTION` actions, they are cheap, they are deterministic, and they are the correct first move on a SYCL kernel. An agent asked to rewrite a kernel that is simply running in the wrong GRF mode will produce an expensive, complicated, worse answer.

### 11.8 Patch-back for SYCL is still P1

This is the part that makes SYCL support practical rather than theoretical: **the operator-override rung (§13) works for SYCL kernels too.** Because `torch-xpu-ops`, IPEX, vLLM-XPU and `sgl-kernel-xpu` all register their kernels as dispatcher ops, an optimized SYCL kernel ships as a small out-of-tree extension, compiled with icpx, that registers an implementation on the XPU key and shadows the original.

No fork of PyTorch. No fork of vLLM or SGLang. No patched build of `libtorch_xpu`. Revert is not importing the module.

P5 — patch the source tree and rebuild — is reserved for kernels that are not dispatch-registered, and it is the rung to avoid, because it drags the framework's entire build into the loop and binds the artifact to one source revision.

### 11.9 What this requires from Xe-Forge

Xe-Forge already accepts `--dsl sycl` (the choices are `triton`, `gluon`, `sycl`, `cuda`), runs SYCL through `core/sycl_executor.py` — a wrapper over ai_bench's `SYCLCompiler`, with AOT target autodetection and a deliberately reduced stage set — and does LLM-driven tile search over CUTLASS SYCL kernels (`--tile-tune`, `core/tile_search/`). The optimizer side exists. What it needs:

- **A SYCL kernel contract** parallel to the Triton `Model` — source TU, build recipe, and a thin `torch.library` or pybind harness that exposes a dispatcher-registered kernel for benchmarking and correctness, driven by the same spec. Standalone SYCL sources already build and run; the dispatcher-op harness is the missing piece.
- **Extended knowledge-base coverage for Xe SYCL.** `knowledge_base/sycl/xpu/` is already substantial — `cutlass_sycl_framework.yaml`, `xetla_patterns.yaml`, `llm_workload_shapes.yaml`, `real_model_shapes.yaml` — so this is a depth extension, not a new corpus: sub-group semantics and sizes per generation, DPAS/XMX shapes and constraints, 2D block loads and prefetch, SLM sizing and bank behaviour, GRF modes and their occupancy trade-off, work-group shaping, and the memory-order and atomics costs.
- **Compiler-axis awareness**, so an optimization stage can propose a flag change as a candidate rather than only a code change.
- **Build-cost awareness in the trial loop**, so a SYCL run does not silently consume the whole budget in compiles.

### 11.10 Do not let language bias selection

The ranking function in §18 includes `extraction_tractability`, and left unbounded it would quietly turn this into a Triton project: Triton kernels extract cleanly and iterate in seconds, so they win every tie. That is how a 40%-of-GPU-time SYCL GEMM loses to a 5% Triton elementwise kernel and the report calls it optimization.

Two guards:

1. **Cap the tractability term.** It breaks ties; it does not overturn an order-of-magnitude difference in `max_e2e_gain`.
2. **Report what was skipped and why.** Every run emits a "considered but not attempted" list with the reason — build cost, low confidence, opaque provider, no available action. If the top three entries by GPU time are all skipped, the headline number is not the story and the report should say so.

---

## 12. Kernel extraction and dependency closure **[NEW]**

### 12.1 Why this is its own subsystem

Real kernels are not single self-contained files. In the Intel inference stack a single hot kernel can be spread across:

- a `@triton.jit` entrypoint in one module,
- several `@triton.jit` device helpers imported from sibling modules,
- an `@triton.autotune` / `@triton.heuristics` decorator whose winning configuration is decided at runtime,
- a tuned-config JSON keyed by expert count, N, dtype and device name,
- a Python launch wrapper that computes grid, strides, block tables and scales,
- a platform dispatch layer that decided this backend was selected at all,
- or, for SYCL and C++ extension ops, no Python at all — a `.so`, a source tree, and a build system with `-fsycl` device flags.

Assuming "the kernel is a file" is the assumption most likely to break the pipeline on contact with vLLM. Extraction is therefore a first-class subsystem with its own artifact, its own fallback ladder, and its own verification step.

### 12.2 The bundle contract

Everything extraction produces is a `KernelBundle`, regardless of source or language.

```python
@dataclass
class LaunchRecord:
    fq_name: str  # module.path:kernel_fn
    grid: tuple[int, ...]
    num_warps: int | None
    num_stages: int | None
    constexprs: dict[str, Any]
    specialization: dict[str, Any]  # divisibility / equal-to-1 hints
    arg_order: list[str]
    selected_autotune_config: dict | None
    compiled_metadata: dict  # n_regs, n_spills, shared/SLM, grf_mode, binary hash


@dataclass
class BuildRecipe:
    compiler: str  # icpx, dpcpp, nvcc-equivalent
    flags: list[str]  # -fsycl, -fsycl-targets, AOT device flags
    includes: list[Path]
    defines: dict[str, str]
    link: list[str]
    entry_symbol: str


@dataclass
class KernelBundle:
    kernel_id: str
    extraction_level: str  # E0 | E1 | E2 | E3 | E4
    language: str  # triton | sycl | cpp | opaque
    entrypoint: str
    primary_source: Path | None
    closure: list[Path]  # helper jit fns, headers, templates
    data_deps: list[Path]  # autotune JSON, tables, scales
    launch: LaunchRecord
    build: BuildRecipe | None
    inputs: CapturedInvocation
    dispatch_chain: list[str]
    env_pins: dict[str, str]
    verification: ExtractionCheck
```

**Xe-Forge's `Model` convention is the universal container.** There is no base class to import — the executor resolves a module-level `Model` attribute by duck typing and constructs it from spec `init_args`, `get_init_inputs()`, or no-arg, in that order. E1/E2 bundles produce a `Model` that launches the extracted kernel directly; E3 bundles produce a `Model` whose `forward` calls into the installed framework. No new optimizer-facing contract is needed either way (see §9.8).

### 12.3 The extraction ladder

Not everything can be made standalone. Define the fallback explicitly rather than failing.

| Level | Meaning | Loop cost | Typical source |
| --- | --- | --- | --- |
| **E0** | No extraction. Optimize in place, benchmark only in the workload. | Highest | Anything editable, as last resort |
| **E1** | File-local. The kernel's module is self-sufficient. | Low | Simple hand-written Triton |
| **E2** | Closure. Kernel plus transitively reachable helpers, constants and data files, flattened into a bundle. | Low | Inductor Triton, vLLM Triton, sycl-tla |
| **E3** | In-situ harness. Cannot isolate; `Model.forward` imports the framework and drives the single layer/op with captured inputs. | Medium | vLLM attention/MoE with deep dispatch, IPEX C++ ops |
| **E4** | Opaque. No source. Only a reproducer string and library-level actions. | n/a | oneDNN primitives |

**E3 is not a failure mode.** It is the reliable default for tangled kernels: heavier per-iteration than a standalone file, but always available, always faithful to the real dispatch, and still compatible with the existing `Model` + YAML-spec harness. Prefer E2 when the closure is clean; fall back to E3 without hesitation. The extractor reports the level it achieved and why.

`xe-orbit extract <kernel-id> --level auto` attempts E2, verifies (§12.10), and downgrades to E3 on failure rather than emitting a bundle it cannot prove is the right kernel.

### 12.4 Launch-site interception is the primary mechanism

Static analysis alone will not find the right kernel or its real configuration. Extraction is driven by **intercepting the actual launch during the trace run**, not by reading source and guessing.

For Triton, wrap the JIT launch path to record, per launch:

- fully qualified kernel function and its defining source file and line,
- the complete argument list with names, in order,
- every `tl.constexpr` value as specialized,
- specialization hints (divisibility, equal-to-1) — these change the generated code,
- `grid`, `num_warps`, `num_stages`,
- the selected autotune config when a decorator is present,
- compiled-kernel metadata: register count, spill count, SLM/shared usage, GRF mode, binary hash.

The binary hash and register/spill counts matter twice: they are how extraction is verified (§12.10), and register pressure is a first-order concern on Xe.

For C++/SYCL extension ops, intercept at the dispatcher (`torch.ops.<ns>.<op>`) and record the op schema, the resolved implementation, and the shared object it came from.

Record the **dispatch chain** in both cases — platform selection, attention backend choice, layer, op, kernel. Patch-back (§13) needs to know where to intervene, and the chain is also the cheapest explanation of *why* a given kernel ran.

### 12.5 Per-source extractors

**Inductor-generated Triton.**
The kernel body lives in the Inductor cache module; the launch wrapper, grid computation and stream handling live in the generated `output_code.py`. Both are needed. The generated code imports Inductor runtime helpers, so the closure either vendors them or declares a hard torch version pin — prefer the pin, and record it in `env_pins`. Pin `TORCHINDUCTOR_CACHE_DIR`, disable the FX graph cache during extraction so codegen is reproducible, and enable Inductor's trace output so `output_code.py` is written. Typically **E2 with a torch pin**.

**Hand-written Triton in vLLM.**
Real source files inside the installed package, but the kernel is usually decorated with autotune and/or heuristics, calls `@triton.jit` device helpers that may live in sibling modules, and depends on tuned-configuration JSON files selected by expert count, N, dtype and device name. Closure is computed by AST-tracing the `@triton.jit` call graph plus module-level `constexpr` constants, then copying the matching JSON as a data dependency. Typically **E2**; drop to **E3** when the launch wrapper computes block tables or KV-cache metadata that is impractical to reproduce.

**vLLM-XPU / IPEX / torch-xpu-ops / sgl-kernel-xpu SYCL ops.** See §11 for the full treatment — identity via demangled kernel names, closure via `compile_commands.json`, compiler-option actions, and P1 patch-back. In summary:
There is no Python kernel. The entry point is a registered `torch.ops` schema; the implementation is inside a compiled extension. Extraction resolves op → shared object → source tree → build system, and the bundle carries a `BuildRecipe` with the SYCL compiler, `-fsycl` and `-fsycl-targets` flags, AOT device flags for the target Xe device, include paths and defines. Xe-Forge already accepts `--dsl sycl`, so the optimizer side is in place. In practice these start at **E3** (drive the op in place, where the existing build already works) and move to **E2** only once the standalone build is proven to reproduce the same binary.

**sycl-tla / CUTLASS-SYCL (Xe-Fuse territory).**
Templated C++ where the "configuration" is template parameters rather than runtime values. Xe-Forge's tile search (`core/tile_search/`, `docs/TILE.md`) — and, externally, Xe-Fuse — already have compile-and-benchmark harnesses for exactly this; reuse them instead of building a third. **E2 with a template instantiation harness.**

**oneDNN and other opaque libraries.**
No source extraction is possible. Run the workload once with oneDNN verbose enabled and capture, per primitive: kind, propagation, memory descriptors, implementation name, and the problem string. That string *is* the isolated reproducer — it drives the library's own benchmark tool, which gives a real standalone measurement for a kernel with no editable source. **E4**, with actions restricted to `BACKEND_CHANGE`, `LAYOUT_CHANGE`, `LIBRARY_CONFIG` and `REGION_FUSION`.

**Runtime memory operations (v9).**
Level Zero reports `Memcpy D2H`, `Memcpy H2D` and `Memset` in the same stream as kernels, and they are not kernels. Left to the unknown fallback they arrive as "no provenance; needs more profiling" — advice that can never be discharged, because no amount of profiling will produce a source file for a host/device copy. They get their own provider, an **E4** reproducer, and a host-side action space: pinned memory, fewer or larger transfers, overlap with compute. Transfer time on the critical path is a genuine finding; it is simply not a kernel-rewrite finding. The distinction from an opaque library primitive matters for the same reason — oneDNN has a backend to swap and a library to reconfigure, and a memcpy has neither.

**Unknown.**
No provenance. Action is `PROFILE_MORE`, never optimization. An unknown kernel holding significant GPU time is a finding to report, not a target to guess at. Keep this category for things genuinely unattributed: every entry that lands here for a knowable reason — a transfer, a runtime op — makes the category less legible, and its whole value is that anything in it deserves a second look.

### 12.6 Closure resolution for multi-file kernels

Most real kernels are not one file. A vLLM attention or fused-MoE kernel typically reaches across several modules, and the closure has to be computed, not guessed.

Algorithm, starting from the intercepted launch record (§12.4):

1. Resolve the entrypoint to its defining module and source file.
2. Walk the kernel body's AST. Every free name resolves against module globals; every call that lands on a `@triton.jit` function is a **device helper** and is added to the work list — transitively, across modules, including helpers that are re-exported or imported under an alias.
3. Collect module-level constants used as `constexpr` values, and `tl.constexpr` defaults.
4. Collect decorator arguments: the autotune configuration list, and heuristics callables — noting that heuristics lambdas frequently close over module state, which must come along.
5. Follow the launch wrapper as well as the kernel: grid computation, stride derivation, block-table construction and scale lookup often live in the caller and are part of the kernel's real behaviour.
6. Resolve data dependencies the wrapper reads (§12.8), including config JSON selected by device name, expert count and N.
7. Record every file with a content hash.

Packaging: **keep the package structure and add a path shim** rather than flattening into one file. Flattening breaks relative imports and re-export chains, and it destroys the mapping back to the original source that patch-back needs. The bundle is a small tree plus a manifest, not a concatenation.

If any step cannot be resolved — a dynamic import, a helper selected at runtime by a registry, a C-level dependency — the extractor **downgrades to E3 and records which step failed**. A partially resolved closure is worse than an honest in-situ harness, because it looks standalone and is not.

### 12.7 Autotune and heuristics pinning

If the kernel carries an autotune decorator, the baseline and the candidate must be compared under a controlled configuration policy, or the measurement is meaningless.

- Record the config that actually won at runtime, and pin it as the bundle's baseline.
- Report **both** comparisons: pinned-vs-pinned (did the code get better?) and tuned-vs-tuned (did the best achievable get better?). They can disagree, and the disagreement is informative.
- Never let a candidate silently widen or narrow the autotune search space relative to the baseline without recording it as part of the change.

The same applies to heuristics-derived values and to any `constexpr` that depends on input shape: extraction pins the specialization the workload actually produced.

### 12.8 Non-code dependencies

Copy into the bundle as data, never regenerate:

- tuned-config JSON files,
- quantization scale tensors and zero points,
- block tables, sequence-length tables, and other paged-attention metadata,
- KV-cache layout descriptors,
- any lookup table the kernel reads.

These are exactly the things a synthetic-input reconstruction gets wrong, and exactly the things that make a "correct" kernel produce wrong tokens.

### 12.9 Environment pinning

Every bundle records the environment that produced it, because extraction results are not reproducible without it: torch, IPEX, Triton, vLLM and driver versions; Inductor and Triton cache directories; compile-cache enable/disable state; oneDNN verbose state; the attention backend and platform selection variables that steered dispatch; and device identity plus clock state.

A change in any of these invalidates the bundle and forces re-extraction. Silent reuse across versions is the failure mode that produces unexplainable results three months later.

### 12.10 Extraction verification

**An extracted bundle is not trusted until it is proven to be the same kernel the workload ran.** Build the bundle, run it once, and check:

1. The compiled kernel identity matches the runtime event — kernel name, and for Triton the specialization and binary hash.
2. Register count, spill count and SLM usage match the intercepted launch metadata.
3. Grid, `num_warps`, `num_stages` and every `constexpr` match.
4. The bundle's output matches the captured reference output within the tightened tolerance.

If any check fails, `provenance_confidence` drops, the bundle is marked unverified, and the extractor downgrades a level rather than proceeding. The specific failure this catches — and it is common with autotune plus specialization — is optimizing a *different specialization* of the right kernel, producing a real speedup on a variant the workload never executes.

**Version skew is a second route to that same outcome, and it is the common one for SYCL (v9).** The source trees of §11.2 are cloned, not built from; a `git clone` gives you HEAD while the installed wheel was built from a release. Measured here: torch 2.13.0+xpu against a `torch-xpu-ops` checkout whose `FillKernel.cpp` dispatches over `kBComplex32`, a `ScalarType` the installed headers do not define. That bundle is not a variant of the running kernel, it is a *later* kernel — and everything about it looks correct until it is compiled.

So skew gets its own verdict, distinct from an incomplete closure, because the two call for opposite responses: carrying more files fixes a closure and does nothing for skew, where the remedy is checking out the revision matching the installed binary. Recording the tree's revision alongside the framework version in the manifest is what lets the check say which of the two it is rather than reporting a compile error and leaving the reader to guess.

**Resolved, and it was worth more than expected.** The authoritative pin is PyTorch's own
`third_party/xpu.txt`: v2.13.0 pins `torch-xpu-ops` at `bc294243`, while a fresh clone
lands on HEAD — three months and one new `ScalarType` later. Checking the tree out at the
pinned revision took E2 verification from **10 of 22 to 21 of 22**, and total verified
bundles from 14 to 25 of 36. Skew accounted for eleven of the twelve failures, not the five
its error messages named directly; the other six had been reported as ordinary compile
errors because a symbol added upstream fails in whatever way the surrounding code happens
to break.

Two consequences. First, resolving a kernel tree without pinning it to the installed
binary is not a minor imprecision — it was the single largest source of unverified bundles.
Second, the fix is cheap and mechanical, so the pin belongs in `kernel_sources` (§10.6)
alongside the repository URL: a tree is identified by URL *and* revision, and a checkout
that does not name one is a checkout of something else.

**v10 correction, since closed: the pin had never landed.** At audit time none of the
three framework YAMLs carried a `revision:` key and `orbit/languages/sources.py` did
not check one — the conclusion above was recorded and then not applied, so every
checkout was "a checkout of something else" by this section's own standard. Skew
*classification* had landed (`extract/verify.py` distinguishes skew from an incomplete
closure and from a plain compile error); the pin that prevents skew was gap G3 in §24,
now closed: `kernel_sources` entries carry `revision:` (torch-xpu-ops at `bc294243`;
wheels without a known pin declare an explicit empty one), the registry reads each
checkout's actual revision from `.git`, and `xe-orbit sources` reports a per-tree pin
state — ok, skew (naming both revisions), unpinned, or unverified — because "we could
not check" and "it matches" are different claims.

The one bundle still unverified is `xe_fmha_fwd_kernel.h`, a sycl-tla flash-attention
header whose CUTLASS template closure genuinely is not in the bundle — an honest
incomplete closure rather than skew, and exactly the case §11.5 warns about.

### 12.11 Region extraction

A `RegionRecord` bundle contains several kernel bundles plus the glue between them: the intermediate tensors that fusion would eliminate, the producer-consumer edges, and a driver that runs the whole region so a fused replacement can be compared against the unfused sequence as a unit. Region bundles are almost always E3 in the first instance — the region is defined by how the framework strings the kernels together, so the framework is the most faithful driver.

### 12.12 Bundle test rig

Extraction is a correctness problem, so the pipeline tests its own bundles. `xe-orbit bundle test <bundle>` runs, in order:

1. **Isolated import** — build the bundle under a restricted `sys.path` with the source package removed. A `ModuleNotFoundError` here means the closure is incomplete; it is the single most common extraction bug and it is silent otherwise.
2. **Launch-record match** — grid, warps, stages, every `constexpr`, specialization hints, and the compiled binary identity match what was intercepted (§12.10).
3. **Output match** — bundle output equals the captured reference within the tightened tolerance.
4. **Mutation check** — perturb the extracted kernel (inject a deliberate arithmetic change) and confirm the bundle's output changes. If it does not, the bundle is silently executing the *installed* package rather than the extracted source, and every measurement taken from it would be meaningless.
5. **Data-dependency check** — remove each declared data file in turn; each removal must produce a failure. A data dep that can be deleted without effect was not actually a dependency, and one that was missed shows up as a pass that should have failed.

Tests 1 and 4 are the pair that make multi-file extraction trustworthy. Neither is expensive; both belong in CI for every bundle the reference workload produces (§15).

**Tests 1, 2 and 4 are language-specific, and writing them as if they were not is a mistake with a very convincing failure mode (v9).** The list above is the Triton version. Run against a SYCL bundle carrying `Indexing.cpp` it reported `ModuleNotFoundError: No module named 'Indexing'` as "closure is incomplete" — a verdict about Python module resolution, delivered on a C++ translation unit, that read as a plausible extraction defect. The C++ forms ask the same questions of the compiler:

* **Isolated compile** replaces isolated import. Compile the translation unit alone, with only the bundle's files plus the installed framework headers (§11.6). An unresolved `#include` is a hard error, where a Python import might be quietly satisfied by something ambient. A header-primary bundle gets a synthesized one-line TU rather than a skip — skipping it was worse than useless, because a skipped check counted toward "verified".
* **Instantiation match** replaces launch-record match. Grid, warps and stages do not exist for SYCL, so asking for them produced a blanket "no launch record" failure on every native bundle: true, uninformative, and concealing that the answer was available. It is in the demangled name. `IndexFunctor<OpaqueType<8>>` and `IndexFunctor<OpaqueType<4>>` appear in one real trace under the same entry symbol, and the template arguments are the only thing that tells them apart — so the bundle records them (§11.5 item 4), and one that cannot is unverified.
* **Mutation check** injects `#error` instead of an arithmetic change, and confirms the compiler sees it. Cheap by construction: the probe fires before any framework header is read.

Two rules the C++ path made explicit, both of which apply to every language:

* **A skipped check is not a passed check at E1/E2.** The defining claim of those levels is "standalone, with closure", and that claim *is* the closure check. Counting a skip as a pass had ten of fourteen bundles reporting verified at E2 with their closure never tested.
* **Say which failure it is.** "Closure incomplete", "version skew" and "compile error" call for three different responses — carry more files, check out the matching revision, fix the source — and collapsing them into one verdict sends the reader the wrong way about two-thirds of the time.

Measured on the reference vLLM trace: 10 of 22 E2 bundles verify completely (closure proven by compiling, instantiation pinned, mutation observed). The other 12 each fail with a specific diagnostic — 5 version skew, 4 compile errors, 3 incomplete closures — which is the intended output. An unverified bundle is a finding with a filename attached, not a gap in the tool.

**Status (v10): implemented.** `orbit/extract/bundle.py` carries all four extraction paths (E1/E2 closure, E3 in-situ, E4 opaque, SYCL) with autotune pinning and package-preserving copy; `orbit/extract/verify.py` runs both language forms of this rig, classifies skew separately from closure and from plain compile errors, and records E4's identity checks as skipped rather than passed; `orbit/extract/batch.py` reports GPU-time-weighted coverage. Driven by `xe-orbit extract [--all]` and `xe-orbit bundle {test,verify,show}`.

### 12.13 CLI

```text
xe-orbit extract <kernel-id> [--level auto|E1|E2|E3]
xe-orbit extract-region <region-id>
xe-orbit bundle verify <bundle>
xe-orbit bundle test <bundle>
xe-orbit bundle show <bundle>
```

---

## 13. Patch-back: a mechanism ladder **[CHANGED — this is the other hard part]**

v1 gave this one line ("patch candidate into workload"). It is, with extraction, the most likely place for the project to stall, and it needs a design decision before PR 1.

**Do not write into the Inductor cache.** Those paths are content-hashed and regenerate on every recompile, config change and version bump. A patch against generated code is not a durable artifact.

### 13.1 The ladder

There is no single mechanism. Define a ladder and always take the highest rung that works, because higher rungs touch less and revert cleanly.

**P1 covers SYCL as well as Triton** (§11.8): `torch-xpu-ops`, IPEX, vLLM-XPU and `sgl-kernel-xpu` all register dispatcher ops, so an optimized SYCL kernel ships as a small out-of-tree extension compiled with icpx — no fork of PyTorch, vLLM or SGLang, and no patched `libtorch_xpu`.

| Rung | Mechanism | Touches | Use when |
| --- | --- | --- | --- |
| **P1** | **Operator override** through the PyTorch dispatcher — register an implementation for an existing `aten` or custom op on the XPU key, shadowing the default | Nothing in the framework | The kernel sits behind a registered op. Works for plain PyTorch, and for vLLM/SGLang whenever the kernel is dispatched as a custom op |
| **P2** | Custom op + Inductor post-grad pattern replacement | Compile pipeline only | A generated, fused kernel that maps to a subgraph rather than one op |
| **P3** | Framework registry substitution — register an alternative attention backend, MoE implementation or layer through the framework's own plugin point | Framework config | The framework exposes a selection point (adapter reports it via `patch_points`, §10.3) |
| **P4** | Import-time module shim — a recorded, reversible monkey-patch installed by Orbit | Process state | No registry exists, but the target is Python |
| **P5** | Source patch and rebuild | Source tree + build | C++/SYCL extension ops with no override path |

**P1 is the default and should be reached for first.** Operator override means the optimized kernel is a small importable module that registers itself, with the framework left entirely untouched — no fork, no vendored patch, no rebuild, and a revert that is just not importing it. It is also the rung that ports across frameworks for free: the same override works under PyTorch, vLLM and SGLang if all three dispatch through the same op.

Every applied patch records: rung used, target symbol, the registration call, the revert procedure, and the verification result.

**Verification is mandatory and is a dispatch assertion, not an inspection.** After applying, re-profile and confirm the new kernel appears in the trace *and the old one does not*. Overrides that silently fail to take effect — wrong dispatch key, wrong overload, registration after the first call, a `torch.compile` graph captured before registration — are common and produce a clean "no change" result that looks like an honest negative.

Consequences to design for:

- The artifact is version-bound, in exactly the way §12.9 describes. **A version change invalidates every accepted candidate** and triggers re-validation, never silent reuse.
- The dispatch chain recorded in §12.4 tells you which rung is available; the adapter's `patch_points` (§10.3) tells you where.
- For oneDNN and other opaque providers, patch-back is not source replacement at all; it is a `REGION_FUSION` substitution or a config/backend change.

Every candidate is applied in its own **git worktree**, never the working tree. This isolates experiments, makes rollback free, and keeps parallel candidate evaluation possible later without redesign.

**Status (v10): implemented.** The ladder and worktree helpers live in `orbit/patch/ladder.py` (`RUNG_ORDER` P1–P5, `render_operator_override`, an `AppliedPatch` record carrying rung, target, registration call, revert procedure and verification), the dispatch assertion is exactly as specified — `assert_dispatch` reports `took_effect` only when the new kernel is present *and* the old one absent — and the out-of-tree icpx SYCL override of §13.3 is generated by `orbit/patch/sycl_override.py`.

### 13.2 Editing an installed tree in place, survivably (v9)

The worktree rule above covers a candidate in a repository we control. It says nothing about the installed framework in site-packages, which is not a git tree and cannot be worktree'd — and that is where AMD's Hyperloom does all of its work (§5.6). So the question has to be answered directly: **should Orbit edit installed source in place?**

**Default: no.** P1 reaches most of what matters without touching the tree at all, and its revert is not importing a module. Nothing below is as safe, and in-place editing should stay the second choice at every rung.

**But it cannot be avoided entirely** — an E3 harness, a patch to a framework's Python layer, a launch-wrapper change — so the mechanism must exist. What decides whether it is usable is not whether the edit applies, but **what happens when the edit is wrong or the process dies halfway through**. Naive editing of site-packages fails in ways that outlive the run: a process killed mid-write leaves a truncated file and a broken environment for every later run *including the one that would have diagnosed it*; an edit with no recorded original cannot be reverted, only guessed at; a revert that restores blindly destroys whatever someone else changed in the meantime; and a crash before revert leaves the tree modified with nothing on disk saying so, so the next run measures a patched baseline and reports it as clean.

That last one is the dangerous one, because it is silent and it produces a *wrong number from a correct-looking pipeline* — the failure this whole project exists to refuse.

Six properties, each aimed at one of those:

1. **Refuse what cannot be restored** — before touching anything. Not writable, a symlink (the write would follow it out of the sandbox), or outside the declared sandbox roots. A patch that cannot be undone is not a patch, it is damage with an optimistic name.
2. **Record the original first**, bytes and digest, in the run directory — outside the tree being modified, so the record survives whatever happens to the tree.
3. **Replace atomically.** Temp file in the target's own directory, fsync, `os.replace`, fsync the directory. A reader sees the old file or the new one, never a partial one, even on `SIGKILL`. The temp file must be a sibling: staged elsewhere, the replace becomes a copy, and a copy can tear.
4. **Journal before, clear after.** The entry is fsynced *before* the edit and removed only *after* the restore, so at every instant the on-disk state is either "nothing to do" or "an entry naming exactly what to put back". `xe-orbit patch recover` reads it; running it at startup is what stops an orphaned edit from poisoning the next baseline.
5. **Verify before reverting.** The file must still hash to what we wrote. If it does not, someone else changed it and restoring would discard their work — report `CONFLICT` and exit non-zero rather than resolving it silently. `--force` exists for an operator who knows better; it is not a default, because the point is that we cannot tell the two cases apart.
6. **Idempotent revert**, and **one record per target**. The second matters more than it looks: re-patching an already-patched file must not journal the *current* content as the original, or a revert restores the previous patch's output and calls the tree pristine. This was a real bug, found by running the demo rather than by reading the code — the second `apply` inherits the first record's original, so revert always reaches the true one.

The general rule this is an instance of: **a mechanism that can damage a shared environment is judged by its recovery path, not its success path.** The success path is the easy half and the one everyone tests.

**Status (v10): implemented** in `orbit/patch/inplace.py` — sibling-temp `atomic_write` with directory fsync, the journal fsynced before the edit and cleared after the restore, digest-verified revert with `CONFLICT` rather than silent overwrite, idempotent one-record-per-target semantics, and sandbox-root refusal — surfaced as `xe-orbit patch status` and `xe-orbit patch recover [--force]`.


### 13.3 SYCL needs no rebuild lane (v9)

The SYCL path was planned as the expensive rung: every trial an icpx build of
`vllm-xpu-kernels` through CMake and Ninja, budgeted at ~8 GB per compile process, needing
Hyperloom's off-loop build lane so a long compile never blocks the loop.

Measured, that is not the shape of the problem. All 33 of vLLM's XPU kernels register as
dispatcher ops — `import vllm_xpu_kernels._C` puts `_C::fused_add_rms_norm`,
`_C::gelu_and_mul`, `_C::mul_and_silu` and the rest on the XPU key — so §11.8's operator
override applies directly. A candidate is **one translation unit** registering a shadowing
implementation, compiled in **7-8 seconds**, which puts a SYCL trial in the same cost class
as a Triton one. Nothing in vLLM or `vllm-xpu-kernels` is modified, and reverting is not
loading the shared object.

Proven end to end on `_C::fused_add_rms_norm` (1.19% of GPU time on the Qwen path). PyTorch
logs the shadow itself — *"Overriding a previously registered kernel for the same operator
and the same dispatch key ... previous kernel: torch_bindings.cpp:16, new kernel:
override.cpp"* — and §13's dispatch assertion passes: the override's kernel appears in the
trace and the original does not.

Two failures on the way, neither discoverable from the generated source:

* **The C++ signature must match the registered schema exactly**, by-value versus
  by-const-reference included. `const std::optional<at::Tensor>&` against a schema
  declaring `std::optional<at::Tensor>` aborts the process at load with "Mismatch in
  kernel C++ signatures". That is the right behaviour — the dispatcher refuses rather
  than accepting a registration that would never shadow — but it is fatal rather than
  diagnosable, so the signature belongs derived from the schema, not written by hand.
* **Use `sycl::reduce_over_group`, not a hand-rolled barrier tree.** A tree reduction with
  a barrier per level hung the device: every work-item must reach every barrier, and
  getting that subtly wrong on Xe does not fault, it stops, and the process needs killing.

The build machinery documented in the kernel repo's `AGENTS.md` — per-family toggles
(`BASIC_KERNELS_ENABLED`, `FA2_KERNELS_ENABLED`, …), `VLLM_XPU_AOT_DEVICES` where empty
values disable AOT, and `*_default.conf` presets "to reduce compile time" — remains the
escape hatch for a change an override cannot express, and those are §11.7 compiler-option
axes with vendor documentation behind them. It is simply not the common path.

### 13.4 Regions: fusion is authored, not routed (v9)

§7.6 says an opaque provider is reached by `REGION_FUSION` rather than source replacement.
That path existed on paper and returned **zero regions on every real trace**, so the
pipeline's advice to "route it to Xe-Fuse" was never actionable. Four defects were stacked
between the trace and a region, each hiding the next:

1. `record_shapes` was never enabled. The profiler option exists; nothing set it.
2. Shapes are recorded on `cpu_op` events, and **0 of 141,947** of them carry a
   correlation id. Kernels correlate to *runtime* events instead, so kernel and shapes
   share no key — they are related by **time containment**, innermost enclosing op on the
   same thread. That join is now `trace.attach_shapes`; 16,859 of 17,326 kernels carry
   shapes after it.
3. The detector read shapes from `event.args` rather than the joined field, so it saw
   none of them.
4. The uplift rule compared a region's share against its largest member's **run-wide**
   share. A region containing a kernel used elsewhere could never clear that bar.

The fourth was a design error rather than a coding one, and fixing the arithmetic was not
enough. Even comparing within-region times, `gemm+activation` failed at 1.069 — because
inside the region the GEMM still dominates. **That is the reason to fuse, not a reason to
reject.** The value is eliminating the intermediate write, not the epilogue's own cost;
requiring the region to beat its largest member asks the epilogue to be expensive, which
is the opposite of the condition that makes it fusable. So the uplift test now applies
only to **unnamed** chains, where nothing suggests fusion helps. A pattern in
`FUSION_PATTERNS` is already an assertion of fusability; `min_share` decides whether it is
worth doing.

Measured on a Qwen2.5-0.5B decode trace, the two regions are `gemm+activation` at 39.6% of
GPU time and `gemm+rmsnorm` at 27.3% — **66.9% together**, against the 6.8% ceiling that
kernel rewriting alone reaches. The eliminable intermediates are `[172, 9728]` and
`[172, 896]` in half precision, 3.7 MB per pass.

**And fusion is authored, not routed.** Making Xe-Fuse the destination made an external
project a hard dependency for the only path that reaches an opaque GEMM. Hyperloom does
not do this — its handler *"authors serving-safe fused kernels and returns a source patch
+ env flags for the integrate gate"*, generated for the shapes at hand. Orbit now defaults
to the same: write the fused kernel for the observed shapes, register it as an operator
override (§11.8), and put it through the ordinary correctness and measurement gates.
Xe-Fuse is one executor among several, preferred when installed and matching a preset;
its absence costs an option rather than the path.

**What authoring a GEMM fusion actually costs, stated plainly.** For these two regions the
dominant member is the oneDNN GEMM, so fusing the epilogue means writing a SYCL GEMM that
beats a vendor-tuned library on its own hardware, and then the eliminated 3.7 MB must
outweigh whatever that costs. That is a different order of difficulty from the single-op
override proven in §13.3, and it is exactly what Xe-Fuse's autotuned sycl-tla templates
exist for. The honest ordering is therefore: prefer Xe-Fuse where a preset matches, author
where the dominant member is not a vendor GEMM, and report the region either way — because
"66.9% of GPU time sits behind two fusable regions" is a finding whether or not this
project is the one that acts on it.

**Measured (v10): Xe-Fuse ran against these exact regions, and the answer is a
specialization set, not a verdict.** At the traced decode shapes (M=16, N=9728, K=896,
bf16) its k2 preset — GEMM, RMSNorm row-scale and SwiGLU in one kernel; the row-scale
commutes through the GEMM, so one preset covers both regions' chain — measured **+3.1%
against vLLM's real unfused sequence (rms_norm 4.0 µs + oneDNN GEMM 345.4 µs +
silu_and_mul 5.4 µs = 361.0 µs), 95% CI [0.24%, 5.94%] excluding zero**, with numerics
verified independently to 0.14% median relative error and 100% of elements within 2%.
At M=128 the same preset **lost by ~25%** even at the best of a swept tile set — oneDNN's
GEMM efficiency at larger M dominates the epilogue saving — which is §14.3's rule
arriving on schedule: a gain at one serving configuration is not a gain, and the
deployable shape is §14.4's specialization set (fused at decode M, vendor GEMM above).
The decomposition also confirms this section's cost analysis: the template GEMM *tied*
oneDNN at M=16 (the win is the eliminated epilogue), and the projected e2e value at the
regions' 67% share is ~+1.7%, above the workload's measured MDE — the first
e2e-resolvable candidate this trace has produced. Three upstream Xe-Fuse findings from
the session, worth fixing there: the generated benchmark parses `--verify` and never
uses it; `initialize_block` zero-fills float scale buffers, so an output-vs-reference
check would pass trivially on D == 0 (caught only because the margin check refused a
too-perfect match); and the tile auto-selector chooses well at M=16 and poorly at
every larger M measured (32, 64, 128 — M-matched tiles beat its picks by 3-26%).

**And then the full loop closed on it — with an honest REJECT (v10).** The fused
chain was integrated into live serving: a torch extension wrapping k2 (two tile
instantiations, M-dispatched; a tiny `add_rms_scale` kernel eliminating the
normalize write; γ folded into the packed weight — numerics verified *closer* to
fp64 truth than the unfused path), applied to `Qwen2DecoderLayer.forward` through
the §13.2 journalled patcher, and A/B'd end to end: **REJECT, −2.52%, 95% CI
[−2.81%, −2.22%], against a 0.19% MDE** from a baseline spread of 0.1%. The
kernel-level win did not convert because the extension launches on cutlass-sycl's
compat queue rather than torch's stream — two `wait()` serializations per layer,
24 layers per decode step, exceeding the fused margin. The revert restored the tree
byte-for-byte. This is §1's founding question — do kernel wins convert to
tokens/s? — answered *no, and instrumented to the microsecond why not* for this
candidate. **And then the instrumented cause was fixed and the answer became yes.**
The cutlass adapter's `run()` accepts a `sycl::queue*`, so round two launches both
ops on torch's in-order XPU stream with no waits — numerics unchanged — and the
fresh two-arm A/B returned **ACCEPT: +0.56% end-to-end, 95% CI [0.21%, 0.91%]
excluding zero, MDE 0.46%**. That is §25's primary criterion met for the first
time, with the honest scope attached — and the reproduction campaign then ran the
same day: at batch 16, **two independent ACCEPTs** (+0.56% CI [0.21, 0.91]; +2.12%
CI [1.25, 2.99]) and two INCONCLUSIVEs with positive point estimates (+1.02% under a
3.19% MDE; +0.47% under 2.13%) — four-of-four positive, mean +1.04%, with desktop
noise the limiter on the other two; batch 32 is unproven (−0.77% under a 4.94% MDE,
consistent with the thinner kernel-level margin there). Session count was fixed
before the reruns rather than extended until a third ACCEPT appeared, because
running-until-significant is choosing the result (§17.5.1). §25's full
three-clean-sessions claim initially remained open pending quiet-machine reruns —
and then closed the next morning under §17.5.1's own remedy. Session 6 ran with a
pre-declared noise gate (baseline first; proceed only if its MDE < 1%): the machine
measured quiet (MDE 0.60%), arm B ran, and the verdict was **ACCEPT, +0.33%, 95%
CI [0.09%, 0.56%] excluding zero**. The complete record, stated in full: six
sessions — three independent ACCEPTs at batch 16 (+0.56% [0.21, 0.91]; +2.12%
[1.25, 2.99]; +0.33% [0.09, 0.56]), two batch-16 INCONCLUSIVEs with positive point
estimates under 2–3% desktop-noise MDEs, one batch-32 INCONCLUSIVE (thin-margin
regime). Five of six point estimates positive; every session's tree reverted
byte-clean. **§25's primary criterion — an end-to-end throughput improvement whose
95% CI excludes zero, reproducible across three independent sessions — is met**,
with the full table as the claim's context rather than a selection from it.

**The served and token-level completions (v10, next day).** The candidate then went
through the two gates the offline campaign had not run. *L3 token gate:* greedy
decode is token-exact on 10 of 16 prompts, with the six divergences starting deep
in generation (tokens 11–38, never at position 0) — the signature of near-tie logit
flips from the declared rounding differences of a path proven *closer* to fp64
truth than the baseline, chaos-amplified by argmax; per §19's own ladder,
acceptance for deployment moves to the bounded-logit-deviation + small-task-eval
branch — which then ran, with both thresholds declared before measurement. The
logit gate passed decisively: all six divergences sit at pristine top-2 gaps of
0.0000 or 0.1250 nats against a 0.25-nat budget — three are *exact ties*; the fused
path flips tokens only where the baseline itself is flipping a coin. The task-eval
gate as declared (fused accuracy ≥ pristine, n=32) **failed by exactly one answer**
(22/32 vs 23/32, near-identical miss sets, one discordant prompt each way — pure
binomial noise at this n), and the declared gate is not relaxed after the fact:
the recorded verdict is that the gate was underpowered as designed — a strict ≥ at
n=32 breaks on a single coin-flip answer — and **deployment acceptance is withheld**
pending a larger pre-declared eval (hundreds of prompts, an equivalence margin
rather than strict dominance). The performance case stands; the quality case is
unproven-not-disproven, stated as exactly that.

**Quantized workloads and the SGLang kernels, measured (v10).** Two more workloads
went through the pipeline, per the two-workload check. GPTQ-Int8: **not runnable on
this stack** — vLLM-XPU routes it to a Marlin path supporting only 4-bit
(`does not support weight_bits = uint8b128`), now a named `quant_capability`
diagnosis in the enablement classifier rather than an `unknown`. AWQ-Int4: runs,
traces, and catalogs cleanly — the AWQ path dequantizes into fp16 oneDNN GEMMs, so
the GEMM still owns 81.4% of GPU time and the same two fusable regions cover 44.9%,
now auto-routed to Xe-Fuse by the availability upgrade; the sampler's share grew 7×
to 4.9% as the GEMMs sped up, Amdahl reshaping the catalog exactly as §18 predicts.
Point-and-start landed alongside: `xe-orbit trace --wrap -- <command>` runs any
single-process torch workload under a shipped profiler wrapper (no workload-side
code; subprocess engines are told to use their framework hook instead of being
handed an empty trace), and a failed workload in the trace stage now carries the
enablement diagnosis. SGLang: the rung-3 scoped-runtime climb ran live and reported
honestly (pip sglang pulls the CUDA dependency stack and ships no XPU extra —
install failed, environment discarded, nothing kept), and the sgl-kernel-xpu source
build failed 14 translation units under parallel icpx on this 13.8 GB shared-memory
machine (~8 GB per compile, §13.3's own budget) — placing SGLang-XPU kernel
measurement at **rung 5, the off-loop build lane**, by the ladder's own taxonomy.
The adapter, knowledge file and source registry (172 indexed symbols) are ready the
day that lane exists; a serial overnight build is the named next attempt. (Precision, stated plainly: the kernel
computes at the standard bf16-data/fp32-accumulate; fp64 exists only as the host
referee, which must out-precision both contestants to rank them.) *Served A/B:*
a real `vllm serve` per arm with `xe-orbit run --framework vllm` driving
`vllm bench serve` clients — the Tier-1 adapter parsed all four declared metrics,
TPOT tight at 25.88 ms CI [25.63, 26.14] — returned INCONCLUSIVE on every metric at
n=5 (TPOT +0.46% under a 1.57% MDE; TTFT's MDE 163.9%, the §17.5 first-request
effect in its full glory). The honest reading: a ~+0.5% effect at this batch is
below what a five-rep served benchmark resolves, the offline A/B was the correct
instrument for this candidate's size — and the served machinery is now proven end
to end for the day a bigger candidate needs it. Everything framework-generic: the
same adapter path serves SGLang unchanged when an install exists. The path is now fully automated in both directions:
`xe-orbit optimize --apply --rounds N` feeds each round's verdicts and reasons back
to the proposer in-process with a shared stall ledger, and `xe-orbit fuse <region>`
runs pattern → preset → trace-derived shapes → deterministic tile sweep (§11.7),
demonstrated live on r0. The four Xe-Fuse findings are fixed on the sibling
checkout's `orbit-findings` branch, including a real `--verify` — validated on the
device (Disposition: Passed, max rel 3.9e-03), and the fixed selector's M=128 pick
beat even the manual sweep's best (526.6 vs 612 µs).

**The full crossover map (v10), fused best-tile vs vLLM's real oneDNN chain, one
device:** at 0.5B shapes (N=9728, K=896) fusion wins the decode regime — +1.5% at
M=16 (the e2e-ACCEPTed candidate), +1.4% at 32, +3.5% at 64 — and loses prefill
badly (−7% at 128, −28% at 1024, −24% at 2048): oneDNN's skinny-K GEMMs at large M
are strong. At qwen25_7b shapes (N=37888, K=3584) — the model class Xe-Fuse's
presets are actually tuned for — the picture inverts with M: −5.4% at M=1024 but
**+4.4% at M=2048 (86.3 vs 90.3 ms)**, single-run measurements. So fusion wins at
the *extremes* — small-M decode on small models, big-M prefill on big models — and
the vendor library owns the middle, on this iGPU. Every big-M sweep also found
256×256×32 beating the fixed selector's auto by 28–32%, the fifth selector data
point for upstream. §14.4's specialization set is not an edge case; it is the
shape of the answer everywhere we have measured. The REJECT→cause→fix→ACCEPT sequence is also the strongest evidence yet
for §17's design: a loop that had collapsed the REJECT into a bare failure would
have discarded a real win one wait-removal away. Reproduction kit:
`examples/fused_mlp_experiment/`.

### 13.5 The agentic loop: the agent proposes, Orbit decides (v9)

§13.2 makes an in-place edit survivable. This is what drives one.

```text
SELECT -> PLAN -> [ per trial: REVIEW -> APPLY -> VERIFY -> MEASURE -> KEEP/REVERT ] -> PROMOTE
          agent               agent    <------------ programmatic ------------>
```

The split is §3's rule, and Hyperloom states the same one for the same reason: *"Kernel work is not handled by an LLM agent... No LLM turn is consumed."* Applying a patch, checking correctness, timing it and reverting are deterministic, so routing them through a model would only make a reproducible answer non-reproducible. The agent is asked what to try — the one part with no closed form.

**Two agent calls, not one.** A cheap PLAN call returns N ranked bounded transformations; each then gets its own Claude Code workspace where the agent edits a *copy* and runs the correctness harness itself. Asking for a ranked list rather than a rewrite is deliberate: a model asked to "make this faster" commits to one answer, while asked for candidates it produces a spread, and a spread is what a search wants. The workspace earns its cost differently — without it, a syntax error or a wrong tensor shape costs a full round trip to discover; with it the agent sees its own traceback and fixes it before Orbit looks.

**And the agent's own result is never the verdict.** The workspace runs on a copy and its harness run is advisory. Orbit re-runs the check in a fresh process against the real tree and believes only that. A fresh process is required rather than preferred: a module imported earlier in the session keeps serving the pre-patch source, so an in-process check would confirm a kernel that is no longer on disk — the §12.12 mutation failure arriving through a different door.

**Gates run cheapest-first, which inverts Hyperloom deliberately.** Their `integrate_handler` grades accuracy only for a candidate that already cleared the throughput bar, because their eval is the expensive half. Ours is the opposite — a correctness check is seconds against a framework engine load of nearly a minute — so correctness gates first and a wrong kernel never costs a measurement. The order is: novelty (free) → path sandbox (free) → critic (one call) → apply → correctness → measure.

Five verdicts, and the distinctions between them are the point:

| Verdict | Meaning |
| --- | --- |
| `KEPT` | Correct, and faster by more than the noise floor. |
| `REVERTED_WRONG` | Applied and numerically wrong. Evidence against the change. |
| `REVERTED_SLOWER` | Correct and not faster enough to distinguish from noise. |
| `UNPROVEN` | Applied, but nothing could be established. Reverted, recorded as a gap in the run — **not** as evidence against the candidate. |
| `REFUSED` | Never applied: the sandbox, the novelty ledger, or the critic stopped it. |

`UNPROVEN` is the one worth defending. A harness that failed to import says nothing about the kernel's numerics, and folding that into "wrong" spends a revert on working code while recording a false negative. It still reverts — an unproven patch must not stay on disk — but the run reports it as something that was not measured.

**Correctness spans the paths that changed, not the paths a reference can reach (v9).** A reference test covers what torch can reproduce. For `gumbel_sample` that is temperature 0, where Gumbel-max degenerates to argmax; above 0 it rides Triton's Philox stream and no torch reference exists. An agent then proposed removing a `tl.where` that lives *only* inside `if temp != 0.0` — a change the reference gate passed at accuracy 1.0000 without executing the changed line once.

So the loop runs a **differential** check alongside the reference one: the same kernel, same seeds, same inputs, before and after the patch, requiring bit-identical output across cases chosen to span branches rather than to be representative. It proves something weaker than correctness — it cannot tell you the original was right — but "bit-identical to what shipped" is exactly the claim a behaviour-preserving optimization makes, and it reaches every path the workload does.

Measured: a defect confined to the temperature>0 branch was reported **correct (1.0000)** by the reference harness and caught at **1/3** by the differential one, with exactly the temperature-0 case passing. Combining them requires all to pass and reports the *weakest* outcome — any `WRONG` dominates, and otherwise any `UNCHECKED` dominates, because a candidate whose changed path was never executed is unproven however many untouched paths passed. Taking the best of the two, or the first, would have shipped it. Verified adversarially: with the measurement rigged to show a 3000x win, the loop still returned `REVERTED_WRONG`.

**Only the winner survives — and making that true took two attempts (v9).** Every other
candidate is reverted before the loop returns, so a crash mid-loop cannot leave an
unaccepted patch behind; §13.2's journal covers the crash itself.

The first live run to produce a `KEPT` verdict exposed how easily this goes wrong. It
reported **two** accepted candidates and finished with an **unmodified tree**: reverting a
losing trial called `revert_all`, which undid the whole journal and took the accepted
patch with it. The loop announced success and had silently discarded the thing it
succeeded at — visible only by running `git diff` rather than trusting the loop's own
verdict.

Scoping the revert to the current candidate did not fix it either, and the reason is a
genuine interaction between two individually-correct rules. §13.2 keeps **one journal
record per target**, so a second `apply` inherits the *first* record's original — which is
what stops a re-patch from recording an intermediate state as the original. Reverting the
loser therefore restores the pristine file, not the previously accepted one. The loop now
re-applies what it accepted after any revert.

The general shape, which is this document's recurring failure mode: two correct
mechanisms composed into an incorrect one, reported as a success. Neither the patcher nor
the loop was wrong on its own terms.

**Status (v10): implemented, and — after the audit — reachable.** The loop is
`orbit/optimize/loop.py` — all five verdicts, the gate order novelty → sandbox → critic →
apply → correctness → measure, `stats.compare` with the fixed floor demoted to fallback
(§17.5.2), and re-apply-the-winner-after-revert — with `orbit/optimize/proposer.py`
(two-call PLAN/IMPLEMENT), `orbit/optimize/harness.py` (the differential check;
`UNCHECKED` distinct from `WRONG`) and `orbit/optimize/session.py` (failed-direction
memory), all under test in `tests/orbit/`. At audit time `xe-orbit optimize` stopped
after `ClaudeProposer.plan()`; gap G1 (§24) closed that: `--apply` drives the full
seam — implement in a workspace copy, journalled apply with the target's directory as
sandbox, a fresh-process `--harness` correctness gate, `--measure`/`--samples`
measurement with the §17 statistics — and planning-only stays the default, so nothing
autonomous runs by accident.

---

## 14. Serving profiles and the shape matrix **[NEW]**

### 14.1 Shapes are a function of the serving configuration

A kernel does not have "a shape distribution." It has one per serving configuration. The GEMM shapes a model produces at ISL 1024 / OSL 256 / concurrency 32 are not the shapes it produces at concurrency 1, and prefill and decode are different regimes of the same kernel — large M versus M in the single digits. A kernel optimized against one and validated against nothing else is a kernel that will regress somebody's deployment.

```python
@dataclass
class ServingProfile:
    id: str
    model: str  # "Qwen/Qwen3-8B"
    dtype: str
    quantization: str | None
    tp: int
    ep: int | None
    isl: int
    osl: int
    concurrency: int
    prefill_decode_ratio: float | None
    attention_backend: str | None
    extra: dict[str, Any]


@dataclass
class WorkloadMatrix:
    profiles: list[ServingProfile]
    weights: dict[str, float]  # business importance, must sum to 1
```

### 14.2 Predicting shapes before running

Shapes are derivable from architecture plus serving profile without executing anything: hidden size, KV head count, FFN dimension, activation and RoPE configuration give the GEMM and attention shapes directly. Xe-Fuse — an external sibling project, not part of this repository — carries model presets for more than ten architectures (LLaMA 2/3/4, Gemma 2, Mistral, Qwen 2.5/3, Phi-3, Mixtral, DeepSeek-V3, DBRX); **reuse that table rather than rebuilding it** where Xe-Fuse is available. In-repo, `knowledge_base/sycl/xpu/real_model_shapes.yaml` and `llm_workload_shapes.yaml` already encode model and workload shapes and are the nearer seed.

This is worth having for two reasons: it lets the planner estimate where time will go before a single run, and it lets Orbit generate a `spec.yaml` covering shapes the current trace did not happen to hit but a neighbouring profile will.

### 14.3 Matrix-aware optimization and acceptance

- Each profile contributes its own shape distribution; distributions are merged into weighted `bench-gpu-N` variants (§8) using the matrix weights.
- **Acceptance requires a weighted win across the matrix *and* no per-profile regression beyond a declared threshold.** A candidate that wins decode by 20% and loses prefill by 8% is not an improvement, it is a trade — and it must be surfaced as one, per profile, in the report.
- `xe-orbit compare` emits a per-profile table, never a single number.

Example matrix:

```yaml
matrix:
  - id: qwen3-8b-chat
    model: Qwen/Qwen3-8B
    dtype: bf16
    tp: 1
    isl: 1024
    osl: 256
    concurrency: 32
    weight: 0.5
  - id: qwen3-8b-longctx
    model: Qwen/Qwen3-8B
    dtype: bf16
    tp: 1
    isl: 8192
    osl: 128
    concurrency: 8
    weight: 0.3
  - id: qwen3-8b-lowlat
    model: Qwen/Qwen3-8B
    dtype: bf16
    tp: 1
    isl: 512
    osl: 512
    concurrency: 1
    weight: 0.2
```

### 14.4 When one kernel cannot win everywhere

Sometimes it cannot, and pretending otherwise produces a candidate that is rejected for the wrong reason. Support **specialization sets**: several kernel variants plus a shape-bucket dispatch chosen at launch.

Rules, because this is easy to get wrong:

- The dispatch itself is measured. A branch on the host path costs launch time, and at decode shapes launch time is a meaningful fraction of the kernel.
- The dispatch is tested: every bucket is exercised by the test rig, and the bucket boundaries are recorded in the artifact.
- Specialization is reported honestly as increased complexity, with the maintenance cost stated. Two variants is usually reasonable; five is a sign the kernel wants a different algorithm rather than more branches.

### 14.5 CLI

```text
xe-orbit matrix show                       # profiles, weights, derived shapes
xe-orbit run --matrix matrix.yaml -- ...   # sweep all profiles
xe-orbit run --profile qwen3-8b-lowlat -- ...
xe-orbit compare baseline candidate --matrix matrix.yaml
```

---

## 15. Reference micro-workload and the pipeline test rig **[NEW]**

### 15.1 Why this exists

Nothing above can be developed against a full vLLM run. A closed loop that takes twenty minutes and burns tokens is not a loop anyone will iterate on, and it cannot run in CI. The project needs a **small workload that exercises the entire pipeline end to end in seconds**, and it should be built early — around PR 4, not as an afterthought.

### 15.2 What it is

`examples/orbit_mini/` — a two-layer decoder block with Qwen-shaped structure at toy dimensions: attention, RMSNorm, SwiGLU MLP, RoPE. Small enough to run in seconds, structured enough to produce a realistic kernel taxonomy under `torch.compile`.

It is **deliberately adversarial for extraction**, because a test workload that extracts cleanly tests nothing:

- the hot kernel's device helpers live in **three separate modules**, one reached through a re-export
- an `@triton.autotune` decorator with several configs, so the winning config must be captured and pinned
- an `@triton.heuristics` callable that closes over a module-level constant
- a tuned-config JSON keyed by device name, as a genuine data dependency
- one input deliberately non-contiguous, so synthetic-input reconstruction fails visibly
- **one hand-written SYCL kernel** registered as a dispatcher op, with its own tiny CMake build — so build-graph closure, the compiler-option sweep, the icpx harness and P1 override on a SYCL op are all covered in CI, not just the Triton path
- one opaque library call, so the E4 path and the `NO_ACTION` path are both exercised
- one region of three fusable kernels, for the Xe-Fuse path

### 15.3 The stub optimizer

CI must not call an LLM. `StubOptimizer` returns, on demand:

- a **known-good** variant with a real, reproducible speedup,
- a **known-bad** variant that is faster in the microbenchmark and slower end to end,
- an **incorrect** variant that passes the loose tolerance and fails the tight one,
- a **no-op** variant, byte-identical in behaviour.

Those four cover every branch of the decision logic deterministically, in seconds, at zero token cost.

### 15.4 What the rig asserts

| Assertion | Catches |
| --- | --- |
| Full loop completes: trace → kernels → capture → extract → emit → optimize → apply → compare | Integration rot |
| Bundle passes isolated import and mutation check (§12.12) | Incomplete closure; bundle silently using the installed package |
| Autotune config pinned, specialization matches the intercepted launch | Optimizing a variant the workload never runs |
| Operator override takes effect — new kernel in trace, old kernel absent | Silent no-op patches (§13) |
| Revert restores the baseline trace exactly | Leaky experiments |
| Known-good accepted; known-bad rejected on e2e despite winning the microbenchmark | Broken accept/reject logic |
| Incorrect variant rejected at L1, never reaching L4 | Tolerance handling |
| No-op variant returns `INCONCLUSIVE`, not `ACCEPT` | Null test, measurement noise (§10.7) |
| Injected slowdown of known size is detected at the right magnitude | Positive control for the whole measurement chain |
| Same kernel optimized across a two-profile matrix; per-profile regression rejected | Matrix acceptance logic (§14.3) |

### 15.5 Reuse

This rig is not just a test. It is the fixture the adapter conformance suite runs against (§10.7), the demo that runs in a meeting without a GPU queue, and the first thing a new contributor executes to see what the system does. Budget real effort for it.

`xe-orbit selftest` runs the whole thing. It should stay under a few minutes on CPU, with an XPU variant in nightly CI.

**Status (v10): built, and consumed by nothing — the gap is the wiring, not the workload.**
`examples/orbit_mini/` exists with every trap listed above, including the SYCL dispatcher
op with its own CMake build (not built by default) — but no test, no CLI path and no CI
step imports it; it runs only as `python -m examples.orbit_mini`. `xe-orbit selftest`
exists and is valuable, but it is **not** this rig: it runs entirely on synthetic stub
data (the four §15.3 variants as `StubVariant`s through `stats.compare`, the null test
and positive control, MDE/Amdahl gating, schema and store round-trips, GenericTorch
conformance, the gate-order and dispatch-assertion invariants, a core-purity import
scan, and `--chaos` failure injection). The only example actually exercised is
`examples/kernel_replacement/`, by `test_replacement_e2e.py` and `test_xpu_hardware.py`.
At audit time §15.4's table was asserted by nothing. Gap G4 (§24) closed the CPU half:
`tests/orbit/test_orbit_mini.py` arms the non-contiguous-input trap (including capture
round-trip), the three-module closure with the re-export, the tuned-config data
dependency and the SYCL op's fallback, all in T0. The silicon-dependent rows of §15.4
(real traces, launch interception, dispatch assertions) remain hardware-tier work.

**Found by running it on the real device, not by reading it:** the fixture's Triton
path had only ever exercised the no-triton compat shim, and real Triton refused it —
"Cannot access global variable CLAMP_LIMIT from within @jit'ed function": modern
Triton requires module-level constants read inside device functions to be
`tl.constexpr` *instances*. `helpers_c.CLAMP_LIMIT` and `helpers_b.GATE_SCALE` are now
constexpr instances (the compat shim degrades them to plain numbers, so the CPU path
is unchanged), and the trap is intact — the constants still live at module level in
sibling modules and still must travel with the bundle. After the fix the workload runs
on a real XPU with every trap armed, and its real trace reproduces §11.2's taxonomy:
the oneDNN GEMM at 41.9% of GPU time at E4 (§7.2 predicted "~40%"), the hand-written
Triton kernels at E2, and torch-xpu-ops' templated functors at E3 with graded
confidence.

---

## 16. Testability contract — every stage stands alone **[NEW]**

### 16.1 The principle

**A stage that can only be tested by running the whole pipeline is not a stage.** This is the direct consequence of §5.3: if the components are real, each one takes a typed artifact in, emits a typed artifact out, runs from its own CLI, and is testable without the stage before it and without the stage after it.

This is not optional polish. This pipeline has a dozen places where a silent failure produces a plausible-looking number — an incomplete closure that falls back to the installed package, an override that never takes effect, a specialization mismatch, a shape distribution from the wrong profile. Every one of those is caught by a cheap test at one stage and by nothing at all downstream.

### 16.2 Definition of done, per stage

A stage is not complete until all six hold:

1. **Published schema.** Its input and output artifacts have a versioned JSON schema, committed, with a schema version field in every artifact.
2. **Standalone CLI.** It runs from committed fixtures, with no live workload and no preceding stage.
3. **Contract test.** Round-trip: schema → object → schema. Plus a compatibility test against the previous schema version.
4. **Golden fixture.** At least one committed input/output pair from `orbit_mini` (§15), diffed on every run.
5. **At least one negative test.** A malformed, missing or adversarial input that must produce a clean, typed failure — never a crash and never a silent default.
6. **Appears in `xe-orbit selftest`.** If it is not in the self-test, it is not covered.

### 16.3 Record and replay

Every stage supports `--replay <run-id>`, running from stored artifacts instead of live hardware.

This is what makes the project testable on CPU-only CI. Trace fixtures let catalog, provenance, ranking, shape aggregation, matrix acceptance and decision logic all be tested with no GPU, in seconds, on every pull request. Only capture, extraction build, benchmarking and patch verification genuinely need silicon.

Replay is also the debugging tool. When a run produces a surprising decision, the whole downstream chain can be re-run against the exact artifacts that produced it, with a modified ranking function or threshold, without re-running the workload.

### 16.4 Stage test matrix

| # | Stage | In | Out | Golden test | Negative / injection test |
| --- | --- | --- | --- | --- | --- |
| 1 | `run` | `WorkloadSpec` | `manifest`, `measurement` | Fixed command → stable manifest fields | Non-zero exit; missing binary; clock instability flagged `INVALID` |
| 2 | trace ingest | raw profiler output | `events.json` | Committed trace → identical normalized events | Truncated trace; unknown event type; empty GPU stream |
| 3 | launch interception | live process | `launches.json` | mini kernel → expected constexprs, grid, warps | Kernel never launched; autotune changes config between runs |
| 4 | kernel catalog | `events.json` | `catalog.json` | Fixed events → fixed table incl. GPU%, gaps, MDE | All-host-bound trace must yield `NO_ACTION`, not a ranking |
| 5 | provenance | `catalog.json` | catalog + provider/source | Each resolver on a fixture per provider | Ambiguous name → low confidence, not a guess; unknown → `PROFILE_MORE` |
| 6 | input capture | live process | `CapturedInvocation` | Round-trip: reload → identical tensors, strides preserved | Non-contiguous input; missing data dep; tensor too large |
| 7 | extraction / closure | kernel + launches | `KernelBundle` | mini multi-file kernel → complete closure, hashes stable | Dynamic import → must downgrade to E3, not emit partial closure |
| 8 | bundle test | `KernelBundle` | pass/fail report | Isolated import, mutation, data-dep checks (§12.12) | Bundle secretly importing installed pkg **must fail** the mutation check |
| 9 | emit | bundle + shapes | `Model` + `spec.yaml` | Fixed shapes → fixed weighted variants | Shape distribution with no dominant mode; single-sample distribution |
| 10 | optimizer call | candidate dir | optimized kernel | `StubOptimizer` four variants (§15.3) | Agent returns unparseable output; timeout; empty diff |
| 11 | patch apply | candidate | applied + revert record | Each rung P1–P4 on mini | **Override that does not take effect must be detected** (§13) |
| 12 | bench / measure | applied workload | samples + CI | Null test: CI contains zero | Positive control: injected slowdown detected at right magnitude |
| 13 | decide / compare | measurements | `decision.json` | Four stub variants → four expected decisions | Per-profile regression → `REJECT`; noisy input → `INCONCLUSIVE` |
| 13b | language backend | kernel event | source + closure + build | Triton and SYCL fixtures resolve to the right TU/instantiation | Templated lambda resolving to several instantiations → low confidence, not a pick |
| 14 | adapter conformance | adapter | conformance report | Full suite (§10.7) | Adapter over-declaring a capability must fail |
| 15 | report | all artifacts | `report.json` | Fixed run → stable report | Missing upstream artifact → explicit gap, not a blank field |

Rows 8, 11 and 12 are the ones that catch plausible-looking lies. They should be written before the stages they test, not after.

### 16.5 Failure injection

Beyond per-stage negatives, `selftest --chaos` exercises the paths that only appear when something goes wrong: a kernel that fails to compile; a candidate that hangs; an out-of-memory during capture; a framework version bumped underneath a stored bundle (must invalidate, per §12.9); a git worktree left dirty; an agent that modifies files outside its declared scope; a correctness test deleted by an agent (must fail the run, loudly).

The last two are policy tests, not engineering tests, and they matter more than they look — §26 states the rules, and rules that are not tested are documentation.

### 16.6 CI tiers

| Tier | Runs on | Contents | Budget |
| --- | --- | --- | --- |
| **T0** | every PR, CPU only | schemas, contract tests, replay of all committed fixtures, golden diffs, stub decisions | minutes |
| **T1** | every PR, XPU | `xe-orbit selftest` on `orbit_mini` — full loop with `StubOptimizer` | < 10 min |
| **T2** | nightly, XPU | adapter conformance for every registered adapter, chaos suite, one real Xe-Forge optimization | ~1 hour |
| **T3** | weekly | full vLLM matrix run, three sessions, reproducibility check against the last accepted candidates | GPU-hours |

T3 doubles as the regression net for the version-rot problem in §12.9: framework versions move, and accepted candidates have to be re-validated on a schedule rather than trusted indefinitely.

Reality check, revised at v10: **T0 exists.** `.github/workflows/tests.yaml` runs, CPU-only, on every push and PR: `pytest tests/orbit -q` (~680 tests), `xe-orbit selftest --chaos`, adapter conformance for `generic_torch` and `vllm`, a full replay loop over the committed `tests/orbit/fixtures/decode_trace.json` (trace → kernels → regions → pipeline), and a drift check that `orbit/schemas/` matches the models. Pytest configuration landed in pyproject (`testpaths`, the `xpu` marker). Both remaining gaps have since closed: a second `core` CI job installs the full project (CPU torch) and runs every root-level test file — kept separate so the T0 job's minimal install stays the proof that Orbit's analysis path has no heavyweight dependency — and `ruff check` runs alongside the format check. T1 waits on an XPU runner; the golden trace fixture is `decode_trace.json`, and since G4 closed, `orbit_mini`'s CPU-viable traps are armed in T0 as well (§15). T2/T3 remain open. One more audit finding, fixed in place: the workflow's dependency list was missing `python-dotenv` — `xe_forge/__init__.py` eagerly imports `xe_forge.config`, which imports it — so every test failed at collection and this workflow had never been able to pass as committed.

---

## 17. Measurement methodology **[CHANGED]**

v1's example rejected a candidate on a -1.4% end-to-end delta. On XPU, with clock and power-state variation, that is inside noise. Publishing accept/reject decisions made this way is the most obvious attack surface on the eventual paper.

This machinery is new at the kernel level too, not only end-to-end: Xe-Forge's harness today reports a single scalar mean per run, so the weighted-kernel decisions in §9.1 need these statistics as much as the workload comparisons do.

Requirements:

1. **Repetitions.** No accept/reject decision from fewer than 5 measured runs (target 10 for e2e).
2. **Interleaving.** Alternate baseline and candidate runs (A,B,A,B,…). Never all-baselines-then-all-candidates — that confounds the comparison with thermal drift.
3. **Report intervals.** Every reported delta carries a 95% CI.
4. **Minimum detectable effect.** Compute MDE from observed baseline variance before running any optimization. If the Amdahl ceiling for a kernel is below the MDE, the kernel is not worth optimizing regardless of its microbenchmark potential — emit `NO_ACTION`.
5. **Clock state.** Lock GPU frequency where the platform permits; sample and record it where it does not. Flag any run with clock variance above threshold as `INVALID`.
6. **Decision rule:**

```text
CI excludes zero, positive   -> ACCEPT
CI excludes zero, negative   -> REJECT
CI straddles zero            -> INCONCLUSIVE   (not REJECT)
variance/clock anomaly       -> INVALID
```

`INCONCLUSIVE` with a stated MDE is a legitimate, publishable outcome. It is also information: it tells you the workload cannot resolve the gain you produced.

**Status (v10): implemented** — `orbit/stats.py` (stdlib-only: `estimate` with CI95, `minimum_detectable_effect`, Welch and paired intervals, `compare` returning exactly the four verdicts above, with `INVALID` on n<5, a zero baseline or unstable clocks) and `orbit/bench/core.py`'s `interleaved()` (ABBA, per §17.5.2). Under test in `tests/orbit/test_stats.py` and `test_bench.py`, including an ABBA-ordering regression test.

### 17.5 Pin what you are not measuring (v9)

Measured on a Wildcat Lake iGPU running Qwen2.5-0.5B under vLLM, where three things had to be nailed down before any comparison meant anything. All three are instances of one rule: **anything left to float becomes part of what you are measuring.**

* **Derived capacity.** vLLM sizes its KV cache from `gpu_memory_utilization` × total memory. On an iGPU that memory is shared with the desktop, so free memory swung between 5.4 GiB and 0.17 GiB across a single session and the cache came out at 270,400 tokens in one run and 122,048 in the next. Two runs then differ by cache capacity as well as by the change under test. Pinning `kv_cache_memory_bytes` to a fixed count costs one flag and makes the comparison a comparison.
* **Contention.** An orphaned benchmark process from an earlier run survived its supervisor and competed for the same device; free memory collapsed to 0.17 GiB. It failed loudly, which was luck — had it merely been slow, it would have produced plausible numbers attributed to the wrong cause. Arms now refuse to start below a free-memory floor: a refusal is not a measurement, and folding contention into a dtype comparison is exactly the silent-wrong-answer failure this project exists to avoid.
* **First-position effects.** The first engine load of a session measured 2421 tok/s against a steady state near 3400 — a 29% first-position effect from cold page cache and cold clocks. ABBA counterbalancing (§17 item 2) spreads a *gradient* across arms but cannot absorb a single outlier of that size at n=6. The answer is a **declared** warmup cycle, discarded before any measurement. Declaring it up front is what separates it from dropping an outlier after seeing the data, which is choosing the result.

Pinning capacity is only sound if the pinned value is not itself a hidden variable, and that is checkable: at batch 32 the same workload measured 1194.5 tok/s with a 1.5 GB cache and 1196.6 tok/s with a 400 MB one, in separate processes minutes apart — 0.2%. The cache has to be large enough and past that it is spare capacity, so pinning it removes a confound without introducing one. Two independent runs agreeing to 0.2% is also the cheapest available evidence that the harness is reproducible at all.

**The harness is itself a measurement decision (v9).** §5.4 says to use the framework's own benchmark, and that is right — it is the authority on what the workload's performance means. It is not a way to stop thinking about what is being measured. `vllm bench throughput` and a hand-rolled decode loop, given the same model, batch and output length, disagreed by **4x**: 314 output tok/s against 1194. Neither is wrong. The native harness times a single pass with no warmup, and on this device roughly nine of its eleven seconds went to the first prompt — first kernel launches, memory-pool growth, autotune — so it reports a *cold* number that amortizes as the prompt count rises (327 at 32 prompts, 420 at 256). The hand-rolled loop discarded a warmup pass and reports warm steady-state decode.

Both are legitimate; they answer different questions. Steady-state throughput is the right denominator for an A/B between two configurations, and the cold number is the right one for what a user sees on a fresh server. What is not legitimate is comparing one against the other, or quoting either without saying which it is. So a reported figure names its harness *and* whether warmup was excluded — and a cross-check between two harnesses is run once at the start, because discovering a 4x disagreement after building a result on top of it is expensive.

A fourth constraint is specific to integrated graphics and worth naming because it looks like a bug in the tool. `torch.xpu.mem_get_info()` reports *truly free* device memory, which on an iGPU excludes reclaimable page cache — it read 1.91 GiB while `/proc/meminfo` showed 7.3 GiB available and no process held the difference. vLLM's own startup gate uses that same figure, so an engine refuses to launch on a machine with plenty of usable memory, and the refusal names memory pressure that a human inspecting `free -h` cannot see. The practical consequence for a measurement plan: **on shared-memory devices, size the experiment to truly-free memory rather than to available memory**, and expect that budget to move under desktop load. Budgeting from `available` produces a run that launches sometimes.

The general form: before comparing A and B, list what else differs between them. Every item on that list is either pinned, counterbalanced, or reported as a limitation — and "we did not think to check" is not one of the three. Note how the first three were found: not by reading the code, but by a comparison failing in a way that could not be explained. The one that would have been dangerous is the contention case, because it was the only one that could have returned a number.

### 17.5.1 A worked INCONCLUSIVE (v9)

The decision rule earns its keep on a case where the obvious answer is wrong. float16 versus bfloat16, same workload, batch 32, ABBA ×3, n=6 per arm on Qwen2.5-0.5B:

```text
float16    n=6   1212.1 tok/s   95% CI [1203.6, 1220.6]   ±0.7%
bfloat16   n=6   1178.0 tok/s   95% CI [1092.6, 1263.4]   ±7.2%
MDE 1.18%   ->   INCONCLUSIVE, CI [-10.15%, +4.52%] straddles zero
```

Comparing means says *"bfloat16 is 2.8% slower, reject it"*. The intervals say we cannot tell, and the reason is visible in the spread: **one arm is ten times noisier than the other** (stdev 8.1 against 81.4), which came from a single run at 1012 tok/s against a band of 1206–1218. An asymmetry that large is diagnostic — it points at one contaminated measurement rather than at a property of bfloat16.

Two things follow, and the second is the uncomfortable one:

* **The MDE is not the whole story.** Baseline variance supports resolving 1.18%, and the comparison still could not resolve 10% — because MDE is computed from the *baseline* arm and the noise landed in the candidate. A declared MDE bounds what the workload can detect under well-behaved conditions; it does not promise the run was well-behaved.
* **The outlier stays in.** Dropping it yields 1211.1 against 1212.1, a −0.08% clean null — a tidier and more quotable result. There was no independent evidence that run was contended, only that its value was inconvenient: the pre-flight memory guard checks before a run, not during it. Removing a point because of where it landed is choosing the conclusion after seeing the data, and it is the same move as the declared-warmup discipline above, pointed the other way. `INCONCLUSIVE` with the reason attached is the correct output, and the follow-up it implies is *re-run under a quieter machine*, not *report the number I preferred*.

This is also the answer to why `INCONCLUSIVE` is a first-class verdict rather than a soft `REJECT` (§17 item 6). A `REJECT` here would have retired a candidate on the strength of one perturbed measurement.

### 17.5.2 A threshold is not a verdict (v9)

The optimization loop of §13.5 decided with `min_improvement_percent`: a fixed floor
applied to a point estimate. That is the thing §17 exists to forbid, and it was sitting
in the middle of the loop the rest of this design feeds.

The cost was visible in a live run. An agent-authored change measured **+0.63%** and was
rejected — not because it had been shown not to work, but because 0.63 is less than 2.
A threshold has two outputs and the situation has three: the candidate is faster, it is
slower, or this workload cannot resolve the difference. Collapsing the third into the
second turns "we could not tell" into "it did not work", which is the same error as
reading `INCONCLUSIVE` as `REJECT` (§17 item 6) — and it is worse inside a loop, because
the agent is then told a direction failed when nothing of the kind was established.

The loop now decides with `stats.compare` wherever samples are available. `ACCEPT` keeps,
`REJECT` reverts as slower, and `INCONCLUSIVE`/`INVALID` revert as **unproven** with the
MDE reported, so the reader learns what the workload could have detected. The fixed floor
survives only as a fallback for callers that can produce a single number.

**Sampling order matters as much as the test — but interleaving has a precondition, and
ignoring it makes the measurement worse.** Measuring every baseline sample and then every
candidate sample lets drift land entirely on the second arm. So kernel comparisons
interleave ABBA, patching and reverting between samples, ABBA rather than ABAB because
with ABAB the baseline always takes first position and absorbs every first-position
effect.

Applied to an in-place Python patch, that made things nine times worse. A source patch
cannot be swapped inside a live interpreter — the module is already imported — so every
sample needs a **fresh process**, and each one then carries process startup, JIT state
and cache variation. Measured on `gumbel_sample`: seven replicates *within* one process
spread 1.6%, while six ABBA-interleaved samples *across* processes spread 8.5%, taking
the MDE from roughly 1% to **14.31%**. A 1.44% effect that might have been resolvable
became unmeasurable.

The precondition, stated so it is checked rather than assumed: **interleaving only helps
when switching arms costs less variance than the drift it cancels.** Where an arm switch
means a new process, it does not, and the honest move is many replicates within one
process plus a warmup discard, accepting that slow drift remains a limitation rather than
trading it for a larger one. To interleave in-process, both variants must be loadable at
once — two differently-named kernels rather than one patched file — which is available
for Triton and not for a source patch to a framework.

### 17.6 Accepted gains do not add up (v9)

Everything above measures **one** change. It said nothing about what happens when five of them are accepted, and the obvious thing to do with five accepted percentages is add them. That is wrong three times over, and wrong in the direction that flatters the tool:

* **Percentages of different denominators do not add.** +12% then +8% is +20.96%, not +20%. Negligible at two steps; at eight steps of +10% the sum says +80% and the compounding says +114%.
* **Gains overlap.** Two kernels on one critical path are partly the same win counted twice. The Amdahl ceiling of §7.4 binds the stack, not only each entry in it.
* **The stack drifts.** Every accepted change alters what the next one is measured against — cache footprint, launch pattern, dispatch decision — and that drift belongs to no single entry.

So the rule is: **the headline number is measured, never derived.** A cumulative gain is a fresh end-to-end measurement of the full stack against the session baseline. The per-change deltas are kept — they are how each change earned its place — but they are marked unsummable and shown *beside* the validated total rather than instead of it.

Three consequences worth stating, because each is a place the implementation could quietly cheat:

* **A stack that was never re-measured has no cumulative result.** Not a derived one, not a compounded one — none, reported as `NOT ESTABLISHED`. A plausible number nobody took is worse than an admitted gap, and it is the number a reader is most likely to quote.
* **The gap between the parts and the whole is reported as drift, not distributed over the entries.** If the parts claim +44% and the stack measures +31%, those 13 points are the most interesting figure on the page: overlapping wins, interference, or a change nobody attributed. Beyond a few points it is a finding in its own right.
* **Each entry records how its contribution was established** — re-measured end to end, accepted locally but never re-measured, or absent. Rendering the three identically invites the reader to trust them equally.

**Measured instance (v9).** Sweeping vLLM's batch size on Qwen2.5-0.5B from 16 to 256 — five steps of +84.8%, +69.1%, +66.7%, +14.1%, +11.6% — the naive sum says **+246%** and the measured end-to-end gain is **+564%**. Here summing *understated* the result by more than a factor of two, which is worth noting because the failure is usually assumed to run the other way. The reason is the same either way: percentages of different denominators do not add, and only the measurement is the answer. In this particular case the compounded figure matched the measurement to within 0.00 points, because each step was a fresh end-to-end re-measurement of one chain rather than five independent optimizations — which is precisely the condition under which compounding is exact, and it does not generalize.

That sweep is also a reminder of what a throughput number omits: the same change made each batch take **2.41× longer to drain** (1.58s → 3.82s). Reporting the +564% alone would present a latency regression as an unqualified win, which is what §14.3 means by a trade that must be surfaced as one.

This section exists because of a comparison with AMD's **Hyperloom**, which solves the same problem for ROCm and reached the same distinction independently: it separates `gain_pct` ("gain against the session baseline — the only figure that can be summed") from `local_gain_pct` ("the executor's own figure … not summable"), tracks a `chain_continuous` flag for steps whose finishing throughput was never recorded, and headlines `cumulative_gain_pct_validated` taken from a validated end state. Two designs arriving at the same separation is reasonable evidence that it is real rather than a matter of taste.

**Status (v10): implemented** in `orbit/compare/cumulative.py`, under Orbit's own names (its docstring maps them to Hyperloom's): `GainMethod` MEASURED / LOCAL_ONLY / MISSING per entry, `local_delta_percent` marked unsummable, `validated_gain_percent` returning `None` for a stack that was never re-measured — NOT ESTABLISHED, never projected — with `naive_sum_percent`, `compounded_percent` and `drift_percent` reported beside it, and `chain_continuous` kept verbatim. Tested in `tests/orbit/test_cumulative.py`.

---

## 18. Gating and ranking **[CHANGED]**

**Ask whether the workload is GPU-bound before ranking any kernel.** v1 could spend an entire budget optimizing kernels in a workload that is host-bound — common in decode, where launch overhead and scheduler work dominate.

Computed at trace time, before selection:

```text
gpu_busy_percent
launch_gap_total_us            (unitrace / Level Zero)
host_bound_fraction
```

Per-candidate Amdahl ceiling:

```text
max_e2e_gain(k, s) = share(k) * (1 - 1/s) * gpu_busy_fraction
```

If `max_e2e_gain` at a plausible `s` is below the MDE, the correct action is `HOST_OPTIMIZATION`, `GRAPH_CAPTURE`, `CONFIG_CHANGE` or `NO_ACTION` — not a kernel rewrite.

Ranking (deterministic, no LLM):

```text
priority(k) = max_e2e_gain(k, s_est)
            * roofline_headroom(k)
            * action_availability(k)
            * provenance_confidence(k)
            * min(extraction_tractability(k), TRACTABILITY_CAP)
```

`roofline_headroom` comes from measured achieved TFLOPS / bandwidth against the hardware ceiling — you already generate these curves with `scripts/roofline.py`, whose presets cover B580 and Arc Pro B70 alongside Max-1550/1100 and Flex-170. **[CHANGED]** v1 used an unspecified `estimated_headroom` fudge factor; use the real roofline instead.

`extraction_tractability` is the new term: an E2 bundle iterates in seconds, an E3 harness in minutes, an E4 kernel not at all. A slightly less promising kernel that extracts cleanly is often the better first target, and the ranking should say so rather than leaving it to judgement.

**Two guards against language bias (§11.10):** the tractability term is capped so it breaks ties rather than overturning an order-of-magnitude difference in `max_e2e_gain`; and every run emits a **"considered but not attempted"** list with reasons. If the top three kernels by GPU time were all skipped, the headline number is not the story, and the report says so.

**[CHANGED] unitrace moves early.** v1 had Level Zero data at Stage 2 / PR 10. Launch-gap and GPU-busy data determine whether the entire pipeline should run, so it belongs at PR 3–4.

---

## 19. Correctness gates **[CHANGED]**

v1's "L3 — framework/model correctness" is unenforceable as written. For inference workloads, define it concretely.

```text
L0  build / import / registration succeeds
L0b extraction verification passes (§12.10) — right kernel, right specialization
L1  kernel correctness vs captured reference output (tightened tolerance)
L2  weighted kernel latency improves, no variant regresses
L3  model-level numerical gate:
      - greedy decode, fixed seed, fixed prompt set (>= 32 prompts)
      - token-exact match against baseline, OR
      - max abs/rel logit deviation within declared budget
    plus one small task eval
L4  end-to-end performance, with CI, per §17
L5  re-profile: confirm the kernel actually changed in the trace
```

L0b and L5 are not optional. L0b catches optimizing the wrong specialization; L5 catches an apparent e2e gain that came from something other than the change.

Hard rules for any agent in the loop:

- Never remove or weaken a correctness test.
- Never relax tolerance without explicit human authorization recorded in the experiment.
- Never change benchmark shapes, batch size, autotune search space, or methodology mid-experiment.
- Never compare against a re-run baseline from a different configuration or environment.
- Never report a candidate as successful on the basis of generated reasoning alone.
- Record every file modified, including incidental ones.

---

## 20. Execution model **[CHANGED]**

**Local execution only.**

```python
class Executor(Protocol):
    def run(self, cmd: list[str], env: dict, cwd: Path, timeout: float) -> RunResult: ...
```

Implement `LocalExecutor` and nothing else. Running inside an interactive Slurm allocation is local execution from Orbit's perspective — `salloc`, then `xe-orbit` — and needs no special support. The protocol exists so a remote/batch backend can be added later without touching call sites, not because one is being built now.

What this does mean:

- Do not assume the dev machine is the target GPU; the whole pipeline must be runnable from a shell on the target node.
- Do not embed hostnames, absolute cache paths, or scheduler assumptions in artifacts.
- Record device identity, driver version, and clock state in every manifest so results collected from different sessions remain comparable.

Candidate isolation uses git worktrees (§13), which works identically in an interactive allocation.

### 20.4 The stall gate: repeating an attempt is not progress (v9)

A loop that can retry will retry, and the cheapest thing for it to retry is whatever it just did. Unchecked, that produces a run which looks busy — attempts logged, time spent, budget consumed — and ends where it started, with the same failure repeated N times in the report instead of once. This matters more here than in a hand-driven tool, because an agent proposing the next step has every incentive to propose the step it already understands.

The rule is one line: **an attempt identical to one already made does not run again.** Identity is the tuple that determines the outcome — action, target, parameters — normalized so that irrelevant differences (dict ordering) cannot disguise a repeat as something new.

Two distinctions decide whether this helps or merely obstructs:

* **A repeat is a stall; a novel attempt is progress even when it also fails.** Failing *differently* is how a search moves — three GRF settings that all lose have told you something, and the same GRF setting three times has not. Only sameness is refused.
* **A timeout is not a repeat.** A timeout is a statement about the machine, not about the attempt, so the same attempt may succeed with more time or a warmer cache. It gets a bounded retry allowance rather than an exemption: one retry separates "the box was busy" from "this does not finish", and a second tells you nothing the first did not.

A refusal always carries its reason, because a gate whose decisions the caller cannot explain to a user is indistinguishable from a bug.

AMD's Hyperloom carries the same mechanism as its "novelty-ledger stall gate", over the same shape of tuple (component, ref, GPU arch, build command) and with the same timeout carve-out, so that the loop "keeps making forward progress rather than looping on an identical failing build". Two loops arriving at the same rule is a reasonable sign it is load-bearing rather than defensive.

**Status (v10): implemented** — `orbit/novelty.py` (`NoveltyLedger`, verdicts NOVEL / STALL / RETRY, normalized attempt tuples, one bounded timeout retry, every refusal carrying its reason), wired as the loop's first gate in `orbit/optimize/loop.py` and tested in `tests/orbit/test_novelty.py`.

---

## 21. CLI

As shipped (v10) — 22 subcommands, registered as the `xe-orbit` console script:

```text
xe-orbit frameworks                    # list adapters, tiers, capabilities  (§10)
xe-orbit run        -- <command>       # baseline, environment, timing
xe-orbit trace      -- <command>       # torch.profiler + unitrace + launch interception
                                       #   (--from-trace ingests an existing trace)
xe-orbit kernels    --run <id>         # catalog table
xe-orbit regions    --run <id>         # fusable region table
xe-orbit inspect    <kernel-id>        # provenance, shapes, headroom, extraction level
xe-orbit capture    <kernel-id>        # dump real inputs
xe-orbit extract    <kernel-id>        # build KernelBundle [--all --level --no-agent] (§12)
xe-orbit bundle     {test,verify,show} <bundle>   # §12.10, §12.12
xe-orbit emit       <kernel-id>        # write Model + spec.yaml
xe-orbit optimize   <kernel-id>        # ranked proposals by default; --apply runs the
                                       #   §13.5 loop (--harness, --measure, --samples)
xe-orbit apply      <candidate>        # patch-back into workload
xe-orbit patch      {status,recover}   # in-place journal inspection + crash recovery (§13.2)
xe-orbit compare    baseline <candidate> [--matrix matrix.yaml]
xe-orbit matrix     show               # profiles, weights, derived shapes (§14)
xe-orbit sources                       # SYCL source-registry state (§11.2)
xe-orbit support-matrix                # published support matrix (§5.3)
xe-orbit pipeline                      # staged run driver with stop conditions
xe-orbit conformance <adapter>         # §10.7 suite
xe-orbit selftest [--chaos|--quick]    # deterministic invariants — synthetic today (§15)
xe-orbit schemas                       # export/inspect artifact schemas (§16.2)
xe-orbit runs                          # list .orbit/runs
xe-orbit <stage> --replay <run-id>     # re-run any of 11 stages from artifacts (§16.3)
```

Differences from earlier drafts: `optimize-kernel`, `fuse-region` and `arena run` do not
exist — `optimize` is the entry point, fusion is authored inside the loop (§13.4), and
the arena is post-v0.1 (§5.4). `--replay` is a flag on the eleven stage commands, not a
command of its own. `xe-orbit optimize` plans by default and runs the loop only under
`--apply` with an operator-supplied correctness harness (G1, closed), so nothing
autonomous ships by accident.

Example catalog output:

```text
ID    GPU%    Calls   Operator          Provider      Extract  Actions              Conf
----------------------------------------------------------------------------------------
k0    41.2      256   aten.mm           oneDNN        E4       fuse,backend,layout  0.91
k1    24.8     8192   aten.rms_norm     Inductor      E2       rewrite,autotune     0.94
k2    16.1     8192   unified_attn      vLLM/Triton   E3       rewrite,autotune     0.88
k3     9.4     2048   unknown           unknown       --       profile_more         0.20

GPU busy: 71.3%   launch gaps: 18.2%   MDE (e2e): 1.9%
r0    57.3%   k0+k1   gemm+rmsnorm+swiglu   E3  -> Xe-Fuse
```

---

## 22. Primary workload **[CHANGED]**

**v1 said:** start with a `TinyModel` (linear → silu → norm), add vLLM in v0.2.

**v2 says:** vLLM-XPU is the v0.1 target. `TinyModel` is a plumbing smoke test only.

**v4 qualifies this:** first does not mean special. vLLM is reached only through the adapter protocol in §10, alongside `GenericTorchAdapter`, which is built at the same time precisely so vLLM's shape cannot become the core's shape. If something about vLLM cannot be expressed as an adapter capability, that is a boundary defect to fix immediately, not a shortcut to take.

Reasons:

1. Under `torch.compile`, a linear→silu→norm toy will very likely fuse into approximately one kernel and will not reproduce the provenance taxonomy the plan uses to validate itself. It also will not exercise extraction at all — every kernel will be a clean E2.
2. You already have published isolated results on `UnifiedAttention`, `BatchedMoE` and `FusedMoE` across 24 production configurations — the kernels are in-repo at `examples/vllm/`, extracted from vLLM at `ff712f64`. The unanswered question is not whether Xe-Forge can optimize them — it is whether those speedups survive to TTFT/TPOT.
3. That question is answerable in days, not months, and it is the first thing anyone will ask.

**First experiment, before most of the infrastructure exists:** take one kernel already optimized in the paper, patch it into vLLM-XPU by hand using the mechanism in §13, and measure TTFT/TPOT with the statistics in §17. If a 2.8x kernel produces an unresolvable e2e delta, that finding reshapes the entire roadmap and you want it in week one.

A useful second Phase 0 exercise, since it de-risks §12 directly: hand-extract three kernels of deliberately different difficulty — one Inductor Triton kernel, one vLLM Triton kernel with autotune and a JSON config, one IPEX/SYCL extension op — and write down what it actually took. That transcript is the specification for the extractors.

---

## 23. Artifacts

```text
.orbit/
+-- runs/<run-id>/
    +-- manifest.json           # versions, device, driver, clock state, env pins
    +-- workload.json
    +-- environment.json
    +-- measurement.json        # samples + CIs, not point values
    +-- traces/
    |   +-- torch_trace.json
    |   +-- unitrace/
    |   +-- launches.json       # intercepted launch records (§12.4)
    +-- kernels/catalog.json
    +-- regions/catalog.json
    +-- captures/<kernel-id>/
    +-- bundles/<kernel-id>/
    +-- candidates/<kernel-id>/
    +-- experiments/<exp-id>/
    |   +-- worktree_ref
    |   +-- agent_log.json
    |   +-- validation.json
    |   +-- decision.json
    +-- report.json
```

Agents consume artifacts. No agent parses console output when a structured artifact exists.

Every agent action logs: provider, model, task, target files, commands executed, files modified, full diff, validation result, benchmark result with CI, decision.

---

## 24. The remaining delta **[REWRITTEN in v10]**

v9's Phase 0–3 PR ladder described how to build the pipeline. The pipeline is now built —
§4's table says where — so a PR list for it would be a list of things not to do again.
What replaces it is the delta between what exists and the system §5 describes, ordered by
how much each item is worth. Tiers are ordered; within a tier, items are independent.

**Tier A — close the loop.** Small changes, and collectively the difference between a
tested library and the Hyperloom-class system this document describes.

All four Tier A gaps are **closed in-tree** (working tree, after the v10 audit); each
entry keeps its original statement and records what closed it.

- **G1. Wire `OptimizationLoop` into `xe-orbit optimize`.** `cmd_optimize` stopped after
  `ClaudeProposer.plan()`. *Closed:* `--apply` runs the full seam — implement each
  proposal in a workspace copy, apply through the journalled patcher (target's directory
  as the default sandbox root, `--sandbox` to widen), correctness via an
  operator-supplied `--harness` script (0/1/2 exit protocol, run in a fresh process),
  measurement via a `--measure` command with `--samples >= 5` enabling the §17
  statistical decision path, verdicts persisted to
  `experiments/<kernel>/loop_result.json`. Planning-only remains the default. Seam
  tests: `tests/orbit/test_cli_optimize.py` (accept, revert-slower, revert-wrong,
  no-edit, journal recovery, measurement contract).
- **G2. Persist the resolution tier.** *Closed:* `KernelRecord` carries
  `resolution_method` and a nullable `provenance_confidence` mirroring `SourceLocation`
  (schema 1.1, additive); the name-pattern resolvers stamp `NAME_MATCH`; the catalog
  persists the tier and ranks through `confidence_factor` (deterministic → 1.0);
  `inspect` and the catalog table render `exact`/float/`—` instead of a bare float.
- **G3. Revision pins in `kernel_sources`.** *Closed:* all three YAMLs carry
  `revision:` (torch-xpu-ops pinned to `bc294243` per PyTorch's `third_party/xpu.txt`
  for torch 2.13.0; the wheels without a known pin declare an explicit empty pin), and
  the registry reads each checkout's revision from `.git` and reports a per-tree
  `pin_state` — ok / skew / unpinned / unverified — with `xe-orbit sources` printing a
  SKEW warning naming both revisions.
- **G4. Wire `orbit_mini` into the rig.** *Closed:* `tests/orbit/test_orbit_mini.py`
  arms the CPU-viable traps in T0 — non-contiguous input surviving capture round-trip,
  the three-module closure with the re-export (complete, autotune and heuristics
  recorded), the tuned-config data dependency, and the SYCL op's torch fallback. The
  silicon-dependent assertions of §15.4 (real traces, dispatch) remain hardware-tier.

**Tier B — Xe-Forge core items (§9): closed in-tree.** Each entry keeps its original
statement and records what closed it.

- `weight:` + weighted objective (§9.1). *Closed:* `VariantSpec.weight` is parsed (a
  malformed weight raises rather than defaulting), `KernelSpec.weighted_family` walks
  a family's numbered siblings, and `core/weighted.py` implements
  `score(C) = Σ wᵢ·latency(C, vᵢ)` with the hard per-variant no-regression
  constraint — a candidate that wins the weighted total and regresses one family
  member is rejected naming the variant. Exposed as `--objective weighted` (with
  `--required-speedup`) through the dspy engine and `pipeline.optimize[_file]`; the
  per-variant table lands on `OptimizationResult.weighted`, and the headline
  `total_speedup` becomes the weighted figure. Orbit's `optimize_kernel_dir`
  threads `objective="weighted_latency"` straight through instead of apologising.
- Synchronous Claude result path (§9.9). *Closed:* `EngineConfig.synchronous` (+
  `claude_timeout_s`) makes `ClaudeEngine` block on the session and read back the
  edited kernel from the workspace's documented output location — returning it with
  `success=False` and an explicit "not measured" reason, because §19 forbids success
  on generated reasoning alone; measurement stays with the caller. The async default
  is byte-for-byte the legacy behaviour.
- Spec-driven SYCL dispatcher-op harness (§9.7). *Closed:* `orbit/patch/sycl_harness.py`
  renders a self-contained `Model` that drives `torch.ops.<ns>.<op>` (loader module
  and/or `load_library`, unreachable ops fail at construction naming what was tried)
  and emits the candidate directory `optimize_kernel_dir` resolves — nothing
  fabricated: a missing reference is a named note, not a stub passed off as one.
- Legacy-loader guard (§9.5). *Closed:* the loader now records every
  `framework_*.yaml` in `kb.skipped` with a reason naming its actual consumer,
  instead of loading it as a silent no-op.
- CI (§16.6). *Closed:* a second `core` job installs the full project (CPU torch) and
  runs every root test file; `ruff check` joined the format check. Two defects found
  on the way, both fixed: `xe_forge/core/__init__.py` eagerly imported the executor
  (and so ai_bench), which the root tests didn't need — it is now lazy (PEP 562, the
  same pattern `orbit/__init__.py` uses); and `agents/cover.py` imported dspy's
  *private* `_fmt_exc`, removed upstream after 3.3.0b1 while pyproject declares only
  a floor — guarded with an equivalent local fallback.

**Tier C — Hyperloom mechanisms: the bounded halves are closed in-tree; the deferred
halves are named.** Scheduled explicitly, because silently dropping them would repeat
the failure §4.1 exists to prevent.

- **Enablement ladder + runnable gate** (§5.6 item 1). *v0.1 slice closed:*
  `orbit/enablement.py` implements rung 0 deterministically — `diagnose()` classifies
  a failed launch (backend codegen, OOM, missing device, missing op, missing package,
  config; honest `unknown` fallback) and names the lowest ladder rung that could
  address it, with rungs 3–5 (scoped runtime, source localization, off-loop build)
  present in the enum but reported as *deferred* — a different finding from "no
  fix". The **runnable gate** is enforced as specified: KEEP requires boot *and* a
  passed accuracy eval — boot alone, or boot with no eval supplied, is never a KEEP.
  Wired where failures actually surface: `BenchRunner.measure` attaches the diagnosis
  to every workload that produced no samples. The climb itself — rungs 3–5 — remains
  the v0.2 headline.
- **A minimal policy gate.** *Closed:* `orbit/policy.py` — action allowlist, sandbox
  invariants delegated to (not duplicated from) `InPlacePatcher.check`, and an
  advisory single-writer lock per target with stale-holder breaking, every refusal
  naming its invariant. Wired as an optional `policy=` on `OptimizationLoop`; the
  loop's comment now says truthfully that this minimal gate exists and the full
  PRELUDE→CLOSE phase machine of §5.6 remains deferred.
- **`orbit-bench` standalone.** *Closed by implementing the standalone:* a console
  script (`orbit-bench = xe_forge.orbit.bench.cli:main`) with `run` (declared,
  discarded warmup; §17-grade JSON: samples + CI95 + MDE; `valid: false` naming a
  failed repetition; `decision_grade: false` under 5 repetitions) and `compare`
  (`stats.compare`, exit codes 0/1/2/3 for the four verdicts, 4 for refused inputs).
  Stdlib + orbit-internal imports only — proven to run without torch installed.
- **Arena** — *closed:* `orbit/arena.py` + `xe-orbit arena` — same task format the
  pipeline emits, isolated workspace per (contestant, task) pair, resumable with
  an identity check on persisted results, crash-contained, and a leaderboard that
  ranks only commensurable numbers: unmeasured held-out columns say so, and a
  contestant without a measured train mean is listed unranked rather than placed.

**Tier D — hygiene.** Root `AGENTS.md`/`CLAUDE.md` (§26); the `SGLangAdapter` (v0.2, the
portability test — its knowledge file already exists); `orbit/README.md`'s layout block
still names `optimize.py`, which became the `optimize/` package.

**v0.2, in this order:** the enablement ladder's climb — rungs 3–5 on the foundation
`orbit/enablement.py` lays (Tier C), then `orbit-arena` (agent A/B on the task format
the pipeline already emits), then `SGLangAdapter` — before the planner, before
campaigns — with lines-changed-outside-`adapters/` reported as the portability metric.
Then config/backend sweep executor, planner, campaign manager, budget accounting,
learned experiment memory.

---

## 25. Definition of v0.1 success **[CHANGED]**

v1's success criterion was that a sequence of commands runs. That is a demo criterion, not a research one. Replace with:

> On a named vLLM-XPU workload (model, dtype, TP, ISL, OSL, concurrency all fixed and recorded), `xe-orbit` autonomously identifies a kernel or region, extracts it at a verified fidelity level, produces a validated candidate, patches it back, and demonstrates an end-to-end throughput or TPOT improvement whose 95% CI excludes zero — reproducible across three independent sessions — within a stated agent-call and wall-clock budget.

Secondary criteria, all of which are also successes:

- The system correctly emits `NO_ACTION` on a host-bound workload rather than burning budget.
- The system correctly rejects a candidate that wins the microbenchmark and loses e2e.
- The system correctly refuses to optimize an unverified bundle rather than optimizing the wrong specialization.
- The reported MDE for each workload is documented, so `INCONCLUSIVE` results are interpretable.

Cost must be reported alongside: agent calls, tokens, GPU-hours, wall-clock. "Cost per validated percent of end-to-end speedup" is the number that will decide whether this scales across Intel.

---

## 26. Agent instructions

**Create and maintain** `AGENTS.md` (provider-neutral, authoritative) and `CLAUDE.md` (Claude-specific operational notes) at the repository root. Neither exists today. Do not confuse the root `CLAUDE.md` with `src/xe_forge/claude/templates/CLAUDE.md.j2`, the Jinja template rendered into *generated* optimization workspaces — the two serve different audiences and must not be merged.

`AGENTS.md` must state:

- Mission and the Orbit/Forge/Fuse boundary.
- Typed data models are mandatory; no dictionary-passing between subsystems.
- No parsing terminal output when a structured artifact exists.
- No LLM call for a deterministic task (see §3) — dependency closure and launch-record parsing are deterministic.
- Every candidate is developed in an isolated git worktree.
- Every candidate requires experimental validation through L0–L5.
- No single-run performance claims anywhere.
- An unverified bundle is never optimized.
- Correctness rules from §19, verbatim and non-negotiable.

`CLAUDE.md` adds:

- Inspect existing Xe-Forge abstractions before adding new ones — most of what a task appears to need already exists (§4).
- Prefer PR-sized changes; do not restructure adjacent subsystems opportunistically.
- Never bypass a test to make an optimization pass.
- Never relax tolerance, change benchmark methodology, alter input shapes, or widen an autotune search space during an experiment.
- Preserve baseline artifacts; never overwrite a measured run.
- Record provenance for every generated or modified source file.
- When extraction is ambiguous, downgrade a level and say so; do not guess at a closure.

---

## 27. Summary of changes from v1

| # | v1 | v2/v3 |
| --- | --- | --- |
| 1 | New `Xe-Orbit` repository | `xe_forge.orbit` in the Xe-Forge repo |
| 2 | New `KernelOptimizationTask` API | Emit the existing `Model` + YAML-spec contract; add a thin `optimize_kernel_dir` wrapper (§9.9) |
| 3 | Agent reconstructs kernel inputs | Capture real tensors at runtime |
| 4 | TinyModel first, vLLM in v0.2 | vLLM first; TinyModel is a smoke test |
| 5 | `editable: bool` | `actions_available: list[ActionType]` |
| 6 | No region concept; `KERNEL_FUSION` unimplemented | `RegionRecord` + Xe-Fuse backend |
| 7 | Patch-back unspecified | Custom op + post-grad pattern, worktree-isolated |
| 8 | Point-value comparisons | Repetitions, interleaving, CIs, MDE, `INCONCLUSIVE` |
| 9 | No host-bound gating | GPU-busy / launch-gap / Amdahl gate before ranking |
| 10 | unitrace at PR 10 | unitrace at PR 3 |
| 11 | Two knowledge bases | One extended `knowledge_base/` |
| 12 | Slurm/remote implied | Local executor only; thin protocol for later |
| 13 | `estimated_headroom` factor | Measured roofline headroom |
| 14 | Success = commands run | Success = e2e gain with CI excluding zero, reproducible, costed |
| 15 | "Extract the kernel source" (one line, PR 8) | **§12: bundle contract, E0–E4 ladder, launch interception, per-source extractors, autotune pinning, verification** |
| 16 | `FrameworkAdapter` sketched, deferred to v0.2 with vLLM | **§10: two-tier adapter model, declared capabilities, per-framework determinism profiles, knowledge-as-data, conformance suite with null test and positive control, SGLang scheduled as the portability proof** |
| 17 | Single-shape optimization, one workload configuration | **§14: serving profile matrix, per-profile acceptance, architecture-derived shape prediction, specialization sets** |
| 18 | No fast test path; everything validated against a real workload | **§15: `orbit_mini` adversarial reference workload, `StubOptimizer`, deterministic assertions, `xe-orbit selftest` in CI** |
| 19 | Patch-back = "custom op + pattern" | **§13: P1–P5 ladder, operator override preferred, mandatory dispatch assertion, recorded revert** |
| 20 | No external positioning; components implicitly monolithic | **§5: Hyperloom read structurally — standalone `orbit-bench`, agent arena outside the loop, published support matrix** |
| 21 | Testing implied, not specified | **§16: per-stage definition of done, schemas, record/replay, 15-row stage test matrix, chaos suite, T0–T3 CI tiers** |
| 22 | Triton-centric; SYCL treated as a hard case | **§11: `LanguageBackend` layer, build-graph closure, AOT/JIT and instantiation capture, compiler-option action space, P1 override for SYCL ops, ranking guard against language bias** |

---

## 28. The three things that decide this project

Not the model, not the agent framework, not the planner.

> **Extraction:** reliably pulling a kernel out of a framework that scatters it across generated code, imported device helpers, runtime-selected autotune configs, tuned-config data files, dispatch layers and compiled shared objects — and then *proving* the thing you extracted is the thing that ran.
>
> **Reinsertion:** getting the optimized artifact back into the running workload in a way that survives a recompile, and measuring the result well enough that the number is defensible.
>
> **Portability:** doing both of those through a boundary narrow enough that SGLang, and the framework after SGLang, cost a week each instead of a rewrite — and having a conformance suite that proves it rather than asserting it.

Everything between those is scheduling. Everything outside them is Xe-Forge, which already works.