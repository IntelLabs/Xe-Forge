# Adding a New DSL to Xe Forge

This guide explains how to add support for a new kernel **DSL** (domain-specific
language / programming model — e.g. Triton, Gluon, SYCL, CUDA C++) to the Xe Forge
optimization pipeline.

A "DSL" in Xe Forge is the source language the kernels are written in. The pipeline
is DSL-aware end to end: the **analyzer**, **planner**, **optimizer**, **executor**,
**knowledge base**, and **prompt library** all branch on the active DSL so that the
LLM sees the right instructions and the generated code is compiled, run, and verified
with the right toolchain.

Xe Forge currently ships four DSLs, defined in [src/xe_forge/models.py](src/xe_forge/models.py#L11-L21):

| DSL     | Value    | Language | Executor             |
|---------|----------|----------|----------------------|
| Triton  | `triton` | Python   | `KernelBenchExecutor` |
| Gluon   | `gluon`  | Python   | `KernelBenchExecutor` |
| SYCL    | `sycl`   | C++      | `SyclExecutor`        |
| CUDA    | `cuda`   | C++      | `KernelBenchExecutor` |

The DSL is selected at runtime via the `DSL` environment variable or the
`--dsl` CLI flag, both resolved in [src/xe_forge/config.py](src/xe_forge/config.py#L234-L236)
into `config.device_config.dsl`.

---

## Table of Contents

1. [Architecture overview](#architecture-overview)
2. [The pieces you must touch](#the-pieces-you-must-touch)
3. [Step 1 — Register the DSL enum](#step-1--register-the-dsl-enum)
4. [Step 2 — Declare supported stages (DSL registry)](#step-2--declare-supported-stages-dsl-registry)
5. [Step 3 — Provide an executor](#step-3--provide-an-executor)
6. [Step 4 — Wire executor selection](#step-4--wire-executor-selection)
7. [Step 5 — Add prompt-library entries](#step-5--add-prompt-library-entries)
8. [Step 6 — Add analyzer & optimizer signatures](#step-6--add-analyzer--optimizer-signatures)
9. [Step 7 — Add the knowledge base](#step-7--add-the-knowledge-base)
10. [Step 8 — (Optional) issue types & stage mapping](#step-8--optional-issue-types--stage-mapping)
11. [Step 9 — CLI / config plumbing](#step-9--cli--config-plumbing)
12. [Step 10 — Claude engine templates](#step-10--claude-engine-templates)
13. [Testing your DSL](#testing-your-dsl)
14. [Checklist](#checklist)

---

## Architecture overview

A single optimization run flows through [`XeForgePipeline.optimize()`](src/xe_forge/pipeline.py#L166)
roughly as follows:

```
spec (YAML) ──► input shapes, dtypes, FLOPs
kernel code ──► AnalyzerAgent.analyze()        # detects issues (DSL-aware prompt)
                       │
                       ▼
              PlannerAgent.plan()               # orders the stages
                       │
       filtered by  get_stages_for_dsl(dsl)     # dsl_registry.py
                       ▼
   for each stage:  OptimizerAgent.optimize_stage()
                       │
                       ▼  (CoVeR loop: generate → compile → run → compare)
              Executor.compare_kernels()        # KernelBenchExecutor or SyclExecutor
                       │
                       ▼
              re-analyze, next stage …
```

Three knobs make this DSL-aware:

- **`config.device_config.dsl`** — the active DSL string, read everywhere.
- **`DSL.code_language`** — `"python"` or `"cpp"`, used to pick file extensions,
  comment markers, and DSPy `Code[...]` types ([models.py](src/xe_forge/models.py#L17-L21)).
- **The DSL registry** — which optimization stages even apply to this DSL
  ([dsl_registry.py](src/xe_forge/dsl_registry.py)).

The cleanest mental model: **Triton is the reference DSL**. Anything that does not
special-case your DSL falls back to the Triton path, so the more your DSL resembles
Triton's Python+KernelBench flow, the less you have to write.

---

## The pieces you must touch

| # | Concern | File(s) | Required? |
|---|---------|---------|-----------|
| 1 | DSL identity | [models.py](src/xe_forge/models.py#L11-L21) | **Yes** |
| 2 | Supported stages | [dsl_registry.py](src/xe_forge/dsl_registry.py) | **Yes** |
| 3 | Compile / run / verify | [core/executor.py](src/xe_forge/core/executor.py) or [core/sycl_executor.py](src/xe_forge/core/sycl_executor.py) | **Yes** (reuse or new) |
| 4 | Executor selection | [pipeline.py](src/xe_forge/pipeline.py#L65-L80), [core/__init__.py](src/xe_forge/core/__init__.py#L118-L139) | **Yes** |
| 5 | Prompt components | [prompts/device_prompts.py](src/xe_forge/prompts/device_prompts.py) | **Yes** |
| 6 | DSPy signatures | [agents/analyzer_agent.py](src/xe_forge/agents/analyzer_agent.py), [agents/optimizer_agent.py](src/xe_forge/agents/optimizer_agent.py), [agents/react_agent.py](src/xe_forge/agents/react_agent.py) | If code rules differ from Triton |
| 7 | Knowledge base | [knowledge_base/](knowledge_base/) directory | Optional (recommended) |
| 8 | Issue types / mapping | [models.py](src/xe_forge/models.py#L45-L104), [knowledge/patterns.py](src/xe_forge/knowledge/patterns.py) | Only for novel issues |
| 9 | CLI / config | [cli.py](src/xe_forge/cli.py), [config.py](src/xe_forge/config.py) | Usually trivial |
| 10 | Claude engine templates | [claude/templates/](src/xe_forge/claude/templates/) | Only for the Claude engine |

The rest of this document walks each step in order.

---

## Step 1 — Register the DSL enum

Add a member to the `DSL` `StrEnum` in [src/xe_forge/models.py](src/xe_forge/models.py#L11-L21)
and make sure `code_language` returns the correct value for it:

```python
class DSL(StrEnum):
    TRITON = "triton"
    GLUON = "gluon"
    SYCL = "sycl"
    CUDA = "cuda"
    MOJO = "mojo"          # <-- new DSL

    @property
    def code_language(self) -> str:
        if self in (DSL.SYCL, DSL.CUDA):
            return "cpp"
        return "python"     # mojo falls here; add to the cpp tuple if it is C++-like
```

`code_language` drives:

- the saved-file extension and comment marker in [`_save_results`](src/xe_forge/pipeline.py#L582-L599)
  (`.py`/`#` vs `.cpp`/`//`), and
- the `dspy.Code["python"]` / `dspy.Code["cpp"]` type hint used by the CoVeR verify
  callback ([optimizer_agent.py](src/xe_forge/agents/optimizer_agent.py#L472)).

> The enum **value** (`"mojo"`) is the string used everywhere else — env var `DSL=mojo`,
> the `--dsl mojo` flag, knowledge-base directory name, and prompt lookups. Keep it
> lowercase and stable.

---

## Step 2 — Declare supported stages (DSL registry)

[src/xe_forge/dsl_registry.py](src/xe_forge/dsl_registry.py) maps each DSL to the
set of `OptimizationStage`s it supports. The planner's output is filtered against
this set in the pipeline ([pipeline.py](src/xe_forge/pipeline.py#L329-L332)), so any
stage you omit here will never run for your DSL.

```python
DSL_SUPPORTED_STAGES: dict[DSL, set[OptimizationStage]] = {
    ...
    DSL.MOJO: {
        OptimizationStage.ANALYSIS,
        OptimizationStage.ALGORITHMIC,
        OptimizationStage.DTYPE_FIX,
        OptimizationStage.FUSION,
        OptimizationStage.MEMORY_ACCESS,
        OptimizationStage.DEVICE_SPECIFIC,
        OptimizationStage.AUTOTUNING,
        OptimizationStage.DISCOVERY,
    },
}
```

Pick only the stages that make sense for the language. For example,
`BLOCK_POINTERS` and `PERSISTENT_KERNEL` are Triton/CUDA concepts and are
deliberately absent from the SYCL set. The full list of stages lives in the
`OptimizationStage` enum ([models.py](src/xe_forge/models.py#L30-L42)); the canonical
run order is `list(OptimizationStage)`, which `get_stages_for_dsl()` preserves.

If you skip this step, `get_stages_for_dsl()` falls back to the Triton stage set
([dsl_registry.py](src/xe_forge/dsl_registry.py#L57)).

---

## Step 3 — Provide an executor

The **executor** is what makes a DSL real: it takes generated source, compiles it,
runs it on the device, measures time/TFLOPS, and compares the optimized kernel
against the original for correctness. Its `compare_kernels()` feedback string is fed
straight back to the LLM inside the CoVeR loop.

Two executors exist today, and they define the contract you must satisfy:

### Option A — Reuse `KernelBenchExecutor` (Python-based DSLs)

[core/executor.py](src/xe_forge/core/executor.py) handles any DSL whose kernels are
**importable Python modules** exposing either a KernelBench-style `class Model` with
`forward()`, or a named callable. Triton, Gluon, and CUDA (via Python wrappers /
`torch.utils.cpp_extension`) all use it. If your DSL is invoked from Python, you
likely need **no new executor** — just make sure the generated code imports cleanly
and exposes `Model`/`forward`.

Key methods you can rely on:

- `execute(kernel_code, kernel_name, input_shapes, flop=…, dtype=…, init_args=…, input_dtypes=…)`
  → `ExecutionResult`
- `compare_kernels(original_code, optimized_code, …)` → `ComparisonResult` with a
  `feedback_message` for the agent.

### Option B — Write a new executor (compiled / out-of-process DSLs)

If your DSL needs a separate compiler and runs as a subprocess (like SYCL), model it
on [core/sycl_executor.py](src/xe_forge/core/sycl_executor.py). The `SyclExecutor`:

- writes source to a temp `.cpp`, compiles via `ai_bench.sycl.compiler.SYCLCompiler`,
- runs the binary with CLI args, parses `TFlop/s` and `ms` from stdout,
- compares outputs by dumping `D2.bin` files and `numpy.allclose`,
- returns a `SyclComparisonResult` whose `feedback_message` mirrors the
  `KernelBenchExecutor` wording (SUCCESS / REGRESSION / CORRECTNESS FAILURE).

**Contract for any executor** — to drop into the pipeline it must expose:

- `execute(...) -> ExecutionResult` (the model in [models.py](src/xe_forge/models.py#L181-L188))
- `compare_kernels(...) -> <result with .speedup, .feedback_message, .optimized_correct, .is_slower>`

Export your new class from [core/__init__.py](src/xe_forge/core/__init__.py).

---

## Step 4 — Wire executor selection

Two places choose the executor by DSL. Add a branch for yours (or let it fall through
to `KernelBenchExecutor` if you reuse Option A):

1. **Pipeline constructor** — [pipeline.py](src/xe_forge/pipeline.py#L65-L80):

   ```python
   if self.config.device_config.dsl == DSL.SYCL:
       executor = SyclExecutor(verify=…)
   else:
       executor = KernelBenchExecutor(device=…, …)   # triton/gluon/cuda/mojo land here
   ```

2. **`create_executor_from_config()`** — [core/__init__.py](src/xe_forge/core/__init__.py#L118-L139),
   used by skills/scripts.

The pipeline also has SYCL-specific branches around baseline measurement
([pipeline.py](src/xe_forge/pipeline.py#L223-L260) and
[L500-L516](src/xe_forge/pipeline.py#L500-L516)) that test `isinstance(self.executor, SyclExecutor)`.
If your DSL needs dims-based (M/N/K) execution rather than `input_shapes`, follow the
`_is_sycl` pattern; otherwise the `input_shapes` path is used automatically.

---

## Step 5 — Add prompt-library entries

[src/xe_forge/prompts/device_prompts.py](src/xe_forge/prompts/device_prompts.py)
centralizes all DSL/device-aware prompt text via `PromptLibrary(dsl, device_type)`.
At minimum, register a human-readable name so the LLM knows what it is writing:

```python
_DSL_NAMES: dict[str, str] = {
    "triton": "Triton",
    "gluon": "Gluon",
    "sycl": "SYCL/XeTLA",
    "cuda": "CUDA C++",
    "mojo": "Mojo",          # <-- new
}
```

Then review the methods that branch on `self.dsl` and add your cases as needed:

- `code_requirements()` — DSL-specific validation rules surfaced to the agent
  ([device_prompts.py](src/xe_forge/prompts/device_prompts.py#L123-L145)).
- `stage_guidance(stage)` — per-stage hints (e.g. how `block_pointers` or
  `autotuning` should be done in your DSL)
  ([device_prompts.py](src/xe_forge/prompts/device_prompts.py#L81-L121)).
- `optimizer_signature_doc()` / `analyzer_signature_doc()` — the system docstrings.

Anything you don't override degrades to a generic message, which is fine for a first
cut.

---

## Step 6 — Add analyzer & optimizer signatures

The agents pick a **DSPy signature** based on the DSL. Today the split is binary —
SYCL vs "everything else (Triton-shaped)":

- **Analyzer** — [analyzer_agent.py](src/xe_forge/agents/analyzer_agent.py#L332-L335):
  ```python
  sig = SyclAnalysisSignature if self.dsl == DSL.SYCL else AnalysisSignature
  ```
  The issue-category block is also built per-DSL via `_build_issue_categories(dsl)`
  ([analyzer_agent.py](src/xe_forge/agents/analyzer_agent.py#L70-L210)), with SYCL
  skipping Triton-only issue types (`_SYCL_SKIP_ISSUES`).

- **Optimizer** — [optimizer_agent.py](src/xe_forge/agents/optimizer_agent.py#L722-L776)
  selects `SyclOptimizationSignature` / `SyclAlgorithmicOptimizationSignature` for
  SYCL, else the Triton signatures. The CoVeR verify callback also branches: SYCL
  goes through `_verify_sycl`, the Triton path runs `ast.parse` + `@triton.jit`/`Model`
  checks ([optimizer_agent.py](src/xe_forge/agents/optimizer_agent.py#L479-L519)).

- **ReAct optimizer** (alternative strategy) — same pattern in
  [react_agent.py](src/xe_forge/agents/react_agent.py#L395).

**If your DSL is Python + KernelBench `Model` shaped** (like Triton), you can reuse the
default signatures and only adjust `code_requirements()` from Step 5 — the generic
path will validate and run it.

**If your DSL is C++/compiled or has different code rules**, add new signatures
(`MojoOptimizationSignature`, `MojoAnalysisSignature`, …) modeled on the SYCL ones and
extend the `if self.dsl == DSL.MOJO:` branches in the analyzer, optimizer, and react
agents, plus a `_verify_<dsl>` helper for the verify callback if the structural checks
differ.

---

## Step 7 — Add the knowledge base

The knowledge base is **optional** (the pipeline runs on the LLM's built-in knowledge
when it is empty or disabled) but strongly recommended for quality. It is loaded by
[knowledge/loader.py](src/xe_forge/knowledge/loader.py) and enabled with
`KNOWLEDGE_BASE_ENABLED=true` + `KNOWLEDGE_DIR=./knowledge_base`.

### Directory layout

`load_knowledge_base(dir, dsl, device_type)` collects YAML files in priority order
([loader.py](src/xe_forge/knowledge/loader.py#L318-L351)):

```
knowledge_base/
├── common/                     # DSL-agnostic, always loaded
│   ├── algorithmic_patterns.yaml
│   └── correctness.yaml
├── <dsl>/                      # e.g. triton, sycl, gluon  →  your new <dsl>/
│   ├── common/                 # optional: DSL-wide, device-agnostic
│   └── <device_type>/          # e.g. xpu, cuda
│       ├── *.yaml              # patterns + constraints for (dsl, device)
│       └── examples/
│           ├── index.yaml      # reference-kernel manifest
│           └── *.py / *.cpp    # the actual before/after kernels
```

So for a new DSL targeting XPU you would create
`knowledge_base/mojo/xpu/*.yaml` and `knowledge_base/mojo/xpu/examples/`.
Look at [knowledge_base/triton/xpu/](knowledge_base/triton/xpu/) and
[knowledge_base/sycl/xpu/](knowledge_base/sycl/xpu/) as templates.

### Pattern YAML schema

Each YAML file may contain `constraints:` and `patterns:` lists
([loader.py](src/xe_forge/knowledge/loader.py#L359-L391)):

```yaml
patterns:
  - id: large_tiles
    name: Use large tiles on XPU
    stage: device_specific        # must map to an OptimizationStage (aliases allowed)
    description: ...
    rationale: ...
    pattern_before: |             # or "before:"
      ...code...
    pattern_after: |              # or "after:"
      ...code...
    expected_speedup: "2-4x"
    notes: ...

constraints:
  - id: grf_mode_constexpr        # the id keyword routes it to a stage
    name: grf_mode must be constexpr
    severity: critical
    description: ...
```

Notes on the loader's behavior:

- `stage` strings are normalized through `_STAGE_ALIASES`
  ([loader.py](src/xe_forge/knowledge/loader.py#L29-L48)) — e.g. `memory`, `dtype`,
  `xpu_specific`, `stream_k` all resolve to canonical stages. Unmappable stages are
  skipped and logged.
- **Constraints have no `stage` field**; their target stage is inferred from keywords
  in their `id` via `_CONSTRAINT_STAGE_HINTS`
  ([loader.py](src/xe_forge/knowledge/loader.py#L50-L74)). A constraint with no keyword
  match applies to *all* stages.
- Only patterns/constraints for stages your DSL supports (Step 2) will ever be shown.

### Examples manifest

`examples/index.yaml` lists reference kernels with `stages:` tags and points at the
code files ([loader.py](src/xe_forge/knowledge/loader.py#L467-L549)). See
[knowledge_base/triton/xpu/examples/index.yaml](knowledge_base/triton/xpu/examples/index.yaml):

```yaml
examples:
  - id: gemm_activation
    name: GEMM + Activation Fusion
    stages: [algorithmic, fusion, device_specific, autotuning]
    description: ...
    unoptimized: gemm_activation_unoptimized.py   # or "file:" for optimized-only
    optimized: gemm_activation_optimized.py
    optimizations_applied: [ ... ]
    expected_speedup: 2-4x
```

If `stages:` is omitted, the loader infers them from keywords in the description /
`optimizations_applied` ([loader.py](src/xe_forge/knowledge/loader.py#L552-L607)).

How it is consumed: `KnowledgeBase.format_for_stage(stage)` returns only the
constraints, patterns, and examples relevant to the stage currently running
([loader.py](src/xe_forge/knowledge/loader.py#L165-L267)), keeping the context window
lean. The analyzer gets the critical constraints; the optimizer gets the stage's
patterns and examples.

---

## Step 8 — (Optional) issue types & stage mapping

If your DSL has optimization opportunities **not covered** by the existing
`IssueType` enum, you can add them — but in most cases you don't need to, because:

- the `OPEN_ENDED` / `DISCOVERY` mechanism lets the LLM propose novel optimizations
  without a predefined type ([models.py](src/xe_forge/models.py#L100-L104)), and
- new issue strings are auto-routed to a stage by keyword/prefix inference in
  [knowledge/patterns.py](src/xe_forge/knowledge/patterns.py#L98-L190).

If you do add a type:

1. Add the member to `IssueType` ([models.py](src/xe_forge/models.py#L45-L104)).
2. Map it to a stage in `_MAPPING` in
   [knowledge/patterns.py](src/xe_forge/knowledge/patterns.py#L32-L89) — or rely on the
   keyword/prefix layers, or call `register_stage(value, stage)` at runtime
   ([patterns.py](src/xe_forge/knowledge/patterns.py#L199-L213)).
3. Add a one-line description in `_descriptions` inside `_build_issue_categories`
   ([analyzer_agent.py](src/xe_forge/agents/analyzer_agent.py#L100-L172)) so the LLM
   knows when to emit it.
4. If the issue is Triton-only and should be hidden from your DSL, add it to a skip
   set analogous to `_SYCL_SKIP_ISSUES`.

Anything unmapped falls back to `ANALYSIS` and is skipped with a warning, so nothing
breaks silently.

---

## Step 9 — CLI / config plumbing

The `--dsl` flag is already generic: it accepts any string and sets the `DSL` env var
([cli.py](src/xe_forge/cli.py#L110-L114), [L242-L243](src/xe_forge/cli.py#L242-L243)),
which `ConfigManager` reads into `device_config.dsl`
([config.py](src/xe_forge/config.py#L234-L236)). So a new DSL is usable as
`--dsl mojo` with no parser change.

Review these DSL-string checks in the CLI and add your DSL where the behavior should
match a C++/compiled flow rather than the Python/reference flow:

- reference-implementation reading is gated to non-C++ DSLs
  ([cli.py](src/xe_forge/cli.py#L504), [L519-L521](src/xe_forge/cli.py#L519-L521)) —
  `if dsl not in ("sycl", "cuda")`.
- default variant resolution for compiled DSLs
  ([cli.py](src/xe_forge/cli.py#L428-L431)).

If your DSL needs its own device defaults (tile sizes, warps), either reuse
`XPUConfig`/`CUDAConfig` or add a new `DeviceConfig` subclass and branch in
`_build_device_config()` ([config.py](src/xe_forge/config.py#L288-L310)). Note device
config is keyed on **device type** (`xpu`/`cuda`), not DSL, so this is usually
unnecessary.

---

## Step 10 — Claude engine templates

Xe Forge has two engines ([engines/](src/xe_forge/engines/)): the automated **DSPy**
pipeline (everything above) and the **Claude** engine, which generates a ready-to-run
workspace. The Claude generator renders Jinja templates with the DSL
([claude/generator.py](src/xe_forge/claude/generator.py#L47-L69)) from
[claude/templates/](src/xe_forge/claude/templates/) (`CLAUDE.md.j2`,
`optimize-kernel.md.j2`, `tool-runner.md.j2`, `config.yaml.j2`).

If you want the Claude engine to support your DSL, make sure those templates handle
the `dsl` variable (instructions, file extensions, build/run commands). The DSPy
engine and the Claude engine are independent — you can ship a DSL on one without the
other.

---

## Testing your DSL

1. **Unit-level**: the spec loader and validator have tests in
   [tests/](tests/) ([test_spec_loader.py](tests/test_spec_loader.py),
   [test_validator.py](tests/test_validator.py)). Add KB-loading coverage modeled on
   [runners/test_kb_examples.py](runners/test_kb_examples.py) if you add a knowledge base.
2. **Knowledge base sanity**: run with `KNOWLEDGE_BASE_ENABLED=true` and check the
   startup log line `Knowledge base loaded (dsl=…): N patterns, M constraints, K examples`
   plus any "Skipped … unmappable stage" warnings
   ([loader.py](src/xe_forge/knowledge/loader.py#L306-L313)).
3. **End-to-end**: write a small kernel + YAML spec (mirror the pairs in
   [test_kernels/](test_kernels/)) and run:
   ```bash
   python -m xe_forge.cli --dsl mojo --device xpu --kernel my_kernel.<ext> --spec my_kernel.yaml
   ```
   Confirm the baseline measures, the planned stages are filtered to your supported
   set, and the executor's compile/run/compare feedback flows back into the agent.

---

## Checklist

- [ ] **Step 1** — `DSL` enum member + correct `code_language` ([models.py](src/xe_forge/models.py#L11-L21))
- [ ] **Step 2** — `DSL_SUPPORTED_STAGES` entry ([dsl_registry.py](src/xe_forge/dsl_registry.py))
- [ ] **Step 3** — executor: reuse `KernelBenchExecutor` or add a new one ([core/](src/xe_forge/core/))
- [ ] **Step 4** — executor selection in [pipeline.py](src/xe_forge/pipeline.py#L65-L80) and [core/__init__.py](src/xe_forge/core/__init__.py#L118-L139)
- [ ] **Step 5** — `_DSL_NAMES` + relevant `PromptLibrary` branches ([device_prompts.py](src/xe_forge/prompts/device_prompts.py))
- [ ] **Step 6** — analyzer/optimizer/react signatures (only if code rules differ from Triton)
- [ ] **Step 7** — `knowledge_base/<dsl>/<device>/` patterns, constraints, examples (optional)
- [ ] **Step 8** — new `IssueType`s + stage mapping (only if needed)
- [ ] **Step 9** — CLI DSL-string checks / device config (usually trivial)
- [ ] **Step 10** — Claude engine templates (only for the Claude engine)
- [ ] **Test** — KB load log is clean, stages filter correctly, an end-to-end run compiles/runs/compares

For the minimum viable DSL (Python-based, KernelBench `Model`-shaped, XPU), only
Steps 1, 2, and 5 are strictly required — everything else falls back to the Triton
path.
