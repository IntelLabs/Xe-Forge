# Adding a New DSL to Xe Forge

A **DSL** is the source language kernels are written in. Xe Forge is DSL-aware end to
end — analyzer, planner, optimizer, executor, knowledge base, and prompts all branch
on the active DSL. It ships with four, defined in `src/xe_forge/models.py`:

| DSL | Value | Language | Executor |
|-----|-------|----------|----------|
| Triton | `triton` | Python | `KernelBenchExecutor` |
| Gluon | `gluon` | Python | `KernelBenchExecutor` |
| SYCL | `sycl` | C++ | `SyclExecutor` |
| CUDA | `cuda` | Python | `KernelBenchExecutor` |

The DSL is chosen with `--dsl <name>` or the `DSL` env var, ending up in
`config.device_config.dsl`. **Triton is the reference path**: anything that doesn't
special-case a DSL falls back to it. A Python + KernelBench-`Model`-shaped DSL on XPU
only needs Steps 1, 2, and 5.

---

## Step 1 — Register the DSL enum

`src/xe_forge/models.py`:

```python
class DSL(StrEnum):
    TRITON = "triton"
    GLUON = "gluon"
    SYCL = "sycl"
    CUDA = "cuda"
    MOJO = "mojo"          # new

    @property
    def code_language(self) -> str:
        if self in (DSL.SYCL, DSL.CUDA):   # add MOJO here if it is C++-like
            return "cpp"
        return "python"
```

`code_language` decides saved-file extension (`.py`/`.cpp`), comment marker, and the
`dspy.Code["python"|"cpp"]` type. The enum value (`"mojo"`) is the string used for the
flag, env var, and knowledge-base directory — keep it lowercase.

## Step 2 — Declare supported stages

`src/xe_forge/dsl_registry.py`. The planner output is filtered to this set, so omitted
stages never run.

```python
DSL_SUPPORTED_STAGES = {
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

Include only stages that make sense (e.g. SYCL omits `BLOCK_POINTERS` and
`PERSISTENT_KERNEL`). Missing DSL → falls back to the Triton set.

## Step 3 — Executor

The executor compiles, runs, times, and compares kernels; its `compare_kernels()`
feedback string is fed back to the LLM. It must expose:

```python
def execute(...) -> ExecutionResult: ...
def compare_kernels(...):   # result has .speedup, .feedback_message, .optimized_correct, .is_slower
```

- **Python / importable kernels** (Triton, Gluon, CUDA): reuse `KernelBenchExecutor`
  (`src/xe_forge/core/executor.py`). Just emit code that imports cleanly and exposes a
  `class Model` with `forward()` (or a named callable). Usually **no new executor needed**.
- **Compiled / out-of-process** (like SYCL): model a new class on `SyclExecutor`
  (`src/xe_forge/core/sycl_executor.py`) — write source to temp file, compile, run as
  subprocess, parse timing, compare output dumps. Export it from `src/xe_forge/core/__init__.py`.

## Step 4 — Wire executor selection

Two spots pick the executor. Add a branch or let it fall through to
`KernelBenchExecutor`:

```python
# src/xe_forge/pipeline.py  (constructor)  and  src/xe_forge/core/__init__.py (create_executor_from_config)
if config.device_config.dsl == DSL.SYCL:
    executor = SyclExecutor(...)
else:
    executor = KernelBenchExecutor(...)   # triton/gluon/cuda/mojo
```

If your DSL runs from M/N/K dims instead of `input_shapes`, follow the `_is_sycl`
branches in `pipeline.py`.

## Step 5 — Prompt library

`src/xe_forge/prompts/device_prompts.py`. At minimum register the display name:

```python
_DSL_NAMES = {
    "triton": "Triton",
    "sycl": "SYCL/XeTLA",
    "mojo": "Mojo",          # new
}
```

Then add cases as needed in `code_requirements()` (validation rules) and
`stage_guidance(stage)` (per-stage hints). Unhandled cases degrade to generic text.

## Step 6 — Agent signatures (only if code rules differ from Triton)

Agents pick a DSPy signature per DSL — today it's SYCL vs Triton-shaped:

```python
# analyzer_agent.py
sig = SyclAnalysisSignature if self.dsl == DSL.SYCL else AnalysisSignature
# optimizer_agent.py: SyclOptimizationSignature / SyclAlgorithmicOptimizationSignature else Triton
```

Python+`Model`-shaped DSLs reuse the default signatures (only adjust Step 5). For a
C++/compiled DSL, add `MojoAnalysisSignature` / `MojoOptimizationSignature` modeled on
the SYCL ones, extend the `if self.dsl == DSL.MOJO:` branches in
`analyzer_agent.py`, `optimizer_agent.py`, `react_agent.py`, and add a `_verify_<dsl>`
helper for the CoVeR verify callback if the structural checks differ from the Triton
`ast.parse` + `@triton.jit`/`Model` checks.

## Step 7 — Knowledge base (optional, recommended)

Loaded by `src/xe_forge/knowledge/loader.py`, enabled with
`KNOWLEDGE_BASE_ENABLED=true`. Layout (priority: `common` → `<dsl>/common` → `<dsl>/<device>`):

```
knowledge_base/
├── common/                  # DSL-agnostic, always loaded
└── mojo/xpu/                # your <dsl>/<device>
    ├── *.yaml               # patterns + constraints
    └── examples/
        ├── index.yaml
        └── *.py / *.cpp
```

Pattern / constraint YAML:

```yaml
patterns:
  - id: large_tiles
    name: Use large tiles on XPU
    stage: device_specific        # aliases ok: memory, dtype, xpu_specific, stream_k...
    description: ...
    rationale: ...
    pattern_before: |
      ...code...
    pattern_after: |
      ...code...
    expected_speedup: "2-4x"

constraints:
  - id: grf_mode_constexpr        # stage inferred from keywords in the id
    name: grf_mode must be constexpr
    severity: critical
    description: ...
```

Examples manifest (`examples/index.yaml`):

```yaml
examples:
  - id: gemm_activation
    name: GEMM + Activation Fusion
    stages: [algorithmic, fusion, device_specific, autotuning]
    description: ...
    unoptimized: gemm_activation_unoptimized.py    # or "file:" for optimized-only
    optimized: gemm_activation_optimized.py
    expected_speedup: 2-4x
```

`format_for_stage()` shows only the constraints/patterns/examples for the stage
currently running, so context stays lean. Copy `knowledge_base/triton/xpu/` or
`knowledge_base/sycl/xpu/` as a starting template.

## Step 8 — Issue types (only if needed)

Usually skip this: the LLM can propose novel optimizations via the `OPEN_ENDED` /
`DISCOVERY` path, and unknown issue strings are auto-routed by keyword/prefix in
`src/xe_forge/knowledge/patterns.py`. To add a real type: add it to `IssueType`
(`models.py`), map it in `_MAPPING` (`patterns.py`), and give it a description in
`_build_issue_categories` (`analyzer_agent.py`).

## Step 9 — CLI / config

`--dsl` already accepts any string. Just check the DSL-string gates in
`src/xe_forge/cli.py` (e.g. `if dsl not in ("sycl", "cuda")` for reading the reference
implementation and default variant) and add your DSL where it should follow the
compiled-flow path instead of the Python/reference path. Device defaults are keyed on
device type, not DSL, so `config.py` rarely needs changes.

## Step 10 — Skills folder

`src/xe_forge/skills/` is a thin CLI wrapper around the core modules (`validate`,
`benchmark`, `analyze`, `profile`, `trial`). Even though they just call core, their
DSL knowledge is hardcoded and must be updated:

```python
# src/xe_forge/skills/__init__.py — add the value to every --dsl choices list
p_validate.add_argument("--dsl", default="triton",
                        choices=["triton", "sycl", "gluon", "cuda", "mojo"])
```

- `skills/benchmark.py` constructs `KernelBenchExecutor` directly — switch it to
  `create_executor_from_config(...)` (or branch on the DSL) so a compiled DSL gets the
  right executor.
- `skills/validate.py` forwards `--dsl` to `KernelValidator.validate(code, dsl=...)`
  in `src/xe_forge/core/validator.py`, which dispatches `_validate_triton` /
  `_validate_sycl`. Add a `_validate_<dsl>` branch there (else it falls back to the
  Triton checks).

## Step 11 — Claude engine templates (only for the Claude engine)

The Claude engine renders `src/xe_forge/claude/templates/*.j2` with the `dsl`
variable. If you want it to support your DSL, make those templates handle the new
value (build/run commands, extensions). The DSPy and Claude engines are independent.

---

## Test

```bash
python -m xe_forge.cli --dsl mojo --device xpu --kernel my_kernel.<ext> --spec my_kernel.yaml
```

Confirm: baseline measures, planned stages are filtered to your supported set, the KB
load log (`Knowledge base loaded (dsl=mojo): N patterns ...`) is clean with no
"unmappable stage" warnings, and the executor's compile/run/compare feedback flows
back into the agent. Mirror the kernel+spec pairs in `test_kernels/`.

## Checklist

- [ ] Step 1 — `DSL` enum + `code_language` (`models.py`)
- [ ] Step 2 — `DSL_SUPPORTED_STAGES` entry (`dsl_registry.py`)
- [ ] Step 3 — executor: reuse `KernelBenchExecutor` or add one (`core/`)
- [ ] Step 4 — executor selection (`pipeline.py`, `core/__init__.py`)
- [ ] Step 5 — `_DSL_NAMES` + `PromptLibrary` branches (`prompts/device_prompts.py`)
- [ ] Step 6 — agent signatures (only if code rules differ from Triton)
- [ ] Step 7 — `knowledge_base/<dsl>/<device>/` (optional)
- [ ] Step 8 — issue types (only if needed)
- [ ] Step 9 — CLI DSL-string checks (`cli.py`)
- [ ] Step 10 — skills folder: `--dsl` choices, executor, validator (`skills/`, `core/validator.py`)
- [ ] Step 11 — Claude engine templates (only for the Claude engine)
