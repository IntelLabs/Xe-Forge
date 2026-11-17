# Adding a DSL

A DSL is the language kernels are written in. The analyzer, planner, optimizer, executor,
knowledge base, prompts and workspace templates all branch on it. Xe-Forge has five,
defined in `src/xe_forge/models.py`:

| DSL | Value | File | Executor |
|---|---|---|---|
| Triton | `triton` | `.py` | `KernelBenchExecutor` |
| Gluon | `gluon` | `.py` | `KernelBenchExecutor` |
| SYCL | `sycl` | `.cpp` | `SyclExecutor` |
| CUDA | `cuda` | `.cu` | `KernelBenchExecutor` (built by the pipeline) |
| OpenCL | `opencl` | `.py`, holding the OpenCL C source, launched through `xe_forge.opencl` | `KernelBenchExecutor` |

The DSL is chosen with `--dsl` or `DSL` and ends up in `config.device_config.dsl`. Triton
is the reference path: code that does not special-case a DSL falls back to it. A DSL whose
kernels are Python files with a `Model` class needs steps 1, 2, 5 and 9 and, for the
workspace engines, step 10.

## Step 1: Register the enum

`src/xe_forge/models.py`:

```python
class DSL(StrEnum):
    TRITON = "triton"
    GLUON = "gluon"
    SYCL = "sycl"
    CUDA = "cuda"
    OPENCL = "opencl"
    MOJO = "mojo"  # new

    @property
    def code_language(self) -> str:
        if self in (DSL.SYCL, DSL.CUDA):
            return "cpp"
        return "python"

    @property
    def kernel_ext(self) -> str:
        if self == DSL.SYCL:
            return ".cpp"
        if self == DSL.CUDA:
            return ".cu"
        return ".py"
```

`code_language` decides the comment marker and the `dspy.Code["python"|"cpp"]` type.
`kernel_ext` is the suffix a kernel is written to, and the suffix that makes `-i` a
baseline rather than a reference. The value is the string of the flag, the variable and
the knowledge-base directory; keep it lowercase.

## Step 2: Declare the supported stages

`src/xe_forge/dsl_registry.py`. The plan is filtered to this set, so a stage left out never
runs; a DSL with no entry gets the Triton set.

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
        OptimizationStage.DISCOVERY,
    },
}
```

## Step 3: Executor

The executor compiles, runs, times and compares kernels; the feedback of
`compare_kernels()` goes back to the LLM. It exposes `execute(...) -> ExecutionResult` and
`compare_kernels(...)`, whose result carries the time ratio, the feedback message and
whether each arm was correct (`ComparisonResult` in `src/xe_forge/core/executor.py`).

- Importable kernels: reuse `KernelBenchExecutor` (`src/xe_forge/core/executor.py`). Emit
  code that imports cleanly and defines `class Model` with `forward()`.
- Compiled, out-of-process kernels: model a class on `SyclExecutor`
  (`src/xe_forge/core/sycl_executor.py`), which writes the source to a file, compiles it,
  runs it, parses the timing and compares outputs. Export it from
  `src/xe_forge/core/__init__.py`.

## Step 4: Executor selection

Three places pick the executor:

- `src/xe_forge/pipeline.py` (constructor): `DSL.SYCL` goes through
  `create_executor_from_config`, everything else gets `KernelBenchExecutor`.
- `src/xe_forge/core/__init__.py` (`create_executor_from_config`): the same split.
- `src/xe_forge/cli.py`: with a spec, the CLI builds a `KernelBenchExecutor` for the dspy
  engine unless the DSL is `sycl` or `cuda`, which the pipeline builds itself.

A DSL that runs from M/N/K dims rather than input shapes follows the `_is_sycl` branches
in `pipeline.py`.

## Step 5: Prompts

`src/xe_forge/prompts/device_prompts.py`. Register the display name:

```python
_DSL_NAMES = {
    "triton": "Triton",
    "gluon": "Gluon",
    "sycl": "SYCL/Sycl-tla",
    "cuda": "CUDA C++",
    "mojo": "Mojo",  # new
}
```

A DSL missing here is shown upper-cased (`opencl` is). Add cases to `code_requirements()`
and `stage_guidance(stage)` as needed; a DSL without them gets generic text.

## Step 6: Agent signatures (only if the code rules differ from Triton)

The agents pick a DSPy signature by DSL, SYCL or Triton-shaped:

```python
# analyzer_agent.py
sig = SyclAnalysisSignature if self.dsl == DSL.SYCL else AnalysisSignature
```

`optimizer_agent.py` and `react_agent.py` branch the same way, and the optimizer's
`compile_and_verify` tool calls `verify_sycl` (`src/xe_forge/agents/utils.py`) for SYCL and
runs the Triton structural checks (`ast.parse`, `import triton`, `@triton.jit`, `Model`)
otherwise. A compiled DSL adds signatures modeled on the SYCL ones and a branch in each.

## Step 7: Knowledge base

Optional. Read by the dspy engine when `KNOWLEDGE_BASE_ENABLED=true`, loaded by
`src/xe_forge/knowledge/loader.py` in this order: `common/`, `<dsl>/common/`,
`<dsl>/<device>/`.

```
knowledge_base/
  common/                  every DSL
  mojo/xpu/                <dsl>/<device>
    *.yaml                 patterns and constraints
    examples/
      index.yaml
      *.py / *.cpp
```

Pattern and constraint YAML:

```yaml
patterns:
  - id: large_tiles
    name: Use large tiles on XPU
    stage: device_specific        # aliases: memory, dtype, xpu_specific, ...
    description: ...
    rationale: ...
    pattern_before: |
      ...
    pattern_after: |
      ...

constraints:
  - id: grf_mode_constexpr        # stage inferred from keywords in the id
    name: grf_mode must be constexpr
    severity: critical
    description: ...
```

Examples manifest (`<dsl>/<device>/examples/index.yaml`):

```yaml
examples:
  - id: gemm_activation
    name: GEMM + Activation Fusion
    stages: [algorithmic, fusion, device_specific, autotuning]
    description: ...
    unoptimized: gemm_activation_unoptimized.py    # or "file:" for optimized-only
    optimized: gemm_activation_optimized.py
```

`format_for_stage()` shows only the current stage's constraints, patterns and examples.
`knowledge_base/triton/xpu/` and `knowledge_base/sycl/xpu/` are starting templates.

## Step 8: Issue types (only if needed)

Usually skip this: the `discovery` stage handles optimizations no type names, and unknown
issue strings are routed by keyword in `src/xe_forge/knowledge/patterns.py`. To add a type,
add it to `IssueType` (`models.py`), map it in `_MAPPING` (`patterns.py`), and describe it
in `_build_issue_categories` (`analyzer_agent.py`).

## Step 9: CLI and skills

- `src/xe_forge/cli.py`: add the value to the `--dsl` choices, and check the DSL-string
  gates (`dsl in ("sycl",)` for the default variant, `dsl not in ("sycl", "cuda")` for the
  CLI-built executor).
- `src/xe_forge/skills/__init__.py`: add it to the `--dsl` choices of `validate` and
  `benchmark`.
- `src/xe_forge/skills/benchmark.py` builds a `KernelBenchExecutor` directly; a compiled
  DSL needs a branch there, or a host benchmark command.
- `src/xe_forge/core/validator.py`: `validate()` dispatches to `_validate_triton` or
  `_validate_sycl` and falls back to `_validate_generic`. Add a `_validate_<dsl>` if the
  generic checks are not enough.

Device defaults are keyed on the device, not the DSL, so `config.py` rarely changes.

## Step 10: Workspace templates (claude, codex, dspy-agent)

The workspace engines render the `.j2` files in `src/xe_forge/claude/templates/` with `dsl`; codex and
dspy-agent run the same workspace. Make the templates handle the new value: build and run
commands, the file suffix, and how a `Model` launches the kernel (see the `sycl` and
`opencl` branches of `CLAUDE.md.j2`).

## Test

```bash
xe-forge --dsl mojo --device xpu -i my_kernel.<ext> -s my_kernel.yaml
```

Check that the baseline measures, the planned stages are within the supported set, the
knowledge-base log line (`Knowledge base loaded (dsl=mojo, ...)`) comes with no
"unknown/unmappable stage" warning, and the executor's feedback reaches the agent. Add kernel and spec pairs to
`test_kernels/`.

## Checklist

- [ ] 1: `DSL` enum, `code_language`, `kernel_ext` (`models.py`)
- [ ] 2: `DSL_SUPPORTED_STAGES` (`dsl_registry.py`)
- [ ] 3: executor (`core/`)
- [ ] 4: executor selection (`pipeline.py`, `core/__init__.py`, `cli.py`)
- [ ] 5: `_DSL_NAMES` and `PromptLibrary` cases (`prompts/device_prompts.py`)
- [ ] 6: agent signatures, only if the code rules differ
- [ ] 7: `knowledge_base/<dsl>/<device>/`, optional
- [ ] 8: issue types, only if needed
- [ ] 9: `--dsl` choices and DSL gates (`cli.py`, `skills/`, `core/validator.py`)
- [ ] 10: workspace templates
