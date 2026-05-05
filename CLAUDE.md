# Xe-Forge — Triton Kernel Optimizer for Intel XPU

Xe-Forge transforms PyTorch code into optimized Triton kernels for Intel XPU. Kernels must be numerically equivalent to the baseline and faster than it.

There are **two modes** with one shared toolbox:

- **Python / dspy mode** — run `xe-forge …` and the project's pipeline drives a tree-structured trial search in-process, using dspy agents (analyzer, planner, optimizer) and the trial engine under `src/xe_forge/trials/`.
- **Claude Code mode** — you (Claude) drive the workflow by invoking the same tools as stand-alone Python files and updating trial state manually. This is the mode described below.

Both modes share the same tools under `src/xe_forge/core/` and the same knowledge base under `knowledge_base/`.

## CONFIGURATION — use `.env` (not `config.yaml`)

At session start, make sure a `.env` file is present in the repo root (copy from `.env.example` if it isn't). The variables that matter for trial-driven optimization are:

- **`TRIAL_MAX_TRIALS`** — hard cap on trials (including t0). Always run all of them unless speedup > `TRIAL_EARLY_STOP_SPEEDUP`.
- **`TRIAL_EARLY_STOP_SPEEDUP`** — abort early when speedup reaches this value.
- **`TRIAL_PLATEAU_WINDOW`** — number of trials without a new global best before "try harder" kicks in.
- **`VTUNE_ENABLED`** — if `false`, skip ALL VTune profiling.
- **`VTUNE_BIN`** — path to the VTune binary (also settable via `VTUNE_BIN` env var).

Knowledge-base settings (`KNOWLEDGE_BASE_ENABLED`, `KNOWLEDGE_DIR`) and LLM settings (`LLM_MODEL`, `OPENAI_API_BASE`, …) also live in `.env.example`.

## RULES — NEVER VIOLATE

1. **ONLY create** Triton kernel files (`trials/<kernel_name>/t<N>.py` or staging files inside `trials/<kernel_name>/`). The `trial_manager` tool writes to `trials/` and `output/` for you.
2. **NEVER create** benchmark scripts, test scripts, helper utilities, or any other Python files.
3. **NEVER write custom scripts** to measure performance or test correctness — ONLY use the tools under `src/xe_forge/core/`.
4. If a tool fails, **STOP and report the error**. Do NOT work around it with custom scripts.
5. Generated kernels must be **self-contained** — all helper functions inline.
6. You **MUST run all `TRIAL_MAX_TRIALS` trials** from `.env`. Do NOT stop early due to plateau — LLM sampling can discover new ideas at any point. The only valid early stop is speedup > `TRIAL_EARLY_STOP_SPEEDUP`.

## MANDATORY TOOLS — Use these and ONLY these

**Delegate all tool execution** to the `tool-runner` agent to keep the main context clean.

**CRITICAL — Single-XPU serialization**: There is only ONE XPU on this machine. You MUST NOT dispatch multiple `tool-runner` agents in parallel if any of them runs `benchmark.py` or `xpu_profiler.py`. These GPU workloads must execute strictly one at a time — concurrent GPU jobs produce wrong results. CPU-only tools (`analyze_kernel.py`, `validate_triton.py`, `trial_manager.py`) are safe to parallelize with each other and with anything else.

| Task | Command |
|------|---------|
| **Analyze** | `python src/xe_forge/core/analyze_kernel.py <pytorch_file>` |
| **Validate** | `python src/xe_forge/core/validate_triton.py <triton_file>` |
| **Benchmark** | `python src/xe_forge/core/benchmark.py <baseline_file> <triton_file> [--triton-baseline] [--baseline-us <cached>]` |
| **Init trials** | `python src/xe_forge/core/trial_manager.py init <kernel_name> <baseline_file> [--triton-baseline]` |
| **Save trial** | `python src/xe_forge/core/trial_manager.py save <kernel_name> <file> [--parent <parent_id>] [--strategy "..."]` |
| **Record result** | `python src/xe_forge/core/trial_manager.py result <kernel_name> <trial_id> --validation pass --correctness <pass\|fail> --speedup <float> --baseline_us <float> --triton_us <float>` |
| **Check status** | `python src/xe_forge/core/trial_manager.py status <kernel_name>` |
| **Best trial** | `python src/xe_forge/core/trial_manager.py best <kernel_name>` |
| **Baseline time** | `python src/xe_forge/core/trial_manager.py baseline-us <kernel_name>` — cached baseline time for `--baseline-us` |
| **Finalize** | `python src/xe_forge/core/trial_manager.py finalize <kernel_name> <name>_triton.py` |
| **Profile** | `python src/xe_forge/core/xpu_profiler.py <triton_file>` — VTune GPU hardware counters + optimization recommendations |

## WORKFLOW — Follow these steps in order

### Step 1: Analyze
- Read the baseline source file. Identify shapes, dtypes, operations, fusion opportunities.
- If baseline is PyTorch: run `python src/xe_forge/core/analyze_kernel.py <pytorch_file>`.
- If baseline is Triton (`--triton-baseline`): skip `analyze_kernel.py` (it only supports PyTorch). Read the Triton file directly.
- Read relevant KB files: start with `knowledge_base/correctness.yaml` and `knowledge_base/xpu_optimizations.yaml`.
- Check `knowledge_base/examples/index.yaml` for similar patterns.
- Read `knowledge_base/implementation_reference.md` for code templates and the `Model` class pattern.

### Step 2: Initialize
```bash
python src/xe_forge/core/trial_manager.py init <kernel_name> <baseline_file> [--triton-baseline]
```

### Step 3: Trial Loop (always run all `TRIAL_MAX_TRIALS` from `.env`)
For each trial:
1. **Write kernel** — start from a reference in `knowledge_base/examples/` or modify the previous trial. See `knowledge_base/implementation_reference.md`.
2. **Validate** — `python src/xe_forge/core/validate_triton.py <triton_file>` (fix until passing; doesn't count as a trial).
3. **Save** — `python src/xe_forge/core/trial_manager.py save <kernel_name> <triton_file> --parent <parent_id> --strategy "description"`. Omit `--parent` for the first trial (t0).
4. **Benchmark** (MANDATORY every trial):
   - **Trial t0:** `python src/xe_forge/core/benchmark.py <baseline_file> <triton_file> [--triton-baseline]` (measures both baseline and triton).
   - **Trials t1+:** Get cached baseline via `python src/xe_forge/core/trial_manager.py baseline-us <kernel_name>`, then `python src/xe_forge/core/benchmark.py <baseline_file> <triton_file> [--triton-baseline] --baseline-us <cached_value>` (skips baseline perf, saves time).
   - **After `finalize`:** Re-run `benchmark.py` without `--baseline-us` for the final accurate comparison.
5. **Record** — `python src/xe_forge/core/trial_manager.py result <kernel_name> <trial_id> --validation pass --correctness <pass|fail> --speedup <float> --baseline_us <float> --triton_us <float>` (runtimes from benchmark output).
6. **Profile (MANDATORY after t1, if `VTUNE_ENABLED=true`)** — `python src/xe_forge/core/xpu_profiler.py <triton_file>`. Use its output to guide subsequent trial strategies. Run again if speedup plateaus after 2+ additional trials. **Skip this step entirely if `VTUNE_ENABLED=false`.**
7. **Decide next action** (use profiler output from step 6):
   - Speedup > `TRIAL_EARLY_STOP_SPEEDUP` → stop (excellent), finalize.
   - Speedup improved → continue on this branch, try next optimization level.
   - Speedup regressed → branch back to the best trial, try a different strategy.
   - Correctness failed → fix on the same branch.
   - Profiler says low occupancy (if `VTUNE_ENABLED`) → increase tile sizes, check `knowledge_base/xpu_optimizations.yaml`.
   - Profiler says overhead kernels dominate (if `VTUNE_ENABLED`) → pre-pack to bf16, see `knowledge_base/optimization_levels.yaml`.
   - Plateau → do NOT stop. Try a fundamentally different approach (different algorithm, tiling, fusion strategy). LLM sampling can discover new ideas.
   - See `knowledge_base/optimization_strategies.md` for the full "try harder" decision tree.

### Step 4: Finalize
```bash
python src/xe_forge/core/trial_manager.py finalize <kernel_name> <name>_triton.py
```

## CRITICAL CORRECTNESS CONSTRAINTS

- NO default values for `@triton.autotune` meta-parameters in kernel signature.
- Use 1D grid when applying tile swizzling (GROUP_SIZE_M).
- `boundary_check` uses dimension indices `(0, 1)`, not booleans.
- Cast batch indices to `int64` before stride multiplication.
- Prefer tensor descriptors (`tl.make_tensor_descriptor`) over block pointers for all new kernels on XPU.
- Do NOT mix block pointer and tensor descriptor APIs on the same operation.
- Pre-zero output buffers when using atomic accumulation.
- `Model` class must be compatible with ai-bench (`nn.Module` with `nn.Linear`).
- Match `get_inputs()`, `get_init_inputs()`, and module-level constants from `*_pytorch.py`.

## REFERENCE DOCS — Read these during Step 1

| Doc | Contents |
|-----|----------|
| `knowledge_base/implementation_reference.md` | Code templates, `Model` class pattern, GEMM example |
| `knowledge_base/optimization_strategies.md` | Strategy reference, optimization levels, checklist, KB index |
| `knowledge_base/workflow_details.md` | Detailed workflow, decision tree, benchmarking/validation details |
| `knowledge_base/correctness.yaml` | Critical constraints to avoid bugs |
| `knowledge_base/xpu_optimizations.yaml` | XPU-specific patterns (tensor descriptors, GRF, swizzling) |
| `knowledge_base/fusion_patterns.yaml` | When to fuse vs split operations |
| `knowledge_base/optimization_levels.yaml` | Progressive optimization with "try harder" decision tree |
| `knowledge_base/examples/index.yaml` | Reference implementations catalog |
| `knowledge_base/examples/` | Optimized reference kernels |

## Trial workflow integration (Python / dspy mode)

When invoked through `xe-forge` the same trial tree is written by `src/xe_forge/trials/runner.py`, driven by a pluggable search strategy (default `tree_walk`, selectable via `TRIAL_SEARCH`) and a pluggable writer (default `stage_sequence`, selectable via `TRIAL_WRITER`). The trial state — parent/child relationships, speedups, cached `baseline_us`, best trial — is written to the same `trials/<kernel_name>/state.json` file either mode uses, so you can inspect it with `trial_manager.py status <kernel_name>` regardless of which mode produced it.

## EXISTING BASELINES ARE NAIVE

The `test_kernels/*.py` Triton files (non-pytorch) are **unoptimized baselines**. They use manual pointer arithmetic, lack autotune, and miss XPU optimizations. Do NOT copy their patterns. Use `knowledge_base/examples/` as the source of good patterns.
