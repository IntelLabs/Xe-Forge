# Triton Kernel Optimizer for Intel XPU

Transform PyTorch code into optimized Triton kernels for Intel XPU. Kernels must be numerically equivalent and faster than baseline.

## CONFIGURATION — Read `config.yaml` first

At the start of every session, read `config.yaml` in the project root. It controls:
- **`max_trials`** — hard cap on optimization trials; always run all of them (use this instead of hardcoded "10")
- **`vtune_enabled`** — if `false`, skip ALL VTune profiling steps (Step 3.6 and profiler-related decisions)
- **`vtune_bin`** — path to the VTune binary (also settable via `VTUNE_BIN` env var)

## RULES — NEVER VIOLATE

1. **ONLY create** Triton kernel files (`test_kernels/*_triton.py` or trial files `t<trial_id>.py`).
2. **NEVER create** benchmark scripts, test scripts, helper utilities, or any other Python files.
3. **NEVER write custom scripts** to measure performance or test correctness — ONLY use `skills/benchmark.py`.
4. If a tool fails, **STOP and report the error**. Do NOT work around it with custom scripts.
5. Generated kernels must be **self-contained** — all helper functions inline.
6. You **MUST run all `max_trials` trials** from `config.yaml`. Do NOT stop early due to plateau — LLM sampling can discover new ideas at any point. The only valid early stop is speedup > 5x.

## MANDATORY TOOLS — Use these and ONLY these

**Delegate all tool execution** to the `tool-runner` agent to keep the main context clean.

**CRITICAL — Single-XPU serialization**: There is only ONE XPU on this machine. You MUST NOT dispatch multiple tool-runner agents in parallel if any of them runs `benchmark.py` or `xpu_profiler.py`. These GPU workloads must execute strictly one at a time — concurrent GPU jobs produce wrong results. CPU-only tools (`analyze_kernel.py`, `validate_triton.py`, `trial_manager.py`) are safe to parallelize with each other and with anything else.

| Task | Command |
|------|---------|
| **Analyze** | `python skills/analyze_kernel.py <pytorch_file>` |
| **Validate** | `python skills/validate_triton.py <triton_file>` |
| **Benchmark** | `python skills/benchmark.py <baseline_file> <triton_file> [--triton-baseline] [--baseline-us <cached>]` |
| **Init trials** | `python skills/trial_manager.py init <kernel_name> <baseline_file> [--triton-baseline]` |
| **Save trial** | `python skills/trial_manager.py save <kernel_name> <file> [--parent <parent_id>] [--strategy "..."]` |
| **Record result** | `python skills/trial_manager.py result <kernel_name> <trial_id> --validation pass --correctness <pass\|fail> --speedup <float> --baseline_us <float> --triton_us <float>` |
| **Check status** | `python skills/trial_manager.py status <kernel_name>` |
| **Best trial** | `python skills/trial_manager.py best <kernel_name>` |
| **Baseline time** | `python skills/trial_manager.py baseline-us <kernel_name>` — cached baseline time for `--baseline-us` |
| **Finalize** | `python skills/trial_manager.py finalize <kernel_name> <name>_triton.py` |
| **Profile** | `python skills/xpu_profiler.py <triton_file>` — VTune GPU hardware counters + optimization recommendations |

## WORKFLOW — Follow these steps in order

### Step 1: Analyze
- Read the baseline source file. Identify shapes, dtypes, operations, fusion opportunities.
- If baseline is PyTorch: run `python skills/analyze_kernel.py <pytorch_file>`.
- If baseline is Triton (`--triton-baseline`): skip `analyze_kernel.py` (it only supports PyTorch). Read the Triton file directly.
- Read relevant KB files: start with `kb/correctness.yaml` and `kb/xpu_optimizations.yaml`.
- Check `kb/examples/index.yaml` for similar patterns.
- Read `kb/implementation_reference.md` for templates and the Model class pattern.

### Step 2: Initialize
```bash
python skills/trial_manager.py init <kernel_name> <baseline_file> [--triton-baseline]
```

### Step 3: Trial Loop (always run all `max_trials` from config.yaml)
For each trial:
1. **Write kernel** — start from `templates/` or modify previous trial. See `kb/implementation_reference.md`.
2. **Validate** — `python skills/validate_triton.py <triton_file>` (fix until passing; doesn't count as a trial).
3. **Save** — `python skills/trial_manager.py save <kernel_name> <triton_file> --parent <parent_id> --strategy "description"`. Omit `--parent` for the first trial (t0).
4. **Benchmark** (MANDATORY every trial):
   - **Trial t0:** `python skills/benchmark.py <baseline_file> <triton_file> [--triton-baseline]` (measures both baseline and triton).
   - **Trials t1+:** Get cached baseline via `python skills/trial_manager.py baseline-us <kernel_name>`, then run `python skills/benchmark.py <baseline_file> <triton_file> [--triton-baseline] --baseline-us <cached_value>` (skips baseline perf, saves time).
   - **After `finalize`:** Re-run `benchmark.py` without `--baseline-us` for final accurate comparison.
5. **Record** — `python skills/trial_manager.py result <kernel_name> <trial_id> --validation pass --correctness <pass|fail> --speedup <float> --baseline_us <float> --triton_us <float>` (runtimes from benchmark output).
6. **Profile (MANDATORY after t1, if `vtune_enabled` is true in config.yaml)** — Run `python skills/xpu_profiler.py <triton_file>` after your first benchmarked trial. Use its output to guide subsequent trial strategies. Run again if speedup plateaus after 2+ additional trials. **Skip this step entirely if `vtune_enabled` is false.**
7. **Decide next action** (use profiler output from step 6 to inform decisions):
   - Speedup > 5x → stop (excellent), finalize
   - Speedup improved → continue on this branch, try next optimization level
   - Speedup regressed → branch back to best trial, try different strategy
   - Correctness failed → fix on same branch
   - Profiler says low occupancy (if vtune_enabled) → increase tile sizes, check `kb/xpu_optimizations.yaml`
   - Profiler says overhead kernels dominate (if vtune_enabled) → pre-pack to bf16, see `kb/optimization_levels.yaml`
   - Plateau → do NOT stop. Try a fundamentally different approach (different algorithm, tiling, fusion strategy). LLM sampling can discover new ideas.
   - See `kb/optimization_strategies.md` for the full "try harder" decision tree

### Step 4: Finalize
```bash
python skills/trial_manager.py finalize <kernel_name> <name>_triton.py
```

## CRITICAL CORRECTNESS CONSTRAINTS

- NO default values for `@triton.autotune` meta-parameters in kernel signature
- Use 1D grid when applying tile swizzling (GROUP_SIZE_M)
- `boundary_check` uses dimension indices `(0, 1)`, not booleans
- Cast batch indices to `int64` before stride multiplication
- Prefer tensor descriptors (`tl.make_tensor_descriptor`) over block pointers for all new kernels on XPU
- Do NOT mix block pointer and tensor descriptor APIs on same operation
- Pre-zero output buffers when using atomic accumulation
- Model class must be compatible with ai-bench (`nn.Module` with `nn.Linear`)
- Match `get_inputs()`, `get_init_inputs()`, and module-level constants from `*_pytorch.py`

## REFERENCE DOCS — Read these during Step 1

| Doc | Contents |
|-----|----------|
| `kb/implementation_reference.md` | Code templates, Model class pattern, GEMM example |
| `kb/optimization_strategies.md` | Strategy reference, optimization levels, checklist, KB index |
| `kb/workflow_details.md` | Detailed workflow, decision tree, benchmarking/validation details |
| `kb/correctness.yaml` | Critical constraints to avoid bugs |
| `kb/xpu_optimizations.yaml` | XPU-specific patterns (tensor descriptors, GRF, swizzling) |
| `kb/fusion_patterns.yaml` | When to fuse vs split operations |
| `kb/optimization_levels.yaml` | Progressive optimization with "try harder" decision tree |
| `kb/examples/index.yaml` | Reference implementations catalog |
| `templates/` | GEMM, GEMM+epilogue, reduction starting points |

## EXISTING BASELINES ARE NAIVE

The `test_kernels/*.py` Triton files (non-pytorch) are **unoptimized baselines**. They use manual pointer arithmetic, lack autotune, and miss XPU optimizations. Do NOT copy their patterns. Use `templates/` and `kb/examples/` instead.
