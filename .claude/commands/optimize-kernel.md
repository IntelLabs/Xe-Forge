Optimize `$ARGUMENTS` into a Triton kernel for Intel XPU.

Resolve the argument to a file in `test_kernels/` — it could be a full filename, partial name, or just a number. Glob to find the match; if ambiguous, ask.

First read `.env` (or `.env.example`) for session settings — `TRIAL_MAX_TRIALS`, `TRIAL_EARLY_STOP_SPEEDUP`, `VTUNE_ENABLED`, `VTUNE_BIN`.
Follow the CLAUDE.md workflow exactly — every step, in order:

1. Read the kernel to understand the operations.
2. Run `python src/xe_forge/core/analyze_kernel.py test_kernels/<name>_pytorch.py` (skip if Triton baseline).
3. Read these KB files before writing any code:
   - `knowledge_base/correctness.yaml`
   - `knowledge_base/xpu_optimizations.yaml`
   - `knowledge_base/implementation_reference.md`
   - `knowledge_base/examples/index.yaml`
4. Initialize: `python src/xe_forge/core/trial_manager.py init <kernel_name> <baseline_file> [--triton-baseline]`
5. Run ALL `TRIAL_MAX_TRIALS` trials (from `.env`) — do NOT stop early on plateau. For EACH trial you MUST:
   - Validate with `python src/xe_forge/core/validate_triton.py`
   - Save with `python src/xe_forge/core/trial_manager.py save`
   - Benchmark with `python src/xe_forge/core/benchmark.py` — NEVER create custom test scripts
   - Record with `python src/xe_forge/core/trial_manager.py result`
   - If `VTUNE_ENABLED=true` and speedup < 3x after 2+ trials, run `python src/xe_forge/core/xpu_profiler.py` to identify bottlenecks
6. Finalize the best correct trial

CRITICAL REMINDERS:
- ONLY create Triton kernel files — NO benchmark scripts, NO test scripts, NO helpers.
- Use ONLY `src/xe_forge/core/benchmark.py` for testing — if it fails, report the error.
- Always use ALL `TRIAL_MAX_TRIALS` trials — never stop early due to plateau, only stop early if speedup > `TRIAL_EARLY_STOP_SPEEDUP`.
- The output file must be self-contained (all helpers inline), use tensor descriptors (preferred) or block pointers, and have a `Model` class wrapper with `get_inputs()` and `get_init_inputs()` matching the baseline.
