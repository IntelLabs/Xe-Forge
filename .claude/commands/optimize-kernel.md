Optimize `$ARGUMENTS` into a Triton kernel for Intel XPU.

Resolve the argument to a file in `test_kernels/` — it could be a full filename, partial name, or just a number. Glob to find the match; if ambiguous, ask.

First read `config.yaml` for session settings (max_trials, vtune_enabled).
Follow the CLAUDE.md workflow exactly — every step, in order:

1. Read the kernel to understand the operations.
2. Run `python skills/analyze_kernel.py test_kernels/<name>_pytorch.py` (skip if Triton baseline).
3. Read these KB files before writing any code:
   - `kb/correctness.yaml`
   - `kb/xpu_optimizations.yaml`
   - `kb/implementation_reference.md`
   - `kb/examples/index.yaml`
4. Initialize: `python skills/trial_manager.py init <kernel_name> <baseline_file> [--triton-baseline]`
5. Run ALL max_trials trials (from config.yaml) — do NOT stop early on plateau. For EACH trial you MUST:
   - Validate with `python skills/validate_triton.py`
   - Save with `python skills/trial_manager.py save`
   - Benchmark with `python skills/benchmark.py` — NEVER create custom test scripts
   - Record with `python skills/trial_manager.py result`
   - If vtune_enabled and speedup < 3x after 2+ trials, run `python skills/xpu_profiler.py` to identify bottlenecks
6. Finalize the best correct trial

CRITICAL REMINDERS:
- ONLY create Triton kernel files — NO benchmark scripts, NO test scripts, NO helpers
- Use ONLY `skills/benchmark.py` for testing — if it fails, report the error
- Always use ALL max_trials trials — never stop early due to plateau, only stop early if speedup > 5x
- The output file must be self-contained (all helpers inline), use tensor descriptors (preferred) or block pointers, and have a `Model` class wrapper with `get_inputs()` and `get_init_inputs()` matching the baseline.
