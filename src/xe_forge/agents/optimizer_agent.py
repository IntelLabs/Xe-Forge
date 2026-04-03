"""
Optimizer Agent - Uses CoVeR for iterative kernel optimization with tool-based verification.
Relies on LLM built-in knowledge instead of local YAML knowledge base.
The pipeline still builds the list of detected issues and passes them to each stage.
"""

import ast
import logging
import re
from typing import Optional

import dspy

from xe_forge.agents.base import Optimizer
from xe_forge.agents.cover import CoVeR
from xe_forge.knowledge.loader import KnowledgeBase
from xe_forge.models import (
    OptimizationStage,
    StageResult,
)

logger = logging.getLogger(__name__)

SUCCESS_MESSAGE = "Success! Optimization verified and kernel is faster."


class OptimizationSignature(dspy.Signature):
    """Apply optimization transformation to Triton kernel.

    You are an expert Triton kernel optimizer for Intel XPU with deep knowledge
    of GPU programming, numerical linear algebra, and high-performance computing.

    Optimize the kernel for maximum performance while producing numerically
    equivalent outputs. You may change the algorithm if outputs are equivalent.
    Maintain the same Model class signature including weights shapes and names.

    === STAGE-SPECIFIC GUIDANCE ===
    ALGORITHMIC: mathematical simplifications, CSE, loop-invariant hoisting,
      caching intermediates, reorder associative ops, tree reductions,
      exploit GEMM structure (symmetric, triangular, low-rank).
    DTYPE_FIX: float64->float32, proper accumulator precision, remove
      unnecessary type conversions.
    FUSION: fuse kernel launches, elementwise chains, reduction+elementwise.
    MEMORY_ACCESS: fix uncoalesced access, remove transposes from inner loops,
      add boundary checks, reduce register pressure.
    BLOCK_POINTERS: use tl.make_block_ptr(), boundary_check=(0,1) tuple format,
      tl.advance() for pointer updates.
    XPU_SPECIFIC: BLOCK_M=256, BLOCK_N=256, BLOCK_K=32, num_warps=32,
      GROUP_SIZE_M swizzling.
      GRF MODE: grf_mode is a compiler option, NOT a triton.Config() kwarg.
      Declare it as tl.constexpr in the kernel signature:
        grf_mode: tl.constexpr  (values: "default", "128", "256", "auto")
      Use "auto" — it automatically selects 256-GRF when register spill > 1000 bytes.
      256-GRF requires num_warps <= 32 (halved thread occupancy).
    PERSISTENT_KERNEL: persistent kernel pattern, tune NUM_PROGS.
    DISCOVERY: apply the open-ended optimization described in the issues field.
      This is a novel optimization not covered by standard stages. Follow the
      proposal exactly, preserving all numerical equivalences.

    === CODE REQUIREMENTS ===
    - Include ALL imports, @triton.jit decorator, kernel function, Model class
    - num_warps must be power of 2; block sizes must be powers of 2
    - NEVER replace @triton.jit kernels with torch.matmul, torch.mm, torch.bmm,
      or any vendor library (oneDNN, cuBLAS, MKL). Keep all original Triton kernels.
    """

    original_code: str = dspy.InputField(desc="Original Triton kernel code for reference")
    current_code: str = dspy.InputField(desc="Current Triton kernel code to optimize")
    stage: str = dspy.InputField(desc="Optimization stage to apply")
    issues: str = dspy.InputField(desc="Specific issues to fix in this stage")
    xpu_config: str = dspy.InputField(desc="Intel XPU configuration parameters")
    problem_context: str = dspy.InputField(
        desc="Problem context: input tensor shapes, dtype, Model init args, and FLOP count. "
        "Use this to choose appropriate tile sizes, understand memory footprint, "
        "and reason about whether the kernel is compute-bound or memory-bound."
    )
    performance_context: str = dspy.InputField(
        desc="Current execution performance: original baseline time, current time after "
        "previous stages, and speedup achieved so far. Use this to understand how much "
        "headroom remains and whether the kernel is close to hardware peak. "
        "Empty string if not yet measured."
    )
    vtune_report: str = dspy.InputField(
        desc="VTune profiling report (Markdown). Empty string if not available. "
        "Use hotspot and memory-access data to guide which optimizations matter most."
    )
    knowledge_base_context: str = dspy.InputField(
        desc="Relevant optimization patterns, constraints, and examples from the knowledge base "
        "for this stage. Empty string if KB is disabled. "
        "IMPORTANT: follow the patterns and constraints listed here precisely — "
        "they are validated optimizations for Intel XPU."
    )
    optimized_code: dspy.Code["python"] = dspy.OutputField(
        desc="Complete optimized Triton kernel code with all imports, decorators, kernel, and Model class."
    )


class AlgorithmicOptimizationSignature(dspy.Signature):
    """Apply algorithmic / mathematical optimization to a Triton kernel.

    You are an expert in numerical linear algebra, compiler optimizations, and
    high-performance GPU kernel design.

    Transform the kernel to perform FEWER FLOPs and/or FEWER memory accesses
    while producing numerically equivalent results.

    Think about:
    1. Matrix structure exploitation (symmetric, triangular, diagonal, low-rank, sparse)
    2. Associative / distributive law rewrites to reduce FLOPs
    3. Common sub-expression elimination
    4. Loop-invariant code hoisting
    5. Caching intermediates in registers vs recomputing
    6. Tree reductions vs serial reductions
    7. Algebraic simplification of fused computations

    Maintain the Model class signature. Produce equivalent outputs.

    === CODE REQUIREMENTS ===
    - Include ALL imports, @triton.jit decorator, kernel function, Model class
    - NEVER replace @triton.jit kernels with torch.matmul, torch.mm, or any vendor library.
    """

    original_code: str = dspy.InputField(desc="Original Triton kernel code for reference")
    current_code: str = dspy.InputField(desc="Current Triton kernel code to optimize")
    pytorch_code: str = dspy.InputField(
        desc="Original PyTorch implementation for context (may be empty)"
    )
    issues: str = dspy.InputField(desc="Specific algorithmic issues identified by analysis")
    xpu_config: str = dspy.InputField(desc="Intel XPU configuration parameters")
    problem_context: str = dspy.InputField(
        desc="Problem context: input tensor shapes, dtype, Model init args, and FLOP count. "
        "Use this to understand problem scale, memory footprint, and compute intensity."
    )
    performance_context: str = dspy.InputField(
        desc="Current execution performance: original baseline time and speedup so far. "
        "Empty string if not yet measured."
    )
    knowledge_base_context: str = dspy.InputField(
        desc="Relevant algorithmic patterns and examples from the knowledge base. "
        "Empty string if KB is disabled. Follow these patterns precisely."
    )
    optimized_code: dspy.Code["python"] = dspy.OutputField(
        desc="Complete optimized Triton kernel with algorithmic improvements."
    )


class AutotuneSignature(dspy.Signature):
    """Add or improve @triton.autotune configuration for a Triton kernel.

    You are an expert in Triton kernel autotuning for Intel XPU.

    Your task: Add or improve the @triton.autotune decorator so the kernel
    automatically selects the best configuration at runtime.

    You will receive:
    - The current kernel code
    - Hardware information (compute units, memory, capabilities)
    - Problem shapes (M, N, K dimensions)
    - A set of suggested autotune configurations generated from hardware analysis

    Your job:
    1. Add @triton.autotune decorator with a good set of configs to search.
    2. Use the suggested configs as a starting point but ADD more configs
       based on your knowledge of what works well for this kernel type.
    3. Include the key= argument so configs are re-evaluated when shapes change.
    4. Ensure num_warps and num_stages are included in each config.
    5. Ensure BLOCK sizes are powers of 2 and appropriate for the hardware.
    6. For Intel XPU, always include at least one config with num_warps=32
       and large tile sizes (256x256).
    7. Remove any hardcoded meta-parameters that are now covered by autotune.
    8. Keep the kernel functionally equivalent.

    Tips for good autotune configs:
    - Vary BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K across powers of 2
    - Include both small tiles (64x64) for small problems and large tiles
      (256x256) for large problems
    - Vary num_warps: try 4, 8, 16, 32
    - Vary num_stages: try 2, 3, 4
    - Include GROUP_SIZE_M for L2 cache swizzling
    - Use key= with the shape arguments that affect tiling
    - Do NOT put grf_mode in triton.Config() — it causes TypeError at runtime.
      grf_mode is a compiler option: declare it as tl.constexpr in the kernel
      signature. Use grf_mode="auto" (auto-selects 256-GRF if spill > 1000 bytes)
      or grf_mode="256" for large register file. Requires num_warps <= 32.

    === CODE REQUIREMENTS ===
    - Include ALL imports (torch, triton, triton.language as tl)
    - Include @triton.autotune with configs list and key
    - Include @triton.jit on the kernel
    - Include the Model class with forward() method
    """

    original_code: str = dspy.InputField(desc="Original Triton kernel code for reference")
    current_code: str = dspy.InputField(desc="Current Triton kernel code to add autotune to")
    issues: str = dspy.InputField(desc="Specific autotuning issues identified by analysis")
    xpu_config: str = dspy.InputField(desc="Intel XPU hardware info and recommended parameters")
    suggested_autotune_configs: str = dspy.InputField(
        desc="Suggested autotune configurations from hardware/shape analysis (use as starting point)"
    )
    problem_shapes: str = dspy.InputField(
        desc="Problem dimensions (M, N, K) and input shapes for key= argument"
    )
    problem_context: str = dspy.InputField(
        desc="Problem context: input tensor shapes, dtype, Model init args, and FLOP count. "
        "Use this to choose config search space breadth and understand compute intensity."
    )
    performance_context: str = dspy.InputField(
        desc="Current execution performance: original baseline time and speedup so far. "
        "Empty string if not yet measured."
    )
    knowledge_base_context: str = dspy.InputField(
        desc="Relevant autotuning patterns and constraints from the knowledge base. "
        "Empty string if KB is disabled. Follow these patterns precisely."
    )
    optimized_code: dspy.Code["python"] = dspy.OutputField(
        desc="Complete Triton kernel with @triton.autotune. Must include all imports, autotune decorator with configs and key, kernel, and Model class."
    )


def _build_performance_context(perf_context: Optional[dict]) -> str:
    """Format perf_context dict into a human-readable string for the LLM prompt."""
    if not perf_context:
        return ""
    orig_ms = perf_context.get("original_ms")
    orig_tf = perf_context.get("original_tflops")
    curr_ms = perf_context.get("current_ms")
    so_far = perf_context.get("speedup_so_far")

    lines = ["=== Performance Context ==="]
    if orig_ms:
        lines.append(
            f"Original baseline:  {orig_ms:.3f} ms"
            + (f"  ({orig_tf:.2f} TFLOPS)" if orig_tf else "")
        )
    if curr_ms and curr_ms != orig_ms:
        lines.append(f"Current (after previous stages): {curr_ms:.3f} ms")
    if so_far and so_far != 1.0:
        lines.append(f"Speedup so far: {so_far:.2f}x")
        if so_far < 1.0:
            lines.append("  WARNING: previous stages made the kernel slower — be conservative.")
        elif so_far >= 2.0:
            lines.append("  Good progress. Focus on remaining bottlenecks.")
        else:
            lines.append("  Moderate progress. Significant headroom likely remains.")
    elif orig_ms and (not so_far or so_far == 1.0):
        lines.append("No speedup from previous stages yet.")
    stage_best = perf_context.get("stage_best_so_far")
    if stage_best:
        lines.append(
            f"Best achieved this stage so far: {stage_best:.3f}x — "
            f"your next attempt must beat this to be accepted."
        )
    return "\n".join(lines)


def _has_cpu_return(code: str) -> bool:
    """Return True if kernel_function/forward returns a CPU tensor.
    .cpu() in a return is always wrong — output must stay on XPU.
    Note: .xpu() IS valid (moves to XPU) but should not be needed in a return.
    """
    if re.search(r"return\s+\S+\.cpu\(\)", code):
        return True
    if re.search(r"return\s+\S+\.to\(.[Cc][Pp][Uu].", code):
        return True
    return False


def _extract_code_from_response(code_str):
    if code_str is None:
        return ""
    code = str(code_str)
    if "```python" in code:
        m = re.search(r"```python\s*(.*?)\s*```", code, re.DOTALL)
        if m:
            code = m.group(1)
    elif "```" in code:
        m = re.search(r"```\s*(.*?)\s*```", code, re.DOTALL)
        if m:
            code = m.group(1)
    return code.strip()


class OptimizerAgent(Optimizer):
    """Applies optimization transformations using CoVeR with LLM knowledge."""

    def __init__(
        self,
        knowledge_base: "KnowledgeBase | None" = None,
        executor=None,
        validator=None,
        max_iterations=5,
    ):
        self.executor = executor
        self.validator = validator
        self.max_iterations = max_iterations
        self.knowledge_base: KnowledgeBase | None = knowledge_base
        if not executor:
            logger.warning("No executor provided - kernels will NOT be verified at runtime!")

    def _create_verify_tool(
        self,
        original_code,
        kernel_name,
        input_shapes,
        flop,
        dtype=None,
        init_args=None,
        skip_speedup_check=False,
        stage=None,
        baseline_ms: float | None = None,
    ):
        executor = self.executor
        last_accepted = {"comparison": None}

        _verify_call_count = [0]  # mutable counter in closure

        def compile_and_verify(optimized_code: dspy.Code["python"]) -> str:
            _verify_call_count[0] += 1
            logger.debug("compile_and_verify call #%d", _verify_call_count[0])
            code = _extract_code_from_response(
                optimized_code.code if hasattr(optimized_code, "code") else str(optimized_code)
            )

            try:
                ast.parse(code)
            except SyntaxError as e:
                return f"SYNTAX ERROR at line {e.lineno}: {e.msg}"

            for check, msg in [
                ("import triton" in code or "from triton" in code, "MISSING: import triton"),
                (
                    "triton.language" in code or "import triton.language" in code,
                    "MISSING: import triton.language",
                ),
                ("@triton.jit" in code, "MISSING: @triton.jit decorator"),
                ("class Model" in code, "MISSING: class Model"),
            ]:
                if not check:
                    return msg

            warps_match = re.search(r"num_warps\s*=\s*(\d+)", code)
            if warps_match:
                nw = int(warps_match.group(1))
                if nw <= 0 or (nw & (nw - 1)) != 0:
                    return f"INVALID num_warps={nw}: Must be power of 2."

            for bn in [
                "BLOCK_M",
                "BLOCK_N",
                "BLOCK_K",
                "BLOCK_SIZE_M",
                "BLOCK_SIZE_N",
                "BLOCK_SIZE_K",
            ]:
                bm = re.search(rf"{bn}\s*[=:]\s*(\d+)", code)
                if bm:
                    bs = int(bm.group(1))
                    if bs <= 0 or (bs & (bs - 1)) != 0:
                        return f"INVALID {bn}={bs}: Must be power of 2."

            if re.search(r"triton\.Config\s*\([^)]*grf_mode", code):
                return (
                    "INVALID: grf_mode cannot be passed to triton.Config(). "
                    "It is a compiler-level option, not a kernel meta-parameter. "
                    "To use large GRF: declare grf_mode: tl.constexpr in the kernel "
                    "signature (values: 'default', '128', '256', 'auto'). "
                    "'auto' is recommended — it recompiles with 256-GRF only if "
                    "register spill > 1000 bytes. Remove grf_mode from triton.Config()."
                )

            if stage is not None and stage.value == "fusion":
                for _vp in [
                    r"torch\.matmul",
                    r"torch\.mm\b",
                    r"torch\.bmm",
                    r"F\.linear\b",
                    r"torch\.nn\.functional\.linear",
                ]:
                    if re.search(_vp, code):
                        return (
                            "INVALID: fusion replaced a Triton kernel with a vendor call. "
                            "Keep all @triton.jit kernels. "
                            "Do not replace Triton kernels with torch.matmul/mm/bmm."
                        )

            if _has_cpu_return(code):
                return (
                    "INVALID: kernel_function/forward must NOT return a CPU tensor. "
                    "Remove .cpu() from all return statements. "
                    "The output tensor must stay on XPU."
                )

            original_kernels = set(re.findall(r"def\s+(\w+)\s*\(", original_code))
            original_jit_kernels = {
                name
                for name in original_kernels
                if f"@triton.jit" in original_code
                and re.search(
                    rf"@triton\.jit[\s\S]{{0,200}}def\s+{re.escape(name)}\s*\(", original_code
                )
            }
            if original_jit_kernels:
                missing = [k for k in original_jit_kernels if f"def {k}" not in code]
                if missing:
                    return (
                        f"INVALID: Triton kernel(s) {missing} were removed from the optimized code. "
                        "You must keep all original @triton.jit kernels. "
                        "Do NOT replace Triton kernels with torch.matmul, torch.mm, "
                        "oneDNN, or any other vendor library call."
                    )

            if executor and input_shapes:
                try:
                    logger.info(
                        "Measuring: original vs optimized (call #%d)", _verify_call_count[0]
                    )
                    comparison = executor.compare_kernels(
                        original_code=original_code,
                        optimized_code=code,
                        kernel_name=kernel_name,
                        input_shapes=input_shapes,
                        flop=flop,
                        dtype=dtype,
                        init_args=init_args,
                    )
                    if comparison.original_time_us:
                        logger.info("  original : %.2f µs", comparison.original_time_us)
                    if comparison.optimized_time_us:
                        logger.info("  optimized: %.2f µs", comparison.optimized_time_us)
                    if not comparison.optimized_correct:
                        return comparison.feedback_message or "Optimized kernel failed."
                    # Use pipeline baseline for regression check (avoids JIT warmup noise)
                    if not skip_speedup_check:
                        if baseline_ms and comparison.optimized_time_us:
                            true_spd = (baseline_ms * 1000) / comparison.optimized_time_us
                        else:
                            true_spd = comparison.speedup
                        if true_spd < 1.0:
                            sd = 1.0 / true_spd if true_spd > 0 else float("inf")
                            return (
                                f"PERFORMANCE REGRESSION: {sd:.2f}x SLOWER. Try different approach."
                            )
                    # Compute speedup relative to true baseline if available
                    # (avoids JIT warmup / thermal noise in re-measured original)
                    if baseline_ms and comparison.optimized_time_us:
                        true_speedup = (baseline_ms * 1000) / comparison.optimized_time_us
                        logger.info(
                            f"Optimization verified: {true_speedup:.2f}x speedup "
                            f"(vs true baseline {baseline_ms:.3f}ms)"
                        )
                    else:
                        logger.info(f"Optimization verified: {comparison.speedup:.2f}x speedup")
                    last_accepted["comparison"] = comparison
                    last_accepted["baseline_ms"] = baseline_ms
                    return SUCCESS_MESSAGE
                except Exception as e:
                    return f"RUNTIME ERROR: {e}"
            return SUCCESS_MESSAGE

        tool = dspy.Tool(
            func=compile_and_verify,
            name="compile_and_verify",
            desc=f'Compiles and verifies optimized Triton kernel. Returns "{SUCCESS_MESSAGE}" on success.',
        )
        return tool, last_accepted

    def optimize_stage(
        self,
        code,
        stage,
        analysis,
        xpu_config,
        kernel_name=None,
        input_shapes=None,
        flop=None,
        dtype=None,
        pytorch_code="",
        init_args=None,
        vtune_report="",
        perf_context: Optional[dict] = None,
    ):
        logger.info(f"Applying optimization stage: {stage.value}")
        original_code = code

        stage_issues = self._get_stage_issues(analysis, stage)
        if not stage_issues:
            return StageResult(
                stage=stage,
                success=True,
                input_code=code,
                output_code=code,
                changes_made=["No changes needed"],
            )

        issues_text = "\n".join(
            [
                f"- {i.issue_type.value}: {i.description}\n  Fix: {i.suggested_fix}\n  Speedup: {i.estimated_speedup or 'Unknown'}"
                + (
                    f"\n  Proposal: {i.open_ended_proposal}"
                    if hasattr(i, "open_ended_proposal") and i.open_ended_proposal
                    else ""
                )
                for i in stage_issues
            ]
        )

        xpu_text = "Intel XPU Configuration:\n" + "\n".join(
            f"  {k}: {v}" for k, v in xpu_config.items()
        )

        CORRECTNESS_ONLY_STAGES = {}
        skip_speedup = stage in CORRECTNESS_ONLY_STAGES

        _baseline_ms = perf_context.get("original_ms") if perf_context else None

        verify_tool, last_accepted = self._create_verify_tool(
            original_code,
            kernel_name,
            input_shapes,
            flop,
            dtype,
            init_args=init_args,
            skip_speedup_check=skip_speedup,
            stage=stage,
            baseline_ms=_baseline_ms,
        )

        problem_ctx = self._build_problem_context(input_shapes, dtype, init_args, flop)
        perf_ctx = _build_performance_context(perf_context)

        kb_context = self._get_stage_patterns(stage)
        if kb_context:
            logger.info(
                "KB context for %s: %d chars, %d patterns/constraints",
                stage.value,
                len(kb_context),
                kb_context.count("###") + kb_context.count("CONSTRAINT"),
            )
        else:
            logger.debug("No KB context for stage %s (KB disabled or empty)", stage.value)

        if stage == OptimizationStage.ALGORITHMIC:
            sig = AlgorithmicOptimizationSignature
            kwargs = dict(
                original_code=original_code,
                current_code=code,
                pytorch_code=pytorch_code or "",
                issues=issues_text,
                xpu_config=xpu_text,
                problem_context=problem_ctx,
                performance_context=perf_ctx,
                knowledge_base_context=kb_context,
            )
        elif stage == OptimizationStage.AUTOTUNING:
            sig = AutotuneSignature
            suggested_configs = self._build_autotune_configs(xpu_config, input_shapes)
            problem_shapes = self._build_problem_shapes(input_shapes)
            kwargs = dict(
                original_code=original_code,
                current_code=code,
                issues=issues_text,
                xpu_config=xpu_text,
                suggested_autotune_configs=suggested_configs,
                problem_shapes=problem_shapes,
                problem_context=problem_ctx,
                performance_context=perf_ctx,
                knowledge_base_context=kb_context,
            )
        else:
            sig = OptimizationSignature
            kwargs = dict(
                original_code=original_code,
                current_code=code,
                stage=stage.value,
                issues=issues_text,
                xpu_config=xpu_text,
                problem_context=problem_ctx,
                performance_context=perf_ctx,
                vtune_report=vtune_report or "",
                knowledge_base_context=kb_context,
            )

        cover = CoVeR(
            signature=sig,
            tools=[verify_tool],
            success=SUCCESS_MESSAGE,
            max_iters=self.max_iterations,
            use_raw_fixer_output=True,
        )

        best_code = None
        best_spd = None
        best_mb = best_ma = best_traj = None
        iters_used = 0
        attempt_history: list[str] = []  # what each run tried and achieved

        current_code_for_run = code

        try:
            while iters_used < self.max_iterations:
                remaining = self.max_iterations - iters_used
                cover.max_iters = remaining

                run_kwargs = {**kwargs, "current_code": current_code_for_run}

                result = cover(**run_kwargs)

                traj = result.trajectory if hasattr(result, "trajectory") else {}
                thoughts_this_run = sum(1 for k in traj if k.startswith("thought_"))
                iters_used += max(1, thoughts_this_run)

                if not hasattr(result, "optimized_code") or result.optimized_code is None:
                    break

                code_obj = result.optimized_code
                candidate = _extract_code_from_response(
                    code_obj.code if hasattr(code_obj, "code") else str(code_obj)
                )

                ok, spd, mb, ma, err = self._final_verify(
                    original_code,
                    candidate,
                    kernel_name,
                    input_shapes,
                    flop,
                    dtype,
                    init_args=init_args,
                    skip_speedup_check=skip_speedup,
                    cached_comparison=last_accepted["comparison"],
                    baseline_ms=_baseline_ms,
                )

                # Record this attempt for feedback to the next run
                _attempt_thoughts = self._reasoning(traj) if traj else ""
                _attempt_summary = (
                    f"Attempt {len(attempt_history) + 1}: "
                    + (f"achieved {spd:.3f}x" if ok and spd else f"failed ({err})")
                    + (f" | approach: {_attempt_thoughts[:150]}" if _attempt_thoughts else "")
                )
                attempt_history.append(_attempt_summary)

                if not ok:
                    logger.debug(f"Run failed ({err}), stopping best-of loop")
                    break

                # Require 2% improvement on ALL runs (first and subsequent)
                # This eliminates noise-based false improvements from timing variance
                _MIN_IMPROVEMENT = 1.02
                _is_improvement = (
                    spd is not None and spd > _MIN_IMPROVEMENT
                    if best_spd is None
                    else spd is not None and spd > best_spd * _MIN_IMPROVEMENT
                )
                # Also stop if code is identical to previous best (LLM stuck)
                _code_identical = best_code is not None and candidate == best_code
                if _code_identical:
                    logger.info(f"Stage {stage.value}: LLM produced identical code — stopping")
                    break
                if _is_improvement:
                    logger.info(
                        f"Stage {stage.value} new best: {spd:.2f}x"
                        + (f" (was {best_spd:.2f}x)" if best_spd is not None else "")
                    )
                    best_code, best_spd, best_mb, best_ma, best_traj = (
                        candidate,
                        spd,
                        mb,
                        ma,
                        traj,
                    )
                    current_code_for_run = candidate
                    # Only clear cache if code genuinely changed
                    # (keeps original measurement reuse across runs)
                    last_accepted["comparison"] = None
                    # Rebuild performance_context with updated speedup
                    # so the next CoVeR iteration knows where it stands
                    if perf_context:
                        updated_perf = dict(perf_context)
                    else:
                        updated_perf = {}
                    _orig_ms = updated_perf.get("original_ms")
                    if spd and _orig_ms:
                        updated_perf["current_ms"] = _orig_ms / spd
                    updated_perf["speedup_so_far"] = spd
                    updated_perf["stage_best_so_far"] = spd
                    perf_ctx = _build_performance_context(updated_perf)
                    # Update kwargs: fresh performance context + attempt history
                    history_text = "\n".join(attempt_history[-3:])  # last 3 attempts
                    _issues_with_history = issues_text + (
                        f"\n\n=== Previous attempts this stage ===\n{history_text}\n"
                        "Try a DIFFERENT approach to beat the current best."
                        if attempt_history
                        else ""
                    )
                    kwargs = {
                        **kwargs,
                        "performance_context": perf_ctx,
                        "issues": _issues_with_history,
                    }
                else:
                    _best_str = f"{best_spd:.2f}x" if best_spd is not None else "none"
                    _spd_str = f"{spd:.2f}x" if spd is not None else "N/A"
                    logger.info(
                        f"Stage {stage.value} no improvement ({_spd_str} vs best {_best_str}), stopping"
                    )
                    break

            if best_code is not None:
                logger.info(
                    f"Stage {stage.value} OK — best {best_spd:.2f}x"
                    f" ({self.max_iterations - (self.max_iterations - iters_used)} iters used)"
                )
                return StageResult(
                    stage=stage,
                    success=True,
                    input_code=original_code,
                    output_code=best_code,
                    changes_made=self._changes(best_traj or {}),
                    reasoning=self._reasoning(best_traj or {}),
                    speedup=best_spd,
                    metrics_before=best_mb,
                    metrics_after=best_ma,
                )
            else:
                logger.warning(f"Stage {stage.value} failed: no valid result in budget")
                return StageResult(
                    stage=stage,
                    success=False,
                    input_code=original_code,
                    output_code=original_code,
                    error_message="No valid optimization found within iteration budget",
                )

        except Exception as e:
            logger.error(f"CoVeR failed: {e}")
            return StageResult(
                stage=stage,
                success=False,
                input_code=original_code,
                output_code=original_code,
                error_message=str(e),
            )

    @staticmethod
    def _extract_example_code(code: str, max_chars: int = 2500) -> str:
        """Extract key patterns from example code: header comments, __init__, cache method, forward, kernel_function."""
        import re

        sections = []

        # File-level optimization summary
        header = re.match(r"((?:#[^\n]*\n)+)", code)
        if header:
            key = [
                l
                for l in header.group(1).split("\n")
                if any(
                    kw in l.lower()
                    for kw in [
                        "fix",
                        "key",
                        "optim",
                        "cache",
                        "pack",
                        "fuse",
                        "speedup",
                        "important",
                        "fp16",
                        "fp32",
                        "1)",
                        "2)",
                        "3)",
                    ]
                )
            ]
            if key:
                sections.append("# Key optimizations:\n" + "\n".join(key[:10]))

        # Model.__init__
        m = re.search(
            r"(    def __init__\(self[^)]*\):.*?)(?=\n    def |\nclass |\Z)", code, re.DOTALL
        )
        if m:
            sections.append("# __init__:\n" + m.group(1)[:500])

        # Parameter cache/pack method
        m = re.search(
            r"(    def (?:_move_params_once|_ensure_cache|_ensure_device|_build_cache)[^:]*:.*?)"
            r"(?=\n    def |\nclass |\Z)",
            code,
            re.DOTALL,
        )
        if m:
            sections.append("# Parameter caching:\n" + m.group(1)[:900])

        # forward()
        m = re.search(
            r"(    def forward\(self[^)]*\):.*?)(?=\n    def |\nclass |\Z)", code, re.DOTALL
        )
        if m:
            sections.append("# forward():\n" + m.group(1)[:350])

        # kernel_function
        m = re.search(
            r"(def kernel_function\([^)]*\):.*?)(?=\nclass |\ndef (?!kernel)|\Z)", code, re.DOTALL
        )
        if m:
            sections.append("# kernel_function:\n" + m.group(1)[:450])

        # If nothing matched (e.g. pure kernel file), fall back to first max_chars
        if not sections:
            return code[:max_chars]

        result = "\n\n".join(s for s in sections if s.strip())
        return result[:max_chars]

    def _get_stage_patterns(self, stage: OptimizationStage) -> str:
        """Return KB context: constraints + patterns + compact example summaries."""
        if self.knowledge_base is None:
            return ""
        try:
            # 1. Get constraints + patterns (no full code) from format_for_stage
            full = self.knowledge_base.format_for_stage(stage)
            if not full:
                return ""
            # Strip the full code section — replace with compact example summaries
            split_marker = "FULL CODE EXAMPLES FOR"
            if split_marker in full:
                full = full[: full.index(split_marker)].rstrip()

            # 2. Append compact example summaries (name + optimizations, no code)
            examples = self.knowledge_base.examples_for_stage(stage)
            if examples:
                ex_lines = [f"\n\nRELEVANT EXAMPLES FOR {stage.value.upper()}:"]
                ex_lines.append("(these are real optimized kernels — apply the same patterns)")
                for ex in examples[:4]:  # cap at 4 examples
                    ex_lines.append(f"\n## {ex.name}")
                    ex_lines.append(f"Description: {ex.description.strip()[:150]}")
                    if ex.optimizations_applied:
                        ex_lines.append("Optimizations applied:")
                        for opt in ex.optimizations_applied[:8]:
                            ex_lines.append(f"  - {opt}")
                    if ex.expected_speedup:
                        ex_lines.append(f"Expected speedup: {ex.expected_speedup}")
                    # Include code: full if small, smart-extracted if large
                    if ex.optimized_code:
                        code_to_show = self._extract_example_code(ex.optimized_code)
                        if code_to_show:
                            ex_lines.append("Key patterns from optimized code:")
                            ex_lines.append("```python")
                            ex_lines.append(code_to_show)
                            ex_lines.append("```")
                full += "\n".join(ex_lines)

            # Hard cap at 14000 chars (~3500 tokens)
            if len(full) > 14000:
                full = full[:14000] + "\n... [KB content truncated]"
            logger.debug("KB patterns for %s: %d chars", stage.value, len(full))
            return full
        except Exception as e:
            logger.debug("KB context retrieval failed: %s", e)
            return ""

    def _build_problem_context(self, input_shapes, dtype, init_args, flop):
        lines = ["=== Problem Context ==="]

        if input_shapes:
            lines.append(f"Input tensors ({len(input_shapes)}):")
            for i, shape in enumerate(input_shapes):
                numel = 1
                for d in shape:
                    numel *= d
                bytes_per_elem = 2 if dtype and "16" in str(dtype) else 4
                mem_mb = numel * bytes_per_elem / (1024 * 1024)
                lines.append(f"  Input {i}: shape={shape}, elements={numel:,}, ~{mem_mb:.1f} MB")

            total_mem = 0
            bytes_per_elem = 2 if dtype and "16" in str(dtype) else 4
            for shape in input_shapes:
                n = 1
                for d in shape:
                    n *= d
                total_mem += n * bytes_per_elem
            lines.append(f"  Total input memory: ~{total_mem / (1024 * 1024):.1f} MB")
        else:
            lines.append("Input tensors: not available")

        if dtype:
            lines.append(f"Data type: {dtype}")

        if init_args:
            lines.append(f"Model init args: {init_args}")
            if len(init_args) == 1:
                lines.append(f"  (likely head_dim or hidden_dim = {init_args[0]})")

        if flop:
            lines.append(f"FLOP count: {flop:,.0f}")
            if flop > 1e12:
                lines.append(f"  = {flop / 1e12:.2f} TFLOP")
            elif flop > 1e9:
                lines.append(f"  = {flop / 1e9:.2f} GFLOP")

            if input_shapes:
                total_bytes = 0
                bytes_per_elem = 2 if dtype and "16" in str(dtype) else 4
                for shape in input_shapes:
                    n = 1
                    for d in shape:
                        n *= d
                    total_bytes += n * bytes_per_elem
                if total_bytes > 0:
                    ai = flop / total_bytes
                    lines.append(f"  Arithmetic intensity: {ai:.1f} FLOPs/byte")
                    if ai > 100:
                        lines.append(
                            "  -> Compute-bound: focus on algorithmic and compute optimizations"
                        )
                    elif ai > 10:
                        lines.append("  -> Balanced: both compute and memory optimizations matter")
                    else:
                        lines.append(
                            "  -> Memory-bound: focus on memory access patterns and data reuse"
                        )

        return "\n".join(lines)

    def _build_autotune_configs(self, xpu_config, input_shapes):
        try:
            from xe_forge.core.xpu_query import (
                extract_mnk_from_shapes,
                get_autotune_configs,
            )

            if input_shapes and len(input_shapes) >= 1:
                M, N, K = extract_mnk_from_shapes(input_shapes)
                if M and N and K:
                    configs = get_autotune_configs(M, N, K)
                    lines = [f"Suggested configs for M={M}, N={N}, K={K}:"]
                    for i, cfg in enumerate(configs):
                        lines.append(f"  Config {i + 1}: {cfg}")
                    return "\n".join(lines)
        except Exception as e:
            logger.debug(f"Could not generate autotune configs: {e}")

        lines = ["No shape-specific configs available. Suggested search space:"]
        bm = xpu_config.get("BLOCK_SIZE_M", 256)
        bn = xpu_config.get("BLOCK_SIZE_N", 256)
        bk = xpu_config.get("BLOCK_SIZE_K", 32)
        nw = xpu_config.get("num_warps", 32)
        lines.append(f"  Base: BLOCK_M={bm}, BLOCK_N={bn}, BLOCK_K={bk}, num_warps={nw}")
        lines.append("  Also try: BLOCK_M/N in [64, 128, 256], BLOCK_K in [32, 64]")
        lines.append("  Also try: num_warps in [4, 8, 16, 32], num_stages in [2, 3, 4]")
        return "\n".join(lines)

    def _build_problem_shapes(self, input_shapes):
        if not input_shapes:
            return "No input shapes available. Use appropriate key= based on kernel arguments."
        try:
            from xe_forge.core.xpu_query import extract_mnk_from_shapes

            M, N, K = extract_mnk_from_shapes(input_shapes)
            lines = [f"Input shapes: {input_shapes}"]
            if M and N and K:
                lines.append(f"Extracted dimensions: M={M}, N={N}, K={K}")
                lines.append(
                    "Use key= with the stride/shape args that correspond "
                    "to M, N, K so autotune re-runs when problem size changes."
                )
            return "\n".join(lines)
        except Exception:
            return f"Input shapes: {input_shapes}"

    def _final_verify(
        self,
        orig,
        opt,
        kn,
        shapes,
        flop,
        dtype,
        init_args=None,
        skip_speedup_check=False,
        cached_comparison=None,
        baseline_ms: float | None = None,
    ):
        if not self._valid_py(opt):
            return False, None, None, None, "Invalid Python syntax"
        if not self._valid_triton(opt):
            return False, None, None, None, "Not valid Triton"
        if self.executor and shapes:
            try:
                if cached_comparison is not None:
                    c = cached_comparison
                else:
                    c = self.executor.compare_kernels(
                        original_code=orig,
                        optimized_code=opt,
                        kernel_name=kn,
                        input_shapes=shapes,
                        flop=flop,
                        dtype=dtype,
                        init_args=init_args,
                    )
                if not c.optimized_correct:
                    return False, None, None, None, "Incorrect results"
                if c.is_slower and not skip_speedup_check:
                    sd = 1.0 / c.speedup if c.speedup > 0 else float("inf")
                    return False, None, None, None, f"{sd:.2f}x slower"
                mb = None
                if c.original_time_us and c.original_tflops:
                    mb = {"time_us": c.original_time_us, "tflops": c.original_tflops}
                ma = None
                if c.optimized_time_us and c.optimized_tflops:
                    ma = {"time_us": c.optimized_time_us, "tflops": c.optimized_tflops}
                # Use true pipeline baseline for speedup if available
                if baseline_ms and c.optimized_time_us:
                    spd = (baseline_ms * 1000) / c.optimized_time_us
                else:
                    spd = c.speedup
                return True, spd, mb, ma, None
            except Exception as e:
                return False, None, None, None, f"Verify failed: {e}"
        return True, None, None, None, None

    def _get_stage_issues(self, analysis, stage):
        from xe_forge.knowledge.patterns import get_stage_for_issue

        seen_types: set = set()
        result = []
        for i in analysis.detected_issues:
            if get_stage_for_issue(i.issue_type) != stage:
                continue
            # Deduplicate by issue_type — keep the first (highest severity) occurrence
            if i.issue_type not in seen_types:
                seen_types.add(i.issue_type)
                result.append(i)
            else:
                logger.debug("Skipping duplicate issue %s in stage %s", i.issue_type, stage.value)
        return result

    def _dump_kernel(self, stage, code):
        import os
        from datetime import datetime

        d = os.environ.get("TRITON_OPT_DUMP_DIR", "./outputs/kernels")
        os.makedirs(d, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            with open(f"{d}/{stage.value}_failed_{ts}.py", "w") as f:
                f.write(f"# Stage: {stage.value}\n# FAILED\n\n{code}")
        except Exception:
            pass

    def _valid_py(self, code):
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _valid_triton(self, code):
        has_triton = "import triton" in code or "from triton" in code
        has_kernel = "@triton.jit" in code or "class Model" in code
        return has_triton and has_kernel

    def _changes(self, traj):
        cs = []
        keywords = [
            "applied",
            "changed",
            "replaced",
            "added",
            "removed",
            "optimized",
            "fixed",
            "simplified",
            "cached",
            "hoisted",
            "fused",
            "reordered",
        ]
        for k, v in sorted(traj.items()):
            if k.startswith("thought_"):
                t = str(v).strip()
                if any(w in t.lower() for w in keywords):
                    cs.append(t[:500])
        return cs or ["Optimization applied via CoVeR"]

    def _reasoning(self, traj):
        ts = [str(v).strip()[:100] for k, v in sorted(traj.items()) if k.startswith("thought_")]
        return " -> ".join(ts) if ts else "CoVeR optimization completed"
