"""
Optimizer Agent - Uses CoVeR for iterative kernel optimization with tool-based verification.

Relies on LLM built-in knowledge instead of local YAML knowledge base.
The pipeline still builds the list of detected issues and passes them to each stage.
"""

import ast
import logging
import re

import dspy

from xpu_forge.agents.base import Optimizer
from xpu_forge.agents.cover import CoVeR
from xpu_forge.models import (
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
      IMPORTANT: grf_mode is NOT a triton.Config() parameter. Do NOT pass
      grf_mode to triton.Config() or @triton.autotune configs. It is set
      via environment variable (IGC_EnableLargeGRF=1) or compiler options,
      not in kernel code. Only use valid triton.Config kwargs: meta-parameters
      (like BLOCK_SIZE_M, BLOCK_SIZE_N, etc.), num_warps, and num_stages.

    PERSISTENT_KERNEL: persistent kernel pattern, tune NUM_PROGS.

    === CODE REQUIREMENTS ===
    - Include ALL imports, @triton.jit decorator, kernel function, Model class
    - num_warps must be power of 2; block sizes must be powers of 2
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
    - NEVER put grf_mode in triton.Config(). grf_mode is NOT a valid
      triton.Config parameter. It is set via environment variable
      (IGC_EnableLargeGRF=1), not in kernel code. Only valid Config
      kwargs are: meta-parameters (BLOCK_SIZE_*, GROUP_SIZE_*, etc.),
      num_warps, and num_stages.

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
    optimized_code: dspy.Code["python"] = dspy.OutputField(
        desc="Complete Triton kernel with @triton.autotune. Must include all imports, autotune decorator with configs and key, kernel, and Model class."
    )


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

    def __init__(self, knowledge_base=None, executor=None, validator=None, max_iterations=5):
        self.executor = executor
        self.validator = validator
        self.max_iterations = max_iterations
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
    ):
        executor = self.executor
        # Mutable container so optimize_stage can read back the last accepted
        # comparison without re-running a benchmark.
        last_accepted = {"comparison": None}

        def compile_and_verify(optimized_code: dspy.Code["python"]) -> str:
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

            # Check for grf_mode in triton.Config - this is NOT a valid param
            if re.search(r"triton\.Config\s*\([^)]*grf_mode", code):
                return (
                    "INVALID: grf_mode is NOT a triton.Config() parameter. "
                    "Remove grf_mode from all triton.Config() calls. "
                    "grf_mode is set via environment variable IGC_EnableLargeGRF=1, "
                    "not in kernel code."
                )

            if executor and input_shapes:
                try:
                    comparison = executor.compare_kernels(
                        original_code=original_code,
                        optimized_code=code,
                        kernel_name=kernel_name,
                        input_shapes=input_shapes,
                        flop=flop,
                        dtype=dtype,
                        init_args=init_args,
                    )
                    if not comparison.optimized_correct:
                        return comparison.feedback_message or "Optimized kernel failed."
                    if comparison.is_slower and not skip_speedup_check:
                        sd = 1.0 / comparison.speedup if comparison.speedup > 0 else float("inf")
                        return f"PERFORMANCE REGRESSION: {sd:.2f}x SLOWER. Try different approach."
                    logger.info(f"Optimization verified: {comparison.speedup:.2f}x speedup")
                    last_accepted["comparison"] = comparison
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
                for i in stage_issues
            ]
        )
        xpu_text = "Intel XPU Configuration:\n" + "\n".join(
            f"  {k}: {v}" for k, v in xpu_config.items()
        )

        # Stages whose purpose is correctness, not speed — don't penalise them
        # for being slower; the performance stages that follow will recover that.
        CORRECTNESS_ONLY_STAGES = {OptimizationStage.DTYPE_FIX}
        skip_speedup = stage in CORRECTNESS_ONLY_STAGES

        verify_tool, last_accepted = self._create_verify_tool(
            original_code,
            kernel_name,
            input_shapes,
            flop,
            dtype,
            init_args=init_args,
            skip_speedup_check=skip_speedup,
        )

        # Build problem context for all stages
        problem_ctx = self._build_problem_context(input_shapes, dtype, init_args, flop)

        if stage == OptimizationStage.ALGORITHMIC:
            sig = AlgorithmicOptimizationSignature
            kwargs = {
                "original_code": original_code,
                "current_code": code,
                "pytorch_code": pytorch_code or "",
                "issues": issues_text,
                "xpu_config": xpu_text,
                "problem_context": problem_ctx,
            }
        elif stage == OptimizationStage.AUTOTUNING:
            sig = AutotuneSignature
            # Build autotune-specific context
            suggested_configs = self._build_autotune_configs(xpu_config, input_shapes)
            problem_shapes = self._build_problem_shapes(input_shapes)
            kwargs = {
                "original_code": original_code,
                "current_code": code,
                "issues": issues_text,
                "xpu_config": xpu_text,
                "suggested_autotune_configs": suggested_configs,
                "problem_shapes": problem_shapes,
                "problem_context": problem_ctx,
            }
        else:
            sig = OptimizationSignature
            kwargs = {
                "original_code": original_code,
                "current_code": code,
                "stage": stage.value,
                "issues": issues_text,
                "xpu_config": xpu_text,
                "problem_context": problem_ctx,
            }

        cover = CoVeR(
            signature=sig,
            tools=[verify_tool],
            success=SUCCESS_MESSAGE,
            max_iters=self.max_iterations,
            use_raw_fixer_output=True,
        )

        try:
            result = cover(**kwargs)
            if not hasattr(result, "optimized_code") or result.optimized_code is None:
                return StageResult(
                    stage=stage,
                    success=False,
                    input_code=original_code,
                    output_code=original_code,
                    error_message="CoVeR produced no code",
                )

            code_obj = result.optimized_code
            optimized = _extract_code_from_response(
                code_obj.code if hasattr(code_obj, "code") else str(code_obj)
            )
            traj = result.trajectory if hasattr(result, "trajectory") else {}

            ok, spd, mb, ma, err = self._final_verify(
                original_code,
                optimized,
                kernel_name,
                input_shapes,
                flop,
                dtype,
                init_args=init_args,
                skip_speedup_check=skip_speedup,
                cached_comparison=last_accepted["comparison"],
            )

            if ok:
                logger.info(f"Stage {stage.value} OK" + (f" ({spd:.2f}x)" if spd else ""))
                return StageResult(
                    stage=stage,
                    success=True,
                    input_code=original_code,
                    output_code=optimized,
                    changes_made=self._changes(traj),
                    reasoning=self._reasoning(traj),
                    speedup=spd,
                    metrics_before=mb,
                    metrics_after=ma,
                )
            else:
                logger.warning(f"Stage {stage.value} failed: {err}")
                self._dump_kernel(stage, optimized)
                return StageResult(
                    stage=stage,
                    success=False,
                    input_code=original_code,
                    output_code=original_code,
                    error_message=err,
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

    def _build_problem_context(self, input_shapes, dtype, init_args, flop):
        """Build problem context string for the LLM with shapes, dtype, init args, and FLOPs."""
        lines = ["=== Problem Context ==="]

        # Input shapes
        if input_shapes:
            lines.append(f"Input tensors ({len(input_shapes)}):")
            for i, shape in enumerate(input_shapes):
                numel = 1
                for d in shape:
                    numel *= d
                # Estimate memory in MB (assume 2 bytes for float16)
                bytes_per_elem = 2 if dtype and "16" in str(dtype) else 4
                mem_mb = numel * bytes_per_elem / (1024 * 1024)
                lines.append(f"  Input {i}: shape={shape}, elements={numel:,}, ~{mem_mb:.1f} MB")
            # Compute total memory
            total_mem = 0
            for shape in input_shapes:
                n = 1
                for d in shape:
                    n *= d
                total_mem += n * bytes_per_elem
            lines.append(f"  Total input memory: ~{total_mem / (1024 * 1024):.1f} MB")
        else:
            lines.append("Input tensors: not available")

        # Dtype
        if dtype:
            lines.append(f"Data type: {dtype}")

        # Model init args
        if init_args:
            lines.append(f"Model init args: {init_args}")
            # Try to label them if we can infer meaning
            if len(init_args) == 1:
                lines.append(f"  (likely head_dim or hidden_dim = {init_args[0]})")

        # FLOPs
        if flop:
            lines.append(f"FLOP count: {flop:,.0f}")
            if flop > 1e12:
                lines.append(f"  = {flop / 1e12:.2f} TFLOP")
            elif flop > 1e9:
                lines.append(f"  = {flop / 1e9:.2f} GFLOP")

            # Compute arithmetic intensity if we have shapes
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
        """Build suggested autotune configs from hardware info and shapes."""
        try:
            from xpu_forge.core.xpu_query import (
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

        # Fallback: provide generic suggestions from xpu_config
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
        """Build problem shape description for autotune key= argument."""
        if not input_shapes:
            return "No input shapes available. Use appropriate key= based on kernel arguments."
        try:
            from xpu_forge.core.xpu_query import extract_mnk_from_shapes

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
    ):
        """Run final verification on optimized code.

        When CoVeR already accepted the kernel (cached_comparison is set), reuse
        that comparison result instead of re-benchmarking — the timing is noisy
        enough that a second run can flip pass→fail on the same kernel.
        """
        if not self._valid_py(opt):
            return False, None, None, None, "Invalid Python syntax"
        if not self._valid_triton(opt):
            return False, None, None, None, "Not valid Triton"
        if self.executor and shapes:
            try:
                # Use CoVeR's already-accepted comparison if available
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
                    mb = {
                        "time_us": c.original_time_us,
                        "tflops": c.original_tflops,
                    }
                ma = None
                if c.optimized_time_us and c.optimized_tflops:
                    ma = {
                        "time_us": c.optimized_time_us,
                        "tflops": c.optimized_tflops,
                    }
                return True, c.speedup, mb, ma, None
            except Exception as e:
                return False, None, None, None, f"Verify failed: {e}"
        return True, None, None, None, None

    def _get_stage_issues(self, analysis, stage):
        """Get issues relevant to this stage."""
        from xpu_forge.knowledge.patterns import get_stage_for_issue

        return [i for i in analysis.detected_issues if get_stage_for_issue(i.issue_type) == stage]

    def _dump_kernel(self, stage, code):
        """Dump failed kernel code to file for debugging."""
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
        """Check if code is valid Python syntax."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _valid_triton(self, code):
        """Check if code looks like a valid Triton kernel."""
        has_triton = "import triton" in code or "from triton" in code
        has_kernel = "@triton.jit" in code or "class Model" in code
        return has_triton and has_kernel

    def _changes(self, traj):
        """Extract changes made from CoVeR trajectory thoughts."""
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
        """Extract reasoning from CoVeR trajectory thoughts."""
        ts = [str(v).strip()[:100] for k, v in sorted(traj.items()) if k.startswith("thought_")]
        return " -> ".join(ts) if ts else "CoVeR optimization completed"
