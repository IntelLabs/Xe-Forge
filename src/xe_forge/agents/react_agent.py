"""
Optimizer Agent - Uses ReAct for iterative kernel optimization with tool-based verification.
"""

from __future__ import annotations

import ast
import logging
import re
from collections.abc import Callable

import dspy

from xe_forge.agents import Optimizer
from xe_forge.core import KernelBenchExecutor
from xe_forge.knowledge.patterns import get_stage_for_issue

try:
    from xe_forge.knowledge.loader import KnowledgeBase
except ImportError:
    KnowledgeBase = None
from xe_forge.models import (
    DetectedIssue,
    KernelAnalysis,
    OptimizationStage,
    StageResult,
)

logger = logging.getLogger(__name__)


SUCCESS_MESSAGE = "Success! Optimization verified and kernel is faster."


class OptimizationReActSignature(dspy.Signature):
    """Apply optimization transformation to Triton kernel.

    You are an expert Triton kernel optimizer for Intel XPU.

    Your task: Optimize the kernel for maximum performance.
    You may change the algorithm/computation approach if it produces equivalent outputs.
    Maintain the same model signature, including the weights' shapes and names. This is necessary for having identical initialization process for formal verification which is done by a correctness tool.

    === OPTIMIZATION PRIORITIES ===
    1. Apply the specific optimization patterns from knowledge_patterns
    2. Use block pointers with tl.make_block_ptr() for better memory access
    3. Use optimal tile sizes: BLOCK_M=256, BLOCK_N=256, BLOCK_K=32 for XPU
    4. Set num_warps=32 for Intel XPU
    5. Add GROUP_SIZE_M swizzling for better L2 cache utilization
    6. Use boundary_check=(0, 1) tuple format, NOT booleans

    === CODE REQUIREMENTS ===
    - Include ALL imports (torch, triton, triton.language as tl)
    - Include the @triton.jit decorator and kernel function
    - Include the Model class with forward() method
    - num_warps MUST be a power of 2 (1, 2, 4, 8, 16, 32)
    - num_stages MUST be a positive integer
    - Block sizes (BLOCK_M, BLOCK_N, BLOCK_K) MUST be powers of 2
    - Block sizes should not exceed 256 for most cases

    === WHAT TO CHANGE ===
    Focus on the issues listed and apply the patterns from knowledge_patterns.
    Be aggressive with optimizations - the verification tool will check correctness.
    """

    original_code: dspy.Code["python"] = dspy.InputField(  # noqa: UP037
        desc="Original Triton kernel code for reference"
    )
    current_code: dspy.Code["python"] = dspy.InputField(  # noqa: UP037
        desc="Current Triton kernel code to optimize"
    )
    stage: str = dspy.InputField(
        desc="Optimization stage to apply (e.g., dtype_fix, block_pointers, xpu_specific)"
    )
    issues: list[DetectedIssue] = dspy.InputField(desc="Specific issues to fix in this stage")
    knowledge_patterns: str = dspy.InputField(
        desc="Optimization patterns and examples to follow - APPLY THESE"
    )
    xpu_config: str = dspy.InputField(desc="Intel XPU configuration parameters")

    optimized_code: dspy.Code["python"] = dspy.OutputField(  # noqa: UP037
        desc="Complete optimized Triton kernel code. Must include all imports, decorators, kernel function, and Model class."
    )


class OptimizerReActAgent(Optimizer):
    """
    Agent that applies optimization transformations to Triton kernels using ReAct.

    We use existing dspy.ReAct implementation.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        executor: KernelBenchExecutor | None = None,
        validator: Callable | None = None,
        max_iterations: int = 5,
    ):
        """
        Initialize optimizer agent.

        Args:
            knowledge_base: KnowledgeBase instance for optimization patterns
            executor: KernelBenchExecutor instance for runtime verification
            validator: Additional validator function (optional)
            max_iterations: Max iterations per stage
        """
        self.knowledge_base = knowledge_base
        self.executor = executor
        self.validator = validator
        self.max_iterations = max_iterations

        if not executor:
            logger.warning("No executor provided - kernels will NOT be verified at runtime!")

    def _create_verify_tool(
        self,
        original_code: str,
        kernel_name: str | None,
        input_shapes: list[tuple[int, ...]] | None,
        flop: float | None,
        dtype=None,
        skip_speedup_check: bool = False,
    ) -> Callable:
        """
        Create a verification tool for ReAct.

        The tool validates the optimized code and returns:
        - SUCCESS_MESSAGE if valid and faster
        - Detailed error/feedback message otherwise
        """
        executor = self.executor

        def compile_and_verify(optimized_code: dspy.Code["python"]) -> str:  # noqa: UP037
            """Compile and verify the optimized Triton kernel.

            Checks:
            1. Python syntax validity
            2. Triton kernel structure (imports, decorators, Model class)
            3. Runtime execution (if executor available)
            4. Performance (must be faster than original)

            Note: Correctness checking (output comparison) will be added later.

            Returns SUCCESS_MESSAGE on success, or detailed error message.
            """
            code: str = optimized_code.code

            # Step 1: Validate Python syntax
            try:
                ast.parse(code)
            except SyntaxError as e:
                return (
                    f"SYNTAX ERROR at line {e.lineno}: {e.msg}\n"
                    f"Please fix the Python syntax error and try again.\n"
                    f"Problematic line: {e.text.strip() if e.text else 'unknown'}"
                )

            # Step 2: Validate Triton structure
            has_triton_import = "import triton" in code or "from triton" in code
            has_tl_import = "triton.language" in code or "import triton.language" in code
            has_kernel = "@triton.jit" in code
            has_model = "class Model" in code

            if not has_triton_import:
                return (
                    "MISSING IMPORT: Code must include 'import triton'.\n"
                    "Please add: import triton\n"
                    "            import triton.language as tl"
                )

            if not has_tl_import:
                return (
                    "MISSING IMPORT: Code must include 'import triton.language as tl'.\n"
                    "Please add this import for Triton language primitives."
                )

            if not has_kernel:
                return (
                    "MISSING KERNEL: Code must include a @triton.jit decorated kernel function.\n"
                    "Please ensure the kernel function has the @triton.jit decorator."
                )

            if not has_model:
                return (
                    "MISSING MODEL: Code must include a 'class Model(nn.Module)' wrapper.\n"
                    "Please ensure the Model class with forward() method is present."
                )

            # Step 2b: Validate Triton constraints
            warps_match = re.search(r"num_warps\s*=\s*(\d+)", code)
            if warps_match:
                num_warps = int(warps_match.group(1))
                if num_warps <= 0 or (num_warps & (num_warps - 1)) != 0:
                    return (
                        f"INVALID num_warps={num_warps}: Must be a power of 2.\n"
                        f"Valid values: 1, 2, 4, 8, 16, 32\n"
                        f"Please fix num_warps to be a power of 2."
                    )

            # Check block sizes are powers of 2
            for block_name in [
                "BLOCK_M",
                "BLOCK_N",
                "BLOCK_K",
                "BLOCK_SIZE_M",
                "BLOCK_SIZE_N",
                "BLOCK_SIZE_K",
            ]:
                block_match = re.search(rf"{block_name}\s*[=:]\s*(\d+)", code)
                if block_match:
                    block_size = int(block_match.group(1))
                    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
                        return (
                            f"INVALID {block_name}={block_size}: Must be a power of 2.\n"
                            f"Valid values: 16, 32, 64, 128, 256\n"
                            f"Please fix {block_name} to be a power of 2."
                        )

            # Step 3: Runtime verification (if executor available)
            if executor and input_shapes:
                try:
                    comparison = executor.compare_kernels(
                        original_code=original_code,
                        optimized_code=code,
                        kernel_name=kernel_name,
                        input_shapes=input_shapes,
                        flop=flop,
                        dtype=dtype,
                    )

                    # Debug: log what executor returned
                    logger.debug(
                        f"compare_kernels result: speedup={comparison.speedup}, "
                        f"orig_time={getattr(comparison, 'original_time_us', 'N/A')}, "
                        f"opt_time={getattr(comparison, 'optimized_time_us', 'N/A')}, "
                        f"orig_tflops={comparison.original_tflops}, "
                        f"opt_tflops={comparison.optimized_tflops}, "
                        f"correct={comparison.optimized_correct}, "
                        f"is_slower={comparison.is_slower}"
                    )

                    # Check if optimized kernel failed to compile/run
                    if not comparison.optimized_correct:
                        # Use executor's feedback message if available
                        if comparison.feedback_message:
                            return comparison.feedback_message
                        return (
                            "COMPILATION/RUNTIME ERROR: Optimized kernel failed.\n"
                            "Please check for syntax errors, missing imports, or invalid Triton operations."
                        )

                    # Check performance regression
                    if not skip_speedup_check and comparison.is_slower:
                        slowdown = (
                            1.0 / comparison.speedup if comparison.speedup > 0 else float("inf")
                        )
                        orig_tflops = comparison.original_tflops or 0
                        opt_tflops = comparison.optimized_tflops or 0
                        return (
                            f"PERFORMANCE REGRESSION: Optimized kernel is {slowdown:.2f}x SLOWER.\n"
                            f"Original: {comparison.original_time_us:.2f}μs ({orig_tflops:.2f} TFLOPS)\n"
                            f"Optimized: {comparison.optimized_time_us:.2f}μs ({opt_tflops:.2f} TFLOPS)\n"
                            f"The optimization made performance worse. Please try a different approach:\n"
                            f"- Check tile sizes (prefer 256x256 for XPU)\n"
                            f"- Ensure num_warps=32 for XPU\n"
                            f"- Use block pointers with boundary_check\n"
                            f"- Add GROUP_SIZE_M swizzling"
                        )

                    # Success!
                    speedup = comparison.speedup
                    orig_tflops = comparison.original_tflops or 0
                    opt_tflops = comparison.optimized_tflops or 0

                    logger.info(
                        f"✓ Optimization verified: {speedup:.2f}x speedup "
                        f"({orig_tflops:.2f} → {opt_tflops:.2f} TFLOPS)"
                    )
                    return SUCCESS_MESSAGE

                except Exception as e:
                    return (
                        f"RUNTIME ERROR: Kernel execution failed.\n"
                        f"Error: {e!s}\n"
                        f"Please fix the runtime error. Common issues:\n"
                        f"- Missing imports (torch, nn)\n"
                        f"- Incorrect tensor shapes or strides\n"
                        f"- Invalid Triton operations\n"
                        f"- Device mismatch (ensure XPU compatibility)"
                    )

            # No executor - accept if syntax and structure are valid
            logger.warning("No executor available - accepting based on static checks only")
            return SUCCESS_MESSAGE

        return compile_and_verify

    def optimize_stage(
        self,
        code: str,
        stage: OptimizationStage,
        analysis: KernelAnalysis,
        xpu_config: dict,
        kernel_name: str | None = None,
        input_shapes: list[tuple[int, ...]] | None = None,
        flop: float | None = None,
        dtype=None,
        pytorch_code: str = "",
        init_args: list | None = None,
        perf_context: dict | None = None,
        correctness_only_stages: set | None = None,
    ) -> StageResult:
        """
        Apply a single optimization stage using ReAct.

        The  agent iteratively generates and validates optimizations until
        the compile_and_verify tool returns SUCCESS_MESSAGE or max_iterations
        is reached.

        Args:
            code: Current Triton code
            stage: Stage to apply (e.g., DTYPE_FIX, BLOCK_POINTERS, XPU_SPECIFIC)
            analysis: Kernel analysis results with detected issues
            xpu_config: XPU configuration (num_warps, tile sizes, etc.)
            kernel_name: Kernel function name (optional for Model-based kernels)
            input_shapes: Input shapes for runtime testing
            flop: FLOP count for TFLOPS calculation
            dtype: Torch dtype for input tensors

        Returns:
            StageResult with optimized code and metrics
        """
        logger.info(f"Applying optimization stage: {stage.value}")
        logger.info(f"  input_shapes: {input_shapes}")
        logger.info(f"  flop: {flop}")
        logger.info(f"  dtype: {dtype}")

        original_code = code

        # Get relevant issues for this stage
        stage_issues = self._get_stage_issues(analysis, stage)
        if not stage_issues:
            logger.info(f"No issues for stage {stage.value}, skipping")
            return StageResult(
                stage=stage,
                success=True,
                input_code=code,
                output_code=code,
                changes_made=["No changes needed"],
                reasoning="No optimization opportunities found for this stage",
            )

        # Get knowledge patterns for this stage
        knowledge_patterns = self._get_stage_patterns(stage)

        xpu_config_text = "Intel XPU Configuration:\n" + "\n".join(
            [f"  {k}: {v}" for k, v in xpu_config.items()]
        )

        skip_speedup = stage in (correctness_only_stages or set())

        # Create verification tool
        verify_tool = self._create_verify_tool(
            original_code=original_code,
            kernel_name=kernel_name,
            input_shapes=input_shapes,
            flop=flop,
            dtype=dtype,
            skip_speedup_check=skip_speedup,
        )

        # Create ReAct agent for this optimization
        react_agent = dspy.ReAct(
            signature=OptimizationReActSignature,
            tools=[verify_tool],
            max_iters=self.max_iterations,
        )

        try:
            logger.info(f"Starting ReAct optimization (max {self.max_iterations} iterations)")

            result = react_agent(
                original_code=original_code,
                current_code=code,
                stage=stage.value,
                issues=stage_issues,
                knowledge_patterns=knowledge_patterns,
                xpu_config=xpu_config_text,
            )

            # Extract optimized code from result
            if not hasattr(result, "optimized_code") or result.optimized_code is None:
                logger.error("Agent didn't return code")
                return StageResult(
                    stage=stage,
                    success=False,
                    input_code=original_code,
                    output_code=original_code,
                    error_message="Agent failed to produce optimized code",
                )

            optimized_code: str = result.optimized_code.code

            trajectory = result.trajectory if hasattr(result, "trajectory") else {}

            # Verify the final code directly (don't rely on trajectory)

            success = False
            speedup = None
            metrics_before = None
            metrics_after = None
            last_error = None

            # Check syntax first
            if not self._is_valid_python(optimized_code):
                last_error = "Final code has invalid Python syntax"
            elif not self._is_valid_triton(optimized_code):
                last_error = "Final code is not valid Triton kernel"
            elif self.executor and input_shapes:
                # Runtime verification
                try:
                    comparison = self.executor.compare_kernels(
                        original_code=original_code,
                        optimized_code=optimized_code,
                        kernel_name=kernel_name,
                        input_shapes=input_shapes,
                        flop=flop,
                        dtype=dtype,
                    )

                    if not comparison.optimized_correct:
                        last_error = "Optimized kernel produces incorrect results"
                    elif not skip_speedup and comparison.is_slower:
                        slowdown = (
                            1.0 / comparison.speedup if comparison.speedup > 0 else float("inf")
                        )
                        last_error = f"Optimized kernel is {slowdown:.2f}x slower"
                    else:
                        # Success!
                        success = True
                        speedup = comparison.speedup

                        # Build metrics dicts
                        metrics_before = None
                        metrics_after = None

                        # Only create metrics if we have all values
                        if (
                            comparison.original_time_us is not None
                            and comparison.original_tflops is not None
                        ):
                            metrics_before = {
                                "time_us": comparison.original_time_us,
                                "tflops": comparison.original_tflops,
                            }

                        if (
                            comparison.optimized_time_us is not None
                            and comparison.optimized_tflops is not None
                        ):
                            metrics_after = {
                                "time_us": comparison.optimized_time_us,
                                "tflops": comparison.optimized_tflops,
                            }
                except Exception as e:
                    last_error = f"Runtime verification failed: {e}"
            else:
                # No executor - accept if syntax valid
                success = True
                logger.warning("No executor available - accepting based on syntax check only")

            if success:
                logger.info(f"Stage {stage.value} completed successfully")
                if speedup:
                    logger.info(f"  Speedup: {speedup:.2f}x")

                return StageResult(
                    stage=stage,
                    success=True,
                    input_code=original_code,
                    output_code=optimized_code,
                    changes_made=self._extract_changes_from_trajectory(trajectory),
                    reasoning=self._extract_reasoning_from_trajectory(trajectory),
                    speedup=speedup,
                    metrics_before=metrics_before,
                    metrics_after=metrics_after,
                )
            else:
                # Optimization failed
                logger.warning(f"Stage {stage.value} failed after {self.max_iterations} iterations")
                logger.warning(f"Last error: {last_error}")

                # Dump failed kernel for debugging
                self._dump_kernel(stage, optimized_code)

                return StageResult(
                    stage=stage,
                    success=False,
                    input_code=original_code,
                    output_code=original_code,  # Keep original on failure
                    error_message=f"Failed after {self.max_iterations} iterations: {last_error}",
                )

        except Exception as e:
            logger.error(f"ReAct optimization failed with exception: {e}")
            import traceback

            logger.debug(traceback.format_exc())

            return StageResult(
                stage=stage,
                success=False,
                input_code=original_code,
                output_code=original_code,
                error_message=str(e),
            )

    def _dump_kernel(self, stage: OptimizationStage, code: str) -> None:
        """Dump failed kernel code to file for debugging."""
        import os
        from datetime import datetime

        dump_dir = os.environ.get("TRITON_OPT_DUMP_DIR", "./outputs/kernels")
        os.makedirs(dump_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dump_dir}/{stage.value}_failed_{timestamp}.py"

        try:
            with open(filename, "w") as f:
                f.write(f"# Stage: {stage.value}\n")
                f.write("# Status: FAILED\n")
                f.write(f"# Timestamp: {timestamp}\n\n")
                f.write(code)
            logger.info(f"Dumped failed kernel to: {filename}")
        except Exception as e:
            logger.warning(f"Failed to dump kernel: {e}")

    def _is_valid_python(self, code: str) -> bool:
        """Check if code is valid Python syntax."""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            logger.debug(f"Syntax error at line {e.lineno}: {e.msg}")
            return False

    def _is_valid_triton(self, code: str) -> bool:
        """Check if code looks like a valid Triton kernel."""
        has_triton_import = "import triton" in code or "from triton" in code
        has_kernel = "@triton.jit" in code or "class Model" in code
        return has_triton_import and has_kernel

    def _extract_changes_from_trajectory(self, trajectory: dict) -> list[str]:
        """Extract changes made from the agent's trajectory thoughts."""
        changes = []
        for key, value in sorted(trajectory.items()):
            if key.startswith("thought_"):
                thought = str(value).strip()
                # Extract meaningful parts of thoughts
                if any(
                    word in thought.lower()
                    for word in [
                        "applied",
                        "changed",
                        "replaced",
                        "added",
                        "removed",
                        "optimized",
                        "fixed",
                        "converted",
                        "updated",
                    ]
                ):
                    # Truncate very long thoughts
                    if len(thought) > 500:
                        thought = thought[:500] + "..."
                    changes.append(thought)

        return changes if changes else ["Optimization applied via ReAct"]

    def _extract_reasoning_from_trajectory(self, trajectory: dict) -> str:
        """Extract reasoning from the agent's trajectory thoughts."""
        thoughts = []
        for key, value in sorted(trajectory.items()):
            if key.startswith("thought_"):
                thought = str(value).strip()
                if len(thought) > 100:
                    thought = thought[:100] + "..."
                thoughts.append(thought)

        if not thoughts:
            return "ReAct optimization completed"

        return " → ".join(thoughts)

    def _get_stage_issues(
        self, analysis: KernelAnalysis, stage: OptimizationStage
    ) -> list[DetectedIssue]:
        """Get issues relevant to this stage."""
        return [
            issue
            for issue in analysis.detected_issues
            if get_stage_for_issue(issue.issue_type) == stage
        ]

    def _get_stage_patterns(self, stage: OptimizationStage) -> str:
        """Get knowledge patterns for a stage."""
        if self.knowledge_base:
            return self.knowledge_base.format_for_stage(stage)
        else:
            return (
                f"No specific patterns available for {stage.value}. Apply general best practices."
            )
