"""
Translate Agent - Translates a Triton kernel to a target backend using CoVeR


"""

import ast
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import dspy

from xpu_forge.agents.cover import CoVeR

logger = logging.getLogger(__name__)

SUCCESS_MESSAGE = "Success! Translation verified."

# ---------------------------------------------------------------------------
# Per-backend guidance injected into the signature instructions and verify tool
# ---------------------------------------------------------------------------
BACKEND_GUIDANCE: dict[str, dict] = {
    "gluon": {
        "display_name": "Gluon (Intel XPU)",
        "required_imports": ["import gluon", "from gluon"],
        "required_decorators": ["@gluon.jit"],
        "forbidden_patterns": [
            (r"triton\.Config\([^)]*grf_mode", "grf_mode must not appear in gluon.Config()"),
        ],
        "translation_guidance": """\
You are translating a Triton kernel to Gluon, the Intel XPU kernel language.

Key Gluon / Intel XPU translation rules:
- Replace `import triton` / `import triton.language as tl` with `import gluon` / `import gluon.language as gl`
- Replace `@triton.jit` with `@gluon.jit`
- Replace `@triton.autotune` with `@gluon.autotune`
- Replace `triton.Config(...)` with `gluon.Config(...)`; valid kwargs: meta-parameters, num_warps, num_stages
- Replace `tl.*` intrinsics with `gl.*` equivalents:
    tl.load            → gl.load
    tl.store           → gl.store
    tl.dot             → gl.dot
    tl.arange          → gl.arange
    tl.zeros           → gl.zeros
    tl.constexpr       → gl.constexpr
    tl.program_id      → gl.program_id
    tl.cdiv            → gl.cdiv
    tl.make_block_ptr  → gl.make_block_ptr
    tl.advance         → gl.advance
    tl.static_assert   → gl.static_assert
    tl.float16/float32 → gl.float16/float32
    tl.math.*          → gl.math.*
    tl.range           → gl.range
- Replace `triton.cdiv` with `gluon.cdiv` in the host (Model.forward) code
- Replace `triton.next_power_of_2` with `gluon.next_power_of_2` if present
- XPU tile sizes: prefer BLOCK_M=256, BLOCK_N=256, BLOCK_K=32, num_warps=32
- Add GROUP_SIZE_M swizzling if not present (improves L2 cache utilisation)
- grf_mode is set via environment variable IGC_EnableLargeGRF=1, NOT in kernel code
- Keep the Model class interface IDENTICAL: same __init__ args, same forward() signature,
  same output tensor shapes and dtypes
- Update all triton.cdiv grid calculations to gluon.cdiv
""",
    },
    # Future backends slot in here:
    # "blackwell": { ... },
    # "sycl": { ... },
}


# ---------------------------------------------------------------------------
# DSPy signature
# ---------------------------------------------------------------------------


class TranslateSignature(dspy.Signature):
    """Translate a Triton kernel to an equivalent kernel in the target backend.

    You are an expert in GPU/XPU kernel programming and backend translation.

    Translate the given Triton kernel to the target backend while:
    - Preserving the exact Model class interface (same forward() args and return type)
    - Using idiomatic constructs for the target backend
    - Applying hardware-appropriate tile sizes and configurations
    - Keeping all algorithmic logic identical

    You will receive backend-specific translation guidance in the
    `translation_guidance` field. Follow it precisely.

    Previous failed attempts and their error messages appear in the trajectory.
    Use that feedback to fix the specific issues in the current attempt.

    === CODE REQUIREMENTS ===
    - Include ALL required imports for the target backend
    - Include the translated kernel function with the correct decorator
    - Include the Model class with forward() returning the same tensor shapes
    - Do NOT include any Triton-specific imports or decorators in the output
    """

    original_triton_code: str = dspy.InputField(desc="Original Triton kernel source to translate")
    target_backend: str = dspy.InputField(desc="Target backend name (e.g. 'gluon')")
    translation_guidance: str = dspy.InputField(
        desc="Backend-specific translation rules and idioms to follow"
    )
    translated_code: dspy.Code["python"] = dspy.OutputField(
        desc="Complete translated kernel with all imports, kernel function, and Model class"
    )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TranslateResult:
    success: bool
    translated_code: str = ""
    error_message: str = ""
    attempts: int = 0


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _extract_code(code_obj) -> str:
    """Unwrap a dspy.Code object or plain string, stripping fences."""
    code = code_obj.code if hasattr(code_obj, "code") else str(code_obj)
    if "```python" in code:
        m = re.search(r"```python\s*(.*?)\s*```", code, re.DOTALL)
        if m:
            return m.group(1).strip()
    if "```" in code:
        m = re.search(r"```\s*(.*?)\s*```", code, re.DOTALL)
        if m:
            return m.group(1).strip()
    return code.strip()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class TranslateAgent:
    """
    Translates a Triton kernel to a target backend using CoVeR.

    Uses the same iterative verify-and-fix loop as OptimizerAgent:
    each attempt is compiled (syntax check) and structurally validated;
    if an executor is provided the translated kernel is also run for
    correctness against the original Triton output.
    """

    def __init__(self, executor=None, max_iterations: int = 5):
        self.executor = executor
        self.max_iterations = max_iterations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(
        self,
        triton_code: str,
        target: str = "gluon",
        kernel_name: str | None = None,
        input_shapes=None,
        flop: float | None = None,
        dtype=None,
        init_args=None,
    ) -> TranslateResult:
        """
        Translate *triton_code* to *target* backend using CoVeR.

        Parameters mirror OptimizerAgent.optimize_stage so the pipeline
        can wire them up identically.
        """
        target = target.lower()
        if target not in BACKEND_GUIDANCE:
            supported = ", ".join(BACKEND_GUIDANCE)
            return TranslateResult(
                success=False,
                error_message=f"Unknown target '{target}'. Supported: {supported}",
            )

        guidance = BACKEND_GUIDANCE[target]
        logger.info(f"Translating kernel to {guidance['display_name']} …")

        verify_tool, attempt_counter = self._build_verify_tool(
            triton_code=triton_code,
            target=target,
            guidance=guidance,
            kernel_name=kernel_name,
            input_shapes=input_shapes,
            flop=flop,
            dtype=dtype,
            init_args=init_args,
        )

        cover = CoVeR(
            signature=TranslateSignature,
            tools=[verify_tool],
            success=SUCCESS_MESSAGE,
            max_iters=self.max_iterations,
            use_raw_fixer_output=True,
        )

        try:
            result = cover(
                original_triton_code=triton_code,
                target_backend=target,
                translation_guidance=guidance["translation_guidance"],
            )
        except Exception as e:
            logger.error(f"CoVeR failed during translation: {e}")
            return TranslateResult(success=False, error_message=str(e))

        if not hasattr(result, "translated_code") or result.translated_code is None:
            return TranslateResult(
                success=False,
                error_message="CoVeR produced no translated_code",
                attempts=attempt_counter["count"],
            )

        translated = _extract_code(result.translated_code)

        # Final structural check
        ok, err = self._structural_check(translated, guidance)
        if not ok:
            return TranslateResult(
                success=False,
                translated_code=translated,
                error_message=f"Final check failed: {err}",
                attempts=attempt_counter["count"],
            )

        logger.info(
            f"Translation to {guidance['display_name']} succeeded "
            f"in {attempt_counter['count']} attempt(s)."
        )
        return TranslateResult(
            success=True,
            translated_code=translated,
            attempts=attempt_counter["count"],
        )

    def translate_file(
        self,
        input_path: Path,
        target: str = "gluon",
        out_dir: Path | None = None,
        overwrite: bool = False,
        kernel_name: str | None = None,
        input_shapes=None,
        flop: float | None = None,
        dtype=None,
        init_args=None,
    ) -> TranslateResult:
        """
        Read a kernel file, translate it, and write <stem>_<target>.py.

        The YAML spec is intentionally left unchanged — the translated kernel
        uses the same input shapes, dtypes, and flop counts.
        """
        input_path = Path(input_path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Kernel file not found: {input_path}")

        source = input_path.read_text(encoding="utf-8")
        result = self.translate(
            triton_code=source,
            target=target,
            kernel_name=kernel_name,
            input_shapes=input_shapes,
            flop=flop,
            dtype=dtype,
            init_args=init_args,
        )

        if not result.success:
            return result

        target_dir = Path(out_dir) if out_dir else input_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / f"{input_path.stem}_{target}.py"

        if out_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {out_path}  (pass overwrite=True to replace)"
            )

        out_path.write_text(result.translated_code, encoding="utf-8")
        logger.info(f"Written: {out_path}")
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_verify_tool(
        self,
        triton_code,
        target,
        guidance,
        kernel_name,
        input_shapes,
        flop,
        dtype,
        init_args,
    ):
        """Build the CoVeR verify tool and a shared attempt counter."""
        executor = self.executor
        attempt_counter = {"count": 0}

        def compile_and_verify(translated_code: dspy.Code["python"]) -> str:
            attempt_counter["count"] += 1
            code = _extract_code(translated_code)

            # 1. Syntax
            try:
                ast.parse(code)
            except SyntaxError as e:
                return f"SYNTAX ERROR at line {e.lineno}: {e.msg}"

            # 2. Structural checks
            ok, err = self._structural_check(code, guidance)
            if not ok:
                return err

            # 3. Correctness against original Triton kernel (if executor available)
            if executor and input_shapes:
                try:
                    comparison = executor.compare_kernels(
                        original_code=triton_code,
                        optimized_code=code,
                        kernel_name=kernel_name,
                        input_shapes=input_shapes,
                        flop=flop,
                        dtype=dtype,
                        init_args=init_args,
                    )
                    if not comparison.optimized_correct:
                        return (
                            comparison.feedback_message
                            or "Translated kernel produces incorrect results vs the original Triton kernel."
                        )
                    logger.info(
                        f"Translation correctness verified "
                        f"(relative perf: {comparison.speedup:.2f}x)"
                    )
                except Exception as e:
                    return f"RUNTIME ERROR during correctness check: {e}"

            return SUCCESS_MESSAGE

        tool = dspy.Tool(
            func=compile_and_verify,
            name="compile_and_verify",
            desc=f'Validates the translated kernel. Returns "{SUCCESS_MESSAGE}" on success.',
        )
        return tool, attempt_counter

    @staticmethod
    def _structural_check(code: str, guidance: dict) -> tuple[bool, str]:
        """Check imports, decorators, and forbidden patterns."""
        has_required_import = any(imp in code for imp in guidance["required_imports"])
        if not has_required_import:
            return False, (
                f"MISSING required import. Expected one of: {guidance['required_imports']}"
            )

        has_decorator = any(dec in code for dec in guidance["required_decorators"])
        if not has_decorator:
            return False, (
                f"MISSING kernel decorator. Expected one of: {guidance['required_decorators']}"
            )

        if "class Model" not in code:
            return False, "MISSING: class Model"

        if "import triton" in code or "triton.jit" in code:
            return False, (
                "OUTPUT STILL CONTAINS Triton imports/decorators. "
                "Replace all triton.* references with the target backend equivalents."
            )

        for pattern, msg in guidance.get("forbidden_patterns", []):
            if re.search(pattern, code):
                return False, f"FORBIDDEN PATTERN: {msg}"

        return True, ""
