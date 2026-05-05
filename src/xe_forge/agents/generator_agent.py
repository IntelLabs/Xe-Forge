"""GeneratorAgent — PyTorch baseline -> initial Triton kernel (t0).

Three steps per generation:
    1. Run `analyze_pytorch_kernel` on the PyTorch source to extract a
       grounded structural summary (kernel type, shapes, operations,
       fusion opportunities, recommended optimizations). This is
       deterministic AST analysis, not LLM output.
    2. Call a dspy Predict over `GenerationSignature` with the PyTorch code,
       the static-analysis summary, the spec summary (from
       xe_forge.core.spec_loader.Spec), and KB hints.
    3. Run `validate_triton_kernel` on the emitted code. On failure, retry
       with the validation errors appended to the prompt, up to
       `max_iterations`.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import dspy

from xe_forge.core.analyze_kernel import analyze_pytorch_kernel
from xe_forge.core.validate_triton import validate_triton_kernel

logger = logging.getLogger(__name__)


class GenerationSignature(dspy.Signature):
    """Generate a complete Triton kernel on Intel XPU from a PyTorch baseline.

    Produce a self-contained Python file with:
    - All imports (torch, triton, triton.language as tl).
    - One or more `@triton.jit` kernel functions implementing the same
      computation as the PyTorch baseline.
    - A `Model` class that subclasses `torch.nn.Module`, launches the
      kernel(s), and mirrors the baseline's `get_inputs()` and
      `get_init_inputs()` / module-level constants exactly.

    The kernel does NOT need to be fully optimized — subsequent trials will
    optimize it. Aim for correctness and a sensible tile-based structure.
    Use tensor descriptors (`tl.make_tensor_descriptor`) when applicable on
    XPU. Respect the constraints in `knowledge_base_context`.

    When `validation_errors` is non-empty, the previous attempt failed
    `validate_triton`; fix the listed issues.
    """

    pytorch_code: str = dspy.InputField(desc="PyTorch baseline implementation")
    spec_summary: str = dspy.InputField(
        desc="Input shapes, dtype, FLOP, and Model init args from the spec"
    )
    static_analysis: str = dspy.InputField(
        desc="AST-derived structural summary of the PyTorch code "
        "(kernel type, operations, fusion opportunities, recommendations)"
    )
    knowledge_base_context: str = dspy.InputField(
        desc="Relevant optimization patterns, templates, and constraints"
    )
    validation_errors: str = dspy.InputField(
        desc="Validation errors from the previous attempt, empty on first try"
    )

    triton_code: dspy.Code["python"] = dspy.OutputField(
        desc="Complete self-contained Triton kernel file + Model class"
    )
    strategy_notes: str = dspy.OutputField(
        desc="One-line description of the kernel structure chosen "
        "(tiling, fusion boundaries, etc.)"
    )


class GeneratorAgent:
    """PyTorch -> initial Triton kernel, with validation feedback loop."""

    def __init__(self, knowledge_base=None, max_iterations: int = 3):
        self.knowledge_base = knowledge_base
        self.max_iterations = max_iterations
        self.predictor = dspy.Predict(GenerationSignature)

    def generate(
        self,
        pytorch_code: str,
        pytorch_path: str | None = None,
        spec_summary: str = "",
        kb_context: str | None = None,
    ) -> tuple[str, str, dict]:
        """Generate an initial Triton kernel.

        Returns ``(triton_code, strategy_notes, static_analysis)``.
        Raises ``RuntimeError`` if no valid kernel could be produced.
        """
        static = self._run_static_analysis(pytorch_code, pytorch_path)
        static_text = _format_static_analysis(static)
        kb_text = kb_context if kb_context is not None else self._load_kb_context()

        last_code = ""
        last_strategy = ""
        validation_errors = ""

        for attempt in range(1, self.max_iterations + 1):
            logger.info("[generator] attempt %d/%d", attempt, self.max_iterations)
            try:
                result = self.predictor(
                    pytorch_code=pytorch_code,
                    spec_summary=spec_summary or "(no spec summary)",
                    static_analysis=static_text,
                    knowledge_base_context=kb_text or "",
                    validation_errors=validation_errors,
                )
            except Exception as e:
                logger.warning("[generator] predictor failed: %s", e)
                if attempt == self.max_iterations:
                    raise RuntimeError(
                        f"GeneratorAgent failed after {attempt} attempts: {e}"
                    ) from e
                validation_errors = f"Previous call errored: {e}"
                continue

            raw_code = getattr(result, "triton_code", None)
            triton_code = str(raw_code) if raw_code is not None else ""
            strategy = getattr(result, "strategy_notes", "") or ""
            if not triton_code.strip():
                validation_errors = "Empty kernel returned; emit complete code."
                continue

            errors = self._validate(triton_code)
            critical = [e for e in errors if e.level == "ERROR"]
            last_code, last_strategy = triton_code, strategy

            if not critical:
                logger.info("[generator] passed validation on attempt %d", attempt)
                return triton_code, strategy, static

            logger.info(
                "[generator] %d validation error(s) on attempt %d", len(critical), attempt
            )
            validation_errors = "\n".join(str(e) for e in critical)

        if last_code:
            logger.warning(
                "[generator] returning last attempt despite validation errors "
                "(GeneratorAgent exhausted retries)"
            )
            return last_code, last_strategy, static
        raise RuntimeError(
            f"GeneratorAgent failed to produce valid code after "
            f"{self.max_iterations} attempts"
        )

    # ------------------------------------------------------------------ helpers

    def _run_static_analysis(self, pytorch_code: str, pytorch_path: str | None) -> dict:
        try:
            if pytorch_path is not None and Path(pytorch_path).exists():
                return analyze_pytorch_kernel(Path(pytorch_path))
            # Fall back to a tempfile so analyze_pytorch_kernel (which reads
            # from disk) still works when only the source string is available.
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(pytorch_code)
                tmp_path = Path(f.name)
            try:
                return analyze_pytorch_kernel(tmp_path)
            finally:
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
        except Exception as e:
            logger.warning("[generator] static analysis failed: %s", e)
            return {}

    def _validate(self, triton_code: str):
        """Write to a temp file and run validate_triton_kernel on it."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(triton_code)
            path = Path(f.name)
        try:
            return validate_triton_kernel(path)
        except Exception as e:
            logger.warning("[generator] validate_triton_kernel raised: %s", e)
            return []
        finally:
            try:
                path.unlink()
            except Exception:
                pass

    def _load_kb_context(self) -> str:
        if self.knowledge_base is None:
            return ""
        for method in ("format_all", "format_for_generator", "summary"):
            fn = getattr(self.knowledge_base, method, None)
            if callable(fn):
                try:
                    return fn()
                except Exception as e:
                    logger.debug("[generator] KB.%s failed: %s", method, e)
        return ""


def _format_static_analysis(analysis: dict[str, Any]) -> str:
    if not analysis:
        return "(static analysis unavailable)"
    # Prefer JSON so the LLM gets named fields it can reason about.
    try:
        serializable = {
            k: (list(v) if isinstance(v, (set, tuple)) else v)
            for k, v in analysis.items()
        }
        return json.dumps(serializable, indent=2, default=str)
    except Exception:
        return str(analysis)
