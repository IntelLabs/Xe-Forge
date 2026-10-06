"""Classify one lowering attempt and turn it into feedback for the next.

The four questions are kept apart and answered in order: did it finalize, did it
run, was it correct, was it faster. A later question is only asked once the
earlier one is answered yes, so a speedup is never reported for a kernel that
was wrong.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import yaml


class FailureCategory(StrEnum):
    FINALIZER_ERROR = "finalizer_error"  # the vISA did not parse or finalize
    RUNTIME_ERROR = "runtime_error"  # it built, but launching or running it failed
    TIMEOUT = "timeout"  # it hung (counted as a runtime failure)
    INCORRECT = "incorrect"  # it ran, and produced the wrong values
    CORRECT_SLOWER = "correct_slower"
    CORRECT_FASTER = "correct_faster"
    CORRECT = "correct"  # correct, performance not measured
    # Not the model's fault; never fed back as a repair request.
    INFRA = "infra"
    LEAK_DETECTED = "leak_detected"

    @property
    def is_correct(self) -> bool:
        return self in (FailureCategory.CORRECT, FailureCategory.CORRECT_FASTER, FailureCategory.CORRECT_SLOWER)

    @property
    def is_model_failure(self) -> bool:
        return self in (
            FailureCategory.FINALIZER_ERROR,
            FailureCategory.RUNTIME_ERROR,
            FailureCategory.TIMEOUT,
            FailureCategory.INCORRECT,
        )


@dataclass
class CompileResult:
    success: bool
    error: str = ""
    override_applied: bool | None = None
    simd_size: int | None = None
    grf_count: int | None = None
    spill_size: int | None = None
    private_size: int | None = None
    binary_size: int | None = None
    compile_s: float | None = None


@dataclass
class RuntimeResult:
    success: bool
    error: str = ""
    timed_out: bool = False
    device_lost: bool = False


@dataclass
class CorrectnessResult:
    success: bool
    shapes_checked: int = 0
    shapes_skipped: int = 0
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    failing_elements: int | None = None
    total_elements: int | None = None
    failing_case: dict[str, Any] | None = None
    first_failures: list[dict[str, Any]] = field(default_factory=list)
    nan_or_inf: bool = False
    guard_violations: list[str] = field(default_factory=list)
    inputs_modified: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class PerfResult:
    baseline_us: float
    candidate_us: float
    timer: str

    @property
    def speedup(self) -> float:
        return self.baseline_us / self.candidate_us if self.candidate_us > 0 else 0.0


@dataclass
class Evaluation:
    """Everything known about one attempt."""

    category: FailureCategory
    compile: CompileResult
    runtime: RuntimeResult | None = None
    correctness: CorrectnessResult | None = None
    performance: PerfResult | None = None
    detail: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        if self.performance is not None:
            d["performance"]["speedup"] = round(self.performance.speedup, 4)
        return d

    def feedback_yaml(self) -> str:
        """The structured feedback shown to the model (the shape plan.md §5 asks for)."""
        doc: dict[str, Any] = {"category": self.category.value}
        c = self.compile
        doc["compile"] = {"success": c.success}
        if c.error:
            doc["compile"]["error"] = c.error
        if self.runtime is not None:
            doc["runtime"] = {"success": self.runtime.success}
            if self.runtime.error:
                doc["runtime"]["error"] = self.runtime.error
            if self.runtime.timed_out:
                doc["runtime"]["timed_out"] = True
        if self.correctness is not None:
            k = self.correctness
            doc["correctness"] = {
                key: value
                for key, value in {
                    "success": k.success,
                    "shapes_checked": k.shapes_checked,
                    "max_abs_error": _round(k.max_abs_error),
                    "max_rel_error": _round(k.max_rel_error),
                    "failing_elements": k.failing_elements,
                    "total_elements": k.total_elements,
                    "failing_case": k.failing_case,
                    "first_failures": k.first_failures or None,
                    "nan_or_inf": k.nan_or_inf or None,
                    "out_of_bounds_writes": k.guard_violations or None,
                    "inputs_modified": k.inputs_modified or None,
                    "error": k.error or None,
                }.items()
                if value is not None
            }
        if self.performance is not None:
            p = self.performance
            doc["performance"] = {
                "baseline_us": round(p.baseline_us, 3),
                "candidate_us": round(p.candidate_us, 3),
                "speedup": round(p.speedup, 3),
            }
        hw = {k: v for k, v in {"grf": c.grf_count, "spills": c.spill_size, "simd": c.simd_size}.items() if v is not None}
        if hw:
            doc["hardware"] = hw
        return yaml.safe_dump(doc, sort_keys=False, width=100)


def _round(x: float | None) -> float | None:
    return None if x is None else float(f"{x:.4g}")


# Lines a diagnostic may carry that say where things live on this machine, not
# what was wrong with the kernel.
_PATH_RE = re.compile(r"(/[\w.+-]+){2,}")
_NOISE = (
    "Compilation from IR - skipping loading of FCL",
    "Build succeeded.",
    "OVERRIDEN:",
)


def sanitize(text: str, *, forbidden_lines: set[str] | None = None, limit: int = 4000) -> str:
    """Make compiler or runtime output safe and short enough to show the model.

    Paths are replaced with their base name, known noise is dropped, any line
    present in ``forbidden_lines`` (text of a sealed compiler artefact) is removed,
    and the result is clipped.
    """
    out = []
    for line in text.splitlines():
        if any(n in line for n in _NOISE):
            continue
        if forbidden_lines and re.sub(r"\s+", " ", line).strip() in forbidden_lines:
            continue
        line = _PATH_RE.sub(lambda m: "<path>/" + m.group(0).rsplit("/", 1)[-1], line)
        out.append(line.rstrip())
    text = "\n".join(line for line in out if line.strip())
    if len(text) > limit:
        text = text[: limit // 2] + "\n...[truncated]...\n" + text[-limit // 2 :]
    return text
