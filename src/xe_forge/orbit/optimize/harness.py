"""
The correctness harness for an in-place kernel (plan §19.6).

A kernel that was never extracted is checked the way the workload reaches it: import it
from the installed tree and compare against a reference computed in higher precision.
This module renders that harness as a standalone script and runs it in a fresh process.

The fresh process is not incidental. The harness has to observe the *file on disk*, and
a module already imported in this interpreter would keep serving the pre-patch version
for the rest of the session — so an in-process check would report the old kernel passing
after a patch that broke it. That is the same failure the mutation check in §12.12 exists
to catch, arriving through a different door.

The verdict is a fraction of rows within tolerance rather than `allclose`, for the
reasons in `compare/accuracy.py`. Three exit codes, and the third matters:

* 0 — checked and correct
* 1 — checked and wrong
* 2 — could not be checked

"Could not be checked" must never collapse into either of the others. A harness that
failed to import reads as a passing kernel if 2 is folded into 0, and as a broken one if
folded into 1; both are wrong, and the second wastes a revert on a working candidate.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

# A correctness check is a handful of kernel launches. Beyond this something is hung —
# usually a JIT compile that will not finish — and waiting longer will not help.
DEFAULT_TIMEOUT_S = 600.0

# Below this fraction of rows matching, the kernel is wrong.
DEFAULT_PASS_FRACTION = 0.999


# Paths a reference cannot reach.
#
# `gumbel_sample` at temperature 0 degenerates to argmax, which torch reproduces exactly.
# Above 0 it depends on Triton's Philox stream, which torch cannot reproduce at all — so
# a reference test can only ever cover the temperature-0 path.
#
# That is not a small corner. An agent proposed removing a `tl.where` that lives inside
# `if temp != 0.0`, i.e. exclusively in the uncovered branch: the gate would have returned
# accuracy 1.0000 without executing the changed line once.
#
# The answer where no reference exists is a DIFFERENTIAL check: run the pre-patch kernel
# and the post-patch kernel with identical seeds and inputs, and require bit-identical
# output. It establishes something weaker than correctness — it cannot tell you the
# original was right — but it is exactly the claim a behaviour-preserving optimization
# makes, and it reaches every path the workload does.
DIFFERENTIAL_NOTE = (
    "differential: same seed, same inputs, output must be bit-identical to the pre-patch kernel"
)


class CheckOutcome(StrEnum):
    CORRECT = "correct"
    WRONG = "wrong"
    # Distinct from WRONG on purpose: nothing was established about the kernel.
    UNCHECKED = "unchecked"


@dataclass
class CheckResult:
    outcome: CheckOutcome
    accuracy: float | None = None
    detail: str = ""

    @property
    def correct(self) -> bool:
        return self.outcome is CheckOutcome.CORRECT

    def format(self) -> str:
        if self.accuracy is None:
            return f"[{self.outcome.value}] {self.detail}"
        return f"[{self.outcome.value}] accuracy {self.accuracy:.4f} — {self.detail}"


# Shapes a correctness check must span, not just the convenient one.
#
# A single-shape harness answers "correct at n=256" and reports it as "correct". The
# gap is not hypothetical: an index-width change (hoisting a row base out of per-lane
# arithmetic so the lane offset can be int32) is exact at n=256, where
# 256 x 151936 = 38.9M fits an int32 with room to spare, and silently corrupts above
# ~14,138 tokens where token_idx * vocab crosses 2^31. Chunked prefill reaches that.
#
# So a shape sweep is part of the check rather than an optional extra, and it includes
# at least one shape chosen to stress index width rather than to be representative.
DEFAULT_SHAPE_SWEEP = (256, 4096, 16384)


@dataclass
class HarnessSpec:
    """What it takes to check one kernel against a reference.

    `reference_expr` is written by hand rather than derived, because a reference derived
    from the kernel under test would agree with its bugs. It must be independent, and in
    higher precision than the kernel — that is what makes it a reference rather than a
    second opinion.
    """

    kernel_id: str
    import_statement: str
    setup: str
    call_expr: str
    reference_expr: str
    comparison: str = "exact"
    notes: str = ""
    # Values substituted for `{n}` in `setup`, one run each. A change that is correct at
    # one shape and wrong at another passes a single-shape harness.
    shapes: tuple[int, ...] = DEFAULT_SHAPE_SWEEP


def render_harness(spec: HarnessSpec) -> str:
    """Render a standalone correctness script for one kernel."""
    compare_block = (
        "    match = (actual.to(torch.int64) == reference.to(torch.int64))\n"
        "    correct = int(match.sum().item()); total = int(match.numel())\n"
        if spec.comparison == "exact"
        else (
            "    from xe_forge.orbit.compare.accuracy import compare_tensors\n"
            "    result = compare_tensors(actual, reference)\n"
            "    correct, total = result.correct, result.total\n"
        )
    )
    return f'''"""Correctness check for {spec.kernel_id}, generated by Xe-Orbit (§19.6).

The kernel is imported from the installed tree exactly as the workload imports it, so
whatever is on disk is what gets checked. The reference is independent and computed in
higher precision; a reference derived from the kernel would agree with its bugs.

{spec.notes}
"""
import json, sys
from pathlib import Path


def main() -> int:
    import torch
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("UNCHECKED: no XPU device", file=sys.stderr)
        return 2

    try:
        {spec.import_statement}
    except Exception as exc:
        # An import failure says nothing about the kernel's numerics, so it must not be
        # reported as a wrong answer.
        print(f"UNCHECKED: import failed: {{exc}}", file=sys.stderr)
        return 2

    torch.manual_seed(0)
{_indent(spec.setup, 4)}

    try:
        actual = {spec.call_expr}
        reference = {spec.reference_expr}
    except Exception as exc:
        print(f"UNCHECKED: kernel or reference raised: {{exc}}", file=sys.stderr)
        return 2

{compare_block}
    accuracy = correct / total if total else 0.0
    Path("accuracy.json").write_text(json.dumps(
        {{"accuracy": accuracy, "correct": correct, "total": total,
          "kernel_id": "{spec.kernel_id}"}}, indent=2))
    print(f"ACCURACY {{accuracy:.6f}} {{correct}}/{{total}}")
    return 0 if accuracy >= {DEFAULT_PASS_FRACTION} else 1


if __name__ == "__main__":
    sys.exit(main())
'''


def run_harness(
    script: Path,
    python: str | None = None,
    cwd: Path | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> CheckResult:
    """Run a rendered harness in a fresh process and read its verdict.

    A fresh process is required, not preferred: a module imported earlier in this
    interpreter would keep serving the pre-patch source, so an in-process check would
    happily confirm a kernel that is no longer the one on disk.
    """
    try:
        completed = subprocess.run(
            [python or sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(cwd) if cwd else None,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            CheckOutcome.UNCHECKED,
            detail=f"harness did not finish within {timeout_s:.0f}s",
        )
    except OSError as exc:
        return CheckResult(CheckOutcome.UNCHECKED, detail=f"could not run harness: {exc}")

    accuracy = _parse_accuracy(completed.stdout)

    if completed.returncode == 2:
        return CheckResult(
            CheckOutcome.UNCHECKED,
            accuracy=accuracy,
            detail=_last_line(completed.stderr) or "harness reported it could not check",
        )
    if completed.returncode == 0:
        return CheckResult(CheckOutcome.CORRECT, accuracy=accuracy, detail="matches the reference")
    if completed.returncode == 1:
        # Exit 1 is ambiguous: the harness uses it for "wrong", and an uncaught Python
        # exception produces it too. A numerical verdict requires numerical evidence, so
        # without an ACCURACY line this is a crash, not a failing kernel — reverting a
        # working candidate because the harness threw would be the wrong repair.
        if accuracy is None:
            return CheckResult(
                CheckOutcome.UNCHECKED,
                detail=f"harness crashed before reporting: {_last_line(completed.stderr)}",
            )
        return CheckResult(
            CheckOutcome.WRONG,
            accuracy=accuracy,
            detail="output does not match the reference",
        )
    # A code the harness does not define — a crash, a signal. Not a numerical verdict.
    return CheckResult(
        CheckOutcome.UNCHECKED,
        accuracy=accuracy,
        detail=f"harness exited {completed.returncode}: {_last_line(completed.stderr)}",
    )


def _parse_accuracy(stdout: str) -> float | None:
    for line in stdout.splitlines():
        if line.startswith("ACCURACY "):
            try:
                return float(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def _last_line(text: str) -> str:
    lines = [line for line in (text or "").strip().splitlines() if line.strip()]
    return lines[-1][:200] if lines else ""


def _indent(block: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line for line in block.splitlines())


@dataclass
class DifferentialSpec:
    """Check a patched kernel against the pre-patch kernel, not against a reference.

    Needed wherever a reference cannot reach. `gumbel_sample` above temperature 0 rides
    Triton's Philox stream, which torch cannot reproduce, so a reference test covers only
    the temperature-0 path — and an agent promptly proposed removing a line that exists
    *only* in the other branch.

    Two runs of the same kernel with the same seed produce the same stream, so a
    differential check reaches every path the workload does. It proves something weaker
    than correctness — it cannot tell you the original was right — but "bit-identical to
    what shipped" is exactly the claim a behaviour-preserving optimization makes, and it
    is the strongest claim available here.
    """

    kernel_id: str
    import_statement: str
    setup: str
    call_expr: str
    # Each case names one execution path. The point is to span branches a reference
    # cannot, so a case that only re-runs the covered path adds nothing.
    cases: list[dict[str, object]] = field(default_factory=list)
    notes: str = ""


def render_differential(spec: DifferentialSpec) -> str:
    """Render a save/compare harness: snapshot before the patch, verify after."""
    cases = json.dumps(spec.cases or [{}], indent=8)
    return f'''"""Differential check for {spec.kernel_id}, generated by Xe-Orbit (§19.6).

Run with --save before patching and --compare after. Identical seeds mean identical RNG
streams, so any difference is the patch. This reaches paths a torch reference cannot —
which is the point: {spec.notes}
"""
import argparse, json, sys
from pathlib import Path

CASES = {cases}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save"); ap.add_argument("--compare")
    args = ap.parse_args()

    import torch
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("UNCHECKED: no XPU device", file=sys.stderr); return 2
    try:
        {spec.import_statement}
    except Exception as exc:
        print(f"UNCHECKED: import failed: {{exc}}", file=sys.stderr); return 2

    results = []
    for case in CASES:
        torch.manual_seed(0)          # same stream on both sides of the patch
{_indent(spec.setup, 8)}
        try:
            out = {spec.call_expr}
        except Exception as exc:
            print(f"UNCHECKED: case {{case}} raised: {{exc}}", file=sys.stderr); return 2
        results.append(out.detach().to("cpu").tolist())

    if args.save:
        Path(args.save).write_text(json.dumps(results))
        print(f"SAVED {{len(results)}} cases"); return 0

    if not args.compare:
        print("UNCHECKED: neither --save nor --compare given", file=sys.stderr); return 2
    try:
        expected = json.loads(Path(args.compare).read_text())
    except Exception as exc:
        print(f"UNCHECKED: no baseline to compare against: {{exc}}", file=sys.stderr); return 2

    if len(expected) != len(results):
        print("UNCHECKED: case count changed between runs", file=sys.stderr); return 2

    matched = sum(1 for a, b in zip(expected, results) if a == b)
    total = len(results)
    print(f"ACCURACY {{matched / total:.6f}} {{matched}}/{{total}}")
    return 0 if matched == total else 1


if __name__ == "__main__":
    sys.exit(main())
'''


def combined_check(checks: list[CheckResult]) -> CheckResult:
    """Require every check to pass, and report the weakest outcome.

    A loop with more than one correctness check must not accept a candidate because the
    cheapest one passed. The ordering of outcomes matters as much as the conjunction:

    * any WRONG makes the whole thing WRONG — one check proving a difference is proof,
      whatever the others say;
    * otherwise any UNCHECKED makes it UNCHECKED, because a candidate whose changed path
      was never executed is unproven no matter how many untouched paths passed. This is
      the case that motivated the function. A reference harness covering temperature 0
      reported accuracy 1.0000 for a kernel whose only defect lived in the temperature>0
      branch; the differential check caught it at 1/3. Taking the best of the two, or
      even the first, would have shipped it.
    """
    if not checks:
        return CheckResult(CheckOutcome.UNCHECKED, detail="no checks were run")

    for result in checks:
        if result.outcome is CheckOutcome.WRONG:
            return result
    for result in checks:
        if result.outcome is CheckOutcome.UNCHECKED:
            return result

    accuracies = [c.accuracy for c in checks if c.accuracy is not None]
    return CheckResult(
        CheckOutcome.CORRECT,
        accuracy=min(accuracies) if accuracies else None,
        detail=f"all {len(checks)} checks passed, including paths no reference reaches",
    )
