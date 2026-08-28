"""
Enablement: making a workload run at all, before optimizing it (plan §5.6 item 1).

Hyperloom's largest structural lead over this codebase is a six-rung escalation
ladder: diagnose the capability gap, then climb — serve-flag wire-up, in-tree source
patch, attempt-scoped runtime in an isolated venv, source localization, off-loop
compiled build — with a **runnable gate**: a fix earns KEEP only when the workload
boots with it and re-passes the accuracy eval, never on artifact verification alone.

This module is the v0.1 slice of that ladder, and it is deliberately the bottom of
it: **diagnosis** (rung 0, deterministic — classifying a failed launch is trace
parsing, and §3 forbids an LLM where a deterministic answer exists), the two rungs
Orbit can already act on (a serve-flag suggestion; a source patch through §13.2's
journalled patcher), and the runnable gate as an enforced contract. Rungs 3-5 — the
scoped runtime, source localization and the off-loop build lane — are v0.2 (§24
Tier C names enablement the v0.2 headline); they appear in the enum so a diagnosis
can honestly say "the fix for this lives on a rung that is not built yet", which is
a different finding from "there is no fix".

The measured motivation, from §5.6: on Wildcat Lake, `GRAPH_CAPTURE` was unavailable
(`No valid triton configs`, `Internal Triton ZEBIN codegen error`) and the pipeline
correctly reported the dead end — honest, and the end of the road. Diagnosis-then-
climb is the difference between reporting that a lever is unavailable and making it
available.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum


class Rung(IntEnum):
    """The escalation ladder, lowest-touch first. Higher rungs touch more."""

    DIAGNOSE = 0
    # Wire up an existing flag or environment variable; nothing is modified.
    SERVE_FLAG = 1
    # An in-tree source patch, applied through the §13.2 journalled patcher.
    SOURCE_PATCH = 2
    # An attempt-scoped runtime in an isolated venv. Not built in v0.1.
    SCOPED_RUNTIME = 3
    # Check the framework's sources out locally. Not built in v0.1.
    SOURCE_LOCALIZE = 4
    # An off-loop compiled build on a dedicated lane. Not built in v0.1.
    COMPILED_BUILD = 5


# Rungs Orbit can act on today. A gap whose lowest useful rung is above this line is
# reported as deferred, not as hopeless — the distinction §24 Tier C exists to keep.
IMPLEMENTED_RUNGS = frozenset({Rung.DIAGNOSE, Rung.SERVE_FLAG, Rung.SOURCE_PATCH})


@dataclass
class CapabilityGap:
    """One classified reason the workload did not run."""

    kind: str  # missing_package | missing_device | missing_op | backend_codegen | oom | config | unknown
    evidence: str  # the matched output line, verbatim — the claim's receipt
    rung: Rung  # lowest rung that could address it
    suggestion: str  # the concrete next move, in the operator's terms

    @property
    def deferred(self) -> bool:
        return self.rung not in IMPLEMENTED_RUNGS

    def format(self) -> str:
        state = "deferred to v0.2" if self.deferred else "actionable now"
        return (
            f"[{self.kind}] rung {self.rung.value} ({self.rung.name.lower()}, {state})\n"
            f"  evidence:   {self.evidence}\n"
            f"  next move:  {self.suggestion}"
        )


# Classification patterns, most specific first. Each maps a failure signature to the
# lowest rung that could address it. These are deterministic on purpose: the residue
# that no pattern matches is reported as `unknown`, never guessed at (§12.5's rule
# about the unknown category applies here too — its value is that everything in it
# deserves a second look).
_CLASSIFIERS: list[tuple[re.Pattern[str], str, Rung, str]] = [
    (
        re.compile(r"No valid triton configs|ZEBIN codegen error|Triton.*codegen", re.I),
        "backend_codegen",
        Rung.SERVE_FLAG,
        "the Triton backend cannot generate code for this device; run eager "
        "(disable graph capture / torch.compile for this workload) rather than "
        "retrying a compile that cannot succeed",
    ),
    (
        re.compile(
            r"ZE_RESULT_ERROR_OUT_OF_(HOST|DEVICE)_MEMORY|CUDA out of memory|"
            r"XPU out of memory|OutOfMemoryError|not enough memory",
            re.I,
        ),
        "oom",
        Rung.SERVE_FLAG,
        "reduce the resident footprint: smaller batch, pinned KV-cache bytes "
        "(§17.5 — on shared-memory devices size to truly-free memory), or a "
        "lower gpu_memory_utilization",
    ),
    (
        re.compile(
            r"xpu is not available|no XPU devices|ZE_RESULT_ERROR_UNINITIALIZED|"
            r"Intel GPU driver|level.?zero.*(not found|missing)",
            re.I,
        ),
        "missing_device",
        Rung.SERVE_FLAG,
        "no usable XPU: check the driver and Level Zero runtime before anything "
        "else — every higher rung assumes a device that initializes",
    ),
    (
        re.compile(
            r"could not run '.*' with arguments from the '\w+' backend|"
            r"NotImplementedError: .*operator|no kernel registered",
            re.I,
        ),
        "missing_op",
        Rung.SOURCE_PATCH,
        "an operator has no implementation on this dispatch key; §13's operator "
        "override can supply one without forking the framework",
    ),
    (
        re.compile(r"ModuleNotFoundError: No module named '([^']+)'|ImportError: cannot import"),
        "missing_package",
        Rung.SCOPED_RUNTIME,
        "a dependency is absent; the attempt-scoped runtime rung would install it "
        "in isolation (v0.2) — until then, install it in the serving environment",
    ),
    (
        re.compile(r"unrecognized arguments|invalid choice|error: argument", re.I),
        "config",
        Rung.SERVE_FLAG,
        "the launch command itself is wrong for this framework version; fix the "
        "flag before concluding anything about the workload",
    ),
]


def diagnose(returncode: int, stdout: str = "", stderr: str = "") -> list[CapabilityGap]:
    """Classify why a launch failed, deterministically (§3).

    Returns every gap the output evidences, most specific first, or a single
    honest `unknown` when nothing matches — an unclassified failure is a real
    finding, and inventing a classification for it would send the operator the
    wrong way with confidence.
    """
    if returncode == 0:
        return []

    text = f"{stdout}\n{stderr}"
    gaps: list[CapabilityGap] = []
    for pattern, kind, rung, suggestion in _CLASSIFIERS:
        match = pattern.search(text)
        if match is None:
            continue
        line = next(
            (ln.strip() for ln in text.splitlines() if match.group(0).splitlines()[0] in ln),
            match.group(0),
        )
        gaps.append(CapabilityGap(kind=kind, evidence=line, rung=rung, suggestion=suggestion))

    if not gaps:
        tail = (stderr or stdout).strip().splitlines()
        gaps.append(
            CapabilityGap(
                kind="unknown",
                evidence=tail[-1] if tail else f"exit code {returncode} with no output",
                rung=Rung.DIAGNOSE,
                suggestion="no known failure signature matched; read the full output "
                "before acting — an unclassified gap is a finding, not a license to guess",
            )
        )
    return gaps


@dataclass
class GateResult:
    """The runnable gate's verdict on a fix (§5.6).

    `kept` is true only when the workload booted *and* the accuracy eval passed.
    Booting alone is not the gate: Hyperloom is explicit that a build does not earn
    KEEP on artifact verification, and the same discipline applies one rung down —
    a serve flag that boots a model which now answers wrongly is not enablement,
    it is a different workload wearing the same name.
    """

    booted: bool
    evaluated: bool
    eval_passed: bool | None
    kept: bool
    reason: str
    gaps: list[CapabilityGap] = field(default_factory=list)


def runnable_gate(
    boot: Callable[[], tuple[int, str, str]],
    quality: Callable[[], bool] | None = None,
) -> GateResult:
    """Decide whether a fix earns KEEP: boot the workload, then re-run the eval.

    `boot` launches the workload with the fix applied and returns
    (returncode, stdout, stderr) — a fresh process, for the §13.5 reason: an
    already-imported module would keep serving the pre-fix code. `quality` re-runs
    the accuracy eval; when it is not supplied, the verdict is `booted,
    unevaluated, not kept`, stated as such — never a KEEP by omission.
    """
    returncode, stdout, stderr = boot()
    if returncode != 0:
        gaps = diagnose(returncode, stdout, stderr)
        return GateResult(
            booted=False,
            evaluated=False,
            eval_passed=None,
            kept=False,
            reason="the workload did not boot with the fix; diagnosis attached",
            gaps=gaps,
        )

    if quality is None:
        return GateResult(
            booted=True,
            evaluated=False,
            eval_passed=None,
            kept=False,
            reason="boots, but no accuracy eval was supplied; a boot alone does not "
            "earn KEEP (§5.6) — wire a quality gate and re-run",
        )

    passed = bool(quality())
    return GateResult(
        booted=True,
        evaluated=True,
        eval_passed=passed,
        kept=passed,
        reason="boots and re-passes the accuracy eval"
        if passed
        else "boots but fails the accuracy eval; the fix changed what the workload computes",
    )
