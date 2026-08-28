"""
Enablement: making a workload run at all, before optimizing it (plan §5.6 item 1).

Hyperloom's largest structural lead over this codebase is a six-rung escalation
ladder: diagnose the capability gap, then climb — serve-flag wire-up, in-tree source
patch, attempt-scoped runtime in an isolated venv, source localization, off-loop
compiled build — with a **runnable gate**: a fix earns KEEP only when the workload
boots with it and re-passes the accuracy eval, never on artifact verification alone.

This module is the working bottom of that ladder: **diagnosis** (rung 0,
deterministic — classifying a failed launch is trace parsing, and §3 forbids an LLM
where a deterministic answer exists), the rungs Orbit can act on (a serve-flag
suggestion; a source patch through §13.2's journalled patcher; and rung 3, the
attempt-scoped runtime — `create_scoped_runtime` builds an isolated venv named
deterministically from its package set, and `climb_missing_package` installs what
the diagnosis named, re-runs the workload through the runnable gate, and keeps the
environment only on the gate's KEEP), and the runnable gate as an enforced
contract. Rungs 4-5 — source localization and the off-loop build lane — remain
v0.2 (§24 Tier C names enablement the v0.2 headline); they stay in the enum so a
diagnosis can honestly say "the fix for this lives on a rung that is not built
yet", which is a different finding from "there is no fix".

The measured motivation, from §5.6: on Wildcat Lake, `GRAPH_CAPTURE` was unavailable
(`No valid triton configs`, `Internal Triton ZEBIN codegen error`) and the pipeline
correctly reported the dead end — honest, and the end of the road. Diagnosis-then-
climb is the difference between reporting that a lever is unavailable and making it
available.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

from xe_forge.orbit.executor import Executor, LocalExecutor


class Rung(IntEnum):
    """The escalation ladder, lowest-touch first. Higher rungs touch more."""

    DIAGNOSE = 0
    # Wire up an existing flag or environment variable; nothing is modified.
    SERVE_FLAG = 1
    # An in-tree source patch, applied through the §13.2 journalled patcher.
    SOURCE_PATCH = 2
    # An attempt-scoped runtime in an isolated venv (`climb_missing_package`).
    SCOPED_RUNTIME = 3
    # Check the framework's sources out locally. Not built yet.
    SOURCE_LOCALIZE = 4
    # An off-loop compiled build on a dedicated lane. Not built yet.
    COMPILED_BUILD = 5


# Rungs Orbit can act on today. A gap whose lowest useful rung is above this line is
# reported as deferred, not as hopeless — the distinction §24 Tier C exists to keep.
IMPLEMENTED_RUNGS = frozenset(
    {Rung.DIAGNOSE, Rung.SERVE_FLAG, Rung.SOURCE_PATCH, Rung.SCOPED_RUNTIME}
)


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
        "a dependency is absent; climb_missing_package (rung 3) installs it in an "
        "attempt-scoped venv and re-runs the workload through the runnable gate",
    ),
    (
        # A quantization method the installed kernel backend cannot execute, e.g.
        # "Marlin does not support weight_bits = uint8b128. Only types = [uint4...]".
        # Measured live: vLLM-XPU routed GPTQ-Int8 to a 4-bit-only Marlin path. The
        # message names the supported set, so the suggestion sends the operator to a
        # supported variant rather than to a rebuild that will not help.
        re.compile(
            r"does not support weight_bits|quantization.*not supported|"
            r"Only types = \[.*\] are supported",
            re.I,
        ),
        "quant_capability",
        Rung.SERVE_FLAG,
        "the kernel backend cannot execute this quantization format; the error names "
        "the supported set — switch the model to a supported variant (e.g. 4-bit) or "
        "select a different quantization backend, rather than rebuilding",
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


# --- Rung 3: the attempt-scoped runtime -------------------------------------------

# A missing module's *import* name is not always its *distribution* name. This map
# covers the ones this stack actually hits; the identity entries are listed so the
# claim "we know this module" is explicit rather than an accident of passthrough.
# An unknown module passes through as-is, with a note — a guessable pip name beats
# refusing, but the guess must be visible in the result.
KNOWN_DISTRIBUTIONS: dict[str, str] = {
    "vllm": "vllm",
    "torch": "torch",
    "triton": "triton",
    "sglang": "sglang",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
}

# The quoted module in a ModuleNotFoundError line — the same evidence string the
# missing_package classifier matched, re-extracted here so the gap's shape does not
# have to change to carry it.
_MODULE_NAME = re.compile(r"No module named '([^']+)'")

# installer(venv_python, packages) -> (returncode, stdout, stderr)
Installer = Callable[[Path, list[str]], tuple[int, str, str]]


@dataclass
class ScopedRuntime:
    """An attempt-scoped venv: created (or reused) for one enablement climb."""

    venv_path: Path
    python: Path  # the venv's interpreter — what the boot command must run under
    installed: list[str]  # distributions actually installed; empty when a step failed
    created: bool  # False when an existing environment was reused (or creation failed)
    reason: str  # what happened, in the operator's terms — failures are named here


@dataclass
class ClimbResult:
    """Outcome of one rung-3 climb, gated exactly as `runnable_gate` gates.

    `kept` mirrors the gate's verdict: boot AND passed eval, never boot alone.
    A not-kept climb discards the environment — an installed-but-unproven venv
    left behind would be a fix that never faced the gate.
    """

    rung: Rung = Rung.SCOPED_RUNTIME
    runtime: ScopedRuntime | None = None  # None when the climb refused to start
    gate: GateResult | None = None  # None when the workload was never booted
    kept: bool = False
    reason: str = ""


def _tail(text: str, limit: int = 200) -> str:
    """The last non-empty line of `text`, capped — enough to name a failure."""
    lines = text.strip().splitlines()
    return lines[-1][-limit:] if lines else ""


def _uv_pip_install(python: Path, packages: list[str]) -> tuple[int, str, str]:
    """The default installer: `uv pip install` into the venv `python` belongs to."""
    proc = subprocess.run(
        ["uv", "pip", "install", "--python", str(python), *packages],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def create_scoped_runtime(
    base_dir: Path,
    packages: list[str],
    *,
    installer: Installer | None = None,
    python_version: str | None = None,
) -> ScopedRuntime:
    """Build (or reuse) an isolated venv holding exactly `packages`.

    The venv's name is a hash of the sorted package set, so a repeat attempt with
    the same needs reuses the environment instead of re-creating it — and says so.
    Failures are named in `reason`, never raised past the result: a runtime that
    could not be built is a finding for the climb to report, not an exception for
    the caller to unwind.
    """
    installer = installer or _uv_pip_install
    digest = hashlib.sha256("\n".join(sorted(packages)).encode()).hexdigest()[:12]
    venv_path = Path(base_dir) / f"venv-{digest}"
    python = venv_path / "bin" / "python"

    if python.exists():
        created = False
        reason = f"reusing existing scoped runtime {venv_path.name} (same package set)"
    else:
        cmd = ["uv", "venv", str(venv_path)]
        if python_version:
            cmd += ["--python", python_version]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return ScopedRuntime(
                venv_path=venv_path,
                python=python,
                installed=[],
                created=False,
                reason=f"venv creation failed: {_tail(proc.stderr)}",
            )
        created = True
        reason = f"created scoped runtime {venv_path.name}"

    if packages:
        returncode, _stdout, stderr = installer(python, packages)
        if returncode != 0:
            return ScopedRuntime(
                venv_path=venv_path,
                python=python,
                installed=[],
                created=created,
                reason=f"install failed: {', '.join(packages)}: {_tail(stderr)}",
            )
        reason += f"; installed {', '.join(packages)}"

    return ScopedRuntime(
        venv_path=venv_path,
        python=python,
        installed=list(packages),
        created=created,
        reason=reason,
    )


def _with_scoped_python(cmd: list[str], python: Path) -> list[str]:
    """Point `cmd` at the venv's interpreter.

    An argv0 that names a python interpreter (`python`, `python3`, `python3.11`,
    any path to one) is replaced; anything else gets the venv python prepended.
    """
    if cmd and Path(cmd[0]).name.startswith("python"):
        return [str(python), *cmd[1:]]
    return [str(python), *cmd]


def climb_missing_package(
    gaps: list[CapabilityGap],
    boot_cmd: list[str],
    base_dir: Path,
    *,
    quality: Callable[[], bool] | None = None,
    executor: Executor | None = None,
    installer: Installer | None = None,
) -> ClimbResult:
    """The rung-3 climb: install what the diagnosis named, then face the gate.

    Collects the missing modules from `missing_package` gaps, maps them to pip
    distributions (`KNOWN_DISTRIBUTIONS`; unknown names pass through with a note),
    builds the scoped runtime, and re-runs `boot_cmd` under the venv's python
    through `runnable_gate`. The gate's discipline is unchanged: without a
    `quality` callable the result is booted-but-not-kept, stated as such. The
    environment survives only a KEEP; anything less is discarded.
    """
    missing = [g for g in gaps if g.kind == "missing_package"]
    if not missing:
        kinds = ", ".join(sorted({g.kind for g in gaps})) or "none"
        return ClimbResult(
            kept=False,
            reason=f"rung 3 addresses missing packages; these gaps are {kinds} — lower rungs apply",
        )

    modules: list[str] = []
    for gap in missing:
        match = _MODULE_NAME.search(gap.evidence)
        if match is None:
            continue
        top_level = match.group(1).split(".")[0]
        if top_level not in modules:
            modules.append(top_level)
    if not modules:
        return ClimbResult(
            kept=False,
            reason="missing_package gaps carried no extractable module name; "
            "nothing to install — read the evidence lines directly",
        )

    packages: list[str] = []
    notes: list[str] = []
    for module in modules:
        distribution = KNOWN_DISTRIBUTIONS.get(module)
        if distribution is None:
            distribution = module
            notes.append(
                f"module '{module}' is not in the distribution map; "
                f"trying '{module}' as the pip name"
            )
        if distribution not in packages:
            packages.append(distribution)

    runtime = create_scoped_runtime(base_dir, packages, installer=installer)
    if not runtime.installed:
        shutil.rmtree(runtime.venv_path, ignore_errors=True)
        return ClimbResult(
            runtime=runtime,
            kept=False,
            reason=f"scoped runtime unusable ({runtime.reason}); environment discarded",
        )

    run_executor = executor or LocalExecutor()
    scoped_cmd = _with_scoped_python(boot_cmd, runtime.python)

    def boot() -> tuple[int, str, str]:
        result = run_executor.run(scoped_cmd)
        return result.returncode, result.stdout, result.stderr

    gate = runnable_gate(boot, quality)
    if gate.kept:
        reason = f"kept: {gate.reason}; scoped runtime retained at {runtime.venv_path}"
    else:
        shutil.rmtree(runtime.venv_path, ignore_errors=True)
        reason = f"discarded: {gate.reason}"
    if notes:
        reason += "; " + "; ".join(notes)

    return ClimbResult(runtime=runtime, gate=gate, kept=gate.kept, reason=reason)
