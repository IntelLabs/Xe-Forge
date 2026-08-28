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
contract. Rung 5 — the off-loop compiled build — is `BuildLane` at the bottom of
this file (§24 Tier E, E1): a single-slot, journalled, resumable queue whose
SUCCEEDED is never KEPT until the runnable gate passes. Rung 4 — source
localization — remains deferred; it stays in the enum so a diagnosis can honestly
say "the fix for this lives on a rung that is not built yet", which is a
different finding from "there is no fix".

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
from xe_forge.orbit.policy import _pid_alive


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
    # An off-loop compiled build on the single-slot `BuildLane` (§24 Tier E, E1).
    COMPILED_BUILD = 5


# Rungs Orbit can act on today. A gap whose lowest useful rung is above this line is
# reported as deferred, not as hopeless — the distinction §24 Tier C exists to keep.
IMPLEMENTED_RUNGS = frozenset(
    {Rung.DIAGNOSE, Rung.SERVE_FLAG, Rung.SOURCE_PATCH, Rung.SCOPED_RUNTIME, Rung.COMPILED_BUILD}
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
        # A precompiled kernel library that lacks the instantiation this model's
        # configuration needs, e.g. "Paged decode kernel not compiled for this
        # configuration" — measured live: phi-2's head_dim=80 is outside the head
        # sizes the XPU paged-attention build instantiates. Distinct from
        # missing_op (no implementation at all) and from quant_capability (format
        # unsupported by design): the kernel exists, this shape was not compiled.
        re.compile(
            r"not compiled for this configuration|"
            r"unsupported head[_ ]?(dim|size)|head[_ ]?(dim|size).*not supported",
            re.I,
        ),
        "kernel_capability",
        Rung.SERVE_FLAG,
        "the kernel library was built without this model's configuration; try "
        "another backend for the op first (e.g. VLLM_ATTENTION_BACKEND) — if none "
        "covers it, the real fix is rung 5: rebuild the kernel library with this "
        "configuration compiled in (off-loop build lane, deferred)",
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


# ---- rung 5: the off-loop build lane (plan §24 Tier E, item E1) ----------------
#
# A long compile must never block the tick loop, and a build must never earn KEEP
# on artifact verification alone — the runnable gate above stays the only door.
# The lane is deliberately dumb: a file-based queue, one slot, every state change
# on disk so a crashed builder is recovered honestly (a RUNNING job whose pid is
# dead becomes FAILED with a note, never silently re-queued as if it had not run).
# The two live cases this was built against: sgl-kernel-xpu needs a serial
# MAX_JOBS=1 build (parallel icpx exhausts shared memory), and vllm-xpu-kernels
# lacks phi-2's head_dim=80 in its compiled paged-decode set (`kernel_capability`).

_JOB_STATES = ("QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "KEPT", "DISCARDED")


@dataclass
class BuildJob:
    """One build in the lane; every field lives in the job file on disk."""

    id: str
    component: str
    command: list[str]
    cwd: str
    env: dict[str, str] = field(default_factory=dict)
    status: str = "QUEUED"
    submitted: str = ""
    started: str = ""
    finished: str = ""
    returncode: int | None = None
    log: str = ""
    note: str = ""
    pid: int | None = None

    def format(self) -> str:
        line = f"{self.id}  {self.status:<10} {self.component}"
        if self.note:
            line += f"\n{'':14}note: {self.note}"
        return line


class BuildLane:
    """Single-slot, journalled, resumable build queue (rung 5).

    `submit` is novelty-aware in the §20.4 sense: an identical job (same
    component, command and cwd) that is already QUEUED or RUNNING is returned
    rather than duplicated, and one that already FAILED is re-admitted only with
    the prior failure named in its note — the lane never silently loops on an
    identical failing build. `run_next` executes the oldest QUEUED job, streaming
    output to a log file; SUCCEEDED/FAILED comes from the exit code, and a FAILED
    job carries `diagnose()`'s classification. KEPT/DISCARDED belong to the
    caller, who alone can run the runnable gate — the lane records the verdict
    via `mark` and refuses to invent it.
    """

    def __init__(self, lane_dir: Path | None = None) -> None:
        self.lane_dir = Path(lane_dir) if lane_dir else Path.home() / ".cache/orbit-dev/build-lane"
        self.jobs_dir = self.lane_dir / "jobs"
        self.logs_dir = self.lane_dir / "logs"

    # -- persistence -----------------------------------------------------

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _save(self, job: BuildJob) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        import json as _json

        self._job_path(job.id).write_text(_json.dumps(job.__dict__, indent=2), encoding="utf-8")

    def jobs(self) -> list[BuildJob]:
        import json as _json

        out = []
        for path in sorted(self.jobs_dir.glob("*.json")):
            try:
                out.append(BuildJob(**_json.loads(path.read_text(encoding="utf-8"))))
            except (ValueError, TypeError):
                continue
        return sorted(out, key=lambda j: j.submitted)

    # -- queue -----------------------------------------------------------

    def submit(self, component: str, command: list[str], cwd: Path, env: dict[str, str] | None = None) -> BuildJob:
        key = hashlib.sha256("\0".join([component, *command, str(cwd)]).encode()).hexdigest()[:12]
        existing = {j.id: j for j in self.jobs()}
        if key in existing:
            prior = existing[key]
            if prior.status in ("QUEUED", "RUNNING"):
                return prior  # already in flight; a duplicate would race the slot
            if prior.status == "FAILED":
                prior.status = "QUEUED"
                prior.note = f"re-admitted after failure at {prior.finished}: {prior.note}"[:500]
                prior.returncode = None
                self._save(prior)
                return prior
        job = BuildJob(
            id=key,
            component=component,
            command=list(command),
            cwd=str(cwd),
            env=dict(env or {}),
            submitted=_lane_now(),
        )
        self._save(job)
        return job

    def recover(self) -> list[BuildJob]:
        """RUNNING jobs whose builder is dead become FAILED, named — never re-queued silently."""
        recovered = []
        for job in self.jobs():
            if job.status == "RUNNING" and (job.pid is None or not _pid_alive(job.pid)):
                job.status = "FAILED"
                job.note = f"builder pid {job.pid} died before finishing; log ends where it stopped"
                job.finished = _lane_now()
                self._save(job)
                recovered.append(job)
        return recovered

    def run_next(self, timeout: float = 14400.0) -> BuildJob | None:
        """Run the oldest QUEUED job in the single slot; None if queue empty or slot held."""
        import os as _os

        self.recover()
        queued = [j for j in self.jobs() if j.status == "QUEUED"]
        if not queued:
            return None
        slot = self.lane_dir / "slot.lock"
        self.lane_dir.mkdir(parents=True, exist_ok=True)
        try:
            fd = _os.open(str(slot), _os.O_CREAT | _os.O_EXCL | _os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                holder = int(slot.read_text().strip() or 0)
            except (OSError, ValueError):
                holder = 0
            if holder and _pid_alive(holder):
                return None  # one slot, and it is taken by a live builder
            slot.unlink(missing_ok=True)
            return self.run_next(timeout=timeout)
        with _os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(str(_os.getpid()))

        job = queued[0]
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.logs_dir / f"{job.id}.log"
            job.status, job.started, job.pid, job.log = "RUNNING", _lane_now(), _os.getpid(), str(log_path)
            self._save(job)
            env = dict(_os.environ)
            env.update(job.env)
            with open(log_path, "w", encoding="utf-8") as log:
                try:
                    proc = subprocess.run(
                        job.command, cwd=job.cwd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout
                    )
                    job.returncode = proc.returncode
                except subprocess.TimeoutExpired:
                    job.returncode = -1
                    job.note = f"timed out after {timeout:.0f}s"
            job.finished = _lane_now()
            if job.returncode == 0:
                job.status = "SUCCEEDED"
                job.note = job.note or "built; not KEPT until the runnable gate passes (mark via gate caller)"
            else:
                job.status = "FAILED"
                tail = ""
                try:
                    tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                except OSError:
                    pass
                gaps = diagnose(job.returncode or 1, "", tail)
                if gaps and not job.note:
                    job.note = f"[{gaps[0].kind}] {gaps[0].suggestion[:200]}"
            self._save(job)
            return job
        finally:
            slot.unlink(missing_ok=True)

    def mark(self, job_id: str, kept: bool, reason: str) -> BuildJob:
        """Record the runnable gate's verdict on a SUCCEEDED build."""
        jobs = {j.id: j for j in self.jobs()}
        if job_id not in jobs:
            raise ValueError(f"no job {job_id!r} in the lane")
        job = jobs[job_id]
        if job.status != "SUCCEEDED":
            raise ValueError(
                f"job {job_id} is {job.status}, not SUCCEEDED; only a finished build "
                f"can face the runnable gate"
            )
        job.status = "KEPT" if kept else "DISCARDED"
        job.note = reason
        self._save(job)
        return job

    def format(self) -> str:
        jobs = self.jobs()
        if not jobs:
            return "build lane: empty"
        return "\n".join(job.format() for job in jobs)


def _lane_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
