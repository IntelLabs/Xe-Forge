"""
GPU hardware counters via VTune (plan §5.2, §9.5, §18).

unitrace answers what a kernel *is*: AOT or JIT, its compiled SIMD width, its GRF mode,
whether it spills. VTune answers what the device *did with it* — occupancy, and, more
usefully, **which limit is binding**. Those are different questions and the second is the
one an optimizer keeps guessing at.

Measured on this machine, for the oneDNN GEMM that owns 93% of a Qwen decode:

    Peak XVE Threads Occupancy        40.0%
      limited by Work Size            40.0%    <- binding
      limited by SLM Use             100.0%    (not limiting)
      limited by Barriers            100.0%    (not limiting)

"Occupancy is 40%" invites a guess. "Occupancy is 40% and the limiter is work size, not
SLM and not barriers" names the lever. An agent given the first proposed larger blocks and
more warps and measured 2x slower twice; the second forecloses both.

Three setup facts, because each cost real time and none is discoverable from the error:

* VTune cannot collect GPU hardware metrics without Intel's **Metrics Discovery** library
  (`libigdmd.so`), which is not packaged on most distributions and is built from
  `github.com/intel/metrics-discovery` (~100 s). Without it the collection aborts and
  produces no result directory at all — not a partial one.
* `/usr/lib/libmd.so` exists on many systems and is **not** it. That is BSD's
  message-digest library; the name collision is a coincidence and following it wastes time.
* **`-no-follow-child` is required for anything that imports a serving framework.** By
  default VTune follows child processes, and importing vLLM spawns helpers (`ldconfig`
  among them). Under the usual `ptrace_scope=1` it cannot attach to those, and the whole
  collection fails with *"the scope of ptrace system call application is limited"* — an
  error that points at a kernel setting needing root, when the actual fix is a flag. A
  pure-torch script profiles fine and a vLLM one does not, which makes the cause look
  like the framework rather than the follow-child default.

Absence is reported, never guessed around: no VTune, or no Metrics Discovery, means the
occupancy limiter is unknown, and the caller says so rather than inferring one.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

VTUNE_BIN = "vtune"

# oneAPI installs VTune outside PATH unless `setvars.sh` has been sourced, which a
# non-interactive run has usually not done.
VTUNE_SEARCH_DIRS = (
    "/opt/intel/oneapi/vtune/latest/bin64",
    "/usr/local/bin",
)
VTUNE_ENV = "ORBIT_VTUNE"

# Where a source build of intel/metrics-discovery leaves libigdmd.so.
METRICS_DISCOVERY_ENV = "ORBIT_METRICS_DISCOVERY"
_MD_RELATIVE = "metrics-discovery/dump/linux64/release/metrics_discovery"
METRICS_DISCOVERY_DIRS = (
    str(Path.home() / ".cache" / "orbit-dev" / _MD_RELATIVE),
    str(Path.home() / "src" / _MD_RELATIVE),
    "/opt/" + _MD_RELATIVE,
)
METRICS_LIB = "libigdmd.so"

# Occupancy at or below this is worth reporting as a finding rather than context.
LOW_OCCUPANCY_PERCENT = 50.0


@dataclass
class KernelOccupancy:
    """What the device did with one kernel, and what held it back."""

    name: str
    simd_width: int = 0
    spill_bytes: int = 0
    instances: int = 0
    total_time_s: float = 0.0
    global_size: str = ""
    local_size: str = ""
    occupancy_percent: float | None = None
    work_size_limit: float | None = None
    slm_limit: float | None = None
    barrier_limit: float | None = None

    @property
    def limiter(self) -> str:
        """The binding constraint on occupancy, named.

        This is the whole value of the measurement. Each `*_limit` is the occupancy that
        factor alone would permit, so the smallest is what actually binds; anything at
        100% is not limiting at all. Reporting occupancy without the limiter leaves the
        reader to guess which lever to pull, and the guess has been wrong twice.
        """
        candidates = {
            "work size": self.work_size_limit,
            "SLM use": self.slm_limit,
            "barriers": self.barrier_limit,
        }
        known = {k: v for k, v in candidates.items() if v is not None}
        if not known:
            return "unknown"
        binding = min(known, key=lambda k: known[k])
        if known[binding] >= 100.0:
            return "none of the measured limits"
        return binding

    @property
    def low_occupancy(self) -> bool:
        return self.occupancy_percent is not None and (
            self.occupancy_percent <= LOW_OCCUPANCY_PERCENT
        )

    def describe(self) -> str:
        parts = [f"  {self.name[:56]}"]
        if self.global_size:
            parts.append(f"    work size: global {self.global_size}, local {self.local_size}")
        if self.simd_width:
            parts.append(f"    SIMD{self.simd_width}, spills {self.spill_bytes} bytes")
        if self.occupancy_percent is not None:
            parts.append(
                f"    peak XVE thread occupancy {self.occupancy_percent:.1f}% — "
                f"limited by {self.limiter}"
            )
            for label, value in (
                ("work size", self.work_size_limit),
                ("SLM use", self.slm_limit),
                ("barriers", self.barrier_limit),
            ):
                if value is not None:
                    note = "  (not limiting)" if value >= 100.0 else ""
                    parts.append(f"      {label:<10} allows {value:5.1f}%{note}")
        return "\n".join(parts)


@dataclass
class VTuneResult:
    available: bool = False
    kernels: list[KernelOccupancy] = field(default_factory=list)
    device_name: str = ""
    xve_count: int = 0
    max_threads_per_xve: int = 0
    reason: str = ""

    def describe(self) -> str:
        if not self.available:
            return f"  no VTune GPU counters: {self.reason}"
        lines = []
        if self.device_name:
            detail = f" ({self.xve_count} XVEs" if self.xve_count else ""
            if self.max_threads_per_xve:
                detail += f", {self.max_threads_per_xve} threads each"
            lines.append(f"  device: {self.device_name}{detail + ')' if detail else ''}")
        lines.extend(k.describe() for k in self.kernels)
        return "\n".join(lines) if lines else "  VTune ran but reported no GPU kernels"


def resolve_binary(binary: str = VTUNE_BIN) -> str | None:
    override = os.environ.get(VTUNE_ENV)
    if override:
        candidate = Path(override)
        candidate = candidate if candidate.is_file() else candidate / binary
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(binary)
    if found:
        return found
    for directory in VTUNE_SEARCH_DIRS:
        candidate = Path(directory) / binary
        if candidate.is_file():
            return str(candidate)
    return None


def resolve_metrics_discovery() -> str | None:
    """The directory holding `libigdmd.so`, without which GPU counters do not collect."""
    override = os.environ.get(METRICS_DISCOVERY_ENV)
    directories = [override, *METRICS_DISCOVERY_DIRS] if override else list(METRICS_DISCOVERY_DIRS)
    for directory in directories:
        if not directory:
            continue
        path = Path(directory)
        if any(path.glob(f"{METRICS_LIB}*")):
            return str(path)
    return None


def available() -> bool:
    return resolve_binary() is not None and resolve_metrics_discovery() is not None


def unavailable_reason() -> str:
    """Say which half is missing, because the two need different fixes."""
    if resolve_binary() is None:
        return (
            "vtune not found; install intel-oneapi-toolkit or set "
            f"{VTUNE_ENV} to its bin64 directory"
        )
    if resolve_metrics_discovery() is None:
        return (
            "Intel Metrics Discovery (libigdmd.so) not found, so VTune cannot collect GPU "
            "hardware counters and the collection aborts without producing a result. "
            "Build github.com/intel/metrics-discovery and point "
            f"{METRICS_DISCOVERY_ENV} at its output directory. Note that /usr/lib/libmd.so "
            "is BSD's message-digest library, not this."
        )
    return ""


def parse_computing_tasks(csv_text: str) -> list[KernelOccupancy]:
    """Parse `vtune -report hotspots -group-by=computing-task -format=csv`.

    Columns are matched by header name rather than position: VTune's column set varies
    with the collection type and hardware, and a positional parser would silently read
    occupancy out of the transfer-size column on a machine slightly unlike this one.
    """
    lines = [line for line in csv_text.splitlines() if line.strip()]
    if len(lines) < 2:
        return []

    header = [column.strip() for column in lines[0].split(";")]
    index = {name: position for position, name in enumerate(header)}

    def field_of(row: list[str], name: str) -> str:
        position = index.get(name)
        return row[position].strip() if position is not None and position < len(row) else ""

    def number(row: list[str], name: str) -> float | None:
        raw = field_of(row, name).replace(",", "")
        try:
            return float(raw)
        except ValueError:
            return None

    kernels: list[KernelOccupancy] = []
    for line in lines[1:]:
        row = line.split(";")
        name = field_of(row, "Computing Task")
        if not name:
            continue
        kernels.append(
            KernelOccupancy(
                name=name,
                simd_width=int(number(row, "Computing Task:SIMD Width") or 0),
                spill_bytes=int(number(row, "Computing Task:Spill Memory Size") or 0),
                instances=int(number(row, "Computing Task:Instance Count") or 0),
                total_time_s=number(row, "Computing Task:Total Time") or 0.0,
                global_size=field_of(row, "Work Size:Global"),
                local_size=field_of(row, "Work Size:Local"),
                occupancy_percent=number(row, "Peak XVE Threads Occupancy(%)"),
                work_size_limit=number(row, "Peak XVE Threads Occupancy:Work Size Limit(%)"),
                slm_limit=number(row, "Peak XVE Threads Occupancy:SLM Use Limit(%)"),
                barrier_limit=number(row, "Peak XVE Threads Occupancy:Barriers Use Limit(%)"),
            )
        )
    return kernels


def collect(
    command: list[str],
    result_dir: Path,
    timeout_s: float = 1800.0,
) -> VTuneResult:
    """Run a workload under VTune's GPU hotspots collection and parse the result."""
    binary = resolve_binary()
    metrics = resolve_metrics_discovery()
    if binary is None or metrics is None:
        return VTuneResult(available=False, reason=unavailable_reason())

    result_dir = Path(result_dir)
    if result_dir.exists():
        shutil.rmtree(result_dir, ignore_errors=True)

    env = dict(os.environ)
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{metrics}:{existing}" if existing else metrics

    try:
        subprocess.run(
            # -no-follow-child: see the module docstring. Following children makes any
            # framework import fail against a restricted ptrace scope.
            [
                binary,
                "-collect",
                "gpu-hotspots",
                "-no-follow-child",
                "-r",
                str(result_dir),
                "--",
                *command,
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            check=False,
        )
        report = subprocess.run(
            [
                binary,
                "-report",
                "hotspots",
                "-group-by=computing-task",
                "-r",
                str(result_dir),
                "-format=csv",
                "-csv-delimiter=;",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return VTuneResult(available=False, reason=f"vtune could not be run: {exc}")

    if not result_dir.exists():
        return VTuneResult(
            available=False,
            reason="vtune produced no result directory; the collection aborted",
        )

    kernels = parse_computing_tasks(report.stdout)
    return VTuneResult(
        available=bool(kernels),
        kernels=kernels,
        reason="" if kernels else "vtune reported no GPU computing tasks",
    )
