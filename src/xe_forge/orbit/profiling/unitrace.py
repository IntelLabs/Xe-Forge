"""
Level Zero timing via unitrace: launch-gap and GPU-busy data for host-bound gating,
plus the compiled-kernel properties table. unitrace is an external Intel tool that is
frequently absent; every function here degrades explicitly and never guesses a number.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from xe_forge.orbit.executor import Executor, LocalExecutor

UNITRACE_BIN = "unitrace"

# unitrace is not packaged by any distro this project targets; it is built from
# Intel's pti-gpu tree and lives wherever the builder put the checkout, almost never
# on PATH — hence the source-build search directories below.
UNITRACE_SEARCH_ENV = "ORBIT_UNITRACE"
_PTI_RELATIVE = "pti-gpu/tools/unitrace/build"
UNITRACE_SEARCH_DIRS = (
    f"/opt/{_PTI_RELATIVE}",
    "/usr/local/bin",
    str(Path.home() / ".cache" / "orbit-dev" / _PTI_RELATIVE),
    str(Path.home() / "src" / _PTI_RELATIVE),
    str(Path.home() / _PTI_RELATIVE),
)

# The tool library unitrace LD_PRELOADs. Without its directory on LD_LIBRARY_PATH the
# launcher starts, prints "cannot be preloaded", and then profiles nothing — a failure
# that looks like a workload with no device activity.
UNITRACE_TOOL_LIB = "libunitrace_tool.so"

# Real unitrace output puts the unit in the label and the value after the colon,
# e.g. "Total Device Time for L0 backend (ns):    71719869".
_TOTAL_TIME_RE = re.compile(r"Total\s+Execution\s+Time\s*\((ns|us|ms|s)\)\s*:\s*([\d.]+)", re.I)
_DEVICE_TIME_RE = re.compile(
    r"Total\s+Device\s+Time(?:\s+for\s+\S+\s+backend)?\s*\((ns|us|ms|s)\)\s*:\s*([\d.]+)",
    re.I,
)
_UNIT_TO_US = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}

# The "Kernel Properties" table carries the compiled-kernel metadata needed to verify
# an extracted bundle is the kernel that ran: AOT vs JIT, sub-group (SIMD) width,
# GRF mode, and spill memory.
_KERNEL_PROPS_RE = re.compile(
    r'^\s*"(?P<name>[^"]+)",\s*(?P<compiled>AOT|JIT),\s*(?P<simd>\d+),'
    r"\s*(?P<args>\d+),\s*(?P<slm>\d+),\s*(?P<private>\d+),"
    r"\s*(?P<spill>\d+),\s*(?P<grf>\d+)",
    re.M,
)


class KernelProperties(BaseModel):
    """Per-kernel compiled metadata, as Level Zero actually reports it."""

    name: str
    compiled: str = ""  # AOT or JIT — these do not perform or rebuild alike
    simd: int = 0  # sub-group width
    arguments: int = 0
    slm_per_group: int = 0
    private_per_thread: int = 0
    spill_per_thread: int = 0  # non-zero spills are a first-order concern on Xe
    grf_per_thread: int = 0  # register file size (the GRF mode axis)


class UnitraceResult(BaseModel):
    """What Level Zero tracing could establish about a run."""

    available: bool = False
    gpu_busy_us: float | None = None
    total_time_us: float | None = None
    launch_gap_total_us: float | None = None
    output_dir: str | None = None
    raw_output: str = ""
    reason: str = ""
    warnings: list[str] = Field(default_factory=list)

    @property
    def gpu_busy_percent(self) -> float | None:
        if not self.gpu_busy_us or not self.total_time_us or self.total_time_us <= 0:
            return None
        return min(100.0, self.gpu_busy_us / self.total_time_us * 100.0)


def available(binary: str = UNITRACE_BIN) -> bool:
    """True when the unitrace binary can be located."""
    return resolve_binary(binary) is not None


def resolve_binary(binary: str = UNITRACE_BIN) -> str | None:
    """Find unitrace on PATH, or in a known source-build location."""
    import os

    override = os.environ.get(UNITRACE_SEARCH_ENV)
    if override:
        candidate = Path(override)
        candidate = candidate if candidate.is_file() else candidate / binary
        if candidate.is_file():
            return str(candidate)

    found = shutil.which(binary)
    if found:
        return found
    for directory in UNITRACE_SEARCH_DIRS:
        candidate = Path(directory) / binary
        if candidate.is_file():
            return str(candidate)
    return None


def tracing_env(binary_path: str, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for running a workload under unitrace.

    The tool library's directory must be on `LD_LIBRARY_PATH`, or the LD_PRELOAD
    silently fails and unitrace reports a run with no device activity. oneAPI's
    compiler libraries must NOT be prepended: sourcing `setvars.sh` shadows torch's
    bundled runtime and silently forces a torch-xpu workload onto the CPU.
    """
    env = dict(base_env or {})
    tool_dir = str(Path(binary_path).parent)
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{tool_dir}:{existing}" if existing else tool_dir
    return env


def unavailable_result(binary: str = UNITRACE_BIN) -> UnitraceResult:
    return UnitraceResult(
        available=False,
        reason=(
            f"{binary} not found on PATH; GPU-busy and launch-gap data unavailable. "
            "Host-bound gating will fall back to trace-derived estimates, which are "
            "weaker: they cannot see time lost between kernel launches."
        ),
    )


def parse_summary(text: str) -> tuple[float | None, float | None]:
    """Extract (device busy microseconds, total microseconds) from unitrace output."""
    device_us: float | None = None
    total_us: float | None = None

    match = _DEVICE_TIME_RE.search(text)
    if match:
        device_us = float(match.group(2)) * _UNIT_TO_US[match.group(1).lower()]

    match = _TOTAL_TIME_RE.search(text)
    if match:
        total_us = float(match.group(2)) * _UNIT_TO_US[match.group(1).lower()]

    return device_us, total_us


def parse_kernel_properties(text: str) -> list[KernelProperties]:
    """Parse the Kernel Properties table into the metadata extraction verifies against.

    A bundle that AOT-builds a kernel the workload JIT'd, or that lands on a different
    GRF mode or sub-group width, is not the same kernel however well it benchmarks.
    """
    found: list[KernelProperties] = []
    for m in _KERNEL_PROPS_RE.finditer(text):
        found.append(
            KernelProperties(
                name=m.group("name"),
                compiled=m.group("compiled").upper(),
                simd=int(m.group("simd")),
                arguments=int(m.group("args")),
                slm_per_group=int(m.group("slm")),
                private_per_thread=int(m.group("private")),
                spill_per_thread=int(m.group("spill")),
                grf_per_thread=int(m.group("grf")),
            )
        )
    return found


def run(
    command: list[str],
    executor: Executor | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    output_dir: Path | None = None,
    timeout: float = 1800.0,
    binary: str = UNITRACE_BIN,
) -> UnitraceResult:
    """Run a command under unitrace and summarize the Level Zero timing.

    Returns an unavailable result rather than raising when unitrace is missing, so a
    trace stage stays usable on a machine without Intel's tooling installed.
    """
    resolved = resolve_binary(binary)
    if resolved is None:
        return unavailable_result(binary)

    exe = executor or LocalExecutor()
    # The tool library must be reachable, and oneAPI must not shadow torch's runtime.
    env = tracing_env(resolved, env)
    argv = [resolved, "--chrome-device-logging", "--device-timing"]
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        argv += ["--output", str(Path(output_dir) / "unitrace")]
    argv += ["--", *command]

    result = exe.run(argv, env=env, cwd=cwd, timeout=timeout)
    combined = f"{result.stdout}\n{result.stderr}"

    if not result.ok:
        return UnitraceResult(
            available=True,
            raw_output=combined[-4000:],
            output_dir=str(output_dir) if output_dir else None,
            reason=f"unitrace exited {result.returncode}",
        )

    device_us, total_us = parse_summary(combined)
    gaps: float | None = None
    if device_us is not None and total_us is not None:
        gaps = max(0.0, total_us - device_us)

    out = UnitraceResult(
        available=True,
        gpu_busy_us=device_us,
        total_time_us=total_us,
        launch_gap_total_us=gaps,
        output_dir=str(output_dir) if output_dir else None,
        raw_output=combined[-4000:],
    )
    if device_us is None:
        out.warnings.append("could not parse device timing from unitrace output")
    return out
