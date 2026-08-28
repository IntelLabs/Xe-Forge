"""
Environment and device identity capture (plan §12.9, §20).

Every bundle and every measurement records the environment that produced it, because
extraction and comparison results are not reproducible without it. A change in torch,
IPEX, Triton, vLLM, the driver, or the device invalidates stored artifacts; silent
reuse across versions is the failure mode that produces unexplainable results months
later.

Torch is imported lazily and never required. On a machine with no torch, or with torch
but no working GPU runtime, this module reports what it found and what it could not
find, rather than failing or pretending.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from xe_forge.orbit.models import EnvironmentInfo

# Packages whose versions change results and therefore invalidate stored artifacts.
TRACKED_PACKAGES = (
    "torch",
    "intel-extension-for-pytorch",
    "triton",
    "triton-xpu",
    "pytorch-triton-xpu",
    "vllm",
    "sglang",
    "ai-bench",
    "numpy",
    "xe-forge",
)

# Environment variables that steer dispatch and codegen, and so must be pinned.
TRACKED_ENV_VARS = (
    "ONEAPI_DEVICE_SELECTOR",
    "SYCL_CACHE_PERSISTENT",
    "SYCL_DEVICE_FILTER",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCHINDUCTOR_FX_GRAPH_CACHE",
    "TRITON_CACHE_DIR",
    "VLLM_ATTENTION_BACKEND",
    "VLLM_USE_V1",
    "IPEX_XPU_ONEDNN_LAYOUT",
    "ZE_AFFINITY_MASK",
    "ZE_FLAT_DEVICE_HIERARCHY",
    "DNNL_VERBOSE",
    "OMP_NUM_THREADS",
)


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str]:
    """Installed versions of the packages whose changes invalidate artifacts."""
    found: dict[str, str] = {}
    for name in names:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return found


def git_state(cwd: Path | None = None) -> tuple[str | None, bool | None]:
    """(commit sha, dirty) for the repository containing `cwd`, or (None, None)."""
    workdir = str(cwd or Path.cwd())
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if commit.returncode != 0:
            return None, None
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        return commit.stdout.strip(), dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def detect_device(preferred: str | None = None) -> tuple[str, str | None, int]:
    """Return (device_type, device_name, count).

    Prefers XPU when a working Intel GPU runtime is present, then CUDA, then CPU.
    A GPU that is physically present but has no usable userspace runtime reports a
    device count of zero here, and we correctly fall back rather than crashing later.
    """
    try:
        import torch
    except ImportError:
        return "cpu", None, 0

    if preferred in ("xpu", "cuda", "cpu"):
        candidates = (preferred,)
    else:
        candidates = ("xpu", "cuda")

    for kind in candidates:
        if kind == "cpu":
            break
        backend = getattr(torch, kind, None)
        if backend is None:
            continue
        try:
            if not backend.is_available():
                continue
            count = backend.device_count()
            if count <= 0:
                continue
            name = backend.get_device_name(0)
            return kind, name, count
        except (RuntimeError, AssertionError, AttributeError):
            continue

    return "cpu", platform.processor() or "cpu", 0


def driver_version(device_type: str) -> str | None:
    """Best-effort driver identification for the active device."""
    if device_type != "xpu":
        return None
    try:
        import torch

        props = torch.xpu.get_device_properties(0)
        return getattr(props, "driver_version", None) or getattr(props, "version", None)
    except Exception:
        return None


def sample_clocks(device_type: str, samples: int = 3) -> list[float]:
    """Sample GPU clock frequency where the platform exposes it.

    Returns an empty list when clocks cannot be read. That is recorded as a
    limitation, not as instability — `stats.clocks_stable` treats the two differently
    on purpose, because a missing reading must not silently invalidate a run.
    """
    if device_type != "xpu":
        return []
    readings: list[float] = []
    try:
        import torch

        for _ in range(samples):
            props = torch.xpu.get_device_properties(0)
            clock = getattr(props, "clock_rate", None) or getattr(props, "max_clock_rate", None)
            if clock:
                readings.append(float(clock))
    except Exception:
        return []
    return readings


def env_pins(names: tuple[str, ...] = TRACKED_ENV_VARS) -> dict[str, str]:
    """The dispatch-steering environment variables that are actually set."""
    import os

    return {name: os.environ[name] for name in names if name in os.environ}


def capture(cwd: Path | None = None, preferred_device: str | None = None) -> EnvironmentInfo:
    """Snapshot everything needed to decide whether two runs are comparable."""
    commit, dirty = git_state(cwd)
    device_type, device_name, count = detect_device(preferred_device)
    clocks = sample_clocks(device_type)

    return EnvironmentInfo(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        packages=package_versions(),
        git_commit=commit,
        git_dirty=dirty,
        device_name=f"{device_type}:{device_name}" if device_name else device_type,
        device_count=count,
        driver_version=driver_version(device_type),
        env_pins=env_pins(),
        frequency_locked=False,
        clock_samples=clocks,
    )


def compatibility_gap(stored: EnvironmentInfo, current: EnvironmentInfo) -> list[str]:
    """Differences that invalidate a stored artifact (§12.9).

    Returned as human-readable reasons so a caller can refuse reuse *and* explain why.
    """
    gaps: list[str] = []
    for pkg, version in stored.packages.items():
        now = current.packages.get(pkg)
        if now is None:
            gaps.append(f"{pkg} {version} is no longer installed")
        elif now != version:
            gaps.append(f"{pkg} changed {version} -> {now}")
    if stored.device_name and stored.device_name != current.device_name:
        gaps.append(f"device changed {stored.device_name} -> {current.device_name}")
    if stored.driver_version and stored.driver_version != current.driver_version:
        gaps.append(f"driver changed {stored.driver_version} -> {current.driver_version}")
    for key, value in stored.env_pins.items():
        now = current.env_pins.get(key)
        if now != value:
            gaps.append(f"env {key} changed {value!r} -> {now!r}")
    return gaps
