"""
The published support matrix: what this machine has, next to what the project
supports. Measured, not declared.
Design rationale: docs/DESIGN.md
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Kernel languages Orbit can identify, extract and patch back.
SUPPORTED_LANGUAGES = {
    "triton": "identify, closure (AST), E2 bundle, P1 override via torch.library",
    "sycl": "identify (demangle), closure (compile_commands), P1 override via icpx extension",
    "sycl_tla": "identify, template instantiation capture; tile search via Xe-Forge",
    "cpp": "dispatcher-registered ops, same P1 path as SYCL",
    "opaque": "E4 reproducer only; actions are fusion, backend, layout, library config",
}

# Frameworks reached through the adapter protocol.
SUPPORTED_FRAMEWORKS = {
    "generic_torch": "Tier 0 — any torch workload; wall-clock, discovery, provenance, capture",
    "vllm": "Tier 1 — serving metrics, determinism knobs, config axes, P1/P3 patch points",
}

# Minimum versions the code assumes; below these, behaviour is untested.
MINIMUM_VERSIONS = {
    "python": "3.11",
    "torch": "2.8",
    "vllm": "0.11",
}

# Optional tooling that changes what can be measured; absence is always reported.
OPTIONAL_TOOLING = {
    "unitrace": "Level Zero GPU-busy and launch-gap timing; without it GPU-busy is "
    "estimated from trace span, which cannot see time between launches",
    "icpx": "builds SYCL operator overrides; identity, closure and option axes work without it",
    "level-zero-headers": "required before Triton can JIT for XPU",
    "vtune": "kernel-level profiling enrichment",
}


@dataclass
class MatrixRow:
    component: str
    detected: str
    supported: str
    note: str = ""


@dataclass
class SupportMatrix:
    rows: list[MatrixRow] = field(default_factory=list)

    def add(self, component: str, detected: str, supported: str, note: str = "") -> None:
        self.rows.append(MatrixRow(component, detected, supported, note))

    def format(self) -> str:
        width = max((len(r.component) for r in self.rows), default=12)
        detected_width = max((len(r.detected) for r in self.rows), default=12)
        lines = [
            f"{'COMPONENT'.ljust(width)}  {'DETECTED'.ljust(detected_width)}  SUPPORTED",
            "-" * (width + detected_width + 40),
        ]
        for row in self.rows:
            lines.append(
                f"{row.component.ljust(width)}  {row.detected.ljust(detected_width)}  {row.supported}"
            )
            if row.note:
                lines.append(f"{' ' * (width + detected_width + 4)}{row.note}")
        return "\n".join(lines)


def _onednn_version() -> tuple[str, str]:
    """oneDNN version as torch reports it (via `torch.__config__`), plus how it is linked."""
    try:
        import re

        import torch

        config = torch.__config__.show()
        match = re.search(r"(?:oneDNN|MKL-DNN)\s+v?([0-9][0-9.]*)", config, re.I)
        version = match.group(1) if match else "unknown"
        return version, "statically linked into libtorch_xpu.so; upgrade requires a torch rebuild"
    except Exception:
        return "not detected", ""


def _onemkl_version() -> str:
    try:
        import re

        import torch

        match = re.search(r"Math Kernel Library Version ([0-9][0-9.]*\S*)", torch.__config__.show())
        return match.group(1) if match else "unknown"
    except Exception:
        return "not detected"


def build_matrix() -> SupportMatrix:
    """Measure this machine and lay it beside what the project supports."""
    import platform
    import shutil

    from xe_forge.orbit.runtime import environment

    matrix = SupportMatrix()
    packages = environment.package_versions()
    device_type, device_name, count = environment.detect_device()

    matrix.add("python", platform.python_version(), f">= {MINIMUM_VERSIONS['python']}")
    matrix.add("platform", f"{platform.system()} {platform.machine()}", "Linux x86_64", "")
    matrix.add(
        "device",
        f"{device_name or 'none'} (x{count})" if count else "none",
        "Intel XPU (Arc, Arc Pro, Max, integrated)",
        ""
        if count
        else "no working GPU runtime: install intel-compute-runtime + level-zero-loader",
    )
    matrix.add(
        "driver",
        environment.driver_version(device_type) or "n/a",
        "Level Zero via intel-compute-runtime",
    )
    matrix.add(
        "torch",
        packages.get("torch", "not installed"),
        f">= {MINIMUM_VERSIONS['torch']}, XPU build for GPU work",
    )

    onednn, onednn_note = _onednn_version()
    matrix.add("oneDNN", onednn, "whatever the torch build ships", onednn_note)
    matrix.add("oneMKL", _onemkl_version(), "whatever the torch build ships")

    triton = (
        packages.get("triton")
        or packages.get("triton-xpu")
        or packages.get("pytorch-triton-xpu", "not installed")
    )
    matrix.add(
        "triton",
        triton,
        "XPU build for GPU work",
        "simple kernels JIT; torch.compile autotuning needs a recognized GPU architecture",
    )
    matrix.add(
        "vllm",
        packages.get("vllm", "not installed"),
        f">= {MINIMUM_VERSIONS['vllm']}, XPU build for GPU serving",
    )

    for name, description in OPTIONAL_TOOLING.items():
        binary = {"icpx": "icpx", "unitrace": "unitrace", "vtune": "vtune"}.get(name)
        if binary:
            if name == "icpx":
                from xe_forge.orbit.patch.sycl_override import available_compiler

                detected = "yes" if available_compiler() else "no"
            elif name == "unitrace":
                from xe_forge.orbit.profiling import unitrace as unitrace_mod

                detected = "yes" if unitrace_mod.available() else "no"
            else:
                detected = "yes" if shutil.which(binary) else "no"
        else:
            from pathlib import Path

            detected = "yes" if Path("/usr/include/level_zero/ze_api.h").is_file() else "no"
        matrix.add(f"optional: {name}", detected, "optional", description)

    return matrix


def format_languages() -> str:
    lines = ["KERNEL LANGUAGES", "-" * 60]
    for name, capability in SUPPORTED_LANGUAGES.items():
        lines.append(f"  {name:<10} {capability}")
    lines.append("")
    lines.append("FRAMEWORKS")
    lines.append("-" * 60)
    for name, capability in SUPPORTED_FRAMEWORKS.items():
        lines.append(f"  {name:<16} {capability}")
    return "\n".join(lines)
