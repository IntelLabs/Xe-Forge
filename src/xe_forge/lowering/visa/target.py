"""Target descriptions for vISA lowering. Xe2 (Battlemage) only, by design."""

from __future__ import annotations

from dataclasses import dataclass

from xe_forge.lowering.context import TargetInfo


@dataclass(frozen=True)
class VisaTarget:
    name: str
    ocloc_device: str
    simd_widths: tuple[int, ...]
    grf_bytes: int
    grf_modes: tuple[int, ...]
    knowledge_dirs: tuple[str, ...]

    def info(self) -> TargetInfo:
        return TargetInfo(
            name=self.name,
            device=self.ocloc_device,
            simd_widths=self.simd_widths,
            grf_bytes=self.grf_bytes,
            grf_modes=self.grf_modes,
        )


# Architectural facts of the generation, not measurements: the ISA's execution
# widths, the register size and the two register-file modes the finalizer offers.
XE2 = VisaTarget(
    name="xe2",
    ocloc_device="bmg",
    simd_widths=(16, 32),
    grf_bytes=64,
    grf_modes=(128, 256),
    knowledge_dirs=("common", "xe2"),
)

TARGETS = {"xe2": XE2}


def get_target(name: str) -> VisaTarget:
    try:
        return TARGETS[name]
    except KeyError:
        raise ValueError(f"unknown vISA lowering target {name!r}; supported: {sorted(TARGETS)}") from None
