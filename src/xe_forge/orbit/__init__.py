"""
Xe-Orbit — workload-level performance optimization inside Xe-Forge.

Orbit is the **control plane**. It decides *what to work on* and *whether it worked*.
It never contains kernel transformation logic: kernel rewrites go to Xe-Forge, region
fusion goes to Xe-Fuse, and everything deterministic (config sweeps, backend swaps,
dependency closure, ranking arithmetic) is executed here without an LLM.

The pipeline, in the order the plan builds it:

    run      -> environment, versions, device, timing with repetitions
    trace    -> torch.profiler ingestion, unitrace, launch-site interception
    kernels  -> catalog with GPU-busy, launch gaps, MDE, Amdahl ceilings, ranking
    capture  -> real input tensors with strides and data dependencies preserved

Everything imports lazily: `import xe_forge.orbit` costs nothing and requires neither
torch nor a GPU, which is what lets the analysis stages run in CPU-only CI from stored
artifacts (§16.3).
"""

from __future__ import annotations

__all__ = [
    "SCHEMA_VERSION",
    "ArtifactError",
    "RunStore",
    "__version__",
]

__version__ = "0.1.0.dev0"


def __getattr__(name: str):
    """Resolve the common entry points lazily, keeping import cost at zero."""
    if name == "SCHEMA_VERSION":
        from xe_forge.orbit.models import SCHEMA_VERSION

        return SCHEMA_VERSION
    if name == "RunStore":
        from xe_forge.orbit.artifacts import RunStore

        return RunStore
    if name == "ArtifactError":
        from xe_forge.orbit.artifacts import ArtifactError

        return ArtifactError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
