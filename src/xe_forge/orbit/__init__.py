"""
Xe-Orbit — workload-level performance optimization inside Xe-Forge.

Everything imports lazily so `import xe_forge.orbit` needs neither torch nor a GPU.
Design rationale: docs/DESIGN.md
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
