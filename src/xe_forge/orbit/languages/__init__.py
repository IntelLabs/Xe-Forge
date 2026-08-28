"""Language backends: one per kernel language, same protocol (plan §11.3).

Registry resolution picks the backend with the highest identification confidence, so a
mangled SYCL symbol goes to the SYCL backend and an Inductor-generated name goes to
Triton, without either being the default.
"""

from xe_forge.orbit.languages.base import (
    BaseLanguageBackend,
    BuildResult,
    CompilerAxis,
    CostProfile,
    LanguageBackend,
)
from xe_forge.orbit.languages.sycl_backend import SyclBackend
from xe_forge.orbit.languages.triton_backend import TritonBackend

_BACKENDS: dict[str, type] = {
    "triton": TritonBackend,
    "sycl": SyclBackend,
}


def available_backends() -> dict[str, type]:
    return dict(_BACKENDS)


def get_backend(name: str, **kwargs) -> BaseLanguageBackend:
    if name not in _BACKENDS:
        raise KeyError(f"unknown language backend {name!r}; available: {sorted(_BACKENDS)}")
    return _BACKENDS[name](**kwargs)


def resolve_backend(kernel_name: str) -> tuple[BaseLanguageBackend | None, float]:
    """Pick the backend most confident about this kernel, or (None, 0.0)."""
    best: BaseLanguageBackend | None = None
    best_score = 0.0
    for cls in _BACKENDS.values():
        backend = cls()
        score = backend.identify(kernel_name)
        if score > best_score:
            best, best_score = backend, score
    return best, best_score


__all__ = [
    "BaseLanguageBackend",
    "BuildResult",
    "CompilerAxis",
    "CostProfile",
    "LanguageBackend",
    "SyclBackend",
    "TritonBackend",
    "available_backends",
    "get_backend",
    "resolve_backend",
]
