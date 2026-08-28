"""Shared fixtures for the Orbit test suite.

Puts `src/` on the path so the suite runs from a checkout without an editable
install, which is what T0 CPU-only CI does (plan §16.6).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def decode_trace_path() -> Path:
    """The committed golden trace: a GPU-bound decode step, every provenance class."""
    return FIXTURES / "decode_trace.json"


@pytest.fixture
def store(tmp_path):
    """A fresh run store rooted in a temporary directory."""
    from xe_forge.orbit.artifacts import RunStore

    return RunStore.create(base=tmp_path / ".orbit")


@pytest.fixture
def stable_samples() -> list[float]:
    """Low-variance samples: a real difference should be resolvable against these."""
    return [1.000, 1.004, 0.997, 1.002, 0.999, 1.001, 0.998, 1.003]


@pytest.fixture
def noisy_samples() -> list[float]:
    """High-variance samples: only a large effect should clear this noise floor."""
    return [1.0, 1.42, 0.68, 1.31, 0.75, 1.28, 0.81, 1.19]
