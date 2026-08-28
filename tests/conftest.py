"""Path shim for the whole test tree.

Puts `src/` on the path so the suite runs from a checkout without an editable
install — the same arrangement T0 CPU-only CI uses (plan §16.6). Lives at the
tests root so the three core test modules collect too; `tests/orbit/conftest.py`
carries the Orbit-specific fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
