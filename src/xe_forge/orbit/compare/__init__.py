"""Comparison: the L0-L5 correctness ladder and matrix acceptance (plan §17, §19, §14.3)."""

from xe_forge.orbit.compare.gates import (
    GATE_DESCRIPTION,
    Gate,
    GateLadder,
    GateResult,
    MatrixDecision,
    ProfileOutcome,
    decide_matrix,
    run_ladder,
)

__all__ = [
    "GATE_DESCRIPTION",
    "Gate",
    "GateLadder",
    "GateResult",
    "MatrixDecision",
    "ProfileOutcome",
    "decide_matrix",
    "run_ladder",
]
