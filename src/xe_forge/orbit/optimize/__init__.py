"""Optimization via two routes: `kernel_dir` (standalone bundle through Xe-Forge's
pipeline) and `loop`/`proposer`/`harness` (in-place patching measured through the
framework). Design rationale: docs/DESIGN.md."""

from xe_forge.orbit.optimize.kernel_dir import (
    Budget,
    OptimizeError,
    OptimizeOutcome,
    OptimizeRequest,
    optimize_kernel_dir,
    resolve_candidate,
)
from xe_forge.orbit.optimize.loop import (
    LoopResult,
    OptimizationLoop,
    Proposal,
    TrialRecord,
    TrialVerdict,
)
from xe_forge.orbit.optimize.xe_fuse_executor import (
    XeFuseResult,
    checkout_available,
    run_preset,
    run_region,
)

__all__ = [
    "Budget",
    "LoopResult",
    "OptimizationLoop",
    "OptimizeError",
    "OptimizeOutcome",
    "OptimizeRequest",
    "Proposal",
    "TrialRecord",
    "TrialVerdict",
    "XeFuseResult",
    "checkout_available",
    "optimize_kernel_dir",
    "resolve_candidate",
    "run_preset",
    "run_region",
]
