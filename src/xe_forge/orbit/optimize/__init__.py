"""
Optimization, by two routes that suit different kernels (plan §9.9, §13.7).

* `kernel_dir` — the **standalone** route: emit the `Model` + spec contract and hand the
  directory to Xe-Forge's own pipeline, which times it with `ai_bench` and rewrites it
  with DSPy or Claude. Iterates in seconds, and needs the kernel to have been extracted
  first (E1/E2), which is the hard part.
* `loop` / `proposer` / `harness` — the **in-place** route: patch the installed tree,
  check correctness by importing the patched kernel, measure through the framework's own
  benchmark. Needs no extraction and no `ai_bench`, which is why it reaches the E3
  kernels that are the majority.

Neither supersedes the other. The first is cheaper per candidate; the second is the only
one available when extraction cannot produce a standalone bundle.
"""

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
