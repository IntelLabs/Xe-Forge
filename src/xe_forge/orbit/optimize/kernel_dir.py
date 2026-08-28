"""
The directory-level optimizer entry point (plan §8, §9.9).

§8 hands Orbit's candidates to Xe-Forge through the file contract Xe-Forge already
consumes. What did *not* exist was a directory-level entry point: the real API is
`XeForgePipeline.optimize()` / `optimize_file()` plus `create_engine()`, all of which
take a kernel file and an explicit spec path. This module is the thin wrapper §9.9
asks for — it resolves the candidate directory's conventional layout and calls into
the existing pipeline.

Three details of the actual repository shape this, and each one would be a silent
failure if assumed away:

* **`kernel_pytorch.py` is resolved by name substitution**, so the reference file's
  name is load-bearing rather than conventional.
* **`target_speedup` exists in config but nothing reads it.** Passing an accept
  threshold therefore means threading it to the caller's own decision, not setting a
  config value and trusting the pipeline to honour it.
* **`ClaudeEngine` is fire-and-forget.** It generates a workspace, optionally spawns
  `claude -p`, and returns success immediately with no measured speedup. Any caller
  that treats its result as a completed optimization gets an empty answer that looks
  like a successful one, so `engine="claude"` requires an explicit synchronous path.

Xe-Forge itself is imported lazily: Orbit's analysis stages must stay importable
without torch, ai_bench or DSPy present (§16.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# The candidate directory layout from §8.
KERNEL_FILE = "kernel.py"
REFERENCE_FILE = "kernel_pytorch.py"
SPEC_FILE = "spec.yaml"


class OptimizeError(RuntimeError):
    """Raised when a candidate cannot be handed to Xe-Forge."""


@dataclass
class Budget:
    """What a caller is willing to spend on one candidate.

    Xe-Forge expresses this today as `TrialConfig.max_trials`; the wall-clock and
    agent-call limits are carried here so the cost accounting §25 asks for has
    somewhere to live, even before the optimizer enforces them.
    """

    trials: int = 10
    wall_clock_s: float | None = None
    agent_calls: int | None = None


@dataclass
class OptimizeRequest:
    path: Path
    engine: str = "dspy"
    device: str = "xpu"
    objective: str = "weighted_latency"
    budget: Budget = field(default_factory=Budget)
    required_speedup: float | None = None
    variant: str | None = None
    dsl: str = "triton"


@dataclass
class OptimizeOutcome:
    success: bool
    candidate_dir: Path
    engine: str
    optimized_path: Path | None = None
    speedup: float | None = None
    detail: str = ""
    synchronous: bool = True
    notes: list[str] = field(default_factory=list)


def resolve_candidate(path: Path) -> dict[str, Path]:
    """Locate the pieces of a candidate directory, failing clearly on a gap."""
    directory = Path(path)
    if not directory.is_dir():
        raise OptimizeError(f"not a candidate directory: {directory}")

    kernel = directory / KERNEL_FILE
    if not kernel.is_file():
        raise OptimizeError(
            f"no {KERNEL_FILE} in {directory}; run `xe-orbit emit <kernel-id>` first"
        )

    spec = directory / SPEC_FILE
    if not spec.is_file():
        raise OptimizeError(f"no {SPEC_FILE} in {directory}; the spec must be passed explicitly")

    reference = directory / REFERENCE_FILE
    resolved = {"kernel": kernel, "spec": spec}
    if reference.is_file():
        resolved["reference"] = reference
    return resolved


def _reference_is_a_stub(reference: Path) -> bool:
    """A generated stub raises rather than computing; treating it as real is a trap."""
    try:
        return "NotImplementedError" in reference.read_text(encoding="utf-8")
    except OSError:
        return False


def optimize_kernel_dir(
    path: Path | str,
    engine: str = "dspy",
    device: str = "xpu",
    objective: str = "weighted_latency",
    budget: Budget | None = None,
    required_speedup: float | None = None,
    variant: str | None = None,
    dsl: str = "triton",
    dry_run: bool = False,
) -> OptimizeOutcome:
    """Optimize a candidate directory through Xe-Forge's existing pipeline.

    `dry_run` resolves and validates the candidate without invoking an engine, which is
    what lets this be exercised in CPU-only CI where neither DSPy nor ai_bench is
    installed.
    """
    request = OptimizeRequest(
        path=Path(path),
        engine=engine,
        device=device,
        objective=objective,
        budget=budget or Budget(),
        required_speedup=required_speedup,
        variant=variant,
        dsl=dsl,
    )
    resolved = resolve_candidate(request.path)
    notes: list[str] = []

    reference = resolved.get("reference")
    if reference is None:
        notes.append(
            f"no {REFERENCE_FILE}: Xe-Forge resolves the reference by name substitution, "
            f"so correctness cannot be checked without it"
        )
    elif _reference_is_a_stub(reference):
        notes.append(
            f"{REFERENCE_FILE} is still the generated stub; the correctness gate will "
            f"raise rather than compare. Supply the eager-mode equivalent before trusting "
            f"any result."
        )

    # 'weighted_latency' maps onto Xe-Forge's weighted objective (§9.1): the pipeline
    # scores the winner across the whole variant family, applies the per-variant
    # no-regression constraint, and enforces `required_speedup` itself.

    if dry_run:
        return OptimizeOutcome(
            success=True,
            candidate_dir=request.path,
            engine=request.engine,
            detail="dry run: candidate resolved, no engine invoked",
            notes=notes,
        )

    if request.engine == "claude":
        return _optimize_with_claude(request, resolved, notes)
    return _optimize_with_pipeline(request, resolved, notes)


def _optimize_with_pipeline(
    request: OptimizeRequest, resolved: dict[str, Path], notes: list[str]
) -> OptimizeOutcome:
    """Drive `XeForgePipeline.optimize_file`, the real file-based entry point."""
    try:
        from xe_forge.config import get_config
        from xe_forge.pipeline import XeForgePipeline
    except ImportError as exc:
        raise OptimizeError(
            f"Xe-Forge's pipeline is not importable ({exc}). It needs ai_bench, DSPy and "
            f"torch; Orbit's analysis stages deliberately do not."
        ) from exc

    config = get_config()
    pipeline = XeForgePipeline(config)
    variant = request.variant or ("bench-xpu" if request.dsl == "sycl" else "bench-gpu")
    weighted = request.objective == "weighted_latency"

    result = pipeline.optimize_file(
        str(resolved["kernel"]),
        spec_path=str(resolved["spec"]),
        variant_type=variant,
        objective="weighted" if weighted else "single",
        required_speedup=request.required_speedup if weighted else None,
    )

    speedup = getattr(result, "total_speedup", None)
    success = bool(getattr(result, "success", False))

    if weighted:
        # §9.1 landed: the pipeline enforced the threshold and the per-variant
        # no-regression constraint itself; its reason is the authoritative one.
        detail = getattr(result, "error_message", "") or ""
        if not success and detail:
            notes.append(f"weighted objective: {detail}")
    elif request.required_speedup is not None and speedup is not None:
        # Single-variant path: the threshold is applied here rather than inside the
        # pipeline, because `target_speedup` in config is inert (§9.2).
        if speedup < request.required_speedup:
            notes.append(
                f"speedup {speedup:.2f}x is below the required {request.required_speedup:.2f}x "
                f"derived from the Amdahl ceiling; not worth accepting"
            )
            success = False

    return OptimizeOutcome(
        success=success,
        candidate_dir=request.path,
        engine=request.engine,
        speedup=speedup,
        detail=getattr(result, "error_message", "") or "",
        notes=notes,
    )


def _optimize_with_claude(
    request: OptimizeRequest, resolved: dict[str, Path], notes: list[str]
) -> OptimizeOutcome:
    """Handle the fire-and-forget engine honestly rather than reporting a false success.

    `ClaudeEngine.optimize` returns `success=True` immediately with no measured
    speedup: it generates a workspace and optionally spawns `claude -p` without
    waiting. Reporting that as a completed optimization would put an unmeasured
    candidate into the decision path, which §19 forbids ("never report a candidate as
    successful on the basis of generated reasoning alone").
    """
    try:
        from xe_forge.config import get_config
        from xe_forge.engines import create_engine
    except ImportError as exc:
        raise OptimizeError(f"Xe-Forge's engine layer is not importable: {exc}") from exc

    config = get_config()
    config.engine.engine = "claude"
    engine = create_engine(config)

    kernel_code = resolved["kernel"].read_text(encoding="utf-8")
    reference_code = (
        resolved["reference"].read_text(encoding="utf-8") if "reference" in resolved else None
    )

    engine.optimize(
        kernel_code=kernel_code,
        reference_code=reference_code,
        kernel_name=request.path.name,
        spec_path=str(resolved["spec"]),
    )

    notes.append(
        "ClaudeEngine is fire-and-forget: it generated a workspace and returned without "
        "measuring anything. This outcome is reported as NOT synchronous, and the "
        "speedup is unknown rather than assumed. Run the trial results through "
        "`xe-orbit compare` before treating this candidate as an improvement."
    )

    return OptimizeOutcome(
        # Deliberately not `result.success`: the engine reports True unconditionally.
        success=False,
        candidate_dir=request.path,
        engine="claude",
        speedup=None,
        synchronous=False,
        detail="workspace generated; result not measured in-process",
        notes=notes,
    )
