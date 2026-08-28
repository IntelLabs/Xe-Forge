"""
The full loop (plan §24, PR 13).

    trace -> kernels -> capture -> extract -> bundle test -> emit -> optimize -> apply -> compare

Each stage reads the previous stage's artifact and writes its own, so the loop is
resumable, replayable, and inspectable at every step. It is also *stoppable*: most of
the value of this orchestration is in refusing to continue.

The stop conditions are the design, not error handling:

* Gating says the workload is host-bound or has no headroom -> stop. Optimizing a
  kernel in a host-bound workload is the single easiest way to waste a week (§18).
* The bundle cannot be verified -> stop. An unverified bundle is never optimized,
  because a real speedup on a specialization the workload never runs is worse than no
  speedup at all (§12.10).
* No patch point exists -> stop, and say what action *would* apply instead.

Every stop is reported with its reason and the arithmetic behind it. `NO_ACTION` and
`INCONCLUSIVE` are first-class results here: a framework that cannot credibly say
"there is no headroom" is not a measurement instrument (§7.6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xe_forge.orbit.artifacts import EVENTS, KERNEL_CATALOG, MEASUREMENT, RunStore
from xe_forge.orbit.models import (
    ActionType,
    Decision,
    ExtractionLevel,
    KernelCatalog,
    WorkloadMeasurement,
)


@dataclass
class StageOutcome:
    name: str
    ok: bool
    detail: str = ""
    stopped: bool = False


@dataclass
class PipelineResult:
    run_id: str
    stages: list[StageOutcome] = field(default_factory=list)
    kernel_id: str | None = None
    decision: Decision | None = None
    stop_reason: str = ""

    @property
    def completed(self) -> bool:
        return bool(self.stages) and not any(s.stopped for s in self.stages)

    def record(self, name: str, ok: bool, detail: str = "", stopped: bool = False) -> None:
        self.stages.append(StageOutcome(name, ok, detail, stopped))

    def format(self) -> str:
        lines = [f"pipeline: {self.run_id}", "=" * 72]
        for stage in self.stages:
            mark = "STOP" if stage.stopped else ("OK" if stage.ok else "FAIL")
            lines.append(f"  [{mark:>4}] {stage.name}")
            if stage.detail:
                lines.append(f"         {stage.detail}")
        lines.append("=" * 72)
        if self.stop_reason:
            lines.append(f"STOPPED: {self.stop_reason}")
        elif self.decision is not None:
            lines.append(f"DECISION: {self.decision.value}")
        else:
            lines.append("COMPLETED" if self.completed else "INCOMPLETE")
        return "\n".join(lines)


def run_pipeline(
    store: RunStore,
    *,
    kernel_id: str | None = None,
    extraction_level: str = "auto",
    emit_candidates: bool = True,
    stop_before_optimize: bool = True,
) -> PipelineResult:
    """Drive the loop over an existing run's artifacts.

    `stop_before_optimize` defaults to True because invoking an optimizer costs tokens
    and GPU time: the pipeline prepares a verified candidate and hands it over rather
    than spending a budget on the caller's behalf.
    """
    from xe_forge.orbit.analysis.catalog import build_catalog
    from xe_forge.orbit.emit import emit_candidate
    from xe_forge.orbit.extract import Extractor, verify_bundle
    from xe_forge.orbit.profiling.trace import TraceEvents

    result = PipelineResult(run_id=store.run_id)

    # --- trace ------------------------------------------------------------
    if not store.exists(EVENTS):
        result.record("trace", False, "no normalized trace in this run", stopped=True)
        result.stop_reason = "run has no trace; start with `xe-orbit trace`"
        return result
    events = store.load(EVENTS, TraceEvents)
    result.record("trace", True, f"{len(events.kernels)} kernel events")

    # --- kernels: catalog and gating -------------------------------------
    mde = 0.0
    if store.exists(MEASUREMENT):
        mde = store.load(MEASUREMENT, WorkloadMeasurement).minimum_detectable_effect

    catalog = (
        store.load(KERNEL_CATALOG, KernelCatalog)
        if store.exists(KERNEL_CATALOG)
        else build_catalog(events, run_id=store.run_id, minimum_detectable_effect=mde)
    )
    store.save(KERNEL_CATALOG, catalog)
    result.record(
        "kernels",
        True,
        f"{len(catalog.kernels)} kernels, GPU busy {catalog.gpu_busy_percent:.1f}%",
    )

    # The gate is the point of running it first.
    if catalog.gating_action in (
        ActionType.NO_ACTION,
        ActionType.HOST_OPTIMIZATION,
        ActionType.PROFILE_MORE,
    ):
        result.record("gate", True, catalog.gating_reason, stopped=True)
        result.stop_reason = f"{catalog.gating_action.value}: {catalog.gating_reason}"
        return result
    result.record("gate", True, catalog.gating_reason)

    # --- select a target --------------------------------------------------
    candidates = [k for k in catalog.kernels if not k.skip_reason]
    if kernel_id:
        candidates = [k for k in catalog.kernels if k.id == kernel_id]
    if not candidates:
        skipped = len(catalog.considered_but_not_attempted)
        result.record(
            "select",
            False,
            f"no actionable kernel ({skipped} considered and skipped)",
            stopped=True,
        )
        result.stop_reason = "every kernel was skipped; see the considered-but-not-attempted list"
        return result

    target = candidates[0]
    result.kernel_id = target.id
    result.record(
        "select",
        True,
        f"{target.id} ({target.runtime_name}), {target.gpu_time_share * 100:.1f}% GPU, "
        f"Amdahl ceiling {target.max_e2e_gain:.2f}%",
    )

    # --- extract ----------------------------------------------------------
    extractor = Extractor(output_root=store.subdir("bundles"))
    extraction = extractor.extract(target, level=extraction_level)
    store.save(f"bundles/{target.id}/manifest.json", extraction.bundle)
    detail = f"level {extraction.level.value}"
    if extraction.downgraded:
        detail += f" (downgraded from {extraction.downgraded_from.value}: {extraction.reasons[0]})"
    result.record("extract", True, detail)

    # --- bundle test ------------------------------------------------------
    report = verify_bundle(extraction.bundle)
    extraction.bundle.verification = report.to_extraction_check()
    store.save(f"bundles/{target.id}/manifest.json", extraction.bundle)

    if not report.passed:
        failures = "; ".join(
            f"{c.name}: {c.detail}" for c in report.checks if not c.passed and not c.skipped
        )
        result.record("bundle test", False, failures, stopped=True)
        result.stop_reason = (
            f"bundle for {target.id} could not be verified. An unverified bundle is "
            f"never optimized: a speedup on the wrong specialization is worse than none."
        )
        return result

    # "Passed" and "proven" are not the same claim. An E4 bundle passes because every
    # substantive check was skipped — there is no source to import or mutate — and
    # reporting that as proof would be exactly the kind of confident overstatement the
    # verification step exists to prevent.
    # Only these three establish that the bundle *is* the kernel that ran. A passing
    # reproducer-present check says a text file exists, which is not the same claim.
    identity_checks = {"isolated import", "launch-record match", "mutation check"}
    proven = [c for c in report.checks if c.name in identity_checks and not c.skipped]
    skipped = [c for c in report.checks if c.skipped]

    if proven:
        detail = (
            f"identity established by {len(proven)} check(s): {', '.join(c.name for c in proven)}"
        )
        if skipped:
            detail += f" ({len(skipped)} not applicable at {extraction.level.value})"
    else:
        detail = (
            f"no identity check applies at {extraction.level.value}; the bundle carries a "
            f"reproducer but nothing was proven about which kernel it corresponds to"
        )
    result.record("bundle test", True, detail)

    # An E4 bundle is verified but has no source to optimize; the actions available are
    # fusion, backend and config changes, which are a different pipeline.
    if extraction.level is ExtractionLevel.E4:
        # A dead end for source optimization is not a dead end for the workload. If
        # this kernel participates in a fusable region, that is the actionable next
        # step, and naming it here is the difference between a stop and a handoff.
        suggestion = ""
        try:
            from xe_forge.orbit.analysis.regions import detect_regions

            regions = detect_regions(
                events, catalog.kernels, total_gpu_time_us=catalog.total_gpu_time_us
            )
            containing = [r for r in regions if target.id in r.kernel_ids]
            if containing:
                region = containing[0]
                suggestion = (
                    f" This kernel is part of region {region.id} "
                    f"({region.fusion_pattern or 'unclassified'}, "
                    f"{region.gpu_time_share * 100:.1f}% of GPU time) — "
                    f"run `xe-orbit regions` and route it to Xe-Fuse."
                )
        except Exception:
            # Region detection is an enhancement to the message, never a failure path.
            suggestion = ""

        if target.provider.value == "runtime":
            # A transfer is not an opaque kernel, and saying so would send the reader
            # looking for a library to reconfigure. The action space is host-side.
            detail = (
                "runtime memory operation, not a kernel: there is no source at any "
                "level. Applicable actions are pinned memory, fewer or larger "
                "transfers, and overlapping the copy with compute."
            )
            stop_reason = "target is a host/device transfer; route to a host-side action"
        else:
            detail = (
                "opaque provider: no source to optimize. Applicable actions are region "
                "fusion, backend change, layout change and library config."
            )
            stop_reason = "kernel is an opaque library primitive; route to a non-source action"

        result.record("emit", True, detail + suggestion, stopped=True)
        result.stop_reason = stop_reason
        return result

    # --- emit -------------------------------------------------------------
    if emit_candidates:
        target_dir = store.subdir("candidates", target.id)
        summary = emit_candidate(target, extraction.bundle, target_dir)
        result.record(
            "emit",
            True,
            f"{summary['variants']} variant(s), {summary['coverage'] * 100:.1f}% shape coverage",
        )

    if stop_before_optimize:
        result.record(
            "optimize",
            True,
            "candidate ready; not invoking an optimizer (costs tokens and GPU time). "
            f"Run: xe-orbit emit {target.id} then hand the candidate to Xe-Forge.",
            stopped=True,
        )
        result.stop_reason = "prepared a verified candidate; optimization is the caller's call"
        return result

    return result
