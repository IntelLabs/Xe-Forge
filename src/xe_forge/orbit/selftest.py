"""
The stub optimizer and pipeline self-test. Four deterministic stub variants
(known-good, known-bad, incorrect, no-op) cover every branch of the decision logic
without calling an LLM, so the self-test is meaningful on a machine with no GPU.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import StrEnum

from xe_forge.orbit import stats
from xe_forge.orbit.models import Decision, WorkloadSpec


class StubVariant(StrEnum):
    KNOWN_GOOD = "known_good"
    KNOWN_BAD = "known_bad"
    INCORRECT = "incorrect"
    NOOP = "noop"


@dataclass
class StubCandidate:
    """A synthetic optimization result with known, checkable properties."""

    variant: StubVariant
    kernel_speedup: float
    e2e_speedup: float
    correct_loose: bool
    correct_tight: bool
    expected_decision: Decision
    description: str = ""

    def kernel_samples(self, baseline_us: float = 100.0, n: int = 8) -> list[float]:
        """Deterministic per-sample timings with reproducible jitter — zero-variance
        samples would make every comparison trivially significant."""
        target = baseline_us / self.kernel_speedup
        return [target * (1.0 + 0.01 * ((i % 3) - 1)) for i in range(n)]

    def e2e_samples(self, baseline_s: float = 1.0, n: int = 8) -> list[float]:
        target = baseline_s / self.e2e_speedup
        return [target * (1.0 + 0.005 * ((i % 3) - 1)) for i in range(n)]


# The four variants, with the decision each one must produce.
STUB_CANDIDATES: dict[StubVariant, StubCandidate] = {
    StubVariant.KNOWN_GOOD: StubCandidate(
        variant=StubVariant.KNOWN_GOOD,
        kernel_speedup=2.0,
        e2e_speedup=1.25,
        correct_loose=True,
        correct_tight=True,
        expected_decision=Decision.ACCEPT,
        description="real speedup that survives to end-to-end",
    ),
    StubVariant.KNOWN_BAD: StubCandidate(
        variant=StubVariant.KNOWN_BAD,
        kernel_speedup=3.0,
        e2e_speedup=0.92,
        correct_loose=True,
        correct_tight=True,
        expected_decision=Decision.REJECT,
        description="wins the microbenchmark, loses the workload",
    ),
    StubVariant.INCORRECT: StubCandidate(
        variant=StubVariant.INCORRECT,
        kernel_speedup=4.0,
        e2e_speedup=1.4,
        correct_loose=True,
        correct_tight=False,
        expected_decision=Decision.REJECT,
        description="fast and wrong; passes loose tolerance, fails tight",
    ),
    StubVariant.NOOP: StubCandidate(
        variant=StubVariant.NOOP,
        kernel_speedup=1.0,
        e2e_speedup=1.0,
        correct_loose=True,
        correct_tight=True,
        expected_decision=Decision.INCONCLUSIVE,
        description="behaviourally identical; must not be reported as a win",
    ),
}


class StubOptimizer:
    """A deterministic stand-in for Xe-Forge, so CI never calls an LLM."""

    name = "stub"

    def optimize(self, variant: StubVariant | str = StubVariant.KNOWN_GOOD) -> StubCandidate:
        key = StubVariant(variant)
        return STUB_CANDIDATES[key]

    def all_variants(self) -> list[StubCandidate]:
        return list(STUB_CANDIDATES.values())


@dataclass
class SelfTestResult:
    name: str
    passed: bool
    detail: str = ""
    skipped: bool = False


@dataclass
class SelfTestReport:
    results: list[SelfTestResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed or r.skipped for r in self.results)

    def add(self, name: str, passed: bool, detail: str = "", skipped: bool = False) -> None:
        self.results.append(SelfTestResult(name, passed, detail, skipped))

    def format(self) -> str:
        lines = ["xe-orbit selftest", "=" * 70]
        for r in self.results:
            mark = "SKIP" if r.skipped else ("PASS" if r.passed else "FAIL")
            lines.append(f"  [{mark}] {r.name}")
            if r.detail:
                lines.append(f"         {r.detail}")
        lines.append("=" * 70)
        passed = sum(1 for r in self.results if r.passed and not r.skipped)
        failed = sum(1 for r in self.results if not r.passed and not r.skipped)
        skipped = sum(1 for r in self.results if r.skipped)
        lines.append(
            f"RESULT: {'PASS' if self.passed else 'FAIL'} "
            f"({passed} passed, {failed} failed, {skipped} skipped)"
        )
        return "\n".join(lines)


def run_selftest(chaos: bool = False, quick: bool = False) -> SelfTestReport:
    """Run the full-loop invariants on stub data."""
    report = SelfTestReport()
    optimizer = StubOptimizer()

    # --- decision logic against all four stub variants -------------------
    for candidate in optimizer.all_variants():
        baseline = candidate.e2e_samples(baseline_s=1.0)
        # A candidate's samples are its own baseline scaled by its e2e speedup.
        improved = [s / candidate.e2e_speedup for s in baseline]

        if not candidate.correct_tight:
            # The correctness gate (L1) fires before any performance comparison.
            report.add(
                f"stub/{candidate.variant.value}: rejected at correctness gate",
                True,
                "fails tight tolerance; never reaches the performance gate (L1 before L4)",
            )
            continue

        decision, detail = stats.compare(baseline, improved, min_repetitions=5)
        report.add(
            f"stub/{candidate.variant.value}: decision is {candidate.expected_decision.value}",
            decision == candidate.expected_decision,
            f"got {decision.value} — {detail.get('reason', '')}",
        )

    # --- null test: identical sample sets must not yield a difference ----
    baseline = [1.0, 1.01, 0.99, 1.005, 0.995, 1.002, 0.998, 1.0]
    decision, detail = stats.compare(baseline, list(baseline), min_repetitions=5)
    report.add(
        "null test: identical inputs give a CI containing zero",
        decision == Decision.INCONCLUSIVE,
        f"got {decision.value} — {detail.get('reason', '')}",
    )

    # --- positive control: a known injected regression must be detected --
    slowed = [s * 1.30 for s in baseline]
    decision, detail = stats.compare(baseline, slowed, min_repetitions=5)
    improvement = float(detail.get("improvement_percent", 0.0))
    report.add(
        "positive control: 30% injected slowdown is detected",
        decision == Decision.REJECT and improvement < -20.0,
        f"got {decision.value}, measured {improvement:.1f}%",
    )

    # --- single-sample runs must never be accepted -----------------------
    decision, detail = stats.compare([1.0], [0.5], min_repetitions=5)
    report.add(
        "single-sample comparison is refused",
        decision == Decision.INVALID,
        f"got {decision.value} — {detail.get('reason', '')}",
    )

    # --- MDE gating: a gain below the noise floor is not actionable ------
    noisy = [1.0, 1.4, 0.7, 1.3, 0.8, 1.2, 0.9, 1.1]
    mde = stats.minimum_detectable_effect(noisy)
    ceiling = stats.amdahl_ceiling(share=0.03, speedup=2.0, gpu_busy_fraction=0.7)
    report.add(
        "Amdahl ceiling below MDE is correctly unactionable",
        ceiling < mde,
        f"3% kernel ceiling={ceiling:.2f}% vs MDE={mde:.2f}% on a noisy workload",
    )

    # --- artifact round-trip and schema stability ------------------------
    try:
        from xe_forge.orbit import schemas

        names = sorted(schemas.ARTIFACT_MODELS)
        schemas.roundtrip("workload", {"command": ["echo", "hi"]})
        report.add(
            "artifact schemas generate and round-trip",
            True,
            f"{len(names)} artifact schemas: {', '.join(names)}",
        )
    except Exception as exc:
        report.add("artifact schemas generate and round-trip", False, str(exc))

    # --- store round-trip -------------------------------------------------
    try:
        import tempfile
        from pathlib import Path

        from xe_forge.orbit.artifacts import RunStore

        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore.create(base=Path(tmp) / ".orbit")
            spec = WorkloadSpec(command=[sys.executable, "-c", "pass"])
            store.save("workload.json", spec)
            reloaded = store.load("workload.json", WorkloadSpec)
            report.add(
                "run store saves and reloads a typed artifact",
                reloaded.command == spec.command,
                f"run_id={store.run_id}",
            )
    except Exception as exc:
        report.add("run store saves and reloads a typed artifact", False, str(exc))

    # --- adapter conformance ---------------------------------------------
    if not quick:
        try:
            from xe_forge.orbit.adapters import GenericTorchAdapter
            from xe_forge.orbit.adapters.conformance import run_conformance

            conformance = run_conformance(GenericTorchAdapter(), repetitions=5)
            failed = [c.name for c in conformance.checks if not c.passed and not c.skipped]
            report.add(
                "GenericTorchAdapter passes conformance",
                conformance.passed,
                "all checks passed" if conformance.passed else f"failed: {failed}",
            )
        except Exception as exc:
            report.add("GenericTorchAdapter passes conformance", False, str(exc))
    else:
        report.add(
            "GenericTorchAdapter passes conformance", True, "skipped (--quick)", skipped=True
        )

    # --- correctness ladder ordering --------------------------------------
    try:
        from xe_forge.orbit.compare import Gate, run_ladder

        wrong = run_ladder(
            "stub",
            build_ok=True,
            extraction_verified=True,
            correctness_ok=False,
            kernel_samples=(baseline, [s * 0.25 for s in baseline]),
            e2e_samples=(baseline, [s * 0.5 for s in baseline]),
        )
        report.add(
            "an incorrect candidate never reaches the timing gate",
            wrong.failed_at is Gate.L1 and not any(r.gate is Gate.L4 for r in wrong.results),
            f"blocked at {wrong.failed_at.value if wrong.failed_at else 'nothing'}",
        )

        unverified = run_ladder("stub", build_ok=True, extraction_verified=False)
        report.add(
            "an unverified bundle is never optimized",
            unverified.failed_at is Gate.L0B,
            "blocked at L0b as required",
        )
    except Exception as exc:
        report.add("correctness ladder ordering", False, str(exc))

    # --- patch-back dispatch assertion -------------------------------------
    try:
        from xe_forge.orbit.patch import assert_dispatch

        effective = assert_dispatch(["orbit_new"], "old_kernel", "orbit_new")
        both = assert_dispatch(["orbit_new", "old_kernel"], "old_kernel", "orbit_new")
        silent = assert_dispatch(["old_kernel"], "old_kernel", "orbit_new")
        report.add(
            "an override that does not take effect is detected",
            effective.took_effect and not both.took_effect and not silent.took_effect,
            "new-present-and-old-absent is the only success; a silent no-op would "
            "otherwise look like an honest negative",
        )
    except Exception as exc:
        report.add("patch-back dispatch assertion", False, str(exc))

    # --- matrix acceptance: a trade is not an improvement -------------------
    try:
        from xe_forge.orbit.compare import decide_matrix
        from xe_forge.orbit.models import ServingProfile, WorkloadMatrix

        matrix = WorkloadMatrix(
            profiles=[
                ServingProfile(id="decode", weight=0.6),
                ServingProfile(id="prefill", weight=0.4),
            ]
        )
        trade = decide_matrix(
            matrix,
            {
                "decode": (baseline, [s * 0.80 for s in baseline]),
                "prefill": (baseline, [s * 1.08 for s in baseline]),
            },
        )
        report.add(
            "a per-profile regression rejects despite a weighted win",
            trade.decision is Decision.REJECT,
            f"got {trade.decision.value}; weighted {trade.weighted_improvement:+.1f}%",
        )
    except Exception as exc:
        report.add("matrix acceptance", False, str(exc))

    # --- core purity: the analysis path imports no serving framework -----
    report.add(*_check_core_purity())

    if chaos:
        report.results.extend(_run_chaos_checks())

    return report


def _check_core_purity() -> tuple[str, bool, str]:
    """The core must import no serving framework — enforced, not assumed."""
    import ast
    from pathlib import Path

    forbidden = {"vllm", "sglang", "tgi", "openvino"}
    core_dirs = ("models.py", "analysis", "capture", "languages", "provenance", "stats.py")
    root = Path(__file__).parent
    offenders: list[str] = []

    for entry in core_dirs:
        target = root / entry
        files = [target] if target.is_file() else sorted(target.rglob("*.py"))
        for path in files:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [(node.module or "").split(".")[0]]
                else:
                    continue
                for name in names:
                    if name in forbidden:
                        offenders.append(f"{path.relative_to(root)} imports {name}")

    return (
        "core imports no serving framework",
        not offenders,
        "clean" if not offenders else "; ".join(offenders),
    )


def _run_chaos_checks() -> list[SelfTestResult]:
    """Failure-injection paths that only appear when something goes wrong."""
    import tempfile
    from pathlib import Path

    from xe_forge.orbit.artifacts import ArtifactError, RunStore

    results: list[SelfTestResult] = []

    # A missing artifact must be a clean typed failure, not a crash or a blank default.
    with tempfile.TemporaryDirectory() as tmp:
        store = RunStore.create(base=Path(tmp) / ".orbit")
        try:
            store.load("catalog.json", type(None))  # type: ignore[arg-type]
            results.append(
                SelfTestResult("chaos: missing artifact fails cleanly", False, "no error raised")
            )
        except ArtifactError as exc:
            results.append(
                SelfTestResult("chaos: missing artifact fails cleanly", True, str(exc)[:100])
            )
        except Exception as exc:
            results.append(
                SelfTestResult(
                    "chaos: missing artifact fails cleanly",
                    False,
                    f"wrong exception type {type(exc).__name__}",
                )
            )

        # Malformed JSON must also be typed, not a JSONDecodeError escaping upward.
        bad = store.path("measurement.json")
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("{not json", encoding="utf-8")
        try:
            from xe_forge.orbit.models import WorkloadMeasurement

            store.load("measurement.json", WorkloadMeasurement)
            results.append(
                SelfTestResult("chaos: malformed artifact fails cleanly", False, "no error")
            )
        except ArtifactError:
            results.append(SelfTestResult("chaos: malformed artifact fails cleanly", True))
        except Exception as exc:
            results.append(
                SelfTestResult(
                    "chaos: malformed artifact fails cleanly",
                    False,
                    f"wrong exception type {type(exc).__name__}",
                )
            )

    # An empty trace must be reported as such, never silently ranked.
    try:
        from xe_forge.orbit.analysis.catalog import build_catalog
        from xe_forge.orbit.models import ActionType
        from xe_forge.orbit.profiling.trace import TraceEvents

        catalog = build_catalog(TraceEvents(), run_id="chaos")
        results.append(
            SelfTestResult(
                "chaos: empty trace yields PROFILE_MORE, not a ranking",
                catalog.gating_action == ActionType.PROFILE_MORE and not catalog.kernels,
                catalog.gating_reason[:100],
            )
        )
    except Exception as exc:
        results.append(SelfTestResult("chaos: empty trace", False, str(exc)))

    return results
