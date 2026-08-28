"""Extract every kernel in a run and report per-level coverage weighted by GPU time.
Design rationale: docs/DESIGN.md."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from xe_forge.orbit.extract.bundle import ExtractionResult, Extractor
from xe_forge.orbit.extract.verify import verify_bundle
from xe_forge.orbit.models import ExtractionLevel, KernelCatalog, KernelRecord, LaunchRecord

# What each extraction level buys, shown in the coverage table.
LEVEL_MEANING = {
    ExtractionLevel.E1: "standalone bundle, file-local",
    ExtractionLevel.E2: "standalone bundle with closure",
    ExtractionLevel.E3: "in-situ harness (framework drives dispatch)",
    ExtractionLevel.E4: "opaque reproducer, no source",
    ExtractionLevel.E0: "in-place only",
}

# Levels that permit a source-level rewrite at all.
REWRITABLE = {ExtractionLevel.E1, ExtractionLevel.E2, ExtractionLevel.E3}
STANDALONE = {ExtractionLevel.E1, ExtractionLevel.E2}


@dataclass
class KernelExtraction:
    kernel: KernelRecord
    result: ExtractionResult | None = None
    verified: bool = False
    error: str | None = None

    @property
    def level(self) -> ExtractionLevel | None:
        return self.result.level if self.result else None

    @property
    def share(self) -> float:
        return self.kernel.gpu_time_share


@dataclass
class ExtractionCoverage:
    """Per-level counts and GPU-time shares for a whole run."""

    run_id: str
    extractions: list[KernelExtraction] = field(default_factory=list)

    def by_level(self) -> dict[str, list[KernelExtraction]]:
        grouped: dict[str, list[KernelExtraction]] = {}
        for item in self.extractions:
            key = item.level.value if item.level else "none"
            grouped.setdefault(key, []).append(item)
        return grouped

    def share_at(self, levels: set[ExtractionLevel]) -> float:
        return sum(e.share for e in self.extractions if e.level in levels)

    @property
    def standalone_share(self) -> float:
        return self.share_at(STANDALONE)

    @property
    def rewritable_share(self) -> float:
        return self.share_at(REWRITABLE)

    @property
    def unattributed_share(self) -> float:
        return sum(e.share for e in self.extractions if e.level is None)

    @property
    def verified_count(self) -> int:
        return sum(1 for e in self.extractions if e.verified)

    def format(self) -> str:
        lines = [
            f"extraction coverage for {self.run_id}",
            "=" * 78,
            f"{'LEVEL':<6} {'KERNELS':>8} {'GPU%':>8} {'VERIFIED':>9}  MEANING",
            "-" * 78,
        ]
        grouped = self.by_level()
        order = ["E1", "E2", "E3", "E4", "E0", "none"]
        for key in order:
            items = grouped.get(key)
            if not items:
                continue
            share = sum(i.share for i in items) * 100
            verified = sum(1 for i in items if i.verified)
            meaning = (
                LEVEL_MEANING.get(ExtractionLevel(key), "")
                if key != "none"
                else "no provenance — a finding to report, not a target"
            )
            lines.append(f"{key:<6} {len(items):>8} {share:>7.1f}% {verified:>9}  {meaning}")

        lines.append("-" * 78)
        lines.append(
            f"source-rewritable (E1/E2/E3): {self.rewritable_share * 100:.1f}% of GPU time"
        )
        lines.append(
            f"standalone bundle (E1/E2):    {self.standalone_share * 100:.1f}% of GPU time"
        )
        if self.unattributed_share > 0:
            lines.append(
                f"unattributed:                 {self.unattributed_share * 100:.1f}% of GPU time "
                f"— profile further before drawing conclusions"
            )
        lines.append(
            f"verified bundles:             {self.verified_count} of {len(self.extractions)}"
        )
        lines.append("")
        lines.append("Every kernel reaches some level — E3 and E4 are always available — so a bare")
        lines.append("'100% extracted' would be true and useless. The rows above are what differs.")
        return "\n".join(lines)


def extract_all(
    catalog: KernelCatalog,
    output_root: Path,
    *,
    launches: dict[str, LaunchRecord] | None = None,
    level: str = "auto",
    verify: bool = True,
    skip_unattributed: bool = True,
    agent: object | None = None,
) -> ExtractionCoverage:
    """Extract every kernel in the catalog and report what each one achieved.

    `skip_unattributed` (default True) skips kernels with no provenance rather than
    producing a harness for something we cannot name.
    """
    extractor = Extractor(output_root=Path(output_root), agent=agent)
    coverage = ExtractionCoverage(run_id=catalog.run_id)
    launch_map = launches or {}

    for kernel in catalog.kernels:
        if skip_unattributed and kernel.provider.value == "unknown":
            coverage.extractions.append(
                KernelExtraction(kernel=kernel, error="no provenance; not attempted")
            )
            continue

        try:
            result = extractor.extract(kernel, launch=launch_map.get(kernel.id), level=level)
        except Exception as exc:  # one bad kernel must not abort the sweep
            coverage.extractions.append(KernelExtraction(kernel=kernel, error=str(exc)))
            continue

        verified = False
        if verify:
            report = verify_bundle(result.bundle)
            result.bundle.verification = report.to_extraction_check()
            verified = report.passed

        coverage.extractions.append(
            KernelExtraction(kernel=kernel, result=result, verified=verified)
        )

    return coverage
