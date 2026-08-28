"""orbit-bench: the standalone measurement backbone. The `orbit-bench` console script
runs a command with declared warmup and repetitions, emits one structured JSON
document, and compares two such documents — no torch, no GPU required."""

from xe_forge.orbit.bench.cli import main
from xe_forge.orbit.bench.core import BenchResult, BenchRunner, MetricExtractor

__all__ = ["BenchResult", "BenchRunner", "MetricExtractor", "main"]
