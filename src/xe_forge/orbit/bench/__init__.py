"""orbit-bench: the standalone measurement backbone (plan §5.4).

Standalone in fact, not just in name (§24 Tier C): the `orbit-bench` console script
(`cli.main`) runs a command with declared warmup and repetitions, emits one structured
JSON document, and compares two such documents with the §17 verdicts — usable on its
own by someone who wants nothing else from Orbit, with no torch and no GPU. Everything
downstream consumes its output rather than re-deriving a measurement.
"""

from xe_forge.orbit.bench.cli import main
from xe_forge.orbit.bench.core import BenchResult, BenchRunner, MetricExtractor

__all__ = ["BenchResult", "BenchRunner", "MetricExtractor", "main"]
