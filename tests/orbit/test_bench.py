"""
The measurement backbone (plan §5.4, §17).

The ordering test here is a regression guard, not a style preference: strict ABAB
alternation produced a real, reproducible bias in the null test, because the baseline
always ran first within each pair and absorbed every first-position effect.
"""

from __future__ import annotations

import pytest

from xe_forge.orbit.bench.core import BenchRunner
from xe_forge.orbit.executor import RunResult
from xe_forge.orbit.models import WorkloadSpec


class RecordingExecutor:
    """Executor stand-in that records call order and returns scripted durations."""

    def __init__(self, durations: dict[str, list[float]] | None = None) -> None:
        self.calls: list[str] = []
        self.durations = durations or {}
        self._counts: dict[str, int] = {}

    def run(self, cmd, env=None, cwd=None, timeout=1800.0) -> RunResult:
        label = cmd[0]
        self.calls.append(label)
        index = self._counts.get(label, 0)
        self._counts[label] = index + 1
        series = self.durations.get(label, [1.0])
        return RunResult(command=list(cmd), returncode=0, duration_s=series[index % len(series)])


def _spec(label: str, warmup: int = 0, repetitions: int = 5) -> WorkloadSpec:
    return WorkloadSpec(command=[label], warmup_iterations=warmup, repetitions=repetitions)


class TestInterleaving:
    def test_order_is_counterbalanced_not_strict_alternation(self):
        """ABBA, so no arm is permanently in the first-of-pair position.

        Under ABAB the baseline runs first in every pair. Any systematic cost of going
        first — scheduler placement, cache warmth — then lands entirely on one arm and
        paired statistics report it as a real difference. This is exactly what made the
        null test flaky before.
        """
        executor = RecordingExecutor()
        runner = BenchRunner(executor=executor)
        runner.interleaved(_spec("base"), _spec("cand"), repetitions=4)

        assert executor.calls == ["base", "cand", "cand", "base", "base", "cand", "cand", "base"]

    def test_each_arm_leads_equally_often(self):
        executor = RecordingExecutor()
        BenchRunner(executor=executor).interleaved(_spec("base"), _spec("cand"), repetitions=6)

        leaders = [executor.calls[i * 2] for i in range(6)]
        assert leaders.count("base") == leaders.count("cand") == 3

    def test_both_arms_are_sampled_equally(self):
        executor = RecordingExecutor()
        base, cand = BenchRunner(executor=executor).interleaved(
            _spec("base"), _spec("cand"), repetitions=5
        )
        assert len(base) == len(cand) == 5

    def test_warmup_runs_precede_measurement_and_are_excluded(self):
        executor = RecordingExecutor()
        base, cand = BenchRunner(executor=executor).interleaved(
            _spec("base", warmup=2), _spec("cand", warmup=2), repetitions=3
        )
        # 2 warmups per arm, then 3 measured pairs.
        assert len(executor.calls) == 4 + 6
        assert len(base) == len(cand) == 3

    def test_a_constant_first_position_penalty_cancels_out(self):
        """The property the ordering exists to provide.

        Every first-of-pair run is scripted to cost 0.10 and every second 0.09,
        independent of which arm it is. Under ABAB this fabricates a 10% difference;
        under ABBA the two arms see the same mix and the means converge.
        """
        durations = {"base": [0.10, 0.09, 0.10, 0.09], "cand": [0.09, 0.10, 0.09, 0.10]}
        executor = RecordingExecutor(durations)
        base, cand = BenchRunner(executor=executor).interleaved(
            _spec("base"), _spec("cand"), repetitions=4
        )
        assert sum(base) / len(base) == pytest.approx(sum(cand) / len(cand))


class TestMeasure:
    def test_measurement_carries_an_interval(self):
        executor = RecordingExecutor({"w": [0.10, 0.11, 0.09, 0.10, 0.10]})
        measurement = BenchRunner(executor=executor).measure(_spec("w"), repetitions=5)
        assert measurement.wall_time.n == 5
        assert measurement.wall_time.ci95_low < measurement.wall_time.mean
        assert measurement.minimum_detectable_effect > 0

    def test_a_workload_that_never_succeeds_raises(self):
        class Failing(RecordingExecutor):
            def run(self, cmd, env=None, cwd=None, timeout=1800.0) -> RunResult:
                return RunResult(command=list(cmd), returncode=1, stderr="boom")

        with pytest.raises(RuntimeError, match="no usable samples"):
            BenchRunner(executor=Failing()).measure(_spec("w"), repetitions=3)

    def test_a_broken_metric_parser_does_not_kill_the_run(self):
        """An adapter's parser is not allowed to take down the measurement."""

        def exploding(_result):
            raise ValueError("bad parse")

        runner = BenchRunner(executor=RecordingExecutor(), metric_extractor=exploding)
        measurement = runner.measure(_spec("w"), repetitions=5)
        assert measurement.wall_time.n == 5
