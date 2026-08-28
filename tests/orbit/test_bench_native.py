"""
Framework-native measurement (plan §5.4, §10.3).

Xe-Forge times through `ai_bench`, which is right for a standalone extracted kernel and
wrong for the in-place path: there the framework ships its own benchmark, and a
hand-rolled one measures something adjacent to what the workload actually does.
"""

from __future__ import annotations

import pytest

from xe_forge.orbit.bench.native import (
    MIN_REPLICATES,
    FlopResult,
    KernelTiming,
    describe_provenance,
    native_harness_for,
)


class TestHarnessResolution:
    def test_each_framework_resolves_to_its_own_benchmark(self):
        assert native_harness_for("vllm") == ["vllm", "bench", "throughput"]
        assert "sglang.bench_offline_throughput" in " ".join(native_harness_for("sglang"))

    def test_a_declared_harness_wins_over_the_builtin_table(self):
        """Adding a framework's harness must be a YAML change, not an edit here (§10.6)."""
        declared = ["python", "-m", "myframework.bench"]
        assert native_harness_for("vllm", declared) == declared

    def test_an_unknown_framework_reports_a_gap_rather_than_a_default(self):
        """Falling back to a generic timer would measure something else and not say so."""
        assert native_harness_for("mystery") is None

    def test_the_declared_list_is_copied_not_aliased(self):
        declared = ["a", "b"]
        returned = native_harness_for("x", declared)
        returned.append("c")
        assert declared == ["a", "b"]


class TestProvenance:
    def test_a_number_names_the_harness_that_produced_it(self):
        """A throughput figure with no named source cannot be checked."""
        text = describe_provenance("vllm", native_harness_for("vllm"))
        assert "vllm bench throughput" in text

    def test_a_missing_harness_is_stated_plainly(self):
        text = describe_provenance("mystery", None)
        assert "no native benchmark declared" in text
        assert "not measured by the framework's own harness" in text


class TestKernelTiming:
    def test_timing_is_a_distribution_not_a_scalar(self):
        t = KernelTiming(samples_us=[10.0, 12.0, 11.0, 13.0, 11.5], label="k")
        assert t.median_us == 11.5
        assert t.usable

    def test_too_few_replicates_is_not_usable_for_statistics(self):
        """Below this the interval Timer reports is not meaningful."""
        t = KernelTiming(samples_us=[10.0] * (MIN_REPLICATES - 1))
        assert not t.usable

    def test_no_samples_is_reported_rather_than_zero(self):
        assert "no samples" in KernelTiming(label="k").format()
        assert KernelTiming().median_us == 0.0

    def test_the_format_names_the_harness(self):
        t = KernelTiming(samples_us=[1.0] * 5, label="k")
        assert "torch.utils.benchmark.Timer" in t.format()


class TestFlops:
    def test_uncounted_is_distinct_from_zero_flops(self):
        """A roofline plot destroys this distinction if both arrive as 0.0."""
        assert FlopResult(counted=False).tflops_at(0.01) is None
        assert FlopResult(total_flops=0, counted=True).tflops_at(0.01) == 0.0

    def test_tflops_needs_a_positive_duration(self):
        assert FlopResult(total_flops=10**12, counted=True).tflops_at(0.0) is None
        assert FlopResult(total_flops=10**12, counted=True).tflops_at(-1.0) is None

    def test_achieved_rate_is_flops_over_seconds(self):
        result = FlopResult(total_flops=2 * 10**12, counted=True)
        assert result.tflops_at(1.0) == pytest.approx(2.0)


@pytest.mark.xpu
class TestAgainstRealTorch:
    def test_timing_a_real_callable_yields_usable_replicates(self):
        torch = pytest.importorskip("torch")
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            pytest.skip("no XPU device")
        from xe_forge.orbit.bench.native import time_kernel

        a = torch.randn(512, 512, device="xpu")
        timing = time_kernel(lambda: torch.matmul(a, a), label="matmul", min_run_time=0.2)
        assert timing.usable
        assert timing.median_us > 0

    def test_flops_are_counted_through_the_dispatcher(self):
        torch = pytest.importorskip("torch")
        from xe_forge.orbit.bench.native import count_flops

        n = 256
        a = torch.randn(n, n)
        result = count_flops(lambda: torch.matmul(a, a))
        assert result.counted
        # A dense n x n matmul is 2*n^3 MACs-as-FLOPs; the counter should be in range.
        assert result.total_flops == pytest.approx(2 * n**3, rel=0.1)

    def test_a_callable_that_raises_is_uncounted_not_zero(self):
        pytest.importorskip("torch")
        from xe_forge.orbit.bench.native import count_flops

        def boom():
            raise RuntimeError("cannot run under the counter")

        assert not count_flops(boom).counted


class TestTimingProducesADistribution:
    """`blocked_autorange` alone can return a single block, which has no dispersion.

    The first version of `time_kernel` did exactly that: one sample of 220 ms for a
    512x512 matmul — the first call's compile cost, recorded as if it were the kernel.
    """

    def test_enough_independent_samples_for_an_interval(self):
        from xe_forge.orbit.bench.native import MIN_REPLICATES, time_kernel

        timing = time_kernel(lambda: sum(range(100)), label="cheap", min_run_time=0.01)
        assert len(timing.samples_us) >= MIN_REPLICATES
        assert timing.usable

    def test_the_replicate_count_is_caller_controlled(self):
        from xe_forge.orbit.bench.native import time_kernel

        timing = time_kernel(lambda: sum(range(10)), min_run_time=0.01, replicates=7)
        assert len(timing.samples_us) == 7

    def test_warmup_runs_are_discarded_not_recorded(self):
        """A JIT backend's first call compiles; that cost is not the kernel's."""
        from xe_forge.orbit.bench.native import time_kernel

        calls = []
        timing = time_kernel(lambda: calls.append(1), min_run_time=0.001, replicates=2, warmup=4)
        assert len(timing.samples_us) == 2
        assert len(calls) > 4, "warmup should have run in addition to the measured blocks"
