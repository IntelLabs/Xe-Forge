"""
Statistics and the accept/reject arithmetic (plan §17).

These tests exist because the decision rule is the part of Orbit most likely to
produce a confident wrong answer. Each one pins a rule the plan states explicitly.
"""

from __future__ import annotations

import pytest

from xe_forge.orbit import stats
from xe_forge.orbit.models import Decision


class TestEstimate:
    def test_interval_brackets_the_mean(self, stable_samples):
        est = stats.estimate(stable_samples, unit="s")
        assert est.n == len(stable_samples)
        assert est.ci95_low < est.mean < est.ci95_high

    def test_single_sample_yields_a_degenerate_interval(self):
        """One sample is not a measurement; the interval must not pretend otherwise."""
        est = stats.estimate([1.0])
        assert est.n == 1
        assert est.stdev == 0.0
        assert est.ci95_low == est.ci95_high == 1.0

    def test_noisier_samples_give_a_wider_interval(self, stable_samples, noisy_samples):
        tight = stats.estimate(stable_samples)
        loose = stats.estimate(noisy_samples)
        assert (loose.ci95_high - loose.ci95_low) > (tight.ci95_high - tight.ci95_low)

    def test_empty_sample_set_is_an_error(self):
        with pytest.raises(ValueError):
            stats.estimate([])


class TestDecisionRule:
    """CI excludes zero positive -> ACCEPT; negative -> REJECT; straddles -> INCONCLUSIVE."""

    def test_real_improvement_is_accepted(self, stable_samples):
        candidate = [s * 0.80 for s in stable_samples]
        decision, detail = stats.compare(stable_samples, candidate)
        assert decision is Decision.ACCEPT
        assert detail["improvement_percent"] == pytest.approx(20.0, abs=0.5)

    def test_real_regression_is_rejected(self, stable_samples):
        candidate = [s * 1.20 for s in stable_samples]
        decision, _ = stats.compare(stable_samples, candidate)
        assert decision is Decision.REJECT

    def test_identical_inputs_are_inconclusive_not_rejected(self, stable_samples):
        """The null test. INCONCLUSIVE is a real outcome, not a soft REJECT."""
        decision, _ = stats.compare(stable_samples, list(stable_samples))
        assert decision is Decision.INCONCLUSIVE

    def test_effect_buried_in_noise_is_inconclusive(self, noisy_samples):
        candidate = [s * 0.98 for s in noisy_samples]
        decision, detail = stats.compare(noisy_samples, candidate, paired=False)
        assert decision is Decision.INCONCLUSIVE
        assert "straddles zero" in str(detail["reason"])

    def test_too_few_repetitions_is_invalid(self):
        """No accept/reject decision from fewer than five measured runs (§17.1)."""
        decision, detail = stats.compare([1.0, 1.0], [0.5, 0.5])
        assert decision is Decision.INVALID
        assert "repetitions" in str(detail["reason"])

    def test_unstable_clocks_invalidate_the_run(self, stable_samples):
        candidate = [s * 0.5 for s in stable_samples]
        decision, detail = stats.compare(
            stable_samples, candidate, clock_samples=[1000.0, 1400.0, 700.0, 1500.0]
        )
        assert decision is Decision.INVALID
        assert "clock" in str(detail["reason"])

    def test_throughput_direction_is_honoured(self, stable_samples):
        """For a higher-is-better metric, a larger candidate is an improvement."""
        candidate = [s * 1.20 for s in stable_samples]
        decision, detail = stats.compare(stable_samples, candidate, lower_is_better=False)
        assert decision is Decision.ACCEPT
        assert detail["improvement_percent"] > 0

    def test_paired_is_used_for_interleaved_runs(self, stable_samples):
        candidate = [s * 0.9 for s in stable_samples]
        _, detail = stats.compare(stable_samples, candidate)
        assert detail["method"] == "paired"

    def test_unequal_lengths_fall_back_to_welch(self, stable_samples):
        candidate = [s * 0.9 for s in stable_samples[:6]]
        _, detail = stats.compare(stable_samples, candidate)
        assert detail["method"] == "welch"


class TestMinimumDetectableEffect:
    def test_noisy_workload_has_a_higher_floor(self, stable_samples, noisy_samples):
        assert stats.minimum_detectable_effect(noisy_samples) > stats.minimum_detectable_effect(
            stable_samples
        )

    def test_more_repetitions_lower_the_floor(self, noisy_samples):
        few = stats.minimum_detectable_effect(noisy_samples, n_planned=5)
        many = stats.minimum_detectable_effect(noisy_samples, n_planned=50)
        assert many < few

    def test_single_sample_cannot_resolve_anything(self):
        assert stats.minimum_detectable_effect([1.0]) == float("inf")


class TestAmdahlCeiling:
    def test_ceiling_scales_with_share(self):
        small = stats.amdahl_ceiling(share=0.03, speedup=2.0, gpu_busy_fraction=1.0)
        large = stats.amdahl_ceiling(share=0.40, speedup=2.0, gpu_busy_fraction=1.0)
        assert large > small
        # A kernel owning 40% of GPU time, made twice as fast, caps at 20% end to end.
        assert large == pytest.approx(20.0)

    def test_host_bound_workload_caps_the_gain(self):
        """The same kernel win is worth less when the GPU is idle half the time."""
        busy = stats.amdahl_ceiling(share=0.4, speedup=2.0, gpu_busy_fraction=1.0)
        idle = stats.amdahl_ceiling(share=0.4, speedup=2.0, gpu_busy_fraction=0.5)
        assert idle == pytest.approx(busy / 2)

    def test_no_speedup_is_no_gain(self):
        assert stats.amdahl_ceiling(share=0.9, speedup=1.0, gpu_busy_fraction=1.0) == 0.0

    def test_small_kernel_cannot_clear_a_noisy_floor(self, noisy_samples):
        """The §18 gate: ceiling below MDE means NO_ACTION regardless of microbench."""
        ceiling = stats.amdahl_ceiling(share=0.03, speedup=2.0, gpu_busy_fraction=0.7)
        assert ceiling < stats.minimum_detectable_effect(noisy_samples)


class TestClockStability:
    def test_missing_readings_are_not_instability(self):
        """Unreadable clocks are a limitation, not an anomaly — they must not INVALID."""
        assert stats.clocks_stable([]) is True
        assert stats.clocks_stable([1200.0]) is True

    def test_steady_clocks_are_stable(self):
        assert stats.clocks_stable([1200.0, 1201.0, 1199.0]) is True

    def test_swinging_clocks_are_not(self):
        assert stats.clocks_stable([1200.0, 1800.0, 900.0]) is False
