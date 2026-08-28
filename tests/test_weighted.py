"""The weighted multi-variant objective (plan §9.1).

`core/weighted.py` is deliberately duck-typed — any executor with `compare_kernels`,
any spec with the per-variant getters — so these tests drive it with stubs and no
ai_bench, torch device, or LLM. The §9.1 property under test is the hard constraint:
a candidate that wins the weighted total and regresses one family member is REJECTED,
and the rejection names the variant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from xe_forge.core.weighted import (
    WeightedComparison,
    compare_weighted,
    family_base,
)


@dataclass
class FakeVariant:
    weight: float | None = None


class FakeSpec:
    """Duck-types the KernelSpec surface compare_weighted touches."""

    def __init__(self, entries):
        # entries: list of (key, weight)
        self._entries = [(key, 0, FakeVariant(weight)) for key, weight in entries]

    def weighted_family(self, base):
        return list(self._entries)

    # The getters only feed the executor stub, so constants suffice.
    def get_input_shapes(self, key, index=0):
        return [(16, 16)]

    def get_flop(self, key, index=0):
        return 512.0

    def get_dtype(self, key, index=0):
        return None

    def get_init_args(self, key, index=0):
        return []

    def get_input_dtypes(self, key, index=0):
        return []


@dataclass
class FakeComparison:
    original_time_us: float
    optimized_time_us: float
    speedup: float
    optimized_correct: bool = True
    feedback_message: str = ""


class FakeExecutor:
    """Returns a scripted result per variant key, in call order."""

    def __init__(self, by_variant):
        self.by_variant = dict(by_variant)
        self.calls: list[tuple] = []

    def compare_kernels(self, original, optimized, kernel_name=None, input_shapes=None, **kw):
        key = next(iter(self.by_variant))  # consumed in order
        self.calls.append((key, input_shapes))
        return self.by_variant.pop(key)


def _spec3():
    return FakeSpec([("bench-gpu", 0.6), ("bench-gpu-1", 0.3), ("bench-gpu-2", 0.1)])


class TestFamilyBase:
    def test_numbered_variant_maps_to_family(self):
        assert family_base("bench-gpu-3") == "bench-gpu"
        assert family_base("bench-gpu") == "bench-gpu"
        assert family_base("ci") == "ci"


class TestAcceptance:
    def test_all_variants_faster_is_accepted_with_weighted_headline(self):
        executor = FakeExecutor(
            {
                "bench-gpu": FakeComparison(200.0, 100.0, 2.0),
                "bench-gpu-1": FakeComparison(100.0, 80.0, 1.25),
                "bench-gpu-2": FakeComparison(50.0, 45.0, 1.11),
            }
        )
        result = compare_weighted(executor, _spec3(), "orig", "opt")
        assert result.accepted
        # Weighted by time: (0.6*200 + 0.3*100 + 0.1*50) / (0.6*100 + 0.3*80 + 0.1*45)
        assert result.weighted_speedup == pytest.approx(155.0 / 88.5)

    def test_required_speedup_gates_acceptance(self):
        executor = FakeExecutor(
            {
                "bench-gpu": FakeComparison(110.0, 100.0, 1.1),
                "bench-gpu-1": FakeComparison(110.0, 100.0, 1.1),
                "bench-gpu-2": FakeComparison(110.0, 100.0, 1.1),
            }
        )
        result = compare_weighted(executor, _spec3(), "orig", "opt", required_speedup=1.5)
        assert not result.accepted
        assert "below the required 1.50x" in result.reason


class TestTheHardConstraint:
    """§9.1: 'a hard no-regression constraint on every variant'."""

    def test_tail_regression_rejects_a_weighted_win(self):
        # The dominant shape doubles; the 10% tail collapses. The weighted total is a
        # clear win — and it must still be rejected, naming the regressing variant.
        executor = FakeExecutor(
            {
                "bench-gpu": FakeComparison(200.0, 100.0, 2.0),
                "bench-gpu-1": FakeComparison(100.0, 90.0, 1.11),
                "bench-gpu-2": FakeComparison(50.0, 100.0, 0.5),
            }
        )
        result = compare_weighted(executor, _spec3(), "orig", "opt")
        assert result.weighted_speedup is not None and result.weighted_speedup > 1.0
        assert not result.accepted
        assert result.regressions == ["bench-gpu-2"]
        assert "bench-gpu-2" in result.reason

    def test_regression_tolerance_forgives_noise_sized_losses(self):
        executor = FakeExecutor(
            {
                "bench-gpu": FakeComparison(200.0, 100.0, 2.0),
                "bench-gpu-1": FakeComparison(100.0, 100.3, 0.997),
                "bench-gpu-2": FakeComparison(50.0, 45.0, 1.11),
            }
        )
        result = compare_weighted(executor, _spec3(), "orig", "opt")
        assert result.accepted  # 0.3% is inside the 1% default tolerance

    def test_failing_variant_rejects_regardless_of_the_others(self):
        executor = FakeExecutor(
            {
                "bench-gpu": FakeComparison(200.0, 100.0, 2.0),
                "bench-gpu-1": FakeComparison(
                    float("inf"), float("inf"), 0.0, feedback_message="FAILURE: boom"
                ),
                "bench-gpu-2": FakeComparison(50.0, 45.0, 1.11),
            }
        )
        result = compare_weighted(executor, _spec3(), "orig", "opt")
        assert not result.accepted
        assert result.failures == ["bench-gpu-1"]
        assert "failed to run" in result.reason

    def test_incorrect_variant_rejects(self):
        executor = FakeExecutor(
            {
                "bench-gpu": FakeComparison(200.0, 100.0, 2.0),
                "bench-gpu-1": FakeComparison(100.0, 50.0, 2.0, optimized_correct=False),
                "bench-gpu-2": FakeComparison(50.0, 45.0, 1.11),
            }
        )
        result = compare_weighted(executor, _spec3(), "orig", "opt")
        assert not result.accepted
        assert "incorrect" in result.reason


class TestWeights:
    def test_no_declared_weights_means_equal_shares(self):
        spec = FakeSpec([("bench-gpu", None), ("bench-gpu-1", None)])
        executor = FakeExecutor(
            {
                "bench-gpu": FakeComparison(100.0, 50.0, 2.0),
                "bench-gpu-1": FakeComparison(100.0, 100.0, 1.0),
            }
        )
        result = compare_weighted(executor, spec, "orig", "opt")
        assert [o.weight for o in result.outcomes] == [0.5, 0.5]

    def test_mixed_weights_are_refused_not_guessed(self):
        spec = FakeSpec([("bench-gpu", 0.7), ("bench-gpu-1", None)])
        executor = FakeExecutor({})
        with pytest.raises(ValueError, match="mixes weighted and unweighted"):
            compare_weighted(executor, spec, "orig", "opt")

    def test_declared_weights_are_normalized(self):
        spec = FakeSpec([("bench-gpu", 3.0), ("bench-gpu-1", 1.0)])
        executor = FakeExecutor(
            {
                "bench-gpu": FakeComparison(100.0, 50.0, 2.0),
                "bench-gpu-1": FakeComparison(100.0, 50.0, 2.0),
            }
        )
        result = compare_weighted(executor, spec, "orig", "opt")
        assert [o.weight for o in result.outcomes] == [pytest.approx(0.75), pytest.approx(0.25)]


class TestReporting:
    def test_empty_family_is_a_named_rejection(self):
        result = compare_weighted(FakeExecutor({}), FakeSpec([]), "orig", "opt")
        assert not result.accepted
        assert "no variants" in result.reason

    def test_format_is_a_table_never_a_single_number(self):
        executor = FakeExecutor(
            {
                "bench-gpu": FakeComparison(200.0, 100.0, 2.0),
                "bench-gpu-1": FakeComparison(50.0, 100.0, 0.5),
            }
        )
        spec = FakeSpec([("bench-gpu", 0.9), ("bench-gpu-1", 0.1)])
        rendered = compare_weighted(executor, spec, "orig", "opt").format()
        assert "bench-gpu-1" in rendered and "REGRESSION" in rendered
        assert "REJECTED" in rendered

    def test_summary_is_json_serializable(self):
        executor = FakeExecutor({"bench-gpu": FakeComparison(200.0, 100.0, 2.0)})
        spec = FakeSpec([("bench-gpu", 1.0)])
        summary = compare_weighted(executor, spec, "orig", "opt").summary()
        assert json.loads(json.dumps(summary))["accepted"] is True

    def test_dataclass_defaults_stand_alone(self):
        # A bare WeightedComparison must render without blowing up, because the
        # empty-family path returns one.
        assert "REJECTED" in WeightedComparison(family="bench-gpu").format()
