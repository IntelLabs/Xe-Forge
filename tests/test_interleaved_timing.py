"""Interleaved timing: both arms every round, alternating order, and a gate on agreement.

Timing the baseline block and then the trial block put every drift between them on one
arm; identical device code measured 29% faster that way. The device timer itself needs a
GPU, so these check the scheduling and the gate around it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from xe_forge.core import executor as executor_mod
from xe_forge.core.executor import KernelBenchExecutor


def test_rounds_alternate_order_and_cover_both_arms(monkeypatch):
    ex = KernelBenchExecutor(device="cpu", warmup_iters=0, benchmark_iters=40, rounds=4)
    order = []

    def fake_samples(call, args, iters):
        order.append(call.name)
        return [call.us] * iters

    monkeypatch.setattr(ex, "_forward_samples", fake_samples)
    a = SimpleNamespace(name="a", us=10.0)
    b = SimpleNamespace(name="b", us=5.0)
    a_us, b_us, ratios = ex.time_interleaved(a, (), b, ())
    assert order == ["a", "b", "b", "a", "a", "b", "b", "a"]
    assert (a_us, b_us, ratios) == (10.0, 5.0, [2.0] * 4)


@pytest.mark.parametrize(
    "r1,r2,verdict,printed",
    [
        # A real 1.25x under a 2% slot bias: base/trial 1.275, trial/base 0.816.
        ([1.275] * 3, [1.02 / 1.25] * 3, None, 1.25),
        # Identical code under the same bias: both slots read 0.98 -- cancelled, not a result.
        ([0.98, 0.97, 0.99], [0.99, 0.98, 0.97], "INDISTINGUISHABLE", 1.0),
        # Every round agrees, but on 0.99x: inside the minimum effect, so not a regression.
        ([0.98] * 3, [1.0] * 3, "INDISTINGUISHABLE", 0.99),
    ],
)
def test_slot_bias_is_cancelled(monkeypatch, r1, r2, verdict, printed):
    model = SimpleNamespace(cpu=lambda: None)
    workload = SimpleNamespace(
        validate=lambda: None,
        original=model,
        optimized=model,
        prepare_call=lambda m: SimpleNamespace(inputs=[]),
    )
    monkeypatch.setattr(executor_mod.ReferenceWorkload, "prepare", lambda *a, **k: workload)
    ex = KernelBenchExecutor(device="xpu")
    answers = iter([(10.0, 8.0, r1), (8.0, 10.0, r2)])
    monkeypatch.setattr(ex, "time_interleaved", lambda *a: next(answers))
    result = ex.compare_reference_workload("ref", "base", "opt", baseline_us=99.0)
    assert next(answers, None) is None  # one swapped pair per benchmark, nothing more
    assert result.verdict == verdict
    assert result.speedup == pytest.approx(printed, abs=0.01)
    assert result.original_time_us == pytest.approx(10.0)  # a cached baseline is not used on GPU
    assert result.optimized_time_us == pytest.approx(8.0)  # times are printed as measured
