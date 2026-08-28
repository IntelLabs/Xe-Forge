"""
End-to-end proof that reinsertion works (plan §13).

Every other test in this suite runs against fixtures or stubs. This one runs real
processes, registers a real dispatcher op, applies a real operator override, and checks
the dispatch assertion against what actually executed. It is the only test that can
answer the question §13 says decides the project: *did the optimized kernel actually
replace the original in a running workload, and can we prove it.*

It compares the workload's own reported per-iteration time rather than process wall
time. That is not a shortcut — it is what a real adapter's metric extractor does, and
here it is necessary: a ~1s torch import dominates a ~0.1s workload, so a wall-clock
comparison would dilute a genuine 15% kernel improvement into noise and the test would
prove nothing.

Runs on CPU. With Triton and an accelerator present the same flow exercises the Triton
kernel instead, and the assertions are unchanged.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from xe_forge.orbit import stats
from xe_forge.orbit.bench.core import BenchRunner
from xe_forge.orbit.executor import LocalExecutor, RunResult
from xe_forge.orbit.models import Decision, KernelRecord, PatchPoint
from xe_forge.orbit.patch import apply_patch, assert_dispatch

pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[2]

BASELINE_KERNEL = "orbit_demo_rms_norm_naive"
OVERRIDE_MODULE = "examples.kernel_replacement.optimized"

_PER_ITER = re.compile(r"per_iter_ms=([\d.]+)")
_CHECKSUM = re.compile(r"checksum=([-\d.]+)")
_KERNEL = re.compile(r"ORBIT_KERNEL=(\S+)")

# Enough iterations that the kernel dominates the measurement, few enough that ten
# subprocess launches stay tolerable in CI.
#
# 40 iterations was not enough and the failure was instructive: the measured window
# (~0.16s in a freshly started process) was short enough that CPU contention and
# frequency ramp produced samples varying by 2.5x, and the comparison came back
# INCONCLUSIVE with a CI of [-90%, +44%]. Lengthening the window is what makes the
# effect resolvable — the same reasoning §17 applies to repetitions, one level down.
ITERS = 150
WARMUP = 30


def _spec(override: bool, repetitions: int = 5):
    from xe_forge.orbit.models import WorkloadSpec

    env = {"PYTHONPATH": str(REPO_ROOT)}
    if override:
        env["ORBIT_OVERRIDE_MODULE"] = OVERRIDE_MODULE
    return WorkloadSpec(
        command=[
            sys.executable,
            "-m",
            "examples.kernel_replacement.workload",
            "--iters",
            str(ITERS),
            "--warmup",
            str(WARMUP),
        ],
        cwd=REPO_ROOT,
        env=env,
        repetitions=repetitions,
        warmup_iterations=1,
    )


def _extract_metrics(result: RunResult) -> dict[str, float]:
    """Parse the workload's self-reported metrics — the adapter metric-extractor path."""
    match = _PER_ITER.search(result.stdout)
    return {"per_iter_ms": float(match.group(1))} if match else {}


def _observed_kernels(result: RunResult) -> list[str]:
    return _KERNEL.findall(result.stdout)


def _checksum(result: RunResult) -> float | None:
    match = _CHECKSUM.search(result.stdout)
    return float(match.group(1)) if match else None


@pytest.fixture(scope="module")
def baseline_run() -> RunResult:
    return LocalExecutor().run(**_run_kwargs(override=False))


@pytest.fixture(scope="module")
def patched_run() -> RunResult:
    return LocalExecutor().run(**_run_kwargs(override=True))


def _run_kwargs(override: bool) -> dict:
    spec = _spec(override)
    return {"cmd": spec.command, "env": spec.env, "cwd": spec.cwd, "timeout": 300.0}


class TestWorkloadItself:
    def test_baseline_dispatches_the_naive_kernel(self, baseline_run):
        assert baseline_run.ok, baseline_run.stderr[-500:]
        assert _observed_kernels(baseline_run) == [BASELINE_KERNEL]

    def test_override_replaces_it(self, patched_run):
        assert patched_run.ok, patched_run.stderr[-500:]
        observed = _observed_kernels(patched_run)
        assert observed
        assert BASELINE_KERNEL not in observed


class TestDispatchAssertion:
    def test_override_provably_took_effect(self, baseline_run, patched_run):
        """The §13 assertion: new kernel present AND old kernel absent.

        This is the check that separates a real replacement from the failure that looks
        identical to an honest negative — an override that silently never fired.
        """
        replacement = _observed_kernels(patched_run)[0]
        assertion = assert_dispatch(
            _observed_kernels(patched_run),
            original_kernel=BASELINE_KERNEL,
            replacement_marker=replacement,
        )
        assert assertion.took_effect, assertion.detail
        assert "is what executes" in assertion.detail

    def test_the_baseline_run_fails_the_same_assertion(self, baseline_run):
        """Negative control: without the override the assertion must not pass.

        If this passed, the assertion would be vacuous and every patch would look
        successful whether or not it did anything.
        """
        assertion = assert_dispatch(
            _observed_kernels(baseline_run),
            original_kernel=BASELINE_KERNEL,
            replacement_marker="orbit_demo_rms_norm_torch_fused",
        )
        assert not assertion.took_effect
        assert "did not take effect" in assertion.detail


class TestCorrectness:
    def test_the_replacement_is_numerically_equivalent(self, baseline_run, patched_run):
        """L1: a faster kernel that changes the answer is not an optimization.

        Fusing reassociates floating-point work, so the results differ in the last few
        bits. That is exactly why the gate is a tolerance rather than equality — and
        why §9.3 has Orbit emit a *tightened* tolerance instead of trusting a default.
        """
        baseline = _checksum(baseline_run)
        patched = _checksum(patched_run)
        assert baseline is not None and patched is not None

        relative = abs(patched - baseline) / abs(baseline)
        assert relative < 1e-6, f"relative deviation {relative:.2e} is too large"
        # And it is genuinely a different computation, not a no-op rename.
        assert baseline != patched


@pytest.fixture(scope="module")
def samples():
    """Interleaved ABBA samples of the workload's own per-iteration time."""
    runner = BenchRunner(executor=LocalExecutor(), metric_extractor=_extract_metrics)
    return runner.interleaved(
        _spec(override=False), _spec(override=True), repetitions=5, metric="per_iter_ms"
    )


class TestMeasuredImprovement:
    def test_both_arms_produced_samples(self, samples):
        baseline, candidate = samples
        assert len(baseline) == len(candidate) == 5

    def test_the_chain_produces_a_well_formed_decision(self, samples):
        """The whole chain: interleaved ABBA runs, paired interval, a real decision.

        This asserts the *loop* works, not that this particular kernel wins. A
        replacement being slower or unresolvable is a legitimate outcome — it is
        exactly what §17's REJECT and INCONCLUSIVE exist to express, and a test that
        demanded ACCEPT would be asserting a property of one CPU rather than of the
        measurement chain.

        What must hold is that the decision is one of the four defined outcomes, that
        it comes with an interval and a stated reason, and that ACCEPT is never reached
        on an interval containing zero.
        """
        baseline, candidate = samples
        decision, detail = stats.compare(baseline, candidate, min_repetitions=5)

        assert decision in (
            Decision.ACCEPT,
            Decision.REJECT,
            Decision.INCONCLUSIVE,
        ), decision
        assert detail["reason"]
        assert detail["method"] == "paired"

        low = float(detail["ci95_low"])
        high = float(detail["ci95_high"])
        assert low <= float(detail["improvement_percent"]) <= high

        # The invariant that actually matters: a verdict is only decisive when the
        # interval excludes zero.
        if decision is Decision.ACCEPT:
            assert low > 0
        elif decision is Decision.REJECT:
            assert high < 0
        else:
            assert low <= 0 <= high

    def test_the_mde_is_reported_so_a_null_result_is_interpretable(self, samples):
        """The MDE accompanies every comparison, decisive or not (§17.4, §25).

        Note what is deliberately *not* asserted here: that a decisive verdict must
        exceed the MDE. That looks like a sound invariant and is not one. The MDE is the
        effect size this setup could reliably resolve at 80% power; a 95% interval can
        legitimately exclude zero with a point estimate below it, when this particular
        sample happened to have low variance. An earlier version of this test asserted
        the stronger claim and failed intermittently — the statistics were right and the
        assertion was wrong.

        The MDE's real job is predictive, and that is where §18 uses it: a kernel whose
        Amdahl ceiling falls under the MDE is not worth optimizing, because the gain
        could not be measured even if achieved. Using it as a post-hoc filter on a
        result that already cleared significance would discard true findings.
        """
        baseline, candidate = samples
        _, detail = stats.compare(baseline, candidate, min_repetitions=5)

        mde = float(detail["minimum_detectable_effect"])
        assert mde > 0
        assert mde != float("inf")


class TestPatchArtifact:
    def test_apply_records_a_revertible_p1_patch(self, tmp_path):
        kernel = KernelRecord(
            id="k0",
            runtime_name=BASELINE_KERNEL,
            framework_op="orbit_demo::rms_norm",
        )
        record = apply_patch(
            kernel,
            [PatchPoint(rung="P1", target="orbit_demo::rms_norm")],
            candidate_module=OVERRIDE_MODULE,
            output_dir=tmp_path,
        )
        assert record.rung == "P1"
        assert record.applied

        generated = Path(record.module_path).read_text()
        assert "torch.library.Library" in generated
        assert OVERRIDE_MODULE in generated
        # P1's defining property: reverting touches nothing.
        assert "do not import" in record.revert_procedure
