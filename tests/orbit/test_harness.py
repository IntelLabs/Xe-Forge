"""
The generated correctness harness (plan §19.6).

The distinction these tests defend hardest is "could not be checked" versus "wrong".
Collapsing them makes an import failure read as a passing kernel or as a broken one;
both are false, and the second wastes a revert on working code.
"""

from __future__ import annotations

import sys

import pytest

from xe_forge.orbit.optimize.harness import (
    CheckOutcome,
    HarnessSpec,
    render_harness,
    run_harness,
)


@pytest.fixture
def spec():
    return HarnessSpec(
        kernel_id="k5",
        import_statement="from mypkg.kernels import my_kernel",
        setup="x = torch.randn(4, 8)",
        call_expr="my_kernel(x)",
        reference_expr="x.argmax(dim=-1)",
        notes="a note",
    )


class TestRendering:
    def test_the_kernel_is_imported_not_extracted(self, spec):
        """The in-place path checks what is on disk, reached the way the workload does."""
        assert "from mypkg.kernels import my_kernel" in render_harness(spec)

    def test_the_rendered_script_is_valid_python(self, spec):
        compile(render_harness(spec), "harness.py", "exec")

    def test_setup_is_indented_into_the_function_body(self, spec):
        assert "    x = torch.randn(4, 8)" in render_harness(spec)

    def test_a_multiline_setup_stays_valid(self):
        s = HarnessSpec(
            kernel_id="k",
            import_statement="import torch as _t",
            setup="a = 1\nb = 2\nc = a + b",
            call_expr="_t.tensor([c])",
            reference_expr="_t.tensor([3])",
        )
        compile(render_harness(s), "harness.py", "exec")

    def test_the_notes_are_carried_into_the_script(self, spec):
        assert "a note" in render_harness(spec)

    def test_an_import_failure_returns_unchecked_not_wrong(self, spec):
        """Verified by reading the generated source: the import guard must return 2."""
        rendered = render_harness(spec)
        guard = rendered.split("import failed")[1]
        assert "return 2" in guard.split("torch.manual_seed")[0]


class TestRunningTheHarness:
    def _script(self, tmp_path, body):
        path = tmp_path / "h.py"
        path.write_text(body, encoding="utf-8")
        return path

    def test_a_correct_kernel_reports_correct_with_its_accuracy(self, tmp_path):
        script = self._script(tmp_path, "print('ACCURACY 1.000000 256/256')\nraise SystemExit(0)\n")
        result = run_harness(script, python=sys.executable)
        assert result.outcome is CheckOutcome.CORRECT
        assert result.accuracy == 1.0
        assert result.correct

    def test_a_wrong_kernel_reports_wrong(self, tmp_path):
        script = self._script(tmp_path, "print('ACCURACY 0.000000 0/256')\nraise SystemExit(1)\n")
        result = run_harness(script, python=sys.executable)
        assert result.outcome is CheckOutcome.WRONG
        assert result.accuracy == 0.0
        assert not result.correct

    def test_exit_two_is_unchecked_and_keeps_its_reason(self, tmp_path):
        """The distinction the whole module exists to preserve."""
        script = self._script(
            tmp_path,
            "import sys\nprint('UNCHECKED: import failed: no module', file=sys.stderr)\n"
            "raise SystemExit(2)\n",
        )
        result = run_harness(script, python=sys.executable)
        assert result.outcome is CheckOutcome.UNCHECKED
        assert not result.correct
        assert "import failed" in result.detail

    def test_an_unexpected_exit_code_is_unchecked_not_a_verdict(self, tmp_path):
        """A crash or a signal is not a numerical result."""
        script = self._script(tmp_path, "raise SystemExit(137)\n")
        assert run_harness(script, python=sys.executable).outcome is CheckOutcome.UNCHECKED

    def test_a_crash_is_unchecked_even_though_python_exits_one(self, tmp_path):
        """Exit 1 is ambiguous: the harness means "wrong", Python means "raised".

        Distinguished by evidence — a numerical verdict requires an ACCURACY line.
        Without one, reverting the candidate would be repairing the wrong thing.
        """
        script = self._script(tmp_path, "raise RuntimeError('boom')\n")
        result = run_harness(script, python=sys.executable)
        assert result.outcome is CheckOutcome.UNCHECKED
        assert "boom" in result.detail

    def test_exit_one_with_evidence_is_a_real_wrong_verdict(self, tmp_path):
        script = self._script(tmp_path, "print('ACCURACY 0.010000 2/256')\nraise SystemExit(1)\n")
        result = run_harness(script, python=sys.executable)
        assert result.outcome is CheckOutcome.WRONG
        assert result.accuracy == 0.01

    def test_a_hang_is_unchecked_rather_than_waiting_forever(self, tmp_path):
        script = self._script(tmp_path, "import time\ntime.sleep(30)\n")
        result = run_harness(script, python=sys.executable, timeout_s=1.0)
        assert result.outcome is CheckOutcome.UNCHECKED
        assert "did not finish" in result.detail

    def test_a_missing_interpreter_is_unchecked(self, tmp_path):
        script = self._script(tmp_path, "print('ACCURACY 1.0 1/1')\n")
        result = run_harness(script, python="/nonexistent/python")
        assert result.outcome is CheckOutcome.UNCHECKED

    def test_accuracy_survives_a_malformed_line(self, tmp_path):
        script = self._script(tmp_path, "print('ACCURACY notanumber')\nraise SystemExit(0)\n")
        result = run_harness(script, python=sys.executable)
        assert result.outcome is CheckOutcome.CORRECT
        assert result.accuracy is None

    def test_the_verdict_renders_readably(self, tmp_path):
        script = self._script(tmp_path, "print('ACCURACY 0.500000 1/2')\nraise SystemExit(1)\n")
        assert "0.5000" in run_harness(script, python=sys.executable).format()


class TestCoverageOfChangedPaths:
    """A reference test covers the paths a reference can reach, and no others.

    `gumbel_sample` at temperature 0 is argmax, which torch reproduces exactly; above 0
    it rides Triton's Philox stream, which torch cannot reproduce. An agent then proposed
    removing a `tl.where` that exists only in the temperature>0 branch — a change the
    gate would have passed at accuracy 1.0000 without executing the changed line.
    """

    def test_the_shape_sweep_spans_more_than_the_convenient_shape(self):
        from xe_forge.orbit.optimize.harness import DEFAULT_SHAPE_SWEEP

        assert len(DEFAULT_SHAPE_SWEEP) > 1
        # 151936 * 14138 crosses 2**31, so an int32 index change is only wrong above it.
        assert max(DEFAULT_SHAPE_SWEEP) > 14138

    def test_a_spec_carries_its_shapes(self):
        from xe_forge.orbit.optimize.harness import DEFAULT_SHAPE_SWEEP, HarnessSpec

        spec = HarnessSpec(
            kernel_id="k", import_statement="x", setup="", call_expr="f()", reference_expr="g()"
        )
        assert spec.shapes == DEFAULT_SHAPE_SWEEP

    def test_the_differential_alternative_is_stated_for_unreferenceable_paths(self):
        from xe_forge.orbit.optimize.harness import DIFFERENTIAL_NOTE

        assert "bit-identical" in DIFFERENTIAL_NOTE
        assert "same seed" in DIFFERENTIAL_NOTE


class TestDifferentialHarness:
    """Proven on hardware: a bug confined to the temperature>0 branch of gumbel_sample
    was reported CORRECT (accuracy 1.0000, exit 0) by the reference harness and caught
    at 1/3 by the differential one. Exactly the temp=0 case passed."""

    def _spec(self, cases=None):
        from xe_forge.orbit.optimize.harness import DifferentialSpec

        return DifferentialSpec(
            kernel_id="k5",
            import_statement="from m import f",
            setup='n = case["n"]',
            call_expr="f(n)",
            cases=cases if cases is not None else [{"n": 256}, {"n": 4096}],
            notes="covers a path no reference can reach",
        )

    def test_the_rendered_script_is_valid_python(self):
        from xe_forge.orbit.optimize.harness import render_differential

        compile(render_differential(self._spec()), "d.py", "exec")

    def test_every_case_is_carried_into_the_script(self):
        from xe_forge.orbit.optimize.harness import render_differential

        src = render_differential(self._spec([{"n": 256, "temp": 0.0}, {"n": 256, "temp": 1.0}]))
        assert '"temp": 1.0' in src or "'temp': 1.0" in src

    def test_the_seed_is_reset_per_case_so_both_runs_share_a_stream(self):
        """Identical seeds are what make the two sides comparable at all."""
        from xe_forge.orbit.optimize.harness import render_differential

        assert "torch.manual_seed(0)" in render_differential(self._spec())

    def test_a_missing_baseline_is_unchecked_not_a_failure(self):
        """Nothing to compare against says nothing about the patch."""
        from xe_forge.orbit.optimize.harness import render_differential

        src = render_differential(self._spec())
        assert "no baseline to compare against" in src
        assert "return 2" in src.split("no baseline to compare against")[1][:80]

    def test_a_changed_case_count_is_unchecked_rather_than_a_mismatch(self):
        from xe_forge.orbit.optimize.harness import render_differential

        assert "case count changed between runs" in render_differential(self._spec())

    def test_it_reports_on_the_same_ACCURACY_contract_as_the_reference_harness(self):
        """So `run_harness` reads either kind without knowing which it ran."""
        from xe_forge.orbit.optimize.harness import render_differential

        assert "ACCURACY " in render_differential(self._spec())

    def test_an_empty_case_list_still_runs_one_case(self):
        from xe_forge.orbit.optimize.harness import render_differential

        assert "CASES = [\n        {}\n]" in render_differential(
            self._spec([])
        ) or "{}" in render_differential(self._spec([]))


class TestCombinedCheck:
    """A loop with several checks must not accept because the cheapest one passed."""

    def _r(self, outcome, acc=None):
        from xe_forge.orbit.optimize.harness import CheckResult

        return CheckResult(outcome, accuracy=acc, detail=str(outcome.value))

    def test_all_passing_is_correct(self):
        from xe_forge.orbit.optimize.harness import CheckOutcome, combined_check

        result = combined_check(
            [self._r(CheckOutcome.CORRECT, 1.0), self._r(CheckOutcome.CORRECT, 1.0)]
        )
        assert result.correct
        assert "no reference reaches" in result.detail

    def test_any_wrong_dominates(self):
        """One check proving a difference is proof, whatever the others say."""
        from xe_forge.orbit.optimize.harness import CheckOutcome, combined_check

        result = combined_check(
            [self._r(CheckOutcome.CORRECT, 1.0), self._r(CheckOutcome.WRONG, 0.33)]
        )
        assert result.outcome is CheckOutcome.WRONG

    def test_unchecked_beats_correct_when_nothing_is_wrong(self):
        """The exact case: reference passed at 1.0000, changed path never executed."""
        from xe_forge.orbit.optimize.harness import CheckOutcome, combined_check

        result = combined_check(
            [self._r(CheckOutcome.CORRECT, 1.0), self._r(CheckOutcome.UNCHECKED)]
        )
        assert result.outcome is CheckOutcome.UNCHECKED

    def test_wrong_still_beats_unchecked(self):
        from xe_forge.orbit.optimize.harness import CheckOutcome, combined_check

        result = combined_check([self._r(CheckOutcome.UNCHECKED), self._r(CheckOutcome.WRONG, 0.0)])
        assert result.outcome is CheckOutcome.WRONG

    def test_the_reported_accuracy_is_the_weakest_not_the_best(self):
        from xe_forge.orbit.optimize.harness import CheckOutcome, combined_check

        result = combined_check(
            [self._r(CheckOutcome.CORRECT, 1.0), self._r(CheckOutcome.CORRECT, 0.9995)]
        )
        assert result.accuracy == 0.9995

    def test_no_checks_at_all_is_unchecked(self):
        from xe_forge.orbit.optimize.harness import CheckOutcome, combined_check

        assert combined_check([]).outcome is CheckOutcome.UNCHECKED
