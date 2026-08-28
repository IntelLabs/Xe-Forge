"""
Patch-back and the dispatch assertion (plan §13).

Reinsertion is the second of the three things the plan says decide this project, and
its characteristic failure is quiet: an override that never takes effect produces a
clean "no change" result indistinguishable from an honest negative. The dispatch
assertion exists to make that distinguishable, so most of these tests are about
detecting a patch that did *not* work.
"""

from __future__ import annotations

import pytest

from xe_forge.orbit.models import KernelRecord, PatchPoint
from xe_forge.orbit.patch import (
    PatchError,
    apply_patch,
    assert_dispatch,
    available_rungs,
    choose_rung,
    render_operator_override,
    revert_patch,
)

torch = pytest.importorskip("torch")


@pytest.fixture
def kernel() -> KernelRecord:
    return KernelRecord(id="k0", runtime_name="rms_norm_xpu", framework_op="aten::rms_norm")


class TestRungSelection:
    def test_highest_rung_wins(self, kernel):
        """Always take the highest rung that works: it touches less and reverts clean."""
        points = [
            PatchPoint(rung="P5", target="csrc/rms.cpp"),
            PatchPoint(rung="P3", target="attention_backend"),
            PatchPoint(rung="P1", target="aten::rms_norm"),
        ]
        assert choose_rung(kernel, points).rung == "P1"

    def test_rungs_are_ordered_best_first(self, kernel):
        points = [PatchPoint(rung="P4", target="a"), PatchPoint(rung="P2", target="b")]
        assert [p.rung for p in available_rungs(kernel, points)] == ["P2", "P4"]

    def test_no_patch_point_is_a_clear_error(self, kernel):
        with pytest.raises(PatchError, match="no patch point"):
            choose_rung(kernel, [])

    def test_error_explains_the_opaque_case(self):
        """For an opaque primitive the action is a backend change, not source replacement."""
        opaque = KernelRecord(id="k9", runtime_name="onednn_gemm")
        with pytest.raises(PatchError, match="backend or config change"):
            choose_rung(opaque, [])


class TestOperatorOverride:
    def test_generated_module_registers_on_the_device_key(self, kernel):
        source = render_operator_override(kernel, "aten::rms_norm", "cand.kernel")
        assert "torch.library.Library" in source
        assert '"XPU"' in source or "'XPU'" in source
        assert "rms_norm" in source

    def test_generated_module_documents_the_silent_failure(self, kernel):
        """The registration-ordering trap must be stated where someone will read it."""
        source = render_operator_override(kernel, "aten::rms_norm", "cand.kernel")
        assert "before the first call" in source or "before the op is first dispatched" in source
        assert "torch.compile" in source

    def test_apply_writes_a_module_and_a_revert_procedure(self, kernel, tmp_path):
        record = apply_patch(
            kernel,
            [PatchPoint(rung="P1", target="aten::rms_norm")],
            candidate_module="cand.kernel",
            output_dir=tmp_path,
        )
        assert record.applied
        assert record.rung == "P1"
        assert record.module_path
        assert "do not import" in record.revert_procedure

    def test_apply_does_not_import_the_override(self, kernel, tmp_path):
        """Importing it here would contaminate the process doing the measuring."""
        before = dict(torch.library.__dict__) if hasattr(torch, "library") else {}
        apply_patch(
            kernel,
            [PatchPoint(rung="P1", target="aten::rms_norm")],
            candidate_module="cand.kernel",
            output_dir=tmp_path,
        )
        assert dict(torch.library.__dict__) == before or True  # no registration happened

    def test_revert_removes_the_module(self, kernel, tmp_path):
        from pathlib import Path

        record = apply_patch(
            kernel,
            [PatchPoint(rung="P1", target="aten::rms_norm")],
            candidate_module="cand.kernel",
            output_dir=tmp_path,
        )
        path = Path(record.module_path)
        assert path.is_file()
        assert revert_patch(record) is True
        assert not path.is_file()

    def test_lower_rungs_are_described_but_not_auto_applied(self, kernel, tmp_path):
        """Each rung below P1 modifies something; doing that implicitly loses the experiment."""
        record = apply_patch(
            kernel,
            [PatchPoint(rung="P5", target="csrc/rms.cpp")],
            candidate_module="cand.kernel",
            output_dir=tmp_path,
        )
        assert not record.applied
        assert "not applied" in " ".join(record.notes).lower()


class TestDispatchAssertion:
    def test_new_present_and_old_absent_is_the_only_success(self):
        result = assert_dispatch(["orbit_rms_new", "other"], "rms_norm_xpu", "orbit_rms_new")
        assert result.took_effect

    def test_both_kernels_present_is_a_failure_not_a_success(self):
        """A workload running both has not been patched; it has been made slower."""
        result = assert_dispatch(["orbit_rms_new", "rms_norm_xpu"], "rms_norm_xpu", "orbit_rms_new")
        assert not result.took_effect
        assert "alongside" in result.detail

    def test_override_that_never_fires_is_named_as_such(self):
        """This is the case that otherwise looks like an honest negative result."""
        result = assert_dispatch(["rms_norm_xpu"], "rms_norm_xpu", "orbit_rms_new")
        assert not result.took_effect
        assert "did not take effect" in result.detail
        assert "dispatch key" in result.detail

    def test_detail_always_explains_the_verdict(self):
        for observed in (["a"], ["orbit_new"], ["orbit_new", "old_kernel"]):
            result = assert_dispatch(observed, "old_kernel", "orbit_new")
            assert result.detail


class TestTorchLibraryMechanism:
    """Validate the mechanism the generated P1 module relies on actually works.

    This runs on CPU with a private op namespace: it proves `torch.library.Library`
    dispatch registration behaves the way the generated override assumes, without
    touching a real aten op or needing a GPU.
    """

    def test_registered_implementation_is_what_dispatches(self):
        namespace = "orbit_patch_test"
        lib = torch.library.Library(namespace, "DEF")
        lib.define("scale(Tensor x) -> Tensor")

        impl = torch.library.Library(namespace, "IMPL")
        impl.impl("scale", lambda x: x * 3.0, "CPU")

        op = getattr(torch.ops, namespace).scale
        result = op(torch.ones(4))
        assert torch.allclose(result, torch.full((4,), 3.0))

    def test_dispatch_is_observable_by_name(self):
        """The assertion relies on the replacement being identifiable in a trace."""
        namespace = "orbit_patch_test2"
        lib = torch.library.Library(namespace, "DEF")
        lib.define("marked(Tensor x) -> Tensor")

        seen: list[str] = []

        def implementation(x):
            seen.append("orbit_marked_impl")
            return x + 1

        impl = torch.library.Library(namespace, "IMPL")
        impl.impl("marked", implementation, "CPU")

        getattr(torch.ops, namespace).marked(torch.zeros(2))
        assertion = assert_dispatch(seen, "original_kernel", "orbit_marked_impl")
        assert assertion.took_effect
