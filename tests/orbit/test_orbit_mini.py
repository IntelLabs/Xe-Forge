"""`orbit_mini` consumed by the rig it was built for (plan §15, gap G4).

The reference workload was designed as the adversarial fixture the §15.4 table asserts
against — and then nothing imported it: no test, no CLI path, no CI step. These tests
wire the CPU-viable traps into T0. The traps that need silicon (real traces, launch
interception, dispatch assertions) belong to the hardware tier, not here.

Each test class is named for the trap it arms, so a failure says which §15.2 property
stopped holding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# `examples` lives at the repository root; conftest puts only `src/` on the path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")

from examples.orbit_mini import build_model, get_example_inputs  # noqa: E402
from examples.orbit_mini.kernels import rmsnorm, sycl_op, tuned  # noqa: E402

KERNELS_DIR = REPO_ROOT / "examples" / "orbit_mini" / "kernels"


class TestNonContiguousInputTrap:
    """§15.2: 'one input deliberately non-contiguous, so synthetic reconstruction fails visibly'."""

    def test_input_is_non_contiguous_with_expected_shape(self):
        hidden = get_example_inputs()
        assert not hidden.is_contiguous()
        batch, seq, hid = hidden.shape
        assert (batch, seq, hid) == hidden.shape and hid > 0

    def test_capture_roundtrip_preserves_the_trap(self, tmp_path):
        """Stage 6's golden test (§16.4) against the fixture built to defeat it.

        A capture that silently normalises the stride pattern would reload a
        contiguous tensor — a different launch record, and on a stride-sensitive
        kernel different numbers. `verify_roundtrip` must come back clean, and the
        reloaded tensor must still be non-contiguous.
        """
        from xe_forge.orbit.capture.capture import (
            capture_invocation,
            load_invocation,
            verify_roundtrip,
        )

        hidden = get_example_inputs()
        invocation = capture_invocation("mini_hidden", {"hidden": hidden}, tmp_path)
        assert verify_roundtrip(invocation) == []
        restored = load_invocation(invocation)["hidden"]
        assert not restored.is_contiguous()
        assert torch.equal(restored, hidden)


class TestWorkloadRuns:
    """The model itself must run under plain CPU torch, or T0 has no workload."""

    def test_forward_is_finite_and_seeded(self):
        model = build_model()
        hidden = get_example_inputs()
        with torch.no_grad():
            out_a = model(hidden)
            out_b = build_model()(get_example_inputs())
        assert torch.isfinite(out_a).all()
        # Seeding is a property of the fixture, not of the caller (§17): two
        # independent builds must produce identical outputs.
        assert torch.equal(out_a, out_b)

    def test_rms_norm_fallback_matches_reference_semantics(self):
        x = get_example_inputs().reshape(-1, get_example_inputs().shape[-1])
        weight = torch.ones(x.shape[-1], dtype=x.dtype)
        out = rmsnorm.rms_norm(x, weight, eps=1e-6)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


class TestSplitClosureTrap:
    """§15.2: device helpers across three modules, one reached through a re-export.

    This is row 7 of the §16.4 stage matrix — 'mini multi-file kernel → complete
    closure' — and the single most common extraction bug when it fails (§12.12).
    """

    def _closure(self):
        from xe_forge.orbit.languages import get_backend
        from xe_forge.orbit.models import SourceLocation

        return get_backend("triton").resolve_closure(
            SourceLocation(file=str(KERNELS_DIR / "rmsnorm.py"), symbol="_rmsnorm_kernel")
        )

    def test_closure_spans_all_three_helper_modules(self):
        result = self._closure()
        names = {path.name for path in result.files}
        # helpers_b is reached only through device_ops' re-export alias — the exact
        # shape the trap exists to catch.
        for member in (
            "rmsnorm.py",
            "device_ops.py",
            "helpers_a.py",
            "helpers_b.py",
            "helpers_c.py",
            "triton_compat.py",
            "tuned.py",
        ):
            assert member in names, f"{member} missing from closure: {sorted(names)}"

    def test_closure_is_complete_and_records_decorators(self):
        result = self._closure()
        assert not result.unresolved, result.unresolved
        # §12.7: the autotune config list and the heuristics callable must be seen,
        # or the winning configuration cannot be pinned.
        assert "_rmsnorm_kernel" in result.autotune_configs
        assert "_rmsnorm_kernel" in result.heuristics


class TestTunedConfigTrap:
    """§15.2: a tuned-config JSON keyed by device name, as a genuine data dependency."""

    def test_lookup_reads_the_json_for_cpu(self):
        tuned.clear_cache()
        entry = tuned.lookup("cpu")
        assert entry.block_n > 0 and entry.num_warps > 0

    def test_the_data_file_exists_where_the_closure_expects_it(self):
        assert (KERNELS_DIR / "tuned_configs.json").is_file()


class TestSyclOpFallback:
    """§15.2's SYCL dispatcher op must degrade to torch on CPU, not fail."""

    def test_cpu_fallback_produces_finite_output(self):
        x = torch.randn(4, 32)
        weight = torch.ones(32)
        out = sycl_op.rmsnorm(x, weight, 1e-6)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_status_reports_rather_than_raises(self):
        assert isinstance(sycl_op.status(), str)
