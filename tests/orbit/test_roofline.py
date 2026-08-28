"""
Roofline headroom (plan §18).

§18 replaced v1's unspecified `estimated_headroom` fudge factor with a measured
achieved-vs-ceiling ratio. These tests pin the two properties that make that a
replacement rather than a rename: the direction of the term (getting it backwards
inverts the ranking) and the refusal to produce a number when nothing was measured.

The last test parses `scripts/roofline.py` to prove the copied preset table has not
drifted from the standalone tool it came from. The library deliberately does not import
that script — it is a PEP-723 tool with its own dependency set — so the duplication is
guarded here instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from xe_forge.orbit.analysis import roofline
from xe_forge.orbit.analysis.catalog import build_catalog
from xe_forge.orbit.models import KernelRecord, Provider
from xe_forge.orbit.profiling import trace as trace_mod

REPO_ROOT = Path(__file__).resolve().parents[2]

# An Arc B580 (117 TFLOPS, 456 GB/s) at an arithmetic intensity of 1000 FLOP/byte is
# far right of the ridge point, so the flat compute roof applies: the ceiling is the
# full 117 TFLOPS. One millisecond of kernel time at exactly the roof therefore moves
# 117e12 * 1e-3 = 1.17e11 FLOPs.
_AT_ROOF_FLOPS = 1.17e11
_AT_ROOF_BYTES = _AT_ROOF_FLOPS / 1000.0
_TIME_US = 1000.0
_B580 = "Intel(R) Arc(TM) B580 Graphics"


def kernel(name: str = "k", avg_us: float = _TIME_US, calls: int = 1) -> KernelRecord:
    return KernelRecord(
        id="k0",
        runtime_name=name,
        calls=calls,
        total_time_us=avg_us * calls,
        avg_time_us=avg_us,
    )


class TestDeviceResolution:
    @pytest.mark.parametrize(
        ("device", "preset"),
        [
            ("arc-b580", "arc-b580"),
            ("Intel(R) Arc(TM) B580 Graphics", "arc-b580"),
            ("Intel Arc Pro B70", "arc-pro-b70"),
            ("Intel Data Center GPU Max 1550", "max-1550"),
            ("Intel Data Center GPU Flex 170", "flex-170"),
        ],
    )
    def test_reported_device_names_resolve_to_presets(self, device, preset):
        resolved = roofline.resolve_hardware(device)
        assert resolved is not None
        assert resolved.key == preset

    def test_an_unknown_device_resolves_to_nothing_rather_than_the_nearest_one(self):
        """Substituting a 'close enough' GPU puts a wrong roof under every number."""
        assert roofline.resolve_hardware("Intel Arc Pro B50") is None
        assert roofline.resolve_hardware("NVIDIA H100") is None
        assert roofline.resolve_hardware(None) is None


class TestHeadroomDirection:
    def test_a_kernel_at_the_roof_has_no_headroom(self):
        estimate = roofline.estimate_headroom(
            time_us=_TIME_US,
            device_name=_B580,
            flops=_AT_ROOF_FLOPS,
            bytes_moved=_AT_ROOF_BYTES,
        )
        assert estimate.measured
        assert estimate.achieved_tflops == pytest.approx(117.0, rel=1e-3)
        assert estimate.ceiling_tflops == pytest.approx(117.0, rel=1e-3)
        assert estimate.value == pytest.approx(1.0, rel=1e-3)

    def test_a_kernel_far_below_the_roof_has_more(self):
        """The direction that matters: below the roof scores HIGHER, not lower."""
        at_roof = roofline.estimate_headroom(
            time_us=_TIME_US,
            device_name=_B580,
            flops=_AT_ROOF_FLOPS,
            bytes_moved=_AT_ROOF_BYTES,
        )
        quarter = roofline.estimate_headroom(
            time_us=_TIME_US,
            device_name=_B580,
            flops=_AT_ROOF_FLOPS / 4,
            bytes_moved=_AT_ROOF_BYTES / 4,
        )
        assert quarter.value == pytest.approx(4.0, rel=1e-3)
        assert quarter.value > at_roof.value

    def test_a_memory_bound_kernel_is_measured_against_the_sloped_roof(self):
        """At AI 10 the B580's ceiling is 4.56 TFLOPS, not its 117 TFLOPS peak."""
        estimate = roofline.estimate_headroom(
            time_us=_TIME_US, device_name=_B580, flops=2.28e9, bytes_moved=2.28e8
        )
        assert estimate.compute_bound is False
        assert estimate.ceiling_tflops == pytest.approx(4.56, rel=1e-3)
        assert estimate.value == pytest.approx(2.0, rel=1e-3)

    def test_headroom_is_capped(self):
        """Unbounded, a tiny badly-optimized kernel would outrank a dominant one."""
        estimate = roofline.estimate_headroom(
            time_us=_TIME_US,
            device_name=_B580,
            flops=_AT_ROOF_FLOPS / 1000,
            bytes_moved=_AT_ROOF_BYTES / 1000,
        )
        assert estimate.value == roofline.MAX_HEADROOM

    def test_above_the_roof_is_clamped_to_neutral_not_inverted(self):
        """Impossible input means the counts are wrong, not that the term flips."""
        estimate = roofline.estimate_headroom(
            time_us=_TIME_US,
            device_name=_B580,
            flops=_AT_ROOF_FLOPS * 4,
            bytes_moved=_AT_ROOF_BYTES * 4,
        )
        assert estimate.value == roofline.NEUTRAL_HEADROOM
        assert "exceeds" in estimate.reason


class TestUnmeasured:
    def test_missing_counts_return_neutral_and_are_flagged(self):
        """The common case: a torch.profiler trace carries no FLOP or byte counts."""
        estimate = roofline.estimate_headroom(time_us=_TIME_US, device_name=_B580)
        assert estimate.value == roofline.NEUTRAL_HEADROOM
        assert estimate.measured is False
        assert estimate.basis == "unmeasured"
        assert "FLOP and byte counts" in estimate.reason

    def test_only_one_of_the_two_counts_is_not_enough(self):
        """One count cannot say which roof applies, and guessing overstates headroom."""
        flops_only = roofline.estimate_headroom(
            time_us=_TIME_US, device_name=_B580, flops=_AT_ROOF_FLOPS
        )
        bytes_only = roofline.estimate_headroom(
            time_us=_TIME_US, device_name=_B580, bytes_moved=_AT_ROOF_BYTES
        )
        assert flops_only.measured is False
        assert "byte counts" in flops_only.reason
        assert bytes_only.measured is False
        assert "FLOP counts" in bytes_only.reason

    def test_unknown_device_falls_back_to_neutral_and_names_the_device(self):
        estimate = roofline.estimate_headroom(
            time_us=_TIME_US,
            device_name="Intel Arc Pro B50",
            flops=_AT_ROOF_FLOPS,
            bytes_moved=_AT_ROOF_BYTES,
        )
        assert estimate.value == roofline.NEUTRAL_HEADROOM
        assert estimate.measured is False
        assert "B50" in estimate.reason
        assert "arc-b580" in estimate.reason  # the known presets are listed

    def test_no_device_is_neutral_rather_than_an_arbitrary_roof(self):
        estimate = roofline.estimate_headroom(
            time_us=_TIME_US, flops=_AT_ROOF_FLOPS, bytes_moved=_AT_ROOF_BYTES
        )
        assert estimate.measured is False
        assert estimate.value == roofline.NEUTRAL_HEADROOM

    def test_zero_duration_is_neutral_not_a_division_by_zero(self):
        estimate = roofline.estimate_headroom(
            time_us=0.0, device_name=_B580, flops=1.0, bytes_moved=1.0
        )
        assert estimate.measured is False
        assert estimate.value == roofline.NEUTRAL_HEADROOM

    def test_zero_counts_are_neutral(self):
        estimate = roofline.estimate_headroom(
            time_us=_TIME_US, device_name=_B580, flops=0.0, bytes_moved=0.0
        )
        assert estimate.measured is False


class TestHeadroomForKernel:
    def test_headroom_for_uses_the_per_call_duration(self):
        record = kernel(avg_us=_TIME_US)
        value = roofline.headroom_for(record, _B580, _AT_ROOF_FLOPS / 2, _AT_ROOF_BYTES / 2)
        assert value == pytest.approx(2.0, rel=1e-3)

    def test_avg_time_is_derived_when_only_totals_are_present(self):
        record = KernelRecord(id="k0", runtime_name="k", calls=4, total_time_us=4 * _TIME_US)
        value = roofline.headroom_for(record, _B580, _AT_ROOF_FLOPS, _AT_ROOF_BYTES)
        assert value == pytest.approx(1.0, rel=1e-3)

    def test_no_cost_data_yields_neutral(self):
        assert roofline.headroom_for(kernel(), _B580) == roofline.NEUTRAL_HEADROOM


class TestCatalogWiring:
    @pytest.fixture
    def events(self, decode_trace_path):
        return trace_mod.ingest_trace_file(decode_trace_path)

    def test_catalog_headroom_is_neutral_without_cost_data(self, events):
        catalog = build_catalog(events, run_id="t", minimum_detectable_effect=1.0)
        assert all(k.roofline_headroom == roofline.NEUTRAL_HEADROOM for k in catalog.kernels)

    def test_the_language_bias_guard_survives_the_roofline_change(self, events):
        """§11.10 regression guard, re-asserted with a device name threaded through.

        Mirrors `test_pipeline.py::TestCatalogRanking::test_language_bias_guard_holds`.
        Introducing the roofline term must not let the 24.7% Triton kernel overtake the
        40.8% oneDNN GEMM when no roofline data exists to justify it.
        """
        catalog = build_catalog(
            events, run_id="t", minimum_detectable_effect=1.0, device_name=_B580
        )
        gemm = next(k for k in catalog.kernels if k.provider is Provider.ONEDNN)
        triton = next(k for k in catalog.kernels if k.provider is Provider.INDUCTOR)
        assert catalog.kernels.index(gemm) < catalog.kernels.index(triton)

    def test_measured_headroom_reorders_and_is_reported(self, events):
        """A GEMM already at the roof genuinely has no room; the ranking should say so.

        This is the roofline term doing its job — distinct from the tractability term,
        which §11.10 caps precisely so it *cannot* do this.
        """
        gemm_name = "gemm_kernel_onednn_jit_bf16"
        rms_name = "triton_poi_fused_rms_norm_mul_0"
        costs = {
            # The GEMM saturates the B580: no headroom.
            gemm_name: roofline.KernelCost(
                flops_per_call=_AT_ROOF_FLOPS * 4.1, bytes_per_call=_AT_ROOF_BYTES * 4.1
            ),
            # The RMSNorm reaches an eighth of its ceiling.
            rms_name: roofline.KernelCost(
                flops_per_call=_AT_ROOF_FLOPS * 2.48 / 8,
                bytes_per_call=_AT_ROOF_BYTES * 2.48,
            ),
        }
        catalog = build_catalog(
            events,
            run_id="t",
            minimum_detectable_effect=1.0,
            device_name=_B580,
            kernel_costs=costs,
        )
        gemm = next(k for k in catalog.kernels if k.runtime_name == gemm_name)
        rms = next(k for k in catalog.kernels if k.runtime_name == rms_name)
        assert gemm.roofline_headroom == pytest.approx(1.0, rel=1e-2)
        assert rms.roofline_headroom > gemm.roofline_headroom
        assert catalog.kernels.index(rms) < catalog.kernels.index(gemm)

    def test_the_table_says_whether_headroom_was_measured(self, events):
        from xe_forge.orbit.analysis.catalog import format_catalog

        catalog = build_catalog(events, run_id="t", minimum_detectable_effect=1.0)
        assert "Roofline: headroom unmeasured for every kernel" in format_catalog(catalog)


class TestPresetDrift:
    """The copied constants must still agree with `scripts/roofline.py` (§18)."""

    @staticmethod
    def _script_presets() -> dict[str, tuple[float, float]]:
        source = (REPO_ROOT / "scripts" / "roofline.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.AnnAssign | ast.Assign) and "HARDWARE_PRESETS" in ast.dump(n)
        )
        mapping = node.value
        assert isinstance(mapping, ast.Dict)

        presets: dict[str, tuple[float, float]] = {}
        for key_node, value_node in zip(mapping.keys, mapping.values, strict=True):
            assert isinstance(key_node, ast.Constant)
            assert isinstance(value_node, ast.Call)
            fields: dict[str, float] = {}
            positional = [a for a in value_node.args[1:] if isinstance(a, ast.Constant)]
            for index, name in enumerate(("peak_tflops", "peak_bandwidth_gbps")):
                if index < len(positional):
                    fields[name] = float(positional[index].value)
            for keyword in value_node.keywords:
                if keyword.arg in {"peak_tflops", "peak_bandwidth_gbps"}:
                    assert isinstance(keyword.value, ast.Constant)
                    fields[keyword.arg] = float(keyword.value.value)
            presets[key_node.value] = (fields["peak_tflops"], fields["peak_bandwidth_gbps"])
        return presets

    def test_preset_keys_match_the_script(self):
        assert set(self._script_presets()) == set(roofline.HARDWARE_PRESETS)

    def test_preset_values_match_the_script(self):
        for key, (tflops, gbps) in self._script_presets().items():
            hardware = roofline.HARDWARE_PRESETS[key]
            assert hardware.peak_tflops == tflops, f"{key} compute roof drifted"
            assert hardware.peak_bandwidth_gbps == gbps, f"{key} bandwidth roof drifted"

    def test_there_is_no_invented_preset(self):
        """Only devices the standalone tool ships; no 'close enough' additions."""
        assert set(roofline.HARDWARE_PRESETS) == {
            "arc-pro-b70",
            "arc-b580",
            "max-1550",
            "max-1100",
            "flex-170",
        }
