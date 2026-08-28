"""
Adapters, language backends, capture and executor (plan §10, §11, §7.5, §20).

The load-bearing assertions here are the ones that catch a *plausible* lie: an adapter
over-declaring a capability, a closure that silently misses a re-exported helper, a
captured tensor that quietly loses its stride pattern.
"""

from __future__ import annotations

import sys

import pytest

from xe_forge.orbit.adapters import (
    AdapterError,
    GenericTorchAdapter,
    describe_adapters,
    get_adapter,
    resolve_adapter,
)
from xe_forge.orbit.adapters.base import LoadSpec
from xe_forge.orbit.adapters.conformance import run_conformance
from xe_forge.orbit.executor import LocalExecutor
from xe_forge.orbit.languages import get_backend, resolve_backend
from xe_forge.orbit.models import KernelRecord, WorkloadSpec

torch = pytest.importorskip("torch", reason="capture tests need torch")


@pytest.fixture
def sleep_spec():
    return WorkloadSpec(
        command=[sys.executable, "-c", "import time; time.sleep(0.01)"],
        repetitions=5,
        warmup_iterations=1,
    )


class TestLocalExecutor:
    def test_successful_command(self):
        result = LocalExecutor().run([sys.executable, "-c", "print('hi')"])
        assert result.ok
        assert "hi" in result.stdout
        assert result.duration_s > 0

    def test_failing_command_is_reported_not_raised(self):
        result = LocalExecutor().run([sys.executable, "-c", "raise SystemExit(3)"])
        assert not result.ok
        assert result.returncode == 3

    def test_missing_binary_is_reported_cleanly(self):
        result = LocalExecutor().run(["definitely-not-a-real-binary-xyz"])
        assert not result.ok
        assert result.returncode == 127

    def test_timeout_is_flagged(self):
        result = LocalExecutor().run(
            [sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.3
        )
        assert result.timed_out
        assert not result.ok

    def test_env_overlay(self):
        result = LocalExecutor().run(
            [sys.executable, "-c", "import os; print(os.environ['ORBIT_TEST'])"],
            env={"ORBIT_TEST": "value"},
        )
        assert "value" in result.stdout

    def test_empty_command_is_rejected(self):
        with pytest.raises(ValueError):
            LocalExecutor().run([])


class TestAdapterRegistry:
    def test_generic_adapter_is_registered(self):
        assert "generic_torch" in describe_adapters()[0]["name"] or any(
            row["name"] == "generic_torch" for row in describe_adapters()
        )

    def test_unknown_adapter_fails_clearly(self):
        with pytest.raises(AdapterError, match="unknown framework adapter"):
            get_adapter("no_such_framework")

    def test_resolution_always_yields_an_adapter(self, sleep_spec):
        """An unfamiliar framework degrades to Tier 0 rather than failing (§10.2)."""
        adapter = resolve_adapter(sleep_spec, requested=None)
        assert adapter is not None
        assert adapter.tier >= 0

    def test_explicit_request_is_honoured(self, sleep_spec):
        assert resolve_adapter(sleep_spec, requested="generic_torch").name == "generic_torch"


class TestGenericTorchAdapter:
    def test_reports_only_what_it_declares(self, sleep_spec):
        """Capabilities are declared, never assumed (§10.4).

        Tier 0 must not surface a TTFT it cannot measure, even as a null field that a
        downstream stage might treat as real.
        """
        adapter = GenericTorchAdapter()
        handle = adapter.launch(sleep_spec, LocalExecutor())
        measurement = adapter.benchmark(handle, LoadSpec(repetitions=5))
        assert measurement.metrics_available == ["wall_time"]
        assert measurement.ttft_ms is None
        assert measurement.throughput is None

    def test_undeclared_capability_raises_rather_than_no_ops(self, sleep_spec):
        """A silent no-op here would produce a confident wrong measurement."""
        adapter = GenericTorchAdapter()
        handle = adapter.launch(sleep_spec, LocalExecutor())
        assert adapter.capabilities.can_reset_state is False
        with pytest.raises(AdapterError, match="can_reset_state"):
            adapter.reset_state(handle)

    def test_missing_single_layer_capability_blocks_e3(self):
        adapter = GenericTorchAdapter()
        with pytest.raises(AdapterError, match="single layer"):
            adapter.build_in_situ_harness(KernelRecord(id="k0", runtime_name="x"), None)

    def test_determinism_profile_names_what_it_cannot_pin(self):
        """Naming the non-pinnable sources is what lets §17 refuse to ACCEPT."""
        profile = GenericTorchAdapter().determinism_profile()
        assert profile.pinnable == set()
        assert "prefix_cache_reuse" in profile.non_pinnable

    def test_patch_point_offered_only_behind_a_registered_op(self):
        adapter = GenericTorchAdapter()
        assert adapter.patch_points(KernelRecord(id="k0", runtime_name="x")) == []
        with_op = KernelRecord(id="k1", runtime_name="x", framework_op="aten::rms_norm")
        points = adapter.patch_points(with_op)
        assert points and points[0].rung == "P1"

    def test_config_axes_apply_to_the_environment(self, sleep_spec):
        adapter = GenericTorchAdapter()
        updated = adapter.apply_config(sleep_spec, {"onednn_verbose": "1"})
        assert updated.env["DNNL_VERBOSE"] == "1"
        assert sleep_spec.env == {}  # original untouched

    def test_passes_conformance(self):
        """Includes the null test and the positive control (§10.7 items 5 and 6)."""
        report = run_conformance(GenericTorchAdapter(), repetitions=5)
        failures = [c.name for c in report.checks if not c.passed and not c.skipped]
        assert report.passed, f"conformance failures: {failures}"


class TestLanguageBackends:
    def test_backends_resolve_by_confidence(self):
        triton, t_score = resolve_backend("triton_poi_fused_add_0")
        sycl, s_score = resolve_backend("_ZTSN4sycl3_V16detail10KernelImplE")
        assert triton.name == "triton"
        assert sycl.name == "sycl"
        assert t_score > 0.5 and s_score > 0.5

    def test_unrecognized_name_claims_nothing(self):
        backend, score = resolve_backend("mystery_kernel_0000")
        assert backend is None or score == 0.0

    def test_cost_profiles_reflect_reality(self):
        """A SYCL rebuild is an order of magnitude costlier than a Triton JIT (§11.6)."""
        triton = get_backend("triton")
        sycl = get_backend("sycl")
        assert sycl.cost_profile.iteration_seconds > 10 * triton.cost_profile.iteration_seconds

    def test_sycl_offers_the_compiler_option_axes(self):
        names = {axis.name for axis in get_backend("sycl").option_axes()}
        assert {"grf_mode", "sub_group_size", "aot_target"} <= names

    def test_numerics_changing_flags_are_marked(self):
        """fast-math is not a free win; it is gated by the correctness rules (§11.7)."""
        axes = {a.name: a for a in get_backend("sycl").option_axes()}
        assert axes["fp_contract"].changes_numerics is True
        assert axes["grf_mode"].changes_numerics is False

    def test_sycl_ambiguous_name_is_low_confidence(self):
        backend = get_backend("sycl")
        unique = backend.resolve_source("_ZTS10SimpleKern")
        templated = backend.resolve_source("_ZTS7Kernel<float, 128, Layout<RowMajor>>")
        assert templated.confidence < unique.confidence

    def test_aot_target_mapping(self):
        from xe_forge.orbit.languages.sycl_backend import aot_target_for_device

        assert aot_target_for_device("Intel(R) Arc(TM) B580 Graphics") == "bmg-g31"
        assert aot_target_for_device("Intel(R) Data Center GPU Max 1550") == "pvc"
        assert aot_target_for_device("some unknown gpu") is None


class TestTritonClosure:
    """Closure must follow re-exports, and must downgrade rather than half-resolve."""

    def test_follows_helpers_across_modules(self, tmp_path):
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "helpers.py").write_text(
            "import triton\n\n@triton.jit\ndef helper(x):\n    return x + 1\n", encoding="utf-8"
        )
        (pkg / "main.py").write_text(
            "import triton\nfrom pkg.helpers import helper\n\n"
            "BLOCK = 128\n\n@triton.jit\ndef kernel(x):\n    return helper(x)\n",
            encoding="utf-8",
        )
        sys.path.insert(0, str(tmp_path))
        try:
            from xe_forge.orbit.models import SourceLocation

            backend = get_backend("triton")
            result = backend.resolve_closure(
                SourceLocation(file=str(pkg / "main.py"), symbol="kernel")
            )
            assert "helper" in result.helpers
            assert result.constants.get("BLOCK") == 128
        finally:
            sys.path.remove(str(tmp_path))

    def test_dynamic_import_forces_a_downgrade(self, tmp_path):
        """A partially resolved closure is worse than an honest E3 (§12.6)."""
        source = tmp_path / "dyn.py"
        source.write_text(
            "import importlib\n\ndef kernel(x):\n"
            "    mod = importlib.import_module('something')\n    return mod.f(x)\n",
            encoding="utf-8",
        )
        from xe_forge.orbit.models import SourceLocation

        result = get_backend("triton").resolve_closure(
            SourceLocation(file=str(source), symbol="kernel")
        )
        assert not result.complete
        assert any("dynamic import" in u for u in result.unresolved)

    def test_missing_source_is_unresolved_not_empty(self, tmp_path):
        from xe_forge.orbit.models import SourceLocation

        result = get_backend("triton").resolve_closure(
            SourceLocation(file=str(tmp_path / "nope.py"), symbol="k")
        )
        assert not result.complete


class TestCapture:
    def test_non_contiguous_layout_survives_the_round_trip(self, tmp_path):
        """The whole point of capture: a transposed view is not a contiguous copy.

        Regenerating this input from shape and dtype would produce a different memory
        access pattern, and the kernel would benchmark something the workload never runs.
        """
        from xe_forge.orbit.capture import capture_invocation, verify_roundtrip

        base = torch.randn(8, 16)
        transposed = base.t()
        assert not transposed.is_contiguous()

        invocation = capture_invocation("k0", {"x": transposed, "alpha": 0.5}, tmp_path / "cap")
        assert invocation.contiguous_map["x"] is False
        assert invocation.stride_map["x"] == list(transposed.stride())
        assert verify_roundtrip(invocation) == []

    def test_scalars_and_dtypes_are_preserved(self, tmp_path):
        from xe_forge.orbit.capture import capture_invocation, load_invocation

        invocation = capture_invocation(
            "k1",
            {"w": torch.ones(4, 4, dtype=torch.float32), "eps": 1e-5, "name": "rms"},
            tmp_path / "cap",
        )
        restored = load_invocation(invocation)
        assert restored["eps"] == 1e-5
        assert restored["name"] == "rms"
        assert restored["w"].shape == (4, 4)
        assert invocation.dtype_map["w"] == "float32"

    def test_data_dependencies_are_copied_not_regenerated(self, tmp_path):
        """Tuned-config JSON is data; regenerating it is how a kernel goes wrong (§12.8)."""
        from xe_forge.orbit.capture import capture_invocation

        dep = tmp_path / "tuned_configs.json"
        dep.write_text('{"cpu": {"BLOCK_N": 128}}', encoding="utf-8")
        invocation = capture_invocation(
            "k2", {"x": torch.ones(2, 2)}, tmp_path / "cap", data_deps=[dep]
        )
        assert len(invocation.data_deps) == 1
        from pathlib import Path

        assert Path(invocation.data_deps[0]).read_text() == dep.read_text()

    def test_missing_data_dependency_is_an_error(self, tmp_path):
        from xe_forge.orbit.capture import CaptureError, capture_invocation

        with pytest.raises(CaptureError, match="does not exist"):
            capture_invocation(
                "k3", {"x": torch.ones(2)}, tmp_path / "cap", data_deps=[tmp_path / "gone.json"]
            )

    def test_reference_output_is_saved(self, tmp_path):
        from xe_forge.orbit.capture import capture_invocation

        invocation = capture_invocation(
            "k4",
            {"x": torch.ones(2, 2)},
            tmp_path / "cap",
            reference_output=torch.full((2, 2), 3.0),
        )
        assert invocation.output_reference
        assert torch.load(invocation.output_reference, weights_only=False).mean().item() == 3.0

    def test_oversized_capture_is_refused(self, tmp_path):
        from xe_forge.orbit.capture import CaptureError, capture_invocation

        with pytest.raises(CaptureError, match="exceed"):
            capture_invocation(
                "k5", {"big": torch.ones(1024, 1024)}, tmp_path / "cap", max_bytes=1024
            )
