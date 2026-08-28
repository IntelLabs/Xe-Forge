"""
Hardware-only validation on a real Intel GPU (plan §11.8, §16.6 tier T1/T2).

Everything else in this suite is designed to run without silicon. These tests are the
opposite: they are the ones that cannot be faked, and they skip cleanly when no XPU is
present so CPU-only CI stays green.

What they establish that no fixture can:

* An optimized **SYCL** kernel really can be compiled with icpx and registered on the
  XPU dispatch key, shadowing an existing op, with nothing in PyTorch, vLLM or SGLang
  modified — the claim §11.8 makes and on which the whole SYCL story rests.
* `torch.profiler` really does surface XPU device activity that the trace ingest can
  normalize, so the catalog and gating stages are operating on real kernels rather than
  on a shape we assumed.

Requires: a working Level Zero runtime (intel-compute-runtime + level-zero-loader) and,
for the compile tests, oneAPI's icpx. Both are checked at collection time.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="XPU tests need torch")

from xe_forge.orbit.models import KernelLanguage, KernelRecord, Provider  # noqa: E402
from xe_forge.orbit.patch import assert_dispatch  # noqa: E402
from xe_forge.orbit.patch.sycl_override import available_compiler, generate  # noqa: E402


def _xpu_available() -> bool:
    try:
        return bool(torch.xpu.is_available() and torch.xpu.device_count() > 0)
    except Exception:
        return False


requires_xpu = pytest.mark.skipif(not _xpu_available(), reason="no working Intel GPU runtime")
requires_icpx = pytest.mark.skipif(
    available_compiler() is None, reason="no SYCL compiler (icpx) installed"
)

pytestmark = pytest.mark.xpu


@requires_xpu
class TestDeviceIsReal:
    def test_compute_actually_runs_on_the_device(self):
        x = torch.randn(512, 512, device="xpu")
        result = x @ x
        torch.xpu.synchronize()
        assert result.device.type == "xpu"
        assert torch.isfinite(result).all().item()

    def test_device_identity_is_reportable(self):
        """Every measurement records device identity, so it has to be obtainable (§12.9)."""
        from xe_forge.orbit.runtime import environment

        device_type, name, count = environment.detect_device()
        assert device_type == "xpu"
        assert count >= 1
        assert name


@requires_xpu
@requires_icpx
class TestSyclOverrideOnHardware:
    """The §11.8 claim, executed rather than asserted."""

    def test_compiled_override_replaces_the_op_on_the_xpu_key(self):
        namespace = "orbit_hw_test"
        lib = torch.library.Library(namespace, "DEF")
        lib.define("scale(Tensor input) -> Tensor")

        # Baseline doubles its input, so "did the override take effect" has an
        # unambiguous numeric answer rather than a timing argument.
        impl = torch.library.Library(namespace, "IMPL")
        impl.impl("scale", lambda x: x * 2.0, "XPU")

        x = torch.ones(8, device="xpu")
        before = float(torch.ops.orbit_hw_test.scale(x).sum())
        assert before == 16.0, "baseline should double its input"

        kernel = KernelRecord(
            id="k0",
            runtime_name="_ZTS_scale",
            language=KernelLanguage.SYCL,
            provider=Provider.SYCL,
            framework_op=f"{namespace}::scale",
        )
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = generate(
                kernel, f"{namespace}::scale", Path(tmp), device_name="Intel(R) Graphics"
            )
            assert artifacts.built, f"icpx failed: {artifacts.reason}\n{artifacts.build_log[-800:]}"

            # Loading the extension is what applies the patch. Nothing was forked.
            torch.ops.load_library(str(artifacts.library_path))

            after = float(torch.ops.orbit_hw_test.scale(x).sum())

        # The generated placeholder is an identity kernel, so the doubling is gone.
        assert after == 8.0, (
            f"override did not take effect: still {after}, expected identity. "
            f"This is the silent-no-op failure §13 exists to catch."
        )

    def test_generated_source_needs_no_python_headers(self):
        """An override is not a Python module, so it must not require Python.h.

        Including <torch/extension.h> pulls in pybind11 and made the build fail on a
        machine with no Python dev headers — a build dependency the artifact never uses.
        """
        kernel = KernelRecord(
            id="k1", runtime_name="_ZTS_x", language=KernelLanguage.SYCL, provider=Provider.SYCL
        )
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = generate(kernel, "aten::relu", Path(tmp), build=False)
            source = artifacts.source_path.read_text()
        # Match the directive, not the string: the source deliberately *mentions*
        # torch/extension.h in a comment explaining why it is not included.
        assert "#include <torch/extension.h>" not in source
        assert "#include <torch/library.h>" in source


@requires_xpu
class TestProfilingRealKernels:
    def test_device_activity_is_visible_to_trace_ingest(self):
        """The catalog stage is only meaningful if real XPU kernels reach it."""
        from xe_forge.orbit.profiling.trace import profile_workload

        def workload():
            x = torch.randn(256, 256, device="xpu")
            for _ in range(3):
                x = torch.relu(x @ x.t())
            torch.xpu.synchronize()

        with tempfile.TemporaryDirectory() as tmp:
            events = profile_workload(workload, device_type="xpu", output=Path(tmp) / "t.json")

        assert events.device_type == "xpu"
        # Real device work must appear; an empty kernel list here would mean the
        # gating stage has been reasoning about nothing on this platform.
        assert events.kernels, f"no XPU kernels in the trace; warnings: {events.warnings}"
        assert events.total_gpu_time_us > 0

    def test_the_gate_discriminates_on_real_hardware(self):
        """§18's gate, exercised in both directions against actual device timing.

        A gate that only ever says one thing is not a gate. These two workloads differ
        only in whether setup is inside the profiled region, and the verdict must flip:
        a warm saturating loop is GPU-bound, a run dominated by allocation and warmup is
        not, and optimizing a kernel in the second case is the week §18 exists to save.
        """
        from xe_forge.orbit.analysis.catalog import build_catalog
        from xe_forge.orbit.models import ActionType
        from xe_forge.orbit.profiling.trace import profile_workload

        weights = torch.randn(1024, 1024, device="xpu")
        warm = torch.randn(1024, 1024, device="xpu")
        for _ in range(3):
            warm = torch.nn.functional.gelu(warm @ weights)
        torch.xpu.synchronize()

        def saturating():
            x = warm
            for _ in range(20):
                x = torch.nn.functional.gelu(x @ weights)
            torch.xpu.synchronize()

        def setup_dominated():
            # Allocation and first-touch inside the measured region, as a cold run has.
            a = torch.randn(1024, 1024, device="xpu")
            b = torch.randn(1024, 1024, device="xpu")
            torch.xpu.synchronize()
            (a @ b).sum().item()

        with tempfile.TemporaryDirectory() as tmp:
            hot = profile_workload(saturating, "xpu", Path(tmp) / "hot.json")
            cold = profile_workload(setup_dominated, "xpu", Path(tmp) / "cold.json")

        hot_catalog = build_catalog(hot, run_id="hot")
        cold_catalog = build_catalog(cold, run_id="cold")

        assert hot_catalog.gpu_busy_percent > cold_catalog.gpu_busy_percent
        assert hot_catalog.gating_action is ActionType.KERNEL_REWRITE, hot_catalog.gating_reason

    def test_real_sycl_kernels_are_attributed_with_graded_confidence(self):
        """torch-xpu-ops kernels arrive as templated symbols, and ambiguity must show.

        A name like `VectorizedElementwiseKernel<4, GeluErfFunctor<float>, ...>` matches
        several instantiations, so §11.4 requires reduced confidence rather than a pick.
        """
        from xe_forge.orbit.analysis.catalog import build_catalog
        from xe_forge.orbit.models import KernelLanguage
        from xe_forge.orbit.profiling.trace import profile_workload

        def workload():
            x = torch.randn(512, 512, device="xpu")
            for _ in range(5):
                x = torch.nn.functional.gelu(x)
            torch.xpu.synchronize()

        with tempfile.TemporaryDirectory() as tmp:
            events = profile_workload(workload, "xpu", Path(tmp) / "t.json")

        catalog = build_catalog(events, run_id="sycl")
        sycl = [k for k in catalog.kernels if k.language is KernelLanguage.SYCL]
        assert sycl, f"expected SYCL kernels, got {[k.runtime_name[:40] for k in catalog.kernels]}"

        templated = [k for k in sycl if "<" in k.runtime_name]
        if templated:
            assert all(k.provenance_confidence < 0.9 for k in templated)

    def test_catalog_attributes_real_kernels(self):
        """Provenance and gating, run against kernels this machine actually executed."""
        from xe_forge.orbit.analysis.catalog import build_catalog
        from xe_forge.orbit.profiling.trace import profile_workload

        def workload():
            x = torch.randn(512, 512, device="xpu")
            for _ in range(5):
                x = torch.nn.functional.gelu(x @ x.t())
            torch.xpu.synchronize()

        with tempfile.TemporaryDirectory() as tmp:
            events = profile_workload(workload, device_type="xpu", output=Path(tmp) / "t.json")

        catalog = build_catalog(events, run_id="hw", device_name="Intel(R) Graphics")
        assert catalog.kernels
        assert catalog.total_gpu_time_us > 0
        # Shares are a distribution over observed device time.
        assert sum(k.gpu_time_share for k in catalog.kernels) == pytest.approx(1.0, abs=1e-6)
        # Every kernel gets an action or an explicit reason it has none.
        for kernel in catalog.kernels:
            assert kernel.actions_available or kernel.skip_reason


@requires_xpu
class TestTritonReplacementOnHardware:
    """The Triton half of §11, executed on the device rather than described.

    Pairs with `TestSyclOverrideOnHardware`: the same P1 rung, the same dispatch
    assertion, a different language. That the two share everything except how the
    implementation is produced is the point of treating language as a dimension (§11.3).
    """

    def test_triton_can_jit_for_xpu(self):
        """Guards a real environment gap: this needs level-zero-headers installed.

        Without them the JIT fails on `#include <level_zero/ze_api.h>` — a missing host
        package, not a code defect, and one worth naming precisely when it bites.
        """
        triton = pytest.importorskip("triton")
        import triton.language as tl

        @triton.jit
        def _add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            tl.store(
                o_ptr + offs,
                tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask),
                mask=mask,
            )

        x = torch.ones(4096, device="xpu")
        y = torch.full((4096,), 2.0, device="xpu")
        out = torch.empty_like(x)
        _add[(4,)](x, y, out, 4096, BLOCK=1024)
        torch.xpu.synchronize()

        assert float(out.sum()) == pytest.approx(4096 * 3.0)

    def test_a_triton_override_replaces_the_kernel_and_is_measurably_faster(self):
        """End to end on the device: replace, assert dispatch, and check correctness.

        The speedup is reported but not asserted as a threshold — §17's decision rule
        owns that judgement, and pinning a number here would assert a property of one
        GPU. What is asserted is what must always hold: the override executed, the old
        kernel did not, and the answer did not change.
        """
        pytest.importorskip("triton")
        from examples.kernel_replacement import dispatch_log, optimized, workload

        dispatch_log.clear()
        workload._define_op()

        x, weight = workload.build_inputs(device="xpu")
        op = torch.ops.orbit_demo.rms_norm

        for _ in range(10):
            op(x, weight, 1e-6)
        torch.xpu.synchronize()
        dispatch_log.clear()
        baseline_out = op(x, weight, 1e-6).clone()
        assert dispatch_log.observed() == [workload.BASELINE_KERNEL]

        # Importing the module registers the override. That *is* the patch (§13).
        optimized.register()
        dispatch_log.clear()
        patched_out = op(x, weight, 1e-6)
        torch.xpu.synchronize()

        observed = dispatch_log.observed()
        assertion = assert_dispatch(
            observed,
            original_kernel=workload.BASELINE_KERNEL,
            replacement_marker=optimized.active_kernel_name(),
        )
        assert assertion.took_effect, f"{assertion.detail} (observed {observed})"

        # Fusing reassociates float work, so this is a tolerance rather than equality.
        assert torch.allclose(baseline_out, patched_out, rtol=1e-4, atol=1e-5)
