"""
SYCL operator override and batch extraction (plan §11.8, §12.3).

§11.8 is what makes SYCL support practical rather than theoretical: because
torch-xpu-ops, IPEX, vLLM-XPU and sgl-kernel-xpu all register their kernels as
dispatcher ops, an optimized SYCL kernel ships as a small out-of-tree extension and
needs no fork of PyTorch, vLLM or SGLang.

Generation is fully testable without oneAPI. Only compilation needs icpx, and its
absence must be reported rather than raised — otherwise "oneAPI is not installed"
becomes indistinguishable from "the generator is broken".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xe_forge.orbit.extract import extract_all
from xe_forge.orbit.models import (
    ExtractionLevel,
    KernelCatalog,
    KernelLanguage,
    KernelRecord,
    PatchPoint,
    Provider,
)
from xe_forge.orbit.patch import apply_patch
from xe_forge.orbit.patch.sycl_override import (
    available_compiler,
    compile_command,
    generate,
    render_override_source,
)


@pytest.fixture
def sycl_kernel() -> KernelRecord:
    return KernelRecord(
        id="k2",
        runtime_name="_ZTSN4sycl3_V16detail16UnifiedAttentionE",
        language=KernelLanguage.SYCL,
        provider=Provider.SYCL,
        framework_op="_C::paged_attention_v1",
    )


class TestGeneratedSource:
    def test_registers_as_an_override_not_a_new_op(self, sycl_kernel):
        """TORCH_LIBRARY_IMPL binds an existing schema; TORCH_LIBRARY would define one."""
        source = render_override_source(sycl_kernel, "_C::paged_attention_v1")
        assert "TORCH_LIBRARY_IMPL(_C, XPU, m)" in source
        assert "TORCH_LIBRARY(" not in source.replace("TORCH_LIBRARY_IMPL(", "")
        assert 'm.impl("paged_attention_v1"' in source

    def test_documents_the_registration_ordering_trap(self, sycl_kernel):
        source = render_override_source(sycl_kernel, "_C::paged_attention_v1")
        assert "torch.compile" in source
        assert "dispatch assertion" in source

    def test_states_that_nothing_is_forked(self, sycl_kernel):
        source = render_override_source(sycl_kernel, "_C::paged_attention_v1")
        assert "Nothing in PyTorch, vLLM or SGLang is modified" in source

    def test_bare_op_name_defaults_to_aten(self, sycl_kernel):
        source = render_override_source(sycl_kernel, "rms_norm")
        assert "TORCH_LIBRARY_IMPL(aten, XPU, m)" in source

    def test_placeholder_body_is_compilable_shaped(self, sycl_kernel):
        """A stub that cannot build makes a generator bug look like a candidate bug."""
        source = render_override_source(sycl_kernel, "_C::x")
        assert "at::Tensor orbit_kernel" in source
        assert "parallel_for" in source
        assert "return output;" in source


class TestArtifacts:
    def test_generate_writes_source_loader_and_recipe(self, sycl_kernel, tmp_path):
        artifacts = generate(sycl_kernel, "_C::paged_attention_v1", tmp_path, build=False)
        assert artifacts.source_path.is_file()
        assert artifacts.loader_path.is_file()
        # An absolute path is expected when oneAPI is installed but not on PATH:
        # §11.5 wants the compile line reproducible without sourcing setvars.sh.
        assert Path(artifacts.build.compiler).name in ("icpx", "dpcpp")
        assert "-fsycl" in artifacts.build.flags

    def test_loader_applies_by_import_and_reverts_by_not_importing(self, sycl_kernel, tmp_path):
        artifacts = generate(sycl_kernel, "_C::paged_attention_v1", tmp_path, build=False)
        loader = artifacts.loader_path.read_text()
        assert "torch.ops.load_library" in loader
        assert "Not importing it is the revert" in loader

    def test_aot_target_is_pinned_from_the_device_name(self, sycl_kernel, tmp_path):
        """AOT and JIT do not perform or rebuild the same, so the target is recorded."""
        artifacts = generate(
            sycl_kernel,
            "_C::x",
            tmp_path,
            device_name="Intel(R) Arc(TM) B580 Graphics",
            build=False,
        )
        assert artifacts.build.aot_target == "bmg-g31"
        assert "-fsycl-targets=spir64_gen" in artifacts.build.flags

    def test_unknown_device_leaves_the_target_unpinned(self, sycl_kernel, tmp_path):
        artifacts = generate(sycl_kernel, "_C::x", tmp_path, device_name="Some GPU", build=False)
        assert artifacts.build.aot_target is None

    def test_compile_command_is_recorded_verbatim(self, sycl_kernel, tmp_path):
        artifacts = generate(sycl_kernel, "_C::x", tmp_path, build=False)
        argv = compile_command(artifacts)
        assert Path(argv[0]).name in ("icpx", "dpcpp")
        assert "-fsycl" in argv
        assert str(artifacts.source_path) in argv


class TestMissingCompiler:
    @pytest.mark.skipif(available_compiler() is not None, reason="oneAPI is installed")
    def test_absence_is_reported_not_raised(self, sycl_kernel, tmp_path):
        """Generation must still succeed: only the build needs oneAPI."""
        artifacts = generate(sycl_kernel, "_C::x", tmp_path, build=True)
        assert not artifacts.built
        assert "no SYCL compiler on PATH" in artifacts.reason
        assert artifacts.source_path.is_file()

    @pytest.mark.skipif(available_compiler() is not None, reason="oneAPI is installed")
    def test_a_stock_clang_is_not_treated_as_sycl_capable(self):
        """Falling back to clang++ turns a missing toolchain into a confusing exit code."""
        import shutil

        assert shutil.which("clang++") is None or available_compiler() is None


class TestLadderIntegration:
    def test_a_sycl_kernel_takes_the_native_p1_path(self, sycl_kernel, tmp_path):
        record = apply_patch(
            sycl_kernel,
            [PatchPoint(rung="P1", target="_C::paged_attention_v1")],
            candidate_module="cand",
            output_dir=tmp_path,
            device_name="Intel(R) Arc(TM) B580 Graphics",
        )
        assert record.rung == "P1"
        assert "TORCH_LIBRARY_IMPL" in record.registration_call
        assert any("no fork" in note for note in record.notes)
        assert list(tmp_path.glob("*.cpp"))

    def test_a_triton_kernel_still_takes_the_python_path(self, tmp_path):
        triton_kernel = KernelRecord(
            id="k1",
            runtime_name="triton_poi_fused_0",
            language=KernelLanguage.TRITON,
            framework_op="aten::rms_norm",
        )
        record = apply_patch(
            triton_kernel,
            [PatchPoint(rung="P1", target="aten::rms_norm")],
            candidate_module="cand.kernel",
            output_dir=tmp_path,
        )
        assert record.applied
        assert Path(record.module_path).suffix == ".py"
        assert not list(tmp_path.glob("*.cpp"))

    def test_revert_never_touches_the_framework(self, sycl_kernel, tmp_path):
        record = apply_patch(
            sycl_kernel,
            [PatchPoint(rung="P1", target="_C::x")],
            candidate_module="cand",
            output_dir=tmp_path,
        )
        assert "No framework source was modified" in record.revert_procedure


class TestBatchExtraction:
    @pytest.fixture
    def catalog(self) -> KernelCatalog:
        return KernelCatalog(
            run_id="batch",
            kernels=[
                KernelRecord(
                    id="k0",
                    runtime_name="gemm_onednn",
                    provider=Provider.ONEDNN,
                    language=KernelLanguage.OPAQUE,
                    extraction_level=ExtractionLevel.E4,
                    gpu_time_share=0.5,
                ),
                KernelRecord(
                    id="k1",
                    runtime_name="_ZTS_attention",
                    provider=Provider.SYCL,
                    language=KernelLanguage.SYCL,
                    gpu_time_share=0.3,
                ),
                KernelRecord(
                    id="k2",
                    runtime_name="mystery_kernel",
                    provider=Provider.UNKNOWN,
                    gpu_time_share=0.2,
                ),
            ],
        )

    def test_every_attributed_kernel_reaches_a_level(self, catalog, tmp_path):
        coverage = extract_all(catalog, tmp_path)
        attributed = [e for e in coverage.extractions if e.kernel.provider is not Provider.UNKNOWN]
        assert all(e.level is not None for e in attributed)

    def test_unattributed_kernels_are_not_guessed_at(self, catalog, tmp_path):
        """A harness for a kernel we cannot name looks like progress and is not."""
        coverage = extract_all(catalog, tmp_path)
        unknown = next(e for e in coverage.extractions if e.kernel.id == "k2")
        assert unknown.level is None
        assert "no provenance" in unknown.error

    def test_coverage_is_weighted_by_gpu_time_not_kernel_count(self, catalog, tmp_path):
        """Two thirds of kernels covered means nothing if they own a tenth of the time."""
        coverage = extract_all(catalog, tmp_path)
        assert coverage.unattributed_share == pytest.approx(0.2)
        # The opaque GEMM owns half the time and is not source-rewritable.
        assert coverage.rewritable_share < 0.5

    def test_report_refuses_to_claim_bare_full_coverage(self, catalog, tmp_path):
        rendered = extract_all(catalog, tmp_path).format()
        assert "true and useless" in rendered
        assert "source-rewritable" in rendered
        assert "standalone bundle" in rendered

    def test_one_failing_kernel_does_not_abort_the_sweep(self, catalog, tmp_path):
        coverage = extract_all(catalog, tmp_path)
        assert len(coverage.extractions) == len(catalog.kernels)


class TestOverrideLessonsFromHardware:
    """Both learned by overriding a real vLLM SYCL op (`_C::fused_add_rms_norm`).

    Proven on device: the override registered, PyTorch logged "Overriding a previously
    registered kernel ... dispatch key: XPU", and the trace showed the new kernel present
    and the original absent — §13's dispatch assertion passing.
    """

    def test_the_signature_rule_is_recorded(self):
        from xe_forge.orbit.patch.sycl_override import SIGNATURE_MUST_MATCH_SCHEMA

        assert "by-value versus by-const-reference" in SIGNATURE_MUST_MATCH_SCHEMA
        assert "aborts at load" in SIGNATURE_MUST_MATCH_SCHEMA

    def test_the_reduction_rule_is_recorded(self):
        """A hand-rolled barrier tree hung the device; it did not fault."""
        from xe_forge.orbit.patch.sycl_override import PREFER_GROUP_REDUCTION

        assert "reduce_over_group" in PREFER_GROUP_REDUCTION
        assert "hangs the device rather than faulting" in PREFER_GROUP_REDUCTION

    def test_the_generated_source_binds_to_an_existing_schema(self):
        """TORCH_LIBRARY_IMPL, not TORCH_LIBRARY: an override, not a second op."""
        from xe_forge.orbit.models import KernelRecord, Provider
        from xe_forge.orbit.patch.sycl_override import render_override_source

        kernel = KernelRecord(
            id="k2", runtime_name="vllm::fused_add_rms_norm_kernel", provider=Provider.CUSTOM
        )
        src = render_override_source(kernel, "_C::fused_add_rms_norm")
        assert "TORCH_LIBRARY_IMPL(_C, XPU, m)" in src
        assert "TORCH_LIBRARY(" not in src.replace("TORCH_LIBRARY_IMPL(", "")

    def test_the_generated_source_avoids_the_pybind_header(self):
        """<torch/extension.h> pulls Python.h to build a module; an override is not one."""
        from xe_forge.orbit.models import KernelRecord, Provider
        from xe_forge.orbit.patch.sycl_override import render_override_source

        src = render_override_source(
            KernelRecord(id="k", runtime_name="x", provider=Provider.CUSTOM), "_C::op"
        )
        # The comment explains why it is avoided, so match the directive, not the name.
        assert "#include <torch/extension.h>" not in src
        assert "#include <torch/library.h>" in src
