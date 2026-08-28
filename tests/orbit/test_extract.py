"""
Extraction, the bundle test rig and emission (plan §12, §8).

Extraction is the first of the three things the plan says decide this project, and the
failure mode it guards against is specific: a bundle that looks standalone, imports
cleanly, benchmarks fast, and is silently executing the installed package the whole
time. The isolated-import and mutation checks are what make that detectable.
"""

from __future__ import annotations

import json

import pytest

from xe_forge.orbit.emit import build_spec, coverage, emit_candidate, weighted_variants
from xe_forge.orbit.extract import Extractor, verify_bundle
from xe_forge.orbit.models import (
    CapturedInvocation,
    ExtractionLevel,
    KernelLanguage,
    KernelRecord,
    LaunchRecord,
    Provider,
    ShapeObservation,
)


def _launch(source, **overrides) -> LaunchRecord:
    payload = {
        "fq_name": "pkg.main:kernel",
        "source_file": str(source),
        "source_line": 1,
        "grid": [8],
        "num_warps": 4,
        "num_stages": 2,
        "constexprs": {"BLOCK": 128},
        "compiled_metadata": {"n_regs": 42, "n_spills": 0},
    }
    payload.update(overrides)
    return LaunchRecord(**payload)


@pytest.fixture
def multi_file_kernel(tmp_path):
    """A kernel whose helpers live in sibling modules, one reached via a re-export.

    This mirrors the shape real inference kernels have, and it is the shape a
    call-graph-only closure walk gets wrong.
    """
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from .unrelated import missing_helper\n", encoding="utf-8")
    (pkg / "unrelated.py").write_text("missing_helper = 1\n", encoding="utf-8")
    (pkg / "helpers_a.py").write_text("def scale(x):\n    return x * 2\n", encoding="utf-8")
    (pkg / "helpers_b.py").write_text("def clamp(x):\n    return x\n", encoding="utf-8")
    (pkg / "reexport.py").write_text(
        "from .helpers_a import scale as scale_alias\nfrom .helpers_b import clamp\n",
        encoding="utf-8",
    )
    (pkg / "main.py").write_text(
        "from .reexport import clamp, scale_alias\n\n"
        "BLOCK = 128\n\n"
        "def kernel(x):\n"
        "    return clamp(scale_alias(x))\n",
        encoding="utf-8",
    )
    return pkg / "main.py"


class TestClosureExtraction:
    def test_reaches_e2_and_follows_the_reexport_chain(self, multi_file_kernel, tmp_path):
        kernel = KernelRecord(
            id="k0",
            runtime_name="kernel",
            language=KernelLanguage.TRITON,
            extraction_level=ExtractionLevel.E2,
        )
        result = Extractor(tmp_path / "bundles").extract(kernel, launch=_launch(multi_file_kernel))

        assert result.level is ExtractionLevel.E2
        assert not result.downgraded
        names = {p.rsplit("/", 1)[-1] for p in result.bundle.closure}
        # The alias hop must bring both helper modules along, not just the re-exporter.
        assert {"main.py", "reexport.py", "helpers_a.py", "helpers_b.py"} <= names

    def test_bundle_passes_isolation_and_mutation(self, multi_file_kernel, tmp_path):
        """The two checks that make multi-file extraction trustworthy (§12.12)."""
        kernel = KernelRecord(id="k0", runtime_name="kernel", language=KernelLanguage.TRITON)
        result = Extractor(tmp_path / "bundles").extract(kernel, launch=_launch(multi_file_kernel))
        report = verify_bundle(result.bundle)

        by_name = {c.name: c for c in report.checks}
        assert by_name["isolated import"].passed, by_name["isolated import"].detail
        assert by_name["mutation check"].passed, by_name["mutation check"].detail
        assert report.passed

    def test_package_init_is_not_copied_verbatim(self, multi_file_kernel, tmp_path):
        """A package init importing modules outside the closure must not be carried.

        Copying it looks more faithful and makes the bundle depend on files it does not
        ship — which the isolated-import check would then fail on.
        """
        kernel = KernelRecord(id="k0", runtime_name="kernel", language=KernelLanguage.TRITON)
        Extractor(tmp_path / "bundles").extract(kernel, launch=_launch(multi_file_kernel))
        init = tmp_path / "bundles" / "k0" / "src" / "pkg" / "__init__.py"
        assert init.is_file()
        assert "missing_helper" not in init.read_text()

    def test_external_dependencies_are_not_treated_as_missing(self, tmp_path):
        """torch and triton are environment dependencies, not closure members."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "main.py").write_text(
            "import json\nimport torch\n\ndef kernel(x):\n    return torch.relu(x)\n",
            encoding="utf-8",
        )
        kernel = KernelRecord(id="k0", runtime_name="kernel", language=KernelLanguage.TRITON)
        result = Extractor(tmp_path / "bundles").extract(kernel, launch=_launch(pkg / "main.py"))
        # File-local closure is E1 by definition; the point is that importing torch and
        # json did not register as unresolved and force a downgrade.
        assert result.level is ExtractionLevel.E1
        assert not result.downgraded
        assert not result.reasons


class TestDowngrade:
    def test_dynamic_import_downgrades_to_e3_with_a_reason(self, tmp_path):
        """A partially resolved closure is worse than an honest E3 (§12.6)."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "main.py").write_text(
            "import importlib\n\ndef kernel(x):\n"
            "    return importlib.import_module('mystery').f(x)\n",
            encoding="utf-8",
        )
        kernel = KernelRecord(id="k0", runtime_name="kernel", language=KernelLanguage.TRITON)
        result = Extractor(tmp_path / "bundles").extract(kernel, launch=_launch(pkg / "main.py"))
        assert result.level is ExtractionLevel.E3
        assert result.downgraded
        assert any("dynamic import" in r for r in result.reasons)

    def test_unresolvable_source_downgrades_rather_than_failing(self, tmp_path):
        kernel = KernelRecord(id="k0", runtime_name="mystery_kernel_xyz")
        result = Extractor(tmp_path / "bundles").extract(kernel, launch=None)
        assert result.level is ExtractionLevel.E3
        assert result.bundle.downgrade_reason

    def test_e3_harness_declines_to_guess_the_framework_call(self, tmp_path):
        kernel = KernelRecord(id="k0", runtime_name="vllm_attn", framework_op="vllm::attn")
        Extractor(tmp_path / "bundles").extract(kernel, level="E3")
        harness = (tmp_path / "bundles" / "k0" / "harness.py").read_text()
        assert "NotImplementedError" in harness
        assert "get_example_inputs" in harness


class TestOpaqueExtraction:
    def test_opaque_kernel_yields_a_reproducer_not_a_failure(self, tmp_path):
        kernel = KernelRecord(
            id="k0",
            runtime_name="gemm_kernel_onednn_jit_bf16",
            provider=Provider.ONEDNN,
            language=KernelLanguage.OPAQUE,
            calls=256,
            total_time_us=4100.0,
            gpu_time_share=0.41,
        )
        result = Extractor(tmp_path / "bundles").extract(kernel)
        assert result.level is ExtractionLevel.E4

        text = (tmp_path / "bundles" / "k0" / "reproducer.txt").read_text()
        assert "onednn" in text
        assert "DNNL_VERBOSE" in text

    def test_opaque_bundle_reports_checks_as_skipped_not_passed(self, tmp_path):
        """Reporting PASS would imply a verification that never happened."""
        kernel = KernelRecord(
            id="k0",
            runtime_name="onednn_gemm",
            provider=Provider.ONEDNN,
            language=KernelLanguage.OPAQUE,
        )
        result = Extractor(tmp_path / "bundles").extract(kernel)
        report = verify_bundle(result.bundle)
        by_name = {c.name: c for c in report.checks}
        assert by_name["mutation check"].skipped
        assert by_name["isolated import"].skipped


class TestVerificationCatchesLies:
    def test_missing_launch_record_fails_verification(self, multi_file_kernel, tmp_path):
        """Without the intercepted launch, the specialization cannot be confirmed."""
        kernel = KernelRecord(id="k0", runtime_name="kernel", language=KernelLanguage.TRITON)
        result = Extractor(tmp_path / "bundles").extract(
            kernel, launch=_launch(multi_file_kernel, constexprs={}, grid=[])
        )
        report = verify_bundle(result.bundle)
        launch_check = next(c for c in report.checks if c.name == "launch-record match")
        assert not launch_check.passed
        assert "constexpr" in launch_check.detail

    def test_declared_but_absent_data_dep_fails(self, multi_file_kernel, tmp_path):
        kernel = KernelRecord(id="k0", runtime_name="kernel", language=KernelLanguage.TRITON)
        result = Extractor(tmp_path / "bundles").extract(kernel, launch=_launch(multi_file_kernel))
        result.bundle.data_deps = [str(tmp_path / "gone.json")]
        report = verify_bundle(result.bundle)
        dep_check = next(c for c in report.checks if c.name == "data-dependency check")
        assert not dep_check.passed

    def test_report_converts_to_the_bundle_verification_field(self, multi_file_kernel, tmp_path):
        kernel = KernelRecord(id="k0", runtime_name="kernel", language=KernelLanguage.TRITON)
        result = Extractor(tmp_path / "bundles").extract(kernel, launch=_launch(multi_file_kernel))
        check = verify_bundle(result.bundle).to_extraction_check()
        assert check.verified
        assert check.isolated_import is True
        assert check.mutation_detected is True


class TestDataDependencies:
    def test_data_deps_are_copied_into_the_bundle(self, multi_file_kernel, tmp_path):
        """Tuned-config JSON is copied as data, never regenerated (§12.8)."""
        dep = tmp_path / "tuned_configs.json"
        dep.write_text('{"xpu": {"BLOCK_N": 256}}', encoding="utf-8")
        inputs = CapturedInvocation(kernel_id="k0", data_deps=[str(dep)])

        kernel = KernelRecord(id="k0", runtime_name="kernel", language=KernelLanguage.TRITON)
        result = Extractor(tmp_path / "bundles").extract(
            kernel, launch=_launch(multi_file_kernel), inputs=inputs
        )
        assert len(result.bundle.data_deps) == 1
        copied = result.bundle.data_deps[0]
        assert json.loads(open(copied).read())["xpu"]["BLOCK_N"] == 256


class TestWeightedVariants:
    def test_distribution_becomes_normalized_weights(self):
        shapes = [
            ShapeObservation(dims={"M": 4096}, count=61),
            ShapeObservation(dims={"M": 2048}, count=23),
            ShapeObservation(dims={"M": 1024}, count=9),
        ]
        variants = weighted_variants(shapes)
        assert len(variants) == 3
        assert sum(v["weight"] for v in variants) == pytest.approx(1.0, abs=0.01)
        # Most frequent shape first, so bench-gpu is the dominant configuration.
        assert variants[0]["dims"]["M"] == 4096

    def test_tail_shapes_are_dropped_and_reported(self):
        shapes = [ShapeObservation(dims={"M": 4096}, count=100)]
        shapes += [ShapeObservation(dims={"M": i}, count=1) for i in range(20)]
        variants = weighted_variants(shapes)
        assert len(variants) < len(shapes)
        assert coverage(shapes, variants) < 1.0

    def test_empty_distribution_yields_no_variants(self):
        assert weighted_variants([]) == []

    def test_spec_emits_weight_and_tightened_tolerance(self):
        kernel = KernelRecord(
            id="k0",
            runtime_name="k",
            shapes=[ShapeObservation(dims={"M": 4096, "H": 8192}, count=10)],
        )
        spec = build_spec(kernel, tolerance=(1e-4, 1e-7))
        assert "bench-gpu" in spec
        entry = spec["bench-gpu"][0]
        assert entry["weight"] == 1.0
        assert entry["rtol"] == 1e-4
        assert entry["dims"]["M"] == 4096


class TestEmitCandidate:
    def test_writes_the_directory_xe_forge_consumes(self, multi_file_kernel, tmp_path):
        kernel = KernelRecord(
            id="k0",
            runtime_name="kernel",
            language=KernelLanguage.TRITON,
            shapes=[ShapeObservation(dims={"M": 4096}, count=10)],
        )
        result = Extractor(tmp_path / "bundles").extract(kernel, launch=_launch(multi_file_kernel))
        target = tmp_path / "candidates" / "k0"
        summary = emit_candidate(kernel, result.bundle, target)

        assert (target / "spec.yaml").is_file()
        # The reference is resolved by name substitution, so the filename is load-bearing.
        assert (target / "kernel_pytorch.py").is_file()
        assert summary["coverage"] == pytest.approx(1.0)

    def test_reference_stub_refuses_to_guess(self, multi_file_kernel, tmp_path):
        """A plausible but wrong reference passes correctness and is wrong in the model."""
        kernel = KernelRecord(id="k0", runtime_name="kernel", language=KernelLanguage.TRITON)
        result = Extractor(tmp_path / "bundles").extract(kernel, launch=_launch(multi_file_kernel))
        target = tmp_path / "candidates" / "k0"
        emit_candidate(kernel, result.bundle, target)
        assert "NotImplementedError" in (target / "kernel_pytorch.py").read_text()

    def test_spec_documents_the_weighted_objective(self, multi_file_kernel, tmp_path):
        kernel = KernelRecord(
            id="k0",
            runtime_name="kernel",
            language=KernelLanguage.TRITON,
            shapes=[ShapeObservation(dims={"M": 4096}, count=10)],
        )
        result = Extractor(tmp_path / "bundles").extract(kernel, launch=_launch(multi_file_kernel))
        target = tmp_path / "candidates" / "k0"
        emit_candidate(kernel, result.bundle, target)
        text = (target / "spec.yaml").read_text()
        assert "weight" in text
        # §9.1 landed: the header points at the weighted objective rather than
        # apologising for a loader that drops the key.
        assert "--objective weighted" in text


class TestNativeBundleVerification:
    """A C++ bundle must be checked with a compiler, not with Python's import machinery.

    The first version of the rig ran the Python checks against every bundle. On a SYCL
    bundle carrying `Indexing.cpp` that produced `ModuleNotFoundError: No module named
    'Indexing'`, reported as "closure is incomplete" — a verdict about Python module
    resolution, delivered on a C++ translation unit, that happened to look plausible.
    """

    def _native_bundle(self, tmp_path, name="Kernel.cpp", level=None):
        from xe_forge.orbit.models import (
            BuildRecipe,
            ExtractionCheck,
            ExtractionLevel,
            KernelBundle,
            KernelLanguage,
        )

        src = tmp_path / "src"
        src.mkdir(exist_ok=True)
        primary = src / name
        primary.write_text("int orbit_probe() { return 0; }\n", encoding="utf-8")
        return KernelBundle(
            kernel_id="k0",
            extraction_level=level or ExtractionLevel.E2,
            language=KernelLanguage.SYCL,
            entrypoint="OrbitFunctor",
            primary_source=str(primary),
            closure=[str(primary)],
            build=BuildRecipe(compiler="icpx", flags=["-fsycl", "-std=c++20"]),
            verification=ExtractionCheck(),
        )

    def test_a_cpp_bundle_is_not_checked_by_importing_it(self, tmp_path):
        """No failure may mention Python imports; the bundle is not Python."""
        from xe_forge.orbit.extract.verify import verify_bundle

        report = verify_bundle(self._native_bundle(tmp_path))
        names = {c.name for c in report.checks}
        assert "isolated import" not in names
        assert "isolated compile" in names
        joined = " ".join(c.detail for c in report.checks)
        assert "ModuleNotFoundError" not in joined
        assert "sys.path" not in joined

    def test_the_specialization_question_is_answered_from_the_type(self, tmp_path):
        """Grid and warps do not exist for SYCL; template arguments do (§11.5)."""
        from xe_forge.orbit.extract.verify import verify_bundle

        bundle = self._native_bundle(tmp_path)
        bundle.build.entry_symbol = "IndexFunctor<OpaqueType<8> >"
        bundle.build.instantiation = "<OpaqueType<8> >"
        report = verify_bundle(bundle)
        check = next(c for c in report.checks if c.name == "instantiation match")
        assert check.passed and not check.skipped

    def test_an_unrecorded_instantiation_fails_rather_than_passing_quietly(self, tmp_path):
        """Two kernels sharing an entry symbol are not the same kernel (§12.10).

        `IndexFunctor<OpaqueType<8>>` and `IndexFunctor<OpaqueType<4>>` appear in one
        real trace under the same symbol. A bundle that records only the symbol can be
        rebuilt as either, so a speedup measured on it may describe code the workload
        never ran.
        """
        from xe_forge.orbit.extract.verify import verify_bundle

        bundle = self._native_bundle(tmp_path)
        bundle.build.entry_symbol = "IndexFunctor<OpaqueType<8> >"
        bundle.build.instantiation = ""
        report = verify_bundle(bundle)
        check = next(c for c in report.checks if c.name == "instantiation match")
        assert not check.passed and not check.skipped

    def test_a_non_template_kernel_has_no_specialization_to_pin(self, tmp_path):
        from xe_forge.orbit.extract.verify import verify_bundle

        bundle = self._native_bundle(tmp_path)
        bundle.build.entry_symbol = "FillKernel"
        report = verify_bundle(bundle)
        check = next(c for c in report.checks if c.name == "instantiation match")
        assert check.skipped

    def test_an_e2_bundle_with_an_unproven_closure_is_not_verified(self, tmp_path):
        """At E2 the closure IS the claim, so skipping its check cannot count as a pass.

        Ten of fourteen bundles once reported verified at E2 on a skipped closure check,
        which is the precise shape of overclaim the rig exists to prevent.
        """
        from xe_forge.orbit.extract.verify import BundleCheck, BundleReport
        from xe_forge.orbit.models import ExtractionLevel

        report = BundleReport(kernel_id="k0", level=ExtractionLevel.E2)
        report.checks.append(BundleCheck("isolated compile", True, "no compiler", skipped=True))
        report.checks.append(BundleCheck("instantiation match", True, "pinned"))
        assert not report.passed, "a skipped closure check must not yield a verified E2"

        report.checks[0] = BundleCheck("isolated compile", True, "compiled standalone")
        assert report.passed

    def test_an_e4_bundle_may_still_pass_on_skips(self, tmp_path):
        """E4 has no source at all, so its skips are genuinely inapplicable checks."""
        from xe_forge.orbit.extract.verify import BundleCheck, BundleReport
        from xe_forge.orbit.models import ExtractionLevel

        report = BundleReport(kernel_id="k0", level=ExtractionLevel.E4)
        report.checks.append(BundleCheck("isolated import", True, "not applicable", skipped=True))
        report.checks.append(BundleCheck("reproducer present", True, "12 lines"))
        assert report.passed


class TestCompileFailureClassification:
    """Why a bundle failed to compile decides what the reader should do about it."""

    def test_version_skew_is_named_rather_than_called_a_broken_closure(self):
        """The real case: torch-xpu-ops at HEAD against an installed torch release.

        The clone uses `kBComplex32`, a ScalarType the installed headers do not define.
        Carrying more files cannot fix that — the remedy is checking out the matching
        revision — so calling it an incomplete closure would send the reader the wrong
        way entirely.
        """
        from xe_forge.orbit.extract.verify import _classify_compile_failure

        kind, detail = _classify_compile_failure(
            "FillKernel.cpp:43:7: error: use of undeclared identifier 'kBComplex32'"
        )
        assert kind == "skew"
        assert "kBComplex32" in detail

    def test_a_missing_header_is_a_broken_closure(self):
        from xe_forge.orbit.extract.verify import _classify_compile_failure

        kind, _ = _classify_compile_failure(
            "Indexing.cpp:12:10: fatal error: 'comm/xpu_aten.h' file not found"
        )
        assert kind == "closure"

    def test_anything_else_is_quoted_rather_than_characterized(self):
        from xe_forge.orbit.extract.verify import _classify_compile_failure

        kind, detail = _classify_compile_failure("Kernel.cpp:9:1: error: expected ';'")
        assert kind == "other"
        assert "expected ';'" in detail


class TestInstantiationParsing:
    """Recovering the template arguments the workload actually ran (§11.5 item 4)."""

    @pytest.mark.parametrize(
        ("demangled", "expected"),
        [
            ("at::native::xpu::FillFunctor<int>", "<int>"),
            (
                "at::native::xpu::IndexFunctor<at::native::xpu::OpaqueType<8> >",
                "<at::native::xpu::OpaqueType<8> >",
            ),
            ("FillKernel", ""),
            ("", ""),
        ],
    )
    def test_outermost_argument_list_is_recovered(self, demangled, expected):
        from xe_forge.orbit.extract.bundle import _instantiation_of

        assert _instantiation_of(demangled) == expected

    def test_a_truncated_symbol_yields_nothing_rather_than_a_guess(self):
        """Runtimes truncate long names; inventing the closing bracket would be a lie."""
        from xe_forge.orbit.extract.bundle import _instantiation_of

        assert _instantiation_of("VectorizedElementwiseKernel<4, FillFunctor<int>, at::deta") == ""

    def test_two_specializations_of_one_functor_are_distinguishable(self):
        """The exact pair that shares an entry symbol in the real vLLM trace."""
        from xe_forge.orbit.extract.bundle import _instantiation_of

        eight = _instantiation_of("xpu::IndexFunctor<xpu::OpaqueType<8> >")
        four = _instantiation_of("xpu::IndexFunctor<xpu::OpaqueType<4> >")
        assert eight and four and eight != four
