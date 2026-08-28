"""
Multi-framework SYCL source resolution (plan §11.2, §12.5).

§11.2's table says Intel's kernels live in torch-xpu-ops, vllm-xpu-kernels,
sgl-kernel-xpu and IPEX — all open-source SYCL registered as dispatcher ops. The
obstacle to extracting them was never the language; it was that each installs as a
wheel containing only a compiled shared object, so a resolver requiring
`compile_commands.json` reports E3 forever for want of a checkout.

These tests use synthetic trees so they run anywhere. The same code was exercised
against real checkouts during development, resolving genuine kernels from all three
frameworks — including one captured by unitrace from a live decode.
"""

from __future__ import annotations

import pytest

from xe_forge.orbit.languages.sources import (
    KNOWN_SOURCE_TREES,
    SourceRegistry,
    SourceTree,
    discover,
)
from xe_forge.orbit.languages.sycl_backend import identifying_symbols, index_source_tree


def _mangle(symbol: str) -> str:
    """Itanium typeinfo name for a symbol: `_ZTS<length><name>` plus template args.

    Built rather than hardcoded, because the length prefix is load-bearing — the
    resolver reads the count to slice the identifier out exactly, instead of guessing
    at where the template mangling starts.
    """
    return f"_ZTS{len(symbol)}{symbol}IfE"


@pytest.fixture
def fake_trees(tmp_path):
    """Minimal stand-ins for the three frameworks, with realistic symbol shapes."""
    torch_tree = tmp_path / "torch-xpu-ops" / "src"
    torch_tree.mkdir(parents=True)
    (torch_tree / "ActivationGeluKernel.cpp").write_text(
        "template <typename scalar_t>\nstruct GeluErfFunctor {\n"
        "  scalar_t operator()(scalar_t x) const { return x; }\n};\n",
        encoding="utf-8",
    )

    vllm_tree = tmp_path / "vllm-xpu-kernels" / "src"
    vllm_tree.mkdir(parents=True)
    (vllm_tree / "topk_softplus.cpp").write_text(
        "struct TopkGatingSoftplusSqrtKernel {\n  void run() {}\n};\n", encoding="utf-8"
    )

    sgl_tree = tmp_path / "sgl-kernel-xpu" / "src"
    sgl_tree.mkdir(parents=True)
    (sgl_tree / "BiasedTopK.cpp").write_text(
        "class BiasedTopkKernel {\n public:\n  void run() {}\n};\n", encoding="utf-8"
    )
    return tmp_path


class TestIdentifyingSymbols:
    def test_the_functor_beats_the_generic_wrapper(self):
        """The outer template names the launch mechanism, not the kernel.

        `VectorizedElementwiseKernel` is shared by hundreds of ATen XPU kernels. An
        earlier resolver stripped template arguments before matching, which discarded
        `GeluErfFunctor` — the only token that identifies the source file.
        """
        name = (
            "at::native::xpu::VectorizedElementwiseKernel<4, "
            "at::native::xpu::GeluErfFunctor<float>, "
            "at::detail::Array<char*, 2>, TrivialOffsetCalculator<1, unsigned int> >"
        )
        assert identifying_symbols(name)[0] == "GeluErfFunctor"

    def test_generic_wrappers_are_filtered_out(self):
        symbols = identifying_symbols(
            "at::native::xpu::VectorizedElementwiseKernel<4, at::detail::Array<char*, 2> >"
        )
        assert "VectorizedElementwiseKernel" not in symbols
        assert "Array" not in symbols

    def test_a_plain_kernel_name_survives(self):
        assert "BiasedTopkKernel" in identifying_symbols(_mangle("BiasedTopkKernel"))


class TestSourceIndex:
    def test_structs_and_classes_are_indexed(self, fake_trees):
        index = index_source_tree(fake_trees / "torch-xpu-ops" / "src")
        assert "GeluErfFunctor" in index
        assert index["GeluErfFunctor"].name == "ActivationGeluKernel.cpp"

    def test_an_empty_tree_indexes_to_nothing(self, tmp_path):
        assert index_source_tree(tmp_path) == {}


class TestRegistry:
    def test_all_known_frameworks_are_declared(self):
        """§11.2 names these; the registry must know where each one's kernels live."""
        assert {"torch-xpu-ops", "vllm-xpu-kernels", "sgl-kernel-xpu"} <= set(KNOWN_SOURCE_TREES)
        for meta in KNOWN_SOURCE_TREES.values():
            assert meta["repo"].startswith("https://")
            assert meta["provider"]
            assert meta["registers"]

    def test_discovery_indexes_every_present_tree(self, fake_trees):
        registry = discover([fake_trees])
        names = {t.name for t in registry.trees}
        assert names == {"torch-xpu-ops", "vllm-xpu-kernels", "sgl-kernel-xpu"}
        assert registry.total_symbols >= 3

    def test_an_absent_tree_is_reported_not_silently_dropped(self, fake_trees):
        """ "We have not been shown the source" is a different finding from
        "this kernel has no source", and only one is about the kernel (§12.5)."""
        registry = discover([fake_trees])
        assert "intel-extension-for-pytorch" in registry.missing
        rendered = registry.format()
        assert "absent" in rendered
        assert "not a kernel without source" in rendered

    @pytest.mark.parametrize(
        ("kernel", "expected_tree", "expected_file"),
        [
            (
                "at::native::xpu::VectorizedElementwiseKernel<4, "
                "at::native::xpu::GeluErfFunctor<float> >",
                "torch-xpu-ops",
                "ActivationGeluKernel.cpp",
            ),
            (_mangle("TopkGatingSoftplusSqrtKernel"), "vllm-xpu-kernels", "topk_softplus.cpp"),
            (_mangle("BiasedTopkKernel"), "sgl-kernel-xpu", "BiasedTopK.cpp"),
        ],
    )
    def test_kernels_resolve_across_frameworks(
        self, fake_trees, kernel, expected_tree, expected_file
    ):
        """One resolver, three frameworks — which is the §11.2 claim, executed."""
        registry = discover([fake_trees])
        found = registry.resolve(kernel)
        assert found is not None, f"{kernel} did not resolve"
        path, tree, _symbol = found
        assert tree.name == expected_tree
        assert path.name == expected_file

    def test_an_unknown_kernel_resolves_to_nothing(self, fake_trees):
        assert discover([fake_trees]).resolve(_mangle("SomethingNeverSeen")) is None

    def test_lookup_is_deterministic_across_trees(self, fake_trees):
        registry = discover([fake_trees])
        first = registry.resolve(_mangle("BiasedTopkKernel"))
        second = registry.resolve(_mangle("BiasedTopkKernel"))
        assert first[0] == second[0]


class TestBackendIntegration:
    def test_the_sycl_backend_uses_an_explicit_tree(self, fake_trees):
        from xe_forge.orbit.languages.sycl_backend import SyclBackend

        backend = SyclBackend(source_tree=fake_trees / "torch-xpu-ops" / "src")
        location = backend.resolve_source(
            "at::native::xpu::VectorizedElementwiseKernel<4, "
            "at::native::xpu::GeluErfFunctor<float> >"
        )
        assert location.file is not None
        assert location.file.endswith("ActivationGeluKernel.cpp")
        # The file is certain, so there is no confidence figure to report: an exact
        # index hit either matched or it did not. What remains uncertain is which
        # instantiation ran, and that is `candidates`, not a probability.
        from xe_forge.orbit.models import ResolutionMethod

        assert location.method is ResolutionMethod.SYMBOL_INDEX
        assert location.deterministic
        assert location.confidence is None
        assert location.describe_confidence() == "exact"

    def test_registry_absence_degrades_to_low_confidence_not_a_crash(self, tmp_path):
        from xe_forge.orbit.languages.sycl_backend import SyclBackend

        backend = SyclBackend(source_tree=tmp_path)
        location = backend.resolve_source(_mangle("BiasedTopkKernel"))
        assert location.file is None
        assert location.confidence < 0.8


class TestRegistryReporting:
    def test_format_lists_what_each_tree_registers(self, fake_trees):
        rendered = discover([fake_trees]).format()
        assert "aten::* on the XPU dispatch key" in rendered
        assert "torch.ops.sgl_kernel.*" in rendered

    def test_empty_registry_is_reportable(self):
        registry = SourceRegistry(trees=[], missing=list(KNOWN_SOURCE_TREES))
        assert "0 tree(s) indexed" in registry.format()

    def test_symbol_counts_are_reported(self, fake_trees):
        tree = discover([fake_trees]).trees[0]
        assert isinstance(tree, SourceTree)
        assert tree.symbol_count >= 1


class TestKnowledgeDrivenSources:
    """Kernel source locations are framework knowledge, so they live in YAML (§10.6).

    §10.6's target shape is that "most of a new framework is the YAML file", with code
    supplying only what genuinely needs code. Source-tree locations are knowledge about
    a framework, not logic, so hardcoding them in Python meant adding SGLang's kernels
    would have been a code change rather than a data one.
    """

    def test_trees_are_loaded_from_the_knowledge_base(self):
        from xe_forge.orbit.languages.sources import load_known_trees

        trees = load_known_trees()
        assert {"torch-xpu-ops", "vllm-xpu-kernels", "sgl-kernel-xpu"} <= set(trees)

    def test_each_tree_names_its_framework(self):
        from xe_forge.orbit.languages.sources import load_known_trees

        trees = load_known_trees()
        assert trees["torch-xpu-ops"]["framework"] == "pytorch-xpu"
        assert trees["sgl-kernel-xpu"]["framework"] == "sglang"
        assert trees["vllm-xpu-kernels"]["framework"] == "vllm"

    def test_every_declared_tree_is_usable(self):
        """A knowledge entry missing a repo or subdir cannot be acted on."""
        from xe_forge.orbit.languages.sources import load_known_trees

        for name, meta in load_known_trees().items():
            assert meta["repo"].startswith("https://"), name
            assert meta["subdir"], name
            assert meta["provider"], name

    def test_a_missing_knowledge_base_falls_back_rather_than_failing(self, tmp_path):
        """Orbit must work from a wheel with no knowledge directory on disk.

        A missing YAML should reduce what is known, never break resolution outright.
        """
        from xe_forge.orbit.languages.sources import load_known_trees

        trees = load_known_trees(tmp_path / "does-not-exist")
        assert trees, "fallback table should still name the §11.2 trees"
        assert "torch-xpu-ops" in trees

    def test_adding_a_framework_needs_no_code_change(self, tmp_path):
        """The §10.6 claim, executed: a new YAML file is enough to add a source tree."""
        import yaml

        from xe_forge.orbit.languages.sources import load_known_trees

        kb = tmp_path / "kb"
        kb.mkdir()
        (kb / "framework_madeup.yaml").write_text(
            yaml.safe_dump(
                {
                    "framework": "madeup",
                    "kernel_sources": [
                        {
                            "name": "madeup-kernels",
                            "repo": "https://github.com/example/madeup-kernels",
                            "subdir": "src",
                            "provider": "custom",
                            "registers": "torch.ops.madeup.*",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        trees = load_known_trees(kb)
        assert "madeup-kernels" in trees
        assert trees["madeup-kernels"]["framework"] == "madeup"


class TestCrossFrameworkDisambiguation:
    """Identifiers collide across frameworks; the namespace is what settles it.

    `rms_norm_kernel` is defined by both torch-xpu-ops and vllm-xpu-kernels. Scanning
    trees in registry order resolved `vllm::rms_norm_kernel` to torch-xpu-ops'
    `LayerNormKernels.cpp` — the wrong framework's kernel, picked by list position. The
    compile check caught it, but only because that particular file failed to build;
    a collision between two files that both compile would have been silent.
    """

    def _registry(self, tmp_path):
        from xe_forge.orbit.languages.sources import SourceRegistry, SourceTree

        a = tmp_path / "tree-a" / "layernorm.cpp"
        b = tmp_path / "tree-b" / "LayerNormKernels.cpp"
        for p in (a, b):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("// kernel\n", encoding="utf-8")
        return SourceRegistry(
            trees=[
                SourceTree(
                    name="torch-xpu-ops",
                    path=b.parent,
                    provider="sycl",
                    registers="aten::*",
                    framework="pytorch-xpu",
                    namespaces=["at::native::xpu"],
                    symbols={"rms_norm_kernel": b},
                ),
                SourceTree(
                    name="vllm-xpu-kernels",
                    path=a.parent,
                    provider="custom",
                    registers="torch.ops._C.*",
                    framework="vllm",
                    namespaces=["vllm"],
                    symbols={"rms_norm_kernel": a},
                ),
            ]
        )

    def test_the_namespace_picks_the_tree_not_the_registry_order(self, tmp_path):
        registry = self._registry(tmp_path)
        _path, tree = registry.lookup("rms_norm_kernel", "vllm::rms_norm_kernel<half, 2, 8, true>")
        assert tree.name == "vllm-xpu-kernels"

    def test_the_other_namespace_picks_the_other_tree(self, tmp_path):
        registry = self._registry(tmp_path)
        _path, tree = registry.lookup("rms_norm_kernel", "at::native::xpu::rms_norm_kernel<float>")
        assert tree.name == "torch-xpu-ops"

    def test_an_unclaimed_namespace_falls_back_rather_than_failing(self, tmp_path):
        """A symbol no tree claims still resolves; it just resolves ambiguously."""
        registry = self._registry(tmp_path)
        found = registry.lookup("rms_norm_kernel", "someone_else::rms_norm_kernel<float>")
        assert found is not None

    def test_a_unique_symbol_needs_no_namespace(self, tmp_path):
        registry = self._registry(tmp_path)
        registry.trees[0].symbols["OnlyInTorch"] = registry.trees[0].symbols["rms_norm_kernel"]
        _path, tree = registry.lookup("OnlyInTorch", "")
        assert tree.name == "torch-xpu-ops"

    def test_namespaces_come_from_the_knowledge_file(self):
        """Which namespaces a tree owns is knowledge about the framework (§10.6)."""
        from xe_forge.orbit.languages.sources import load_known_trees

        trees = load_known_trees()
        assert "vllm" in trees["vllm-xpu-kernels"]["namespaces"]
        assert "at::native::xpu" in trees["torch-xpu-ops"]["namespaces"]


class TestResolutionIsAuditable:
    """A path alone cannot be reviewed; how it was decided is part of the claim (§11.4).

    "We found `Indexing.cpp`" means something different depending on whether the build
    database said so, a symbol index matched exactly, or a model read the tree and
    formed an opinion. Rendering all three identically invites equal trust in them.
    """

    def test_a_deterministic_tier_reports_no_confidence(self):
        """An exact lookup either matched or it did not; 0.85 would be invented."""
        from xe_forge.orbit.models import ResolutionMethod, SourceLocation

        hit = SourceLocation(file="/x/Fill.cpp", method=ResolutionMethod.SYMBOL_INDEX)
        assert hit.deterministic
        assert hit.confidence is None
        assert hit.describe_confidence() == "exact"

    def test_the_build_graph_is_deterministic_too(self):
        from xe_forge.orbit.models import ResolutionMethod, SourceLocation

        assert SourceLocation(file="/x/Fill.cpp", method=ResolutionMethod.BUILD_GRAPH).deterministic

    def test_an_agent_answer_carries_a_real_estimate(self):
        """The agent genuinely estimated something, so a float is meaningful there."""
        from xe_forge.orbit.models import ResolutionMethod, SourceLocation

        answer = SourceLocation(file="/x/Fill.cpp", method=ResolutionMethod.AGENT, confidence=0.75)
        assert not answer.deterministic
        assert answer.describe_confidence() == "0.75"

    def test_an_exact_hit_does_not_sort_against_an_agent_estimate(self):
        """The category error this prevents: 0.85 exact vs 0.85 self-reported.

        The first is a lookup that matched. The second is a model's opinion about its
        own reliability. Comparing them as numbers treats those as the same kind of
        thing, which is how an agent's guess quietly outranks a fact.
        """
        from xe_forge.orbit.models import ResolutionMethod, SourceLocation

        exact = SourceLocation(file="/a.cpp", method=ResolutionMethod.SYMBOL_INDEX)
        guess = SourceLocation(file="/b.cpp", method=ResolutionMethod.AGENT, confidence=0.85)
        assert exact.confidence is None and guess.confidence == 0.85
        assert exact.deterministic and not guess.deterministic

    def test_an_unresolved_location_is_not_resolved_even_with_a_file(self):
        """Several candidates matched: the candidates are the finding, not a pick."""
        from xe_forge.orbit.models import ResolutionMethod, SourceLocation

        ambiguous = SourceLocation(
            method=ResolutionMethod.UNRESOLVED,
            confidence=0.4,
            candidates=["/a.cpp", "/b.cpp"],
        )
        assert not ambiguous.resolved
        assert len(ambiguous.candidates) == 2

    def test_an_override_keeps_what_it_replaced(self):
        """An agent that overrides a deterministic tier must leave a reversible trail."""
        from xe_forge.orbit.models import ResolutionMethod, SourceLocation

        revised = SourceLocation(
            file="/right.cpp",
            method=ResolutionMethod.AGENT,
            confidence=0.75,
            previous_file="/wrong.cpp",
            previous_method=ResolutionMethod.NAME_MATCH,
        )
        assert revised.previous_file == "/wrong.cpp"
        assert revised.previous_method is ResolutionMethod.NAME_MATCH

    def test_the_default_is_unresolved_not_a_silent_zero(self):
        from xe_forge.orbit.models import ResolutionMethod, SourceLocation

        blank = SourceLocation()
        assert blank.method is ResolutionMethod.UNRESOLVED
        assert not blank.resolved
        assert blank.describe_confidence() == "—"
