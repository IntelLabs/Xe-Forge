"""
Spec-driven bench harness for a dispatcher-registered op (plan §9.7, §11.9).

§9.7's open half: `sycl_override.py` builds the override that shadows an op, but
nothing let a dispatcher-registered op be *driven* — correctness and weighted
benchmarking through the same Model + YAML contract as a Python kernel. These tests
exercise that harness end to end without a GPU or a SYCL toolchain: a real CPU op is
registered through torch.library, the generated `kernel.py` is exec'd against it, and
the emitted directory is handed to the same `resolve_candidate` the optimizer uses.
"""

from __future__ import annotations

import ast

import pytest

from xe_forge.orbit.patch.sycl_harness import (
    KERNEL_FILE,
    REFERENCE_FILE,
    SPEC_FILE,
    emit_dispatcher_candidate,
    render_dispatcher_model,
)

# The harness module itself is stdlib-only; torch is needed to register the test op
# and to exec the generated kernel, so the whole file skips without it.
torch = pytest.importorskip("torch")

NAMESPACE = "orbit_test"
OP = "rms_norm"
QUALIFIED = f"{NAMESPACE}::{OP}"
EPS = 1e-6


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    squared = x * x
    mean = squared.mean(dim=-1, keepdim=True)
    return x / torch.sqrt(mean + eps) * weight


def _ensure_registered() -> None:
    """Register the test op once per process.

    torch.library registration is not idempotent within an interpreter — defining the
    same op twice raises — so repeated collection or reruns in one process must find
    the op already present and do nothing.
    """
    if hasattr(torch.ops.orbit_test, OP):
        return
    library = torch.library.Library(NAMESPACE, "DEF")
    library.define("rms_norm(Tensor x, Tensor weight, float eps) -> Tensor")
    impl = torch.library.Library(NAMESPACE, "IMPL")
    impl.impl(OP, _rms_norm, "CPU")
    # Keep the handles alive: a garbage-collected Library deregisters its ops.
    globals()["_LIBRARY"] = library
    globals()["_IMPL"] = impl


@pytest.fixture(autouse=True)
def registered_op() -> None:
    _ensure_registered()


def _model_class(source: str):
    """Exec the generated harness the way the executor would import it."""
    namespace: dict[str, object] = {}
    exec(compile(source, KERNEL_FILE, "exec"), namespace)
    return namespace["Model"]


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(4, 32, 64, generator=generator)
    weight = torch.randn(64, generator=generator)
    return x, weight


class TestRenderedModel:
    def test_named_forward_matches_the_op_called_directly(self):
        source = render_dispatcher_model(QUALIFIED, arg_names=["x", "weight"], fixed_args=[EPS])
        model = _model_class(source)()
        x, weight = _inputs()
        expected = torch.ops.orbit_test.rms_norm(x, weight, EPS)
        assert torch.equal(model(x, weight), expected)

    def test_varargs_forward_also_drives_the_op(self):
        """Without arg_names the harness passes tensors through positionally."""
        source = render_dispatcher_model(QUALIFIED)
        model = _model_class(source)()
        x, weight = _inputs()
        expected = torch.ops.orbit_test.rms_norm(x, weight, EPS)
        assert torch.equal(model(x, weight, EPS), expected)

    def test_unregistered_op_fails_at_construction_not_first_forward(self):
        """The failure must name the op and land in __init__, not surface as an
        AttributeError inside a measurement loop."""
        source = render_dispatcher_model("orbit_test::does_not_exist")
        model_class = _model_class(source)
        with pytest.raises(RuntimeError, match="orbit_test::does_not_exist"):
            model_class()
        with pytest.raises(RuntimeError, match="not registered"):
            model_class()

    def test_error_reports_the_loader_module_that_was_tried(self):
        """ "Op not registered" is only actionable when the error says what was tried."""
        source = render_dispatcher_model("orbit_test::does_not_exist", loader_module="math")
        with pytest.raises(RuntimeError, match="imported loader module 'math'"):
            _model_class(source)()

    def test_failed_loader_import_is_reported_not_masked(self):
        """A broken loader must not hide behind the generic not-registered message."""
        source = render_dispatcher_model(
            "orbit_test::does_not_exist", loader_module="orbit_no_such_loader_xyz"
        )
        with pytest.raises(RuntimeError, match=r"orbit_no_such_loader_xyz.*failed"):
            _model_class(source)()

    def test_library_path_plumbing_appears_in_the_source(self):
        source = render_dispatcher_model(QUALIFIED, library_path="/tmp/liboverride.so")
        assert "torch.ops.load_library" in source
        assert "/tmp/liboverride.so" in source

    def test_generated_source_imports_only_torch_and_stdlib(self):
        """The candidate travels to machines that have torch but not xe-orbit."""
        source = render_dispatcher_model(QUALIFIED, loader_module="mod", library_path="/x.so")
        imported: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        assert imported <= {"torch", "importlib"}

    def test_no_unfinished_markers_in_the_source(self):
        source = render_dispatcher_model(QUALIFIED, arg_names=["x"], fixed_args=[1])
        lowered = source.lower()
        for marker in ("todo", "fixme", "placeholder", "xxx"):
            assert marker not in lowered

    def test_bare_op_name_defaults_to_aten(self):
        """Same convention as render_override_source, so the two halves agree."""
        source = render_dispatcher_model("rms_norm")
        assert "NAMESPACE = 'aten'" in source

    def test_bad_arg_name_is_rejected_at_render_time(self):
        """A bad name would otherwise fail at exec, far from the call that caused it."""
        with pytest.raises(ValueError, match="not-an-identifier"):
            render_dispatcher_model(QUALIFIED, arg_names=["not-an-identifier"])

    def test_non_literal_fixed_arg_is_rejected(self):
        """Only repr-round-trippable values can be baked into a self-contained file."""
        with pytest.raises(ValueError, match="self-contained"):
            render_dispatcher_model(QUALIFIED, fixed_args=[object()])


class TestEmitDispatcherCandidate:
    def test_writes_the_layout_resolve_candidate_expects(self, tmp_path):
        """The emitted directory is the optimizer's input, so the optimizer's own
        resolver is the arbiter of the layout, not a parallel assertion here."""
        from xe_forge.orbit.optimize.kernel_dir import resolve_candidate

        emit_dispatcher_candidate(
            QUALIFIED,
            tmp_path,
            spec_source="inputs:\n  x:\n    shape: [4, 32, 64]\n    dtype: float32\n",
            reference_source="import torch\n\n\nclass Model(torch.nn.Module):\n"
            "    def forward(self, x, weight):\n"
            "        return x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6) * weight\n",
        )
        resolved = resolve_candidate(tmp_path)
        assert resolved["kernel"] == tmp_path / KERNEL_FILE
        assert resolved["spec"] == tmp_path / SPEC_FILE
        assert resolved["reference"] == tmp_path / REFERENCE_FILE

    def test_missing_reference_is_reported_not_fabricated(self, tmp_path):
        """A guessed reference passes a meaningless correctness gate; refusing to
        write one is the honest behaviour kernel_dir's notes rely on."""
        summary = emit_dispatcher_candidate(QUALIFIED, tmp_path, spec_source="inputs: {}\n")
        assert not (tmp_path / REFERENCE_FILE).exists()
        assert summary["reference_path"] is None
        assert any("correctness cannot be checked" in note for note in summary["notes"])

    def test_stub_reference_is_flagged(self, tmp_path):
        """A reference that raises instead of computing must not count as real."""
        stub = "class Model:\n    def forward(self, *a):\n        raise NotImplementedError\n"
        summary = emit_dispatcher_candidate(QUALIFIED, tmp_path, reference_source=stub)
        assert (tmp_path / REFERENCE_FILE).is_file()
        assert any("stub" in note for note in summary["notes"])

    def test_missing_spec_is_noted(self, tmp_path):
        summary = emit_dispatcher_candidate(QUALIFIED, tmp_path)
        assert not (tmp_path / SPEC_FILE).exists()
        assert summary["spec_path"] is None
        assert any(SPEC_FILE in note for note in summary["notes"])

    def test_emitted_kernel_runs_end_to_end(self, tmp_path):
        """Not just the rendered string: the file on disk, as the executor sees it."""
        summary = emit_dispatcher_candidate(
            QUALIFIED, tmp_path, arg_names=["x", "weight"], fixed_args=[EPS]
        )
        source = (tmp_path / KERNEL_FILE).read_text(encoding="utf-8")
        assert summary["kernel_path"] == str(tmp_path / KERNEL_FILE)
        model = _model_class(source)()
        x, weight = _inputs()
        assert torch.equal(model(x, weight), torch.ops.orbit_test.rms_norm(x, weight, EPS))
