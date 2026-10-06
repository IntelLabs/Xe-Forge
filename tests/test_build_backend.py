"""How a ``module:attr`` build-backend reference becomes a backend.

A runtime-checkable Protocol accepts the class object itself -- it has ``name``
and ``build`` -- so a reference to a class has to be instantiated before the
structural check, or ``build`` is called unbound on the first kernel.
"""

from __future__ import annotations

import pytest

from xe_forge.core.build_backend import BuildError, resolve_build_backend


class ClassBackend:
    name = "class-backend"

    def build(self, source, spec, device):
        return (self, source)


def class_factory():
    return ClassBackend()


INSTANCE = ClassBackend()
NOT_A_BACKEND = object()


@pytest.mark.parametrize("attr", ["ClassBackend", "class_factory", "INSTANCE"])
def test_reference_resolves_to_an_instance(attr):
    backend = resolve_build_backend(f"{__name__}:{attr}")
    assert isinstance(backend, ClassBackend)
    assert backend.build("src", None, "xpu") == (backend, "src")


def test_reference_to_something_else_is_refused():
    with pytest.raises(BuildError, match="does not implement"):
        resolve_build_backend(f"{__name__}:NOT_A_BACKEND")


def test_the_executor_factory_forwards_the_configured_backend():
    """``--build-backend`` reaches the SYCL executor the optimize pipeline builds."""
    from xe_forge.config import Config
    from xe_forge.core import create_executor_from_config

    config = Config()
    config.device_config.dsl = "sycl"
    config.engine.build_backend = f"{__name__}:ClassBackend"
    executor = create_executor_from_config(config)
    assert isinstance(executor._backend, ClassBackend)


def _executor(backend):
    from xe_forge.core.sycl_executor import SyclExecutor

    return SyclExecutor(device_target="bmg", build_backend=backend)


def test_a_backend_rejects_raw_arguments_rather_than_dropping_them():
    class NeverBuilt(ClassBackend):
        def build(self, source, spec, device):
            raise AssertionError("built despite args_str")

    result = _executor(NeverBuilt()).execute_raw(kernel_code="k", args_str="--x 1")
    assert not result.success
    assert "args_str" in result.error_message


@pytest.mark.parametrize(
    ("baseline", "candidate", "correct"),
    [(None, True, True), (True, True, True), (False, True, False), (True, None, False)],
)
def test_an_explicit_baseline_failure_disqualifies_the_comparison(
    tmp_path, monkeypatch, baseline, candidate, correct
):
    from xe_forge.core.sycl_executor import ExecutionResult

    executor = _executor(ClassBackend())
    verdicts = {"original_sycl": baseline, "optimized_sycl": candidate}
    monkeypatch.setattr(
        executor,
        "execute",
        lambda output_name, **_: ExecutionResult(
            success=True, execution_time_ms=1.0, output_correct=verdicts[output_name]
        ),
    )
    result = executor.compare_kernels("a", "b", input_dir=str(tmp_path))
    assert result.optimized_correct is correct
    if baseline is False:
        assert "baseline failed" in result.feedback_message
