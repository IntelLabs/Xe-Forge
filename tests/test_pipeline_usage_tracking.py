from types import SimpleNamespace

import pytest

from xe_forge.models import DSL, KernelAnalysis
from xe_forge.pipeline import XeForgePipeline


class DummyTracker:
    def __init__(self, tokens: dict):
        self._tokens = tokens

    def get_total_tokens(self) -> dict:
        return self._tokens


class DummyUsageContext:
    def __init__(self, tokens: dict):
        self.tracker = DummyTracker(tokens)
        self.exit_calls = 0
        self.exc_type = None

    def __enter__(self):
        return self.tracker

    def __exit__(self, exc_type, exc, tb):
        self.exit_calls += 1
        self.exc_type = exc_type
        return False


class StubAnalyzer:
    def __init__(self, *, raises: Exception | None = None):
        self._raises = raises

    def analyze(self, *args, **kwargs):
        if self._raises is not None:
            raise self._raises
        return KernelAnalysis(kernel_name="k", detected_issues=[])


class StubPlanner:
    def plan(self, **kwargs):
        return []


class StubCoordinator:
    def run(self, **kwargs):
        return kwargs["kernel_code"], 1.25, "ok", []


class StubOptimizer:
    def optimize_stage(self, **kwargs):
        raise AssertionError("optimize_stage should not be called in this test")


def _make_pipeline():
    pipeline = object.__new__(XeForgePipeline)
    pipeline.config = SimpleNamespace(
        optimization=SimpleNamespace(target_dtype=None, best_k=1),
        device_config=SimpleNamespace(device="xpu", dsl=DSL.TRITON),
        logging=SimpleNamespace(save_intermediate=False),
    )
    pipeline.executor = None
    pipeline.trial_manager = None
    pipeline.profiler = None
    pipeline.optimizer = StubOptimizer()
    pipeline.analyzer = StubAnalyzer()
    pipeline.planner = StubPlanner()
    pipeline.coordinator = None
    pipeline._save_results = lambda result: None
    pipeline._resolve_tolerances = lambda *args, **kwargs: (1e-3, 1e-3)
    return pipeline


def test_optimize_sets_token_usage_and_exits_tracker(monkeypatch):
    usage_ctx = DummyUsageContext({"prompt_tokens": 10, "completion_tokens": 2})
    monkeypatch.setattr("xe_forge.pipeline.dspy.track_usage", lambda: usage_ctx)
    monkeypatch.setattr("xe_forge.pipeline.get_device_config_for_pipeline", lambda **kwargs: {})

    pipeline = _make_pipeline()

    result = pipeline.optimize(
        kernel_code="def kernel():\n    return 1\n",
        reference_code="",
        kernel_name="k",
    )

    assert result.token_usage == {"prompt_tokens": 10, "completion_tokens": 2}
    assert usage_ctx.exit_calls == 1
    assert usage_ctx.exc_type is None


def test_optimize_exits_tracker_when_exception_raised(monkeypatch):
    usage_ctx = DummyUsageContext({"prompt_tokens": 0, "completion_tokens": 0})
    monkeypatch.setattr("xe_forge.pipeline.dspy.track_usage", lambda: usage_ctx)
    monkeypatch.setattr("xe_forge.pipeline.get_device_config_for_pipeline", lambda **kwargs: {})

    pipeline = _make_pipeline()
    pipeline.analyzer = StubAnalyzer(raises=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        pipeline.optimize(
            kernel_code="def kernel():\n    return 1\n",
            reference_code="",
            kernel_name="k",
        )

    assert usage_ctx.exit_calls == 1
    assert usage_ctx.exc_type is RuntimeError


def test_optimize_coordinator_path_also_exits_tracker(monkeypatch):
    usage_ctx = DummyUsageContext({"prompt_tokens": 7, "completion_tokens": 3})
    monkeypatch.setattr("xe_forge.pipeline.dspy.track_usage", lambda: usage_ctx)
    monkeypatch.setattr("xe_forge.pipeline.get_device_config_for_pipeline", lambda **kwargs: {})

    pipeline = _make_pipeline()
    pipeline.coordinator = StubCoordinator()

    result = pipeline.optimize(
        kernel_code="def kernel():\n    return 1\n",
        reference_code="",
        kernel_name="k",
    )

    assert result.token_usage == {"prompt_tokens": 7, "completion_tokens": 3}
    assert usage_ctx.exit_calls == 1
    assert usage_ctx.exc_type is None
