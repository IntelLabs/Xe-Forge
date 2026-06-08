"""Tests for the benchmark skill DSL dispatch (Triton vs SYCL).

Platform-independent: the SyclExecutor / KernelBenchExecutor GPU seams are
mocked, so no icpx or GPU is needed.
"""

import types
from argparse import Namespace

import numpy as np
import pytest

from xe_forge.skills import benchmark


class _FakeExecResult:
    def __init__(self, success=True, time_ms=0.5, tflops=10.0, error=""):
        self.success = success
        self.execution_time_ms = time_ms
        self.tflops = tflops
        self.error_message = error


class _FakeSyclComparison:
    def __init__(self, correct=True, opt_ms=0.5, tflops=10.0, msg="ok"):
        self.optimized_correct = correct
        self.optimized_time_ms = opt_ms
        self.optimized_tflops = tflops
        self.feedback_message = msg


def _write_gemm_spec(tmp_path):
    spec = tmp_path / "gemm.yaml"
    spec.write_text(
        "inputs:\n"
        "  A:\n"
        "    shape: [M, K]\n"
        "    dtype: bfloat16\n"
        "bench-xpu:\n"
        "  - params: [A]\n"
        "    dtype: bfloat16\n"
        "    dims: { M: 256, N: 256, K: 256 }\n"
        "    flop: '2*M*N*K'\n"
        "    rtol: 0.02\n"
        "    atol: 0.01\n"
    )
    return spec


def test_sycl_routes_to_sycl_executor(tmp_path, monkeypatch):
    """args.dsl == 'sycl' must use SyclExecutor, never KernelBenchExecutor."""
    spec = _write_gemm_spec(tmp_path)
    baseline = tmp_path / "gemm.cpp"
    baseline.write_text("#include <x>\nint main(){}\n")
    optimized = tmp_path / "t1.cpp"
    optimized.write_text("#include <x>\nint main(){}\n")

    captured = {}

    class FakeSyclExecutor:
        def __init__(self, *a, **k):
            captured["constructed"] = True

        def get_or_create_inputs(self, dims, seed=42, dtype=None):
            captured["dims"] = dims
            return str(tmp_path / "inputs")

        def compare_with_reference(self, **kwargs):
            captured["compare_kwargs"] = kwargs
            return _FakeSyclComparison(correct=True, opt_ms=0.4)

        def execute(self, **kwargs):
            captured.setdefault("execute_calls", []).append(kwargs)
            return _FakeExecResult(success=True, time_ms=0.8)

    # Fail loudly if the SYCL path ever touches KernelBenchExecutor.
    def _boom(*a, **k):
        raise AssertionError("SYCL path must not construct KernelBenchExecutor")

    monkeypatch.setattr("xe_forge.core.sycl_executor.SyclExecutor", FakeSyclExecutor, raising=True)
    monkeypatch.setattr("xe_forge.core.executor.KernelBenchExecutor", _boom, raising=True)
    # No golden reference file -> correctness skipped, exercises the execute() path.
    args = Namespace(
        dsl="sycl",
        baseline=str(baseline),
        optimized=str(optimized),
        spec=str(spec),
        variant="bench-xpu",
        baseline_us=None,
        device="xpu",
    )
    benchmark.run(args)
    assert captured.get("constructed")
    assert captured["dims"] == {"M": 256, "N": 256, "K": 256}


def test_sycl_baseline_us_skips_baseline_rerun(tmp_path, monkeypatch, capsys):
    """When --baseline-us is set, the baseline .cpp is NOT re-run."""
    spec = _write_gemm_spec(tmp_path)
    baseline = tmp_path / "gemm.cpp"
    baseline.write_text("#include <x>\nint main(){}\n")
    optimized = tmp_path / "t1.cpp"
    optimized.write_text("#include <x>\nint main(){}\n")
    # Provide a golden reference so compare_with_reference is taken.
    (tmp_path / "gemm_pytorch.py").write_text(
        "import torch, torch.nn as nn\n"
        "class Model(nn.Module):\n"
        "    def forward(self, A, B0):\n"
        "        return A.float() @ B0.float()\n"
    )

    execute_calls = []

    class FakeSyclExecutor:
        def __init__(self, *a, **k):
            pass

        def get_or_create_inputs(self, dims, seed=42, dtype=None):
            return str(tmp_path / "inputs")

        def compare_with_reference(self, **kwargs):
            return _FakeSyclComparison(correct=True, opt_ms=0.4)

        def execute(self, **kwargs):
            execute_calls.append(kwargs)
            return _FakeExecResult(success=True, time_ms=0.8)

    # Stub the golden computation so we don't need torch-on-bin roundtrip here.
    monkeypatch.setattr("xe_forge.core.sycl_executor.SyclExecutor", FakeSyclExecutor, raising=True)
    monkeypatch.setattr(
        benchmark,
        "_compute_golden",
        lambda *a, **k: np.zeros((256, 256), dtype=np.float32),
    )

    args = Namespace(
        dsl="sycl",
        baseline=str(baseline),
        optimized=str(optimized),
        spec=str(spec),
        variant="bench-xpu",
        baseline_us=123.0,
        device="xpu",
    )
    benchmark.run(args)
    out = capsys.readouterr().out
    # No execute() call at all (compare_with_reference handles the optimized run,
    # baseline rerun is skipped because baseline_us was supplied).
    assert execute_calls == []
    assert "Using cached baseline" in out
    assert "triton_us=" in out
    assert "Correctness: PASSED" in out


def test_triton_path_unchanged(tmp_path, monkeypatch, capsys):
    """args.dsl == 'triton' still uses KernelBenchExecutor.compare_kernels."""
    spec = _write_gemm_spec(tmp_path)
    baseline = tmp_path / "b.py"
    baseline.write_text("class Model: pass\n")
    optimized = tmp_path / "o.py"
    optimized.write_text("class Model: pass\n")

    called = {}

    class FakeKB:
        def __init__(self, *a, **k):
            called["constructed"] = True

        def compare_kernels(self, **kwargs):
            called["compared"] = True
            r = types.SimpleNamespace(
                optimized_correct=True,
                original_time_us=100.0,
                optimized_time_us=50.0,
                speedup=2.0,
                feedback_message="good",
            )
            return r

    monkeypatch.setattr("xe_forge.core.executor.KernelBenchExecutor", FakeKB, raising=True)
    args = Namespace(
        dsl="triton",
        baseline=str(baseline),
        optimized=str(optimized),
        spec=str(spec),
        variant="bench-xpu",
        baseline_us=None,
        device="xpu",
    )
    benchmark.run(args)
    out = capsys.readouterr().out
    assert called.get("compared")
    assert "Correctness: PASSED" in out
    assert "triton_us=50.00" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
