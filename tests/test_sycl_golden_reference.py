"""Tests for SyclExecutor.compare_with_reference (golden-reference path).

The icpx/GPU seam is mocked at SyclExecutor.execute: instead of compiling and
running a kernel, the fake writes a known D2.bin so the Python-side comparison,
reshape, and tolerance logic can be exercised on any platform.
"""

import os
import re

import numpy as np
import pytest
import torch

from xe_forge.core.sycl_executor import SyclExecutor, _save_tensor
from xe_forge.models import ExecutionResult


def _make_executor():
    # device_target="" avoids torch.xpu auto-detection during construction.
    return SyclExecutor(device_target="", verify=False)


def test_bf16_bin_roundtrip_bit_exact(tmp_path):
    """_save_tensor (bf16-as-int16) round-trips bit-exactly via torch view."""
    t = torch.randn(8, 16, dtype=torch.bfloat16)
    path = str(tmp_path / "A.bin")
    _save_tensor(t, path)
    raw = np.fromfile(path, dtype=np.int16)
    back = torch.from_numpy(raw).view(torch.bfloat16).reshape(8, 16)
    assert torch.equal(t, back)


def test_compare_with_reference_passed(tmp_path, monkeypatch):
    ex = _make_executor()
    golden = np.arange(256, dtype=np.float32).reshape(16, 16)

    def fake_execute(**kwargs):
        out_dir = kwargs["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        # Kernel produces exactly the golden values (flat f32).
        golden.astype(np.float32).tofile(os.path.join(out_dir, "D2.bin"))
        return ExecutionResult(success=True, execution_time_ms=0.3, tflops=12.0)

    monkeypatch.setattr(ex, "execute", fake_execute)
    monkeypatch.setattr(ex, "get_or_create_inputs", lambda *a, **k: str(tmp_path / "in"))

    res = ex.compare_with_reference(
        golden_output=golden,
        optimized_path="dummy.cpp",
        dims={"M": 16, "N": 16, "K": 16},
        rtol=1e-2,
        atol=1e-2,
        input_dir=str(tmp_path / "in"),
    )
    assert res.optimized_correct is True
    assert res.optimized_time_ms == 0.3
    assert res.optimized_tflops == 12.0
    assert "PASSED" in res.feedback_message


def test_compare_with_reference_failed(tmp_path, monkeypatch):
    ex = _make_executor()
    golden = np.zeros((16, 16), dtype=np.float32)

    def fake_execute(**kwargs):
        out_dir = kwargs["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        # Wrong output (all ones) -> must fail.
        np.ones((16, 16), dtype=np.float32).tofile(os.path.join(out_dir, "D2.bin"))
        return ExecutionResult(success=True, execution_time_ms=0.3, tflops=12.0)

    monkeypatch.setattr(ex, "execute", fake_execute)

    res = ex.compare_with_reference(
        golden_output=golden,
        optimized_path="dummy.cpp",
        dims={"M": 16, "N": 16, "K": 16},
        rtol=1e-2,
        atol=1e-2,
        input_dir=str(tmp_path / "in"),
    )
    assert res.optimized_correct is False
    assert "CORRECTNESS FAILURE" in res.feedback_message


def test_compare_with_reference_compile_failure_surfaces(tmp_path, monkeypatch):
    ex = _make_executor()
    golden = np.zeros((16, 16), dtype=np.float32)

    def fake_execute(**kwargs):
        return ExecutionResult(success=False, error_message="Compilation failed: boom")

    monkeypatch.setattr(ex, "execute", fake_execute)

    res = ex.compare_with_reference(
        golden_output=golden,
        optimized_path="dummy.cpp",
        dims={"M": 16, "N": 16, "K": 16},
        input_dir=str(tmp_path / "in"),
    )
    assert res.optimized_correct is False
    assert "boom" in res.feedback_message


def test_compare_with_reference_missing_d2(tmp_path, monkeypatch):
    ex = _make_executor()
    golden = np.zeros((16, 16), dtype=np.float32)

    def fake_execute(**kwargs):
        # Success but no D2.bin written — kernel ignored the IO contract.
        os.makedirs(kwargs["output_dir"], exist_ok=True)
        return ExecutionResult(success=True, execution_time_ms=0.3, tflops=12.0)

    monkeypatch.setattr(ex, "execute", fake_execute)

    res = ex.compare_with_reference(
        golden_output=golden,
        optimized_path="dummy.cpp",
        dims={"M": 16, "N": 16, "K": 16},
        input_dir=str(tmp_path / "in"),
    )
    assert res.optimized_correct is False
    assert "D2.bin" in res.feedback_message


def test_printed_format_matches_trial_parser(tmp_path, monkeypatch, capsys):
    """The benchmark skill's SYCL output must match the trial --triton-us regex."""
    from argparse import Namespace

    from xe_forge.skills import benchmark

    spec = tmp_path / "gemm.yaml"
    spec.write_text(
        "inputs:\n  A:\n    shape: [M, K]\n    dtype: bfloat16\n"
        "bench-xpu:\n  - params: [A]\n    dtype: bfloat16\n"
        "    dims: { M: 16, N: 16, K: 16 }\n    flop: '2*M*N*K'\n"
    )
    baseline = tmp_path / "gemm.cpp"
    baseline.write_text("#include <x>\nint main(){}\n")
    optimized = tmp_path / "t1.cpp"
    optimized.write_text("#include <x>\nint main(){}\n")

    class FakeSyclExecutor:
        def __init__(self, *a, **k):
            pass

        def get_or_create_inputs(self, dims, seed=42, dtype=None):
            return str(tmp_path / "in")

        def compare_with_reference(self, **kwargs):
            from xe_forge.core.sycl_executor import SyclComparisonResult

            return SyclComparisonResult(
                original_time_ms=float("inf"),
                optimized_time_ms=0.5,
                speedup=0.0,
                optimized_tflops=10.0,
                optimized_correct=True,
                feedback_message="ok",
            )

        def execute(self, **kwargs):
            return ExecutionResult(success=True, execution_time_ms=1.0, tflops=5.0)

    monkeypatch.setattr("xe_forge.core.sycl_executor.SyclExecutor", FakeSyclExecutor, raising=True)
    monkeypatch.setattr(
        benchmark,
        "_compute_golden",
        lambda *a, **k: np.zeros((16, 16), dtype=np.float32),
    )

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
    out = capsys.readouterr().out
    # Same regex tool-runner / trial result parsing relies on.
    m = re.search(r"baseline_us=([0-9.]+), triton_us=([0-9.]+), speedup=([0-9.]+)x", out)
    assert m, f"perf line did not match expected format:\n{out}"
    assert "Correctness: PASSED" in out
    # No gemm_pytorch.py here -> the no-golden execute() path, which carries the
    # kernel's parsed tflops (5.0; util = 5/160 = 3.1%). Confirms TFLOPS + util
    # are appended after speedup on that path too.
    full = re.search(
        r"baseline_us=[0-9.]+, triton_us=[0-9.]+, speedup=[0-9.]+x, "
        r"tflops=([0-9.]+), util=([0-9.]+)%",
        out,
    )
    assert full, f"perf line missing tflops/util:\n{out}"
    assert full.group(1) == "5.00"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
