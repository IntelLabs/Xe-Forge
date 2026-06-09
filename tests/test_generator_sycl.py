"""Tests for DSL-aware Claude workspace generation (SYCL path).

Platform-independent: no icpx/GPU involved — only template rendering and file
layout are exercised.
"""

from pathlib import Path

import pytest

from xe_forge.claude.generator import generate_workspace
from xe_forge.config import Config, XPUConfig

REFERENCE_PY = """\
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, A, B0):
        return A.float() @ B0.float()
"""

SYCL_CPP = """\
#include "cutlass/gemm/device/gemm_universal.h"
int main(int argc, const char** argv) {
    // reads input_dir A.bin/B0.bin, writes output_dir D2.bin
    return 0;
}
"""

TRITON_PY = """\
import triton
import triton.language as tl


class Model:
    pass
"""


def _sycl_config(workspace: Path) -> Config:
    cfg = Config()
    cfg.device_config = XPUConfig(device="xpu", dsl="sycl")
    cfg.engine.git_init = False
    cfg.engine.workspace = str(workspace)
    cfg.trial.max_trials = 5
    return cfg


def _triton_config(workspace: Path) -> Config:
    cfg = Config()
    cfg.device_config = XPUConfig(device="xpu", dsl="triton")
    cfg.engine.git_init = False
    cfg.trial.max_trials = 5
    return cfg


def test_sycl_writes_cpp_kernel(tmp_path):
    ws = tmp_path / "ws"
    generate_workspace(
        workspace=ws,
        config=_sycl_config(ws),
        kernel_name="gemm",
        kernel_code=SYCL_CPP,
        reference_code=REFERENCE_PY,
    )
    assert (ws / "test_kernels" / "gemm.cpp").exists()
    assert not (ws / "test_kernels" / "gemm.py").exists()
    # PyTorch golden reference written alongside the .cpp.
    assert (ws / "test_kernels" / "gemm_pytorch.py").read_text() == REFERENCE_PY
    # Existing .cpp with #include is kept verbatim (not replaced by the stub).
    assert (ws / "test_kernels" / "gemm.cpp").read_text() == SYCL_CPP


def test_sycl_starter_stub_when_no_include(tmp_path):
    ws = tmp_path / "ws"
    # PyTorch-only input (no #include) -> starter stub is substituted.
    generate_workspace(
        workspace=ws,
        config=_sycl_config(ws),
        kernel_name="gemm",
        kernel_code=REFERENCE_PY,
        reference_code=REFERENCE_PY,
    )
    cpp = (ws / "test_kernels" / "gemm.cpp").read_text()
    assert "#include" in cpp
    assert "int main(" in cpp
    assert "input_dir" in cpp and "output_dir" in cpp and "D2.bin" in cpp
    assert "cutlass" in cpp.lower()


def test_sycl_claude_md_content(tmp_path):
    ws = tmp_path / "ws"
    generate_workspace(
        workspace=ws,
        config=_sycl_config(ws),
        kernel_name="gemm",
        kernel_code=SYCL_CPP,
        reference_code=REFERENCE_PY,
    )
    claude = (ws / "CLAUDE.md").read_text()
    # SYCL-specific contract markers present.
    for token in [
        "input_dir",
        "output_dir",
        "D2.bin",
        "--dsl sycl",
        "bench-xpu",
        "knowledge_base/sycl/xpu",
    ]:
        assert token in claude, f"missing {token!r} in SYCL CLAUDE.md"
    # Triton-isms absent.
    for token in ["@triton.autotune", "GROUP_SIZE_M", "xe-forge-skill analyze"]:
        assert token not in claude, f"unexpected Triton token {token!r} in SYCL CLAUDE.md"


def test_sycl_optimize_command_uses_dsl_flag(tmp_path):
    ws = tmp_path / "ws"
    generate_workspace(
        workspace=ws,
        config=_sycl_config(ws),
        kernel_name="gemm",
        kernel_code=SYCL_CPP,
        reference_code=REFERENCE_PY,
    )
    cmd = (ws / ".claude" / "commands" / "optimize-kernel.md").read_text()
    assert "--dsl sycl" in cmd
    assert ".cpp" in cmd


def test_triton_regression_writes_py(tmp_path):
    ws = tmp_path / "ws"
    generate_workspace(
        workspace=ws,
        config=_triton_config(ws),
        kernel_name="kern",
        kernel_code=TRITON_PY,
        reference_code=REFERENCE_PY,
    )
    assert (ws / "test_kernels" / "kern.py").exists()
    assert not (ws / "test_kernels" / "kern.cpp").exists()
    claude = (ws / "CLAUDE.md").read_text()
    # Triton CLAUDE.md keeps the analyze step.
    assert "analyze" in claude


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
