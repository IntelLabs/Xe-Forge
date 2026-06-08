"""Tests for the profile skill DSL dispatch (Triton vs SYCL).

Platform-independent: the XPUProfiler / SyclProfiler seams are mocked.
"""

from argparse import Namespace

import pytest

from xe_forge.skills import profile


class _FakeResult:
    def __init__(self, text):
        self._text = text

    def format_for_llm(self):
        return self._text


def test_sycl_routes_to_sycl_profiler(tmp_path, monkeypatch, capsys):
    """args.dsl == 'sycl' must use SyclProfiler, never XPUProfiler."""
    spec = tmp_path / "gemm.yaml"
    spec.write_text(
        "inputs:\n  A:\n    shape: [M, K]\n    dtype: bfloat16\n"
        "bench-xpu:\n  - params: [A]\n    dtype: bfloat16\n"
        "    dims: { M: 1024, N: 1024, K: 1024 }\n    flop: '2*M*N*K'\n"
    )
    captured = {}

    class FakeSyclProfiler:
        def __init__(self, *a, **k):
            captured["constructed"] = True

        def profile(self, kernel_path, dims, dtype, **k):
            captured["dims"] = dims
            return _FakeResult("SYCL PROFILE OK")

    def _boom(*a, **k):
        raise AssertionError("SYCL path must not construct XPUProfiler")

    monkeypatch.setattr("xe_forge.core.sycl_profiler.SyclProfiler", FakeSyclProfiler, raising=True)
    monkeypatch.setattr("xe_forge.core.profiler.XPUProfiler", _boom, raising=True)

    args = Namespace(
        dsl="sycl",
        kernel_file=str(tmp_path / "gemm.cpp"),
        spec=str(spec),
        variant="bench-xpu",
        warmup=5,
        iters=200,
        vtune_bin="vtune",
    )
    profile.run(args)
    out = capsys.readouterr().out
    assert captured.get("constructed")
    assert captured["dims"] == {"M": 1024, "N": 1024, "K": 1024}
    assert "SYCL PROFILE OK" in out


def test_triton_routes_to_xpu_profiler(tmp_path, monkeypatch, capsys):
    """args.dsl == 'triton' still uses XPUProfiler."""
    called = {}

    class FakeXPUProfiler:
        def __init__(self, *a, **k):
            called["constructed"] = True

        def profile(self, kernel_file, **k):
            return _FakeResult("TRITON PROFILE OK")

    monkeypatch.setattr("xe_forge.core.profiler.XPUProfiler", FakeXPUProfiler, raising=True)

    args = Namespace(
        dsl="triton",
        kernel_file=str(tmp_path / "k.py"),
        spec=str(tmp_path / "s.yaml"),
        variant="bench-gpu",
        warmup=5,
        iters=20,
        vtune_bin="vtune",
    )
    profile.run(args)
    out = capsys.readouterr().out
    assert called.get("constructed")
    assert "TRITON PROFILE OK" in out


def test_profile_subparser_has_dsl(monkeypatch):
    """The REAL profile subparser must accept --dsl (regression for the missing flag).

    Drives xe_forge.skills.main() with a stubbed profile.run so only argument
    parsing is exercised — proving the flag exists on the actual parser, not a
    reconstruction.
    """
    import sys

    from xe_forge import skills

    seen = {}
    monkeypatch.setattr(profile, "run", lambda args: seen.update(dsl=args.dsl, iters=args.iters))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "xe-forge-skill",
            "profile",
            "k.cpp",
            "--spec",
            "s.yaml",
            "--dsl",
            "sycl",
            "--iters",
            "200",
        ],
    )
    skills.main()
    assert seen["dsl"] == "sycl"
    assert seen["iters"] == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
