"""Tests for SyclProfiler (VTune gpu-hotspots path).

Platform-independent: the vtune subprocess and SyclExecutor.compile/inputs seams
are mocked, so no VTune, icpx, or GPU is needed. Column names and CSV shape match
what VTune 2026.0 emits for a CUTLASS GEMM on the B70.
"""

import subprocess

import pytest

from xe_forge.core import sycl_profiler as sp
from xe_forge.core.sycl_profiler import SyclProfileMetrics, SyclProfiler

# A realistic two-row hotspots CSV: the CUTLASS kernel + an overhead copy task.
# Tab-delimited, with the "(%)" suffix VTune appends to percentage columns.
_CORE_CSV = (
    "Computing Task\tComputing Task:Total Time\tXVE Array:Active(%)\t"
    "XVE Array:Stalled(%)\tXVE Array:Idle(%)\tPeak XVE Threads Occupancy(%)\t"
    "XVE Pipelines:XMX active(%)\tGPU L3:Miss Ratio(%)\n"
    "GemmUniversal<cute::tuple<...>>\t0.0598\t7.3\t39.4\t53.3\t25.0\t5.0\t0.7\n"
    "zeCommandListAppendMemoryCopy\t0.0006\t0.0\t0.0\t100.0\t0.0\t0.0\t100.0\n"
)
_BW_CSV = (
    "Computing Task\tComputing Task:Total Time\t"
    "GPU Memory Bandwidth, GB/sec:Read\tGPU Memory Bandwidth, GB/sec:Write\n"
    "GemmUniversal<cute::tuple<...>>\t0.0598\t120.5\t60.2\n"
    "zeCommandListAppendMemoryCopy\t0.0006\t0.0\t0.0\n"
)


def _profiler(monkeypatch, vtune_ok=True):
    """Build a SyclProfiler without touching torch.xpu / a real executor."""

    # Stub SyclExecutor construction inside the module to a lightweight fake.
    # _dims_to_mnk is a pure static helper the module calls on the class, so the
    # fake provides it with the same semantics as the real one.
    class FakeExecutor:
        def __init__(self, *a, **k):
            pass

        @staticmethod
        def _dims_to_mnk(dims, m=1024, n=1024, k=1024):
            if not dims:
                return m, n, k
            em = int(dims.get("M", dims.get("N", m)))
            return em, int(dims.get("N", em)), int(dims.get("K", em))

    monkeypatch.setattr(sp, "SyclExecutor", FakeExecutor, raising=True)
    prof = SyclProfiler(vtune_bin="vtune")
    monkeypatch.setattr(prof, "available", lambda: vtune_ok)
    return prof


def test_unavailable_vtune_returns_error_without_compiling(tmp_path, monkeypatch):
    prof = _profiler(monkeypatch, vtune_ok=False)
    kernel = tmp_path / "gemm.cpp"
    kernel.write_text("#include <x>\nint main(){}\n")
    compiled = {"called": False}
    monkeypatch.setattr(
        prof._executor, "compile", lambda **k: compiled.__setitem__("called", True), raising=False
    )
    res = prof.profile(kernel, {"M": 1024, "N": 1024, "K": 1024})
    assert res.error and "VTune not found" in res.error
    assert compiled["called"] is False


def test_missing_kernel_file(tmp_path, monkeypatch):
    prof = _profiler(monkeypatch)
    res = prof.profile(tmp_path / "nope.cpp", {"M": 256, "N": 256, "K": 256})
    assert res.error and "not found" in res.error


def test_compile_failure_surfaces(tmp_path, monkeypatch):
    prof = _profiler(monkeypatch)
    kernel = tmp_path / "gemm.cpp"
    kernel.write_text("#include <x>\nint main(){}\n")
    monkeypatch.setattr(
        prof._executor, "compile", lambda **k: (False, "", "boom error"), raising=False
    )
    res = prof.profile(kernel, {"M": 256, "N": 256, "K": 256})
    assert res.error and "boom error" in res.error


def _wire_success(prof, monkeypatch, tmp_path):
    """Wire compile/inputs + a fake vtune subprocess that returns the CSVs."""
    kernel = tmp_path / "gemm.cpp"
    kernel.write_text("#include <x>\nint main(){}\n")
    monkeypatch.setattr(
        prof._executor, "compile", lambda **k: (True, str(tmp_path / "bin"), ""), raising=False
    )
    monkeypatch.setattr(
        prof._executor, "get_or_create_inputs", lambda *a, **k: str(tmp_path / "in"), raising=False
    )

    def fake_run(cmd, **kwargs):
        joined = " ".join(cmd)
        if "-collect" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "collected", "")
        # Report pass: the BW pass requests the comma-free "GB/sec" substring.
        out = _BW_CSV if "GB/sec" in joined else _CORE_CSV
        return subprocess.CompletedProcess(cmd, 0, out, "")

    monkeypatch.setattr(sp.subprocess, "run", fake_run)
    return kernel


def test_profile_success_parses_metrics_and_picks_kernel(tmp_path, monkeypatch):
    prof = _profiler(monkeypatch)
    kernel = _wire_success(prof, monkeypatch, tmp_path)
    res = prof.profile(kernel, {"M": 1024, "N": 1024, "K": 1024})
    assert res.error is None
    # Overhead task must NOT be chosen as primary.
    assert res.primary_kernel.startswith("GemmUniversal")
    m = res.metrics
    assert m.xve_active_pct == 7.3
    assert m.xve_stalled_pct == 39.4
    assert m.peak_occupancy_pct == 25.0
    assert m.xmx_active_pct == 5.0
    assert m.l3_miss_pct == 0.7
    # Bandwidth came from the second (separate) report pass.
    assert m.gpu_mem_bw_read_gbps == 120.5
    assert m.gpu_mem_bw_write_gbps == 60.2


def test_recommendations_memory_bound_and_low_occupancy(tmp_path, monkeypatch):
    prof = _profiler(monkeypatch)
    kernel = _wire_success(prof, monkeypatch, tmp_path)
    res = prof.profile(kernel, {"M": 1024, "N": 1024, "K": 1024})
    cats = {r.category for r in res.recommendations}
    # stalled(39.4) > active(7.3) -> memory_bound; occ 25 < 50 -> low_occupancy;
    # idle 53.3 > 30 -> high_idle; xmx 5 < 20 -> low_xmx.
    assert {"memory_bound", "low_occupancy", "high_idle", "low_xmx"} <= cats
    txt = res.format_for_llm()
    assert "XVE Stalled" in txt and "Recommendations:" in txt


def test_empty_report_is_error(tmp_path, monkeypatch):
    prof = _profiler(monkeypatch)
    kernel = tmp_path / "gemm.cpp"
    kernel.write_text("#include <x>\nint main(){}\n")
    monkeypatch.setattr(
        prof._executor, "compile", lambda **k: (True, str(tmp_path / "bin"), ""), raising=False
    )
    monkeypatch.setattr(
        prof._executor, "get_or_create_inputs", lambda *a, **k: str(tmp_path / "in"), raising=False
    )

    def fake_run(cmd, **kwargs):
        if "-collect" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "ok", "")
        return subprocess.CompletedProcess(cmd, 0, "no data here\n", "")

    monkeypatch.setattr(sp.subprocess, "run", fake_run)
    res = prof.profile(kernel, {"M": 256, "N": 256, "K": 256})
    assert res.error and "No GPU kernel data" in res.error


def test_collection_failure_surfaces(tmp_path, monkeypatch):
    prof = _profiler(monkeypatch)
    kernel = tmp_path / "gemm.cpp"
    kernel.write_text("#include <x>\nint main(){}\n")
    monkeypatch.setattr(
        prof._executor, "compile", lambda **k: (True, str(tmp_path / "bin"), ""), raising=False
    )
    monkeypatch.setattr(
        prof._executor, "get_or_create_inputs", lambda *a, **k: str(tmp_path / "in"), raising=False
    )

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "driver error")

    monkeypatch.setattr(sp.subprocess, "run", fake_run)
    res = prof.profile(kernel, {"M": 256, "N": 256, "K": 256})
    assert res.error and "collection failed" in res.error.lower()


def test_metrics_pct_suffix_fallback():
    # _build_metrics must read both "X" and "X(%)" header spellings.
    cols = {"XVE Array:Active(%)": "7.3", "GPU L3:Miss Ratio": "0.7"}
    m = SyclProfiler._build_metrics(cols)
    assert m.xve_active_pct == 7.3
    assert m.l3_miss_pct == 0.7


def test_no_recommendations_when_healthy():
    # Active > stalled, high occupancy, high xmx, low idle, low l3 -> no recs.
    m = SyclProfileMetrics(
        xve_active_pct=85.0,
        xve_stalled_pct=10.0,
        xve_idle_pct=5.0,
        peak_occupancy_pct=90.0,
        xmx_active_pct=80.0,
        l3_miss_pct=1.0,
    )
    assert SyclProfiler._recommendations(m) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
