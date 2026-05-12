"""VTune GPU hardware counter profiler for XPU kernels.

Collects OA metrics via VTune gpu-offload, extracts XVE/occupancy/cache
counters, and maps bottlenecks to knowledge base optimization patterns.
Gracefully degrades when VTune is not available.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_OVERHEAD_KERNEL_PATTERNS = [
    re.compile(r"VectorizedElementwiseKernel"),
    re.compile(r"UnrolledElementwiseKernel"),
    re.compile(r"zeCommandListAppendMemoryCopy"),
    re.compile(r"ReduceKernelEmptyFunctor"),
    re.compile(r"\[Outside any task\]"),
]

_HOTSPOTS_COLUMNS_PASS1 = ",".join(
    [
        "Computing Task:Total Time",
        "Computing Task:Average Time",
        "Computing Task:Instance Count",
        "Computing Task:SIMD Width",
        "XVE Array:Active",
        "XVE Array:Stalled",
        "XVE Array:Idle",
        "Peak XVE Threads Occupancy",
        "GPU Memory Bandwidth, GB/sec:Read",
        "GPU Memory Bandwidth, GB/sec:Write",
        "GPU L3:Busy",
        "GPU L3:Stalled",
        "GPU L3:Miss Ratio",
        "GPU L3:Average Bandwidth, GB/s:Read",
        "GPU L3:Average Bandwidth, GB/s:Write",
        "GPU Load Store Cache:Miss Ratio",
        "GPU Load Store Cache:L3 Miss Ratio",
        "GPU Shared Local Memory:Bank Conflicts",
        "TLB Misses",
    ]
)

_HOTSPOTS_COLUMNS_PASS2 = ",".join(
    [
        "Computing Task:Total Time",
        "XVE Threads Occupancy",
        "GPU Load Store Cache:Average Bandwidth, GB/s:Read",
        "GPU Load Store Cache:Average Bandwidth, GB/s:Write",
    ]
)


def _is_overhead_kernel(name: str) -> bool:
    return any(pat.search(name) for pat in _OVERHEAD_KERNEL_PATTERNS)


@dataclass
class Recommendation:
    category: str
    message: str
    kb_reference: str = ""


@dataclass
class ProfileMetrics:
    xve_active_pct: float | None = None
    xve_stalled_pct: float | None = None
    xve_idle_pct: float | None = None
    peak_occupancy_pct: float | None = None
    occupancy_limiter: str | None = None
    l3_miss_pct: float | None = None
    gpu_memory_bw_read_gbps: float | None = None
    gpu_memory_bw_write_gbps: float | None = None
    lsc_miss_pct: float | None = None
    lsc_bw_read_gbps: float | None = None
    lsc_bw_write_gbps: float | None = None
    overhead_kernel_pct: float | None = None


@dataclass
class ProfileResult:
    primary_kernel: str = ""
    metrics: ProfileMetrics = field(default_factory=ProfileMetrics)
    recommendations: list[Recommendation] = field(default_factory=list)
    raw_counters: dict = field(default_factory=dict)
    error: str | None = None

    def format_for_llm(self) -> str:
        """Produce a structured digest suitable for passing to an LLM."""
        if self.error:
            return f"Profiling error: {self.error}"
        if not self.primary_kernel:
            return "No profiling data available."

        parts = [
            f"== VTune Profile: {self.primary_kernel} ==",
            "",
            "Metrics:",
        ]
        m = self.metrics
        if m.xve_active_pct is not None:
            parts.append(f"  XVE Active:  {m.xve_active_pct:.1f}%")
        if m.xve_stalled_pct is not None:
            parts.append(f"  XVE Stalled: {m.xve_stalled_pct:.1f}%")
        if m.xve_idle_pct is not None:
            parts.append(f"  XVE Idle:    {m.xve_idle_pct:.1f}%")
        if m.peak_occupancy_pct is not None:
            parts.append(f"  Peak Occupancy: {m.peak_occupancy_pct:.1f}%")
        if m.l3_miss_pct is not None:
            parts.append(f"  L3 Miss Ratio:  {m.l3_miss_pct:.1f}%")
        if m.lsc_miss_pct is not None:
            parts.append(f"  LSC Miss Ratio: {m.lsc_miss_pct:.1f}%")
        if m.gpu_memory_bw_read_gbps is not None:
            parts.append(f"  GPU Mem BW Read:  {m.gpu_memory_bw_read_gbps:.1f} GB/s")
        if m.gpu_memory_bw_write_gbps is not None:
            parts.append(f"  GPU Mem BW Write: {m.gpu_memory_bw_write_gbps:.1f} GB/s")

        if self.recommendations:
            parts.append("")
            parts.append("Recommendations:")
            for rec in self.recommendations:
                parts.append(f"  [{rec.category}] {rec.message}")
                if rec.kb_reference:
                    parts.append(f"    -> {rec.kb_reference}")

        return "\n".join(parts)


class XPUProfiler:
    """VTune-based GPU hardware counter profiler."""

    def __init__(self, vtune_bin: str = "vtune"):
        self.vtune_bin = vtune_bin

    def available(self) -> bool:
        """Check if VTune is accessible."""
        return shutil.which(self.vtune_bin) is not None

    def profile(
        self,
        kernel_file: str | Path,
        spec_path: str | Path | None = None,
        variant: str = "bench-gpu",
        warmup: int = 5,
        iters: int = 20,
    ) -> ProfileResult:
        """Profile a kernel and return structured results.

        Returns an empty ProfileResult with ``error`` set if VTune is not
        available or collection fails.
        """
        if not self.available():
            return ProfileResult(error="VTune not found. Install VTune or set vtune_bin path.")

        kernel_file = Path(kernel_file)
        if not kernel_file.exists():
            return ProfileResult(error=f"Kernel file not found: {kernel_file}")

        try:
            return self._collect_and_parse(kernel_file, spec_path, variant, warmup, iters)
        except Exception as e:
            logger.exception("Profiling failed")
            return ProfileResult(error=str(e))

    def _collect_and_parse(
        self,
        kernel_file: Path,
        spec_path: str | Path | None,
        variant: str,
        warmup: int,
        iters: int,
    ) -> ProfileResult:
        """Run VTune collection and parse the results."""
        with tempfile.TemporaryDirectory(prefix="xeforge_vtune_") as tmpdir:
            result_dir = Path(tmpdir) / "vtune_result"
            runner_script = self._generate_runner_script(kernel_file, warmup, iters, result_dir)
            runner_path = Path(tmpdir) / "runner.py"
            runner_path.write_text(runner_script)

            collect_cmd = [
                self.vtune_bin,
                "-collect",
                "gpu-offload",
                "-start-paused",
                "-result-dir",
                str(result_dir),
                "--",
                sys.executable,
                str(runner_path),
            ]
            proc = subprocess.run(
                collect_cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if proc.returncode != 0:
                return ProfileResult(error=f"VTune collection failed: {proc.stderr[:500]}")

            counters = self._extract_counters(result_dir)
            if not counters:
                return ProfileResult(error="No GPU kernel data in VTune results.")

            primary = self._identify_primary_kernel(counters)
            if not primary:
                return ProfileResult(error="Could not identify primary kernel.")

            metrics = self._build_metrics(counters.get(primary, {}))
            recommendations = self._generate_recommendations(metrics)

            return ProfileResult(
                primary_kernel=primary,
                metrics=metrics,
                recommendations=recommendations,
                raw_counters=counters.get(primary, {}),
            )

    def _generate_runner_script(
        self,
        kernel_file: Path,
        warmup: int,
        iters: int,
        result_dir: Path,
    ) -> str:
        return f"""\
import importlib.util, sys, torch, subprocess

spec = importlib.util.spec_from_file_location("kernel_mod", "{kernel_file}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

Model = mod.Model
get_inputs = mod.get_inputs
get_init_inputs = mod.get_init_inputs

device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cuda"
model = Model(*get_init_inputs()).to(device).eval()
inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_inputs()]

_sync = torch.xpu.synchronize if device == "xpu" else torch.cuda.synchronize

# Warmup (collection paused)
with torch.no_grad():
    for _ in range({warmup}):
        model(*inputs)
        _sync()

# Resume VTune
vtune_bin = "{self.vtune_bin}"
result_dir = "{result_dir}"
subprocess.run([vtune_bin, "-command", "resume", "-r", result_dir],
               capture_output=True, timeout=30)

# Profiled iterations — sync each iter so OA sampling sees every kernel
with torch.no_grad():
    for _ in range({iters}):
        model(*inputs)
        _sync()

subprocess.run([vtune_bin, "-command", "pause", "-r", result_dir],
               capture_output=True, timeout=30)
"""

    def _extract_counters(self, result_dir: Path) -> dict[str, dict]:
        """Extract counters from VTune CSV report.

        Keeps overhead kernels so ``_identify_primary_kernel`` can fall back
        to them when no user compute kernels are captured. Aggregates
        VTune's per-SIMD-width variants of the same kernel name by picking
        the longest-running variant as representative and summing times.
        """
        counters: dict[str, dict] = {}

        def _num(v: str) -> float:
            try:
                return float(str(v).replace(",", "").rstrip("%").strip())
            except (ValueError, TypeError):
                return 0.0

        for columns in (_HOTSPOTS_COLUMNS_PASS1, _HOTSPOTS_COLUMNS_PASS2):
            report_cmd = [
                self.vtune_bin,
                "-report",
                "hotspots",
                "-result-dir",
                str(result_dir),
                "-group-by",
                "computing-task",
                "-column",
                columns,
                "-format",
                "csv",
                "-csv-delimiter",
                "tab",
            ]
            proc = subprocess.run(
                report_cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if proc.returncode != 0:
                logger.warning("VTune hotspots report failed: %s", proc.stderr[:200])
                continue

            # VTune may emit warning lines (e.g. "war:Column filter is ON.")
            # before the actual header; skip to the row starting with
            # "Computing Task".
            lines = proc.stdout.splitlines()
            header_idx = next(
                (i for i, ln in enumerate(lines) if ln.startswith("Computing Task")),
                0,
            )
            payload = "\n".join(lines[header_idx:])

            reader = csv.DictReader(io.StringIO(payload), delimiter="\t")
            for row in reader:
                name = (row.get("Computing Task") or "").strip()
                if not name or name.startswith("["):
                    continue
                new_time = _num(row.get("Computing Task:Total Time", ""))
                existing = counters.get(name)
                if existing is None:
                    counters[name] = {k: v for k, v in row.items() if v and v.strip()}
                    continue
                # Same kernel name across autotune variants: keep the
                # longest-running variant's per-iteration metrics, but sum
                # total time so primary-kernel selection sees the full cost.
                old_time = _num(existing.get("Computing Task:Total Time", ""))
                if new_time > old_time:
                    merged = {k: v for k, v in row.items() if v and v.strip()}
                    # Preserve pass-1 columns that pass-2 doesn't populate
                    for k, v in existing.items():
                        merged.setdefault(k, v)
                    existing = merged
                else:
                    for k, v in row.items():
                        if v and v.strip():
                            existing.setdefault(k, v)
                existing["Computing Task:Total Time"] = f"{old_time + new_time}"
                counters[name] = existing

        return counters

    def _identify_primary_kernel(self, counters: dict[str, dict]) -> str | None:
        """Pick the user compute kernel with the highest total time.

        Falls back to the overall hottest kernel (possibly a PyTorch
        overhead kernel) if no user kernels were captured — better to
        return *something* with a warning than to fail silently.
        """
        best_user: tuple[float, str] | None = None
        best_any: tuple[float, str] | None = None
        for name, cols in counters.items():
            try:
                t = float(str(cols.get("Computing Task:Total Time", 0)).replace(",", ""))
            except (ValueError, TypeError):
                continue
            if best_any is None or t > best_any[0]:
                best_any = (t, name)
            if not _is_overhead_kernel(name):
                if best_user is None or t > best_user[0]:
                    best_user = (t, name)
        if best_user is not None:
            return best_user[1]
        if best_any is not None:
            logger.warning("Only PyTorch overhead kernels captured; using %s", best_any[1])
            return best_any[1]
        return None

    def _build_metrics(self, cols: dict) -> ProfileMetrics:
        def _f(key: str) -> float | None:
            v = cols.get(key)
            if v is None:
                return None
            try:
                return float(str(v).rstrip("%").strip())
            except (ValueError, TypeError):
                return None

        # VTune CSV appends "(%)" to percentage column headers — the
        # request column names do not include it, but the emitted headers
        # do. Fall back to the unsuffixed name for robustness.
        def _pct(key: str) -> float | None:
            return _f(f"{key}(%)") if f"{key}(%)" in cols else _f(key)

        return ProfileMetrics(
            xve_active_pct=_pct("XVE Array:Active"),
            xve_stalled_pct=_pct("XVE Array:Stalled"),
            xve_idle_pct=_pct("XVE Array:Idle"),
            peak_occupancy_pct=_pct("Peak XVE Threads Occupancy"),
            l3_miss_pct=_pct("GPU L3:Miss Ratio"),
            gpu_memory_bw_read_gbps=_f("GPU Memory Bandwidth, GB/sec:Read"),
            gpu_memory_bw_write_gbps=_f("GPU Memory Bandwidth, GB/sec:Write"),
            lsc_miss_pct=_pct("GPU Load Store Cache:Miss Ratio"),
            lsc_bw_read_gbps=_f("GPU Load Store Cache:Average Bandwidth, GB/s:Read"),
            lsc_bw_write_gbps=_f("GPU Load Store Cache:Average Bandwidth, GB/s:Write"),
        )

    def _generate_recommendations(self, m: ProfileMetrics) -> list[Recommendation]:
        recs: list[Recommendation] = []

        if m.xve_stalled_pct is not None and m.xve_active_pct is not None:
            if m.xve_stalled_pct > m.xve_active_pct:
                recs.append(
                    Recommendation(
                        "memory_bound",
                        "XVE Stalled > Active — kernel is memory-bound. "
                        "Use tensor descriptors, bf16 inputs, tile swizzling.",
                        "xpu_optimizations.yaml (xpu_block_pointers, xpu_bf16)",
                    )
                )

        if m.peak_occupancy_pct is not None and m.peak_occupancy_pct < 50:
            recs.append(
                Recommendation(
                    "low_occupancy",
                    f"Peak occupancy {m.peak_occupancy_pct:.0f}% — try larger tiles, "
                    "fewer registers, or persistent kernel.",
                    "xpu_optimizations.yaml (xpu_persistent_kernel)",
                )
            )

        if m.xve_idle_pct is not None and m.xve_idle_pct > 30:
            recs.append(
                Recommendation(
                    "high_idle",
                    f"XVE Idle {m.xve_idle_pct:.0f}% — work distribution issue. "
                    "Check grid dimensions and tile swizzling.",
                    "xpu_optimizations.yaml (xpu_swizzle)",
                )
            )

        if m.l3_miss_pct is not None and m.l3_miss_pct > 50:
            recs.append(
                Recommendation(
                    "l3_thrashing",
                    f"L3 miss ratio {m.l3_miss_pct:.0f}% — cache thrashing. "
                    "Reduce tile sizes or improve data reuse.",
                    "memory_patterns.yaml",
                )
            )

        if m.lsc_miss_pct is not None and m.lsc_miss_pct > 30:
            recs.append(
                Recommendation(
                    "lsc_miss",
                    f"LSC miss ratio {m.lsc_miss_pct:.0f}% — poor cache locality.",
                    "memory_patterns.yaml",
                )
            )

        return recs
