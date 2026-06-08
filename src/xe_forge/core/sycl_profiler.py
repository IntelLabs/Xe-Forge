"""VTune GPU profiler for SYCL/CUTLASS kernels on Intel Xe.

Independent of the Triton profiler (``core/profiler.py``): that one generates a
Python runner that imports a PyTorch ``Model`` and collects ``gpu-offload``.
A SYCL kernel is a compiled ``.cpp`` binary, so this module instead:

  1. compiles the kernel via :class:`SyclExecutor` (reusing the benchmark build),
  2. generates the same deterministic file-IO inputs the benchmark uses,
  3. runs the binary directly under ``vtune -collect gpu-hotspots`` in
     characterization mode (richer Xe hardware metrics than ``gpu-offload``),
  4. parses the per-computing-task hotspots report and maps the metrics to
     CUTLASS tuning knobs (TileShape, SubgroupLayout, PipelineStages, copy
     atoms) via :data:`SYCL_RECOMMENDATION_RULES`.

Verbs, knobs, and column names were confirmed on VTune 2026.0 + Arc Pro B70
(see knowledge_base/sycl/xpu/sycl_vtune.yaml). Gracefully degrades to a result
with ``error`` set when VTune is absent or collection fails, so the trial loop
continues without profiling.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import torch

from xe_forge.core.sycl_executor import KernelType, SyclExecutor

logger = logging.getLogger(__name__)

# VTune gpu-hotspots collection knobs (confirmed on VTune 2026.0 / B70).
_COLLECT_KNOBS = [
    "gpu-profiling-mode=characterization",
    "characterization-mode=overview",
]

# Overhead / non-kernel computing tasks to skip when picking the primary kernel.
_OVERHEAD_PATTERNS = [
    re.compile(r"zeCommandListAppendMemoryCopy"),
    re.compile(r"\[Outside any task\]"),
    re.compile(r"clEnqueue"),
]

# Report columns to request. The memory-bandwidth columns are fetched in a
# SEPARATE pass via a comma-free SUBSTRING filter ("GB/sec"): the full names
# ("GPU Memory Bandwidth, GB/sec:Read") contain a literal comma that VTune's
# -column parser mis-splits, but -column does substring matching, so "GB/sec"
# selects both Read/Write columns cleanly. _build_metrics reads the full
# header names that come back.
_CORE_COLUMNS = [
    "Computing Task:Total Time",
    "XVE Array:Active",
    "XVE Array:Stalled",
    "XVE Array:Idle",
    "Peak XVE Threads Occupancy",
    "XVE Pipelines:XMX active",
    "GPU L3:Miss Ratio",
]
_BW_COLUMNS = ["GB/sec"]


def _is_overhead(name: str) -> bool:
    return any(p.search(name) for p in _OVERHEAD_PATTERNS)


@dataclass
class SyclRecommendation:
    category: str
    message: str
    kb_reference: str = ""


@dataclass
class SyclProfileMetrics:
    xve_active_pct: float | None = None
    xve_stalled_pct: float | None = None
    xve_idle_pct: float | None = None
    peak_occupancy_pct: float | None = None
    xmx_active_pct: float | None = None
    l3_miss_pct: float | None = None
    gpu_mem_bw_read_gbps: float | None = None
    gpu_mem_bw_write_gbps: float | None = None


@dataclass
class SyclProfileResult:
    primary_kernel: str = ""
    metrics: SyclProfileMetrics = field(default_factory=SyclProfileMetrics)
    recommendations: list[SyclRecommendation] = field(default_factory=list)
    raw_counters: dict = field(default_factory=dict)
    error: str | None = None

    def format_for_llm(self) -> str:
        """Structured digest for the agent / tool-runner (mirrors XPUProfiler)."""
        if self.error:
            return f"Profiling error: {self.error}"
        if not self.primary_kernel:
            return "No profiling data available."

        # The CUTLASS kernel name is a giant templated type; show a short tag.
        short = self.primary_kernel.split("<", 1)[0]
        parts = [f"== VTune GPU Profile: {short} ==", "", "Metrics:"]
        m = self.metrics
        rows = [
            ("XVE Active", m.xve_active_pct, "%"),
            ("XVE Stalled", m.xve_stalled_pct, "%"),
            ("XVE Idle", m.xve_idle_pct, "%"),
            ("Peak Occupancy", m.peak_occupancy_pct, "%"),
            ("XMX (DPAS) Active", m.xmx_active_pct, "%"),
            ("L3 Miss Ratio", m.l3_miss_pct, "%"),
            ("GPU Mem BW Read", m.gpu_mem_bw_read_gbps, " GB/s"),
            ("GPU Mem BW Write", m.gpu_mem_bw_write_gbps, " GB/s"),
        ]
        for label, val, unit in rows:
            if val is not None:
                parts.append(f"  {label}: {val:.1f}{unit}")

        if self.recommendations:
            parts.append("")
            parts.append("Recommendations:")
            for rec in self.recommendations:
                parts.append(f"  [{rec.category}] {rec.message}")
                if rec.kb_reference:
                    parts.append(f"    -> {rec.kb_reference}")
        return "\n".join(parts)


class SyclProfiler:
    """VTune ``gpu-hotspots`` profiler for compiled SYCL GEMM kernels."""

    def __init__(
        self,
        vtune_bin: str = "vtune",
        sycl_tla_dir: str | None = None,
        device_target: str | None = None,
        kernel_type: KernelType | str = KernelType.GEMM,
        iterations: int = 200,
        collect_timeout: int = 300,
    ):
        self.vtune_bin = vtune_bin
        self.iterations = iterations
        self.collect_timeout = collect_timeout
        # Reuse the benchmark executor for compile + deterministic inputs.
        executor_kwargs: dict = {"kernel_type": kernel_type, "verify": False}
        if sycl_tla_dir is not None:
            executor_kwargs["sycl_tla_dir"] = sycl_tla_dir
        if device_target is not None:
            executor_kwargs["device_target"] = device_target
        self._executor = SyclExecutor(**executor_kwargs)

    def available(self) -> bool:
        """Whether the VTune binary is on PATH (or an absolute path that exists)."""
        return shutil.which(self.vtune_bin) is not None or os.path.exists(self.vtune_bin)

    def profile(
        self,
        kernel_path: str | Path,
        dims: dict[str, int | float],
        dtype: torch.dtype = torch.bfloat16,
        input_dir: str | None = None,
    ) -> SyclProfileResult:
        """Compile, run under VTune, and return parsed metrics + recommendations.

        Returns a result with ``error`` set (never raises) when VTune is missing
        or any stage fails — the optimization loop treats profiling as advisory.
        """
        if not self.available():
            return SyclProfileResult(
                error=f"VTune not found ({self.vtune_bin}). Set vtune_bin or VTUNE_BIN."
            )
        kernel_path = Path(kernel_path)
        if not kernel_path.exists():
            return SyclProfileResult(error=f"Kernel file not found: {kernel_path}")

        try:
            return self._profile(kernel_path, dims, dtype, input_dir)
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("SYCL profiling failed")
            return SyclProfileResult(error=str(e))

    def _profile(
        self,
        kernel_path: Path,
        dims: dict[str, int | float],
        dtype: torch.dtype,
        input_dir: str | None,
    ) -> SyclProfileResult:
        ok, binary, err = self._executor.compile(
            source_path=str(kernel_path), output_name="kernel_vtune"
        )
        if not ok:
            return SyclProfileResult(error=f"Compilation failed:\n{err[-1500:]}")

        if input_dir is None:
            input_dir = self._executor.get_or_create_inputs(dims, seed=42, dtype=dtype)

        m, n, k = SyclExecutor._dims_to_mnk(dims)
        result_dir = tempfile.mkdtemp(prefix="sycl_vtune_")
        out_dir = tempfile.mkdtemp(prefix="sycl_vtune_out_")
        try:
            collect_err = self._collect(binary, m, n, k, input_dir, out_dir, result_dir)
            if collect_err:
                return SyclProfileResult(error=collect_err)

            counters = self._extract_counters(result_dir)
            if not counters:
                return SyclProfileResult(error="No GPU kernel data in VTune report.")

            primary = self._primary_kernel(counters)
            if primary is None:
                return SyclProfileResult(error="Could not identify the primary GPU kernel.")

            metrics = self._build_metrics(counters[primary])
            recs = self._recommendations(metrics)
            return SyclProfileResult(
                primary_kernel=primary,
                metrics=metrics,
                recommendations=recs,
                raw_counters=counters[primary],
            )
        finally:
            for d in (result_dir, out_dir):
                shutil.rmtree(d, ignore_errors=True)

    def _collect(
        self,
        binary: str,
        m: int,
        n: int,
        k: int,
        input_dir: str,
        out_dir: str,
        result_dir: str,
    ) -> str | None:
        """Run the gpu-hotspots collection. Returns an error string or None."""
        # VTune refuses to write into an existing result-dir.
        shutil.rmtree(result_dir, ignore_errors=True)
        cmd = [self.vtune_bin, "-collect", "gpu-hotspots"]
        for knob in _COLLECT_KNOBS:
            cmd += ["-knob", knob]
        cmd += [
            "-result-dir",
            result_dir,
            "--",
            binary,
            f"--m={m}",
            f"--n={n}",
            f"--k={k}",
            f"--input_dir={input_dir}",
            f"--output_dir={out_dir}",
            f"--iterations={self.iterations}",
            "--verify=0",
        ]
        logger.info("VTune collect: %s", " ".join(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.collect_timeout)
        except subprocess.TimeoutExpired:
            return f"VTune collection timed out after {self.collect_timeout}s"
        except Exception as e:
            return str(e)
        if proc.returncode != 0:
            return f"VTune collection failed (exit {proc.returncode}):\n{proc.stderr[-1500:]}"
        return None

    def _report_csv(self, result_dir: str, columns: list[str]) -> list[dict]:
        """Run one hotspots report pass and return its rows as dicts."""
        cmd = [
            self.vtune_bin,
            "-report",
            "hotspots",
            "-result-dir",
            result_dir,
            "-group-by",
            "computing-task",
            "-column",
            ",".join(columns),
            "-format",
            "csv",
            "-csv-delimiter",
            "tab",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("VTune report failed: %s", e)
            return []
        if proc.returncode != 0:
            logger.warning("VTune report rc=%s: %s", proc.returncode, proc.stderr[:300])
            return []
        # Skip any leading warning lines before the "Computing Task" header.
        lines = proc.stdout.splitlines()
        header_idx = next((i for i, ln in enumerate(lines) if ln.startswith("Computing Task")), 0)
        payload = "\n".join(lines[header_idx:])
        return list(csv.DictReader(io.StringIO(payload), delimiter="\t"))

    def _extract_counters(self, result_dir: str) -> dict[str, dict]:
        """Two report passes (core metrics + bandwidth) merged by task name.

        The bandwidth pass is separate because its column names contain literal
        commas that the -column list parser would mis-split.
        """
        counters: dict[str, dict] = {}
        for columns in (_CORE_COLUMNS, _BW_COLUMNS):
            for row in self._report_csv(result_dir, columns):
                name = (row.get("Computing Task") or "").strip()
                if not name:
                    continue
                entry = counters.setdefault(name, {})
                for key, val in row.items():
                    if val and val.strip():
                        entry.setdefault(key, val.strip())
        return counters

    def _primary_kernel(self, counters: dict[str, dict]) -> str | None:
        """Highest-total-time non-overhead computing task."""

        def total_time(cols: dict) -> float:
            try:
                return float(str(cols.get("Computing Task:Total Time", 0)).replace(",", ""))
            except (ValueError, TypeError):
                return 0.0

        user = [(total_time(c), n) for n, c in counters.items() if not _is_overhead(n)]
        if user:
            return max(user)[1]
        # Fall back to the hottest task overall (with a warning) rather than fail.
        allt = [(total_time(c), n) for n, c in counters.items()]
        if allt:
            logger.warning("Only overhead tasks captured; using hottest task")
            return max(allt)[1]
        return None

    @staticmethod
    def _build_metrics(cols: dict) -> SyclProfileMetrics:
        def num(*keys: str) -> float | None:
            # VTune appends "(%)" to percentage headers; try both spellings.
            for key in keys:
                for variant in (key, f"{key}(%)"):
                    if variant in cols:
                        try:
                            return float(str(cols[variant]).rstrip("%").replace(",", "").strip())
                        except (ValueError, TypeError):
                            return None
            return None

        return SyclProfileMetrics(
            xve_active_pct=num("XVE Array:Active"),
            xve_stalled_pct=num("XVE Array:Stalled"),
            xve_idle_pct=num("XVE Array:Idle"),
            peak_occupancy_pct=num("Peak XVE Threads Occupancy"),
            xmx_active_pct=num("XVE Pipelines:XMX active"),
            l3_miss_pct=num("GPU L3:Miss Ratio"),
            gpu_mem_bw_read_gbps=num("GPU Memory Bandwidth, GB/sec:Read"),
            gpu_mem_bw_write_gbps=num("GPU Memory Bandwidth, GB/sec:Write"),
        )

    @staticmethod
    def _recommendations(m: SyclProfileMetrics) -> list[SyclRecommendation]:
        recs: list[SyclRecommendation] = []
        kb = "knowledge_base/sycl/xpu/sycl_vtune.yaml"

        if (
            m.xve_stalled_pct is not None
            and m.xve_active_pct is not None
            and m.xve_stalled_pct > m.xve_active_pct
        ):
            recs.append(
                SyclRecommendation(
                    "memory_bound",
                    "XVE Stalled > Active — mainloop is memory-bound. Increase "
                    "PipelineStages (prefetch depth), try explicit 2D-block/VNNI "
                    "copy atoms, or reduce TileK.",
                    kb,
                )
            )
        if m.peak_occupancy_pct is not None and m.peak_occupancy_pct < 50:
            recs.append(
                SyclRecommendation(
                    "low_occupancy",
                    f"Peak occupancy {m.peak_occupancy_pct:.0f}% — grid may be too "
                    "small or registers too high. Try a smaller TileShape (e.g. "
                    "256->128) or confirm 256-GRF mode; for small problems expect "
                    "work-size-limited occupancy.",
                    kb,
                )
            )
        if m.xve_idle_pct is not None and m.xve_idle_pct > 30:
            recs.append(
                SyclRecommendation(
                    "high_idle",
                    f"XVE Idle {m.xve_idle_pct:.0f}% — poor work distribution across "
                    "EUs. Revisit TileShape vs M/N (tail effects); consider stream-K "
                    "or a persistent scheduler.",
                    kb,
                )
            )
        if m.xmx_active_pct is not None and m.xmx_active_pct < 20:
            recs.append(
                SyclRecommendation(
                    "low_xmx",
                    f"XMX (DPAS) active {m.xmx_active_pct:.0f}% — the matrix engine is "
                    "underutilized. Raise compute intensity: larger N-per-subgroup, "
                    "check SubgroupLayout vs the DPAS atom.",
                    kb,
                )
            )
        if m.l3_miss_pct is not None and m.l3_miss_pct > 50:
            recs.append(
                SyclRecommendation(
                    "l3_thrashing",
                    f"L3 miss ratio {m.l3_miss_pct:.0f}% — cache thrashing. Reduce tile "
                    "sizes or improve data reuse / K-blocking.",
                    kb,
                )
            )
        return recs
