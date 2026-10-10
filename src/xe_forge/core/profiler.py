"""GPU hardware counter profilers for XPU kernels: VTune and unitrace."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
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


def _runner_script(kernel_file: Path, **options) -> str:
    options = {
        name: str(value) if isinstance(value, Path) else value for name, value in options.items()
    }
    return (
        "from xe_forge.core.profile_runner import run_profile\n"
        f"call = run_profile({str(kernel_file)!r}, **{options!r})\n"
    )


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
    sampled_active_pct: float | None = None
    sampled_stalled_pct: float | None = None


@dataclass
class ProfileResult:
    primary_kernel: str = ""
    metrics: ProfileMetrics = field(default_factory=ProfileMetrics)
    recommendations: list[Recommendation] = field(default_factory=list)
    raw_counters: dict = field(default_factory=dict)
    error: str | None = None
    source: str = "VTune"
    artifacts_dir: str | None = None

    def format_for_llm(self) -> str:
        """Produce a structured digest suitable for passing to an LLM."""
        artifacts = f"\nArtifacts: {self.artifacts_dir}" if self.artifacts_dir else ""
        if self.raw_counters.get("collection_scope") == "forward_only":
            artifacts += "\nCollection scope: forward only; setup, validation, warmup and buffer reset excluded."
        if self.raw_counters.get("stall_temporal_isolation") == "unverified":
            artifacts += "\nEU stall scope exception: temporal isolation is unverified; forward-only scope applies to timings, not guaranteed to stall events."
        assembly = self.raw_counters.get("assembly")
        if assembly:
            artifacts += f"\nAssembly: {assembly['status']} ({assembly['directory']})"
            artifacts += "\nSampled-IP to ISA mapping: unverified"
        if self.error:
            msg = f"Profiling error: {self.error}"
            device_timing = self.raw_counters.get("device_timing")
            if device_timing:
                msg += f"\n\n(partial data was captured)\n{device_timing}"
            return msg + artifacts
        if not self.primary_kernel:
            return "No profiling data available." + artifacts

        parts = [
            f"== {self.source} Profile: {self.primary_kernel} ==",
            "",
            "Metrics:",
        ]
        m = self.metrics
        if self.source == "Unitrace" and not self.raw_counters.get("compute_metrics"):
            parts.append("  EU event shares are not elapsed-time utilization or occupancy.")
        if m.sampled_active_pct is not None:
            parts.append(f"  Active samples:  {m.sampled_active_pct:.1f}%")
        if m.sampled_stalled_pct is not None:
            parts.append(f"  Stalled samples: {m.sampled_stalled_pct:.1f}%")
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

        compute = self.raw_counters.get("compute_metrics")
        if compute:
            parts.append("ComputeBasic sampled intervals (not benchmark timings):")
            for record in compute["records"]:
                if record.get("timing_kernel", record["kernel"]) != self.primary_kernel:
                    continue
                parts.append(
                    f"  {record['device']}: {record['launches']} launches, {record['samples']} samples"
                )
                parts.append(f"  Sampled operation: {record['kernel']}")
                parts.append(f"  Timed launches: {record.get('timed_launches', 'unknown')}")
                parts.append(f"  Sampled us/launch: {record['sampled_us_per_launch']}")
                for direction, memory in record["memory"].items():
                    parts.append(
                        f"  DRAM {direction}: bytes/launch={memory['bytes_per_launch']}, GB/s={memory['gb_per_s']}"
                    )
                for counter in (
                    "GPU_BUSY[%]",
                    "XVE_ACTIVE[%]",
                    "XVE_STALL[%]",
                    "XVE_THREADS_OCCUPANCY_ALL[%]",
                    "XVE_INST_EXECUTED_ALU0_ALL_UTILIZATION[%]",
                    "XVE_INST_EXECUTED_ALU1_ALL_UTILIZATION[%]",
                    "XVE_INST_EXECUTED_ALU2_ALL_UTILIZATION[%]",
                    "L3_STALL[%]",
                    "GPU_MEMORY_REQUEST_QUEUE_FULL[%]",
                ):
                    value = record["counters"].get(counter)
                    parts.append(f"  {counter}: {value if value is not None else 'unknown'}")
            parts.extend(f"  Note: {limitation}" for limitation in compute["limitations"])
            parts.append("Guidance: knowledge_base/sycl/xpu/compute_basic.yaml")

        stalls = self.raw_counters.get("stall_metrics", {}).get(self.primary_kernel)
        if stalls:
            parts.append(
                f"EU stall sampling: {len(stalls['ip_samples'])} instruction-address records"
            )
            parts.append(f"  Active events: {stalls['active_events']:.0f}")
            for cause, events in sorted(
                stalls["stall_events"].items(), key=lambda item: item[1], reverse=True
            ):
                parts.append(f"  {cause}: {events:.0f} events")
            parts.append(
                "Sampled events are not elapsed-time utilization; unsampled launches and IP-to-ISA attribution are unknown."
            )

        timings = self.raw_counters.get("timings", {})
        if timings:
            parts.append("")
            scope = (
                "forward only"
                if self.raw_counters.get("collection_scope") == "forward_only"
                else "collection scope unverified"
            )
            parts.append(f"Top device operations by total time (instrumented; {scope}):")
            total_ns = sum(timing["time_ns"] for timing in timings.values())
            for name in sorted(timings, key=lambda name: timings[name]["time_ns"], reverse=True)[
                :5
            ]:
                timing = timings[name]
                share = 100.0 * timing["time_ns"] / total_ns if total_ns else 0.0
                parts.append(
                    f"  {name}: {timing['time_ns'] / 1000:.2f} us total, "
                    f"{share:.1f}%, {timing['calls']:.0f} calls"
                )

        resources = self.raw_counters.get("kernel_properties", {}).get(self.primary_kernel, [])
        if self.source == "Unitrace":
            parts.append("")
            parts.append("Reported kernel resources (unitrace / driver):")
            if resources:
                for index, properties in enumerate(resources, start=1):
                    parts.append(f"  Record {index}:")
                    for name, value in properties.items():
                        parts.append(f"    {name}: {value if value is not None else 'unknown'}")
            else:
                parts.append("  unknown: no properties matched the hottest operation")
            parts.append("Guidance: knowledge_base/sycl/xpu/resource_feedback.yaml")

        provenance = self.raw_counters.get("build_provenance")
        if provenance:
            parts.append("")
            parts.append(f"Device identity: {provenance.get('device') or 'unknown'}")
            parts.append(
                f"Build environment (not proof of effective flags): {provenance.get('environment', {})}"
            )
            builds = provenance.get("builds", [])
            if not builds:
                parts.append("Extension build: unknown (no observed torch extension build)")
            for build in builds:
                parts.append(f"Observed {build['loader']} arguments: {build['arguments']}")
                for driver, details in build.get("compile_drivers", {}).items():
                    version = details.get("version", "").splitlines()
                    parts.append(
                        f"  Compile driver {driver}: {version[0] if version else 'unknown'}"
                    )
                if build.get("capture_error"):
                    parts.append(f"  Build capture incomplete: {build['capture_error']}")

        comparison = self.raw_counters.get("parent_comparison")
        if comparison:
            parts.append("")
            parts.append(f"Parent resource comparison: {comparison['status']}")
            for field, change in comparison.get("fields", {}).items():
                if change["parent"] == change["current"]:
                    continue
                delta = f" (delta {change['delta']:+d})" if "delta" in change else ""
                parts.append(f"  {field}: {change['parent']} -> {change['current']}{delta}")

        if self.recommendations:
            parts.append("")
            parts.append("Recommendations:")
            for rec in self.recommendations:
                parts.append(f"  [{rec.category}] {rec.message}")
                if rec.kb_reference:
                    parts.append(f"    -> {rec.kb_reference}")

        return "\n".join(parts) + artifacts


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
        variant: str | None = None,
        warmup: int = 5,
        iters: int = 20,
        reference_path: str | Path | None = None,
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
            return self._collect_and_parse(
                kernel_file, spec_path, variant, warmup, iters, reference_path
            )
        except Exception as e:
            logger.exception("Profiling failed")
            return ProfileResult(error=str(e))

    def _collect_and_parse(
        self,
        kernel_file: Path,
        spec_path: str | Path | None,
        variant: str | None,
        warmup: int,
        iters: int,
        reference_path: str | Path | None = None,
    ) -> ProfileResult:
        """Run VTune collection and parse the results."""
        with tempfile.TemporaryDirectory(prefix="xeforge_vtune_") as tmpdir:
            result_dir = Path(tmpdir) / "vtune_result"
            module_path = Path(tmpdir) / "kernel_mod.py"
            module_path.write_text(kernel_file.read_text())
            runner_script = self._generate_runner_script(
                module_path,
                warmup,
                iters,
                result_dir,
                spec_path=spec_path,
                variant=variant,
                reference_path=reference_path,
            )
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
                raw_counters={**counters.get(primary, {}), "collection_scope": "forward_only"},
            )

    def _generate_runner_script(
        self,
        kernel_file: Path,
        warmup: int,
        iters: int,
        result_dir: Path,
        spec_path: str | Path | None = None,
        variant: str | None = None,
        device: str = "xpu",
        reference_path: str | Path | None = None,
    ) -> str:
        return _runner_script(
            kernel_file,
            tool="vtune",
            warmup=warmup,
            iters=iters,
            spec_path=spec_path,
            variant=variant,
            device=device,
            reference_path=reference_path,
            vtune_bin=self.vtune_bin,
            result_dir=result_dir,
        )

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
                        "xpu_optimizations.yaml (xpu_tensor_descriptors, xpu_bf16)",
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


# Stall-cause metric names unitrace's EuStallSampling group reports, in the
# order its CSV emits them. "Active" is time spent executing, not stalled;
# every other column is a specific stall cause sampled at the EU's IP.
_EU_STALL_CAUSES = [
    "PSDepStall",
    "ControlStall",
    "PipeStall",
    "SendStall",
    "DistStall",
    "SbidStall",
    "SyncStall",
    "InstrFetchStall",
    "OtherStall",
]


class UnitraceXPUProfiler:
    """Separate ComputeBasic utilization or opt-in EuStallSampling collection."""

    def __init__(
        self,
        unitrace_bin: str = "unitrace",
        artifacts_dir: str | Path | None = None,
        parent_profile: str | Path | None = None,
        capture_assembly: bool = False,
        sampling_interval_us: int | None = None,
        metric_group: str = "ComputeBasic",
    ):
        self.unitrace_bin = unitrace_bin
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else None
        self.parent_profile = Path(parent_profile) if parent_profile is not None else None
        self.capture_assembly = capture_assembly
        if sampling_interval_us is not None and sampling_interval_us <= 0:
            raise ValueError("Sampling interval must be positive")
        self.sampling_interval_us = sampling_interval_us
        if metric_group not in ("ComputeBasic", "EuStallSampling"):
            raise ValueError(f"Unsupported metric group: {metric_group}")
        self.metric_group = metric_group

    def available(self) -> bool:
        return shutil.which(self.unitrace_bin) is not None

    def _environment(self):
        environment = {**os.environ, "ZE_FLAT_DEVICE_HIERARCHY": "FLAT"}
        binary = Path(shutil.which(self.unitrace_bin) or self.unitrace_bin).resolve()
        library_dir = binary.parent / "lib"
        if (library_dir / "libigdml.so.1").exists():
            environment["LD_LIBRARY_PATH"] = (
                str(library_dir) + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
            )
        return environment

    def profile(
        self,
        kernel_file: str | Path,
        spec_path: str | Path | None = None,
        variant: str | None = None,
        warmup: int = 5,
        iters: int = 20,
        reference_path: str | Path | None = None,
    ) -> ProfileResult:
        """Profile a kernel and return structured results.

        Returns an empty ProfileResult with ``error`` set if unitrace is not
        available or collection fails.
        """
        if not self.available():
            return ProfileResult(
                error="unitrace not found. Install it or set unitrace_bin path.",
                source="Unitrace",
            )

        kernel_file = Path(kernel_file)
        if not kernel_file.exists():
            return ProfileResult(error=f"Kernel file not found: {kernel_file}", source="Unitrace")

        try:
            capability = subprocess.run(
                [self.unitrace_bin, "--metric-list"],
                capture_output=True,
                text=True,
                timeout=30,
                env=self._environment(),
            )
            return self._collect_and_parse(
                kernel_file,
                spec_path,
                variant,
                warmup,
                iters,
                discovery={
                    "command": [self.unitrace_bin, "--metric-list"],
                    "returncode": capability.returncode,
                    "stdout": capability.stdout,
                    "stderr": capability.stderr,
                    "device_hierarchy": "FLAT",
                },
                reference_path=reference_path,
            )
        except Exception as e:
            logger.exception("Unitrace profiling failed")
            return ProfileResult(error=str(e), source="Unitrace")

    def _generate_runner_script(
        self,
        kernel_file: Path,
        warmup: int,
        iters: int,
        spec_path: str | Path | None = None,
        variant: str | None = None,
        device: str = "xpu",
        reference_path: str | Path | None = None,
    ) -> str:
        return _runner_script(
            kernel_file,
            tool="unitrace",
            warmup=warmup,
            iters=iters,
            spec_path=spec_path,
            variant=variant,
            device=device,
            reference_path=reference_path,
        )

    def _collect_and_parse(
        self,
        kernel_file: Path,
        spec_path: str | Path | None,
        variant: str | None,
        warmup: int,
        iters: int,
        discovery: dict | None = None,
        reference_path: str | Path | None = None,
    ) -> ProfileResult:
        """Retain unitrace artifacts for successful and failed collections."""
        root = (
            self.artifacts_dir
            if self.artifacts_dir is not None
            else kernel_file.parent / "profiles"
        )
        root.mkdir(parents=True, exist_ok=True)
        artifacts = Path(
            tempfile.mkdtemp(prefix=f"{kernel_file.stem}_unitrace_", dir=root)
        ).resolve()
        try:
            listing = ""
            if discovery is not None:
                (artifacts / "metric-list.json").write_text(json.dumps(discovery, indent=2))
                listing = discovery["stdout"] + discovery["stderr"]
            if discovery is not None and (
                discovery["returncode"] or not re.search(rf"\b{self.metric_group}\b", listing)
            ):
                result = ProfileResult(
                    error=f"{self.metric_group} unavailable on this device/driver or not accessible. "
                    "Check unitrace --metric-list, Metrics Library loading and counter permissions.\n"
                    + listing[-2000:],
                    source="Unitrace",
                )
            else:
                result = self._run_collection(
                    kernel_file, spec_path, variant, warmup, iters, artifacts, reference_path
                )
        except Exception as exc:
            logger.exception("Unitrace collection failed")
            result = ProfileResult(error=str(exc), source="Unitrace")
        if discovery is not None:
            result.raw_counters["metric_discovery"] = discovery
        if self.capture_assembly:
            from xe_forge.core.resource_feedback import index_assembly

            result.raw_counters["assembly"] = index_assembly(artifacts / "assembly")
        result.raw_counters["primary_kernel"] = result.primary_kernel
        provenance_path = artifacts / "build_provenance.json"
        if provenance_path.exists():
            provenance = json.loads(provenance_path.read_text())
            result.raw_counters["build_provenance"] = provenance
            result.raw_counters["device_identity"] = provenance.get("device")
        context_path = artifacts / "collection.json"
        if context_path.exists():
            collection = json.loads(context_path.read_text())
            result.raw_counters["collection_context"] = collection["context"]
            result.raw_counters["collection_scope"] = collection.get("scope")
        if self.parent_profile is not None:
            from xe_forge.core.resource_feedback import compare_resources

            try:
                parent = json.loads((self.parent_profile / "counters.json").read_text())
                comparison = compare_resources(result.raw_counters, parent)
            except (OSError, ValueError) as exc:
                comparison = {"status": f"parent profile unavailable: {exc}"}
            result.raw_counters["parent_comparison"] = comparison
        result.artifacts_dir = str(artifacts)
        (artifacts / "summary.txt").write_text(result.format_for_llm())
        (artifacts / "counters.json").write_text(json.dumps(result.raw_counters, indent=2))
        return result

    def _run_collection(
        self,
        kernel_file: Path,
        spec_path: str | Path | None,
        variant: str | None,
        warmup: int,
        iters: int,
        tmpdir: Path,
        reference_path: str | Path | None = None,
    ) -> ProfileResult:
        resolved_variant = variant
        if reference_path is not None:
            shutil.copyfile(reference_path, tmpdir / "reference.py")
            reference_path = tmpdir / "reference.py"
        if spec_path is not None:
            from xe_forge.core.spec_loader import load_spec

            shutil.copyfile(spec_path, tmpdir / "spec.yaml")
            spec_path = tmpdir / "spec.yaml"
            resolved_variant = load_spec(spec_path).resolve_variant(variant)
        (tmpdir / "collection.json").write_text(
            json.dumps(
                {
                    "kernel_file": str(kernel_file.resolve()),
                    "source_sha256": hashlib.sha256(kernel_file.read_bytes()).hexdigest(),
                    "spec_path": str(Path(spec_path).resolve()) if spec_path else None,
                    "scope": "forward_only",
                    "context": {
                        "spec_sha256": hashlib.sha256(Path(spec_path).read_bytes()).hexdigest()
                        if spec_path
                        else None,
                        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest()
                        if reference_path
                        else None,
                        "scope": "forward_only",
                        "variant": resolved_variant,
                        "warmup": warmup,
                        "iters": iters,
                        "metric_group": self.metric_group,
                        "sampling_interval_us": self.sampling_interval_us,
                        "device_hierarchy": "FLAT",
                    }
                    if spec_path or reference_path
                    else None,
                },
                indent=2,
            )
        )
        try:
            report_path = Path(tmpdir) / "unitrace_report.txt"
            collection_env = {
                **self._environment(),
                "PTI_ENABLE_COLLECTION": "0",
            }
            if self.capture_assembly:
                assembly_dir = tmpdir / "assembly"
                assembly_dir.mkdir()
                collection_env.update(
                    {
                        "IGC_ShaderDumpEnable": "1",
                        "IGC_DumpToCustomDir": str(assembly_dir),
                        "SYCL_CACHE_PERSISTENT": "0",
                        "NEO_CACHE_PERSISTENT": "0",
                    }
                )
            module_path = Path(tmpdir) / "kernel_mod.py"
            module_path.write_text(kernel_file.read_text())
            runner_script = self._generate_runner_script(
                module_path,
                warmup,
                iters,
                spec_path=spec_path,
                variant=variant,
                reference_path=reference_path,
            )
            runner_path = Path(tmpdir) / "runner.py"
            runner_path.write_text(runner_script)

            # Kernels built with torch.utils.cpp_extension.load_inline fork a
            # build subprocess (ninja/icpx) even on a cache hit (ninja's own
            # freshness check).
            preflight = subprocess.run(
                [sys.executable, str(runner_path)],
                capture_output=True,
                text=True,
                timeout=300,
                env=self._environment(),
            )
            (tmpdir / "preflight.stdout.txt").write_text(preflight.stdout or "")
            (tmpdir / "preflight.stderr.txt").write_text(preflight.stderr or "")
            if preflight.returncode != 0:
                return ProfileResult(
                    error=f"Profiling runner failed before collection: {preflight.stderr[-2000:]}",
                    source="Unitrace",
                )

            collect_cmd = [
                self.unitrace_bin,
                "--start-paused",
                "--device-timing",
                *(
                    ["--stall-sampling"]
                    if self.metric_group == "EuStallSampling"
                    else ["--metric-sampling", "--group", "ComputeBasic"]
                ),
                "--follow-child-process",
                "0",
                "-o",
                str(report_path),
                sys.executable,
                str(runner_path),
            ]
            if self.sampling_interval_us is not None:
                collect_cmd[1:1] = ["--sampling-interval", str(self.sampling_interval_us)]
            (tmpdir / "command.json").write_text(json.dumps(collect_cmd, indent=2))
            proc = subprocess.run(
                collect_cmd,
                capture_output=True,
                text=True,
                timeout=300,
                env=collection_env,
            )
            (tmpdir / "unitrace.stdout.txt").write_text(proc.stdout or "")
            (tmpdir / "unitrace.stderr.txt").write_text(proc.stderr or "")
            if proc.returncode != 0:
                return ProfileResult(
                    error=f"Unitrace collection failed: {proc.stderr[:500]}",
                    source="Unitrace",
                )

            # unitrace always appends a PID to the -o stem (<stem>.<pid><suffix>), and
            # writes the metrics CSV to a separate <stem>.metrics.<pid><suffix>.
            metrics_files = sorted(
                Path(tmpdir).glob(f"{report_path.stem}.metrics.*"), key=lambda p: p.stat().st_size
            )
            report_files = sorted(
                (
                    p
                    for p in Path(tmpdir).glob(f"{report_path.stem}.*")
                    if ".metrics." not in p.name
                ),
                key=lambda p: p.stat().st_size,
            )
            if len(report_files) > 1 or len(metrics_files) > 1:
                return ProfileResult(
                    error="Ambiguous unitrace reports: multiple timing or metrics files; "
                    "refusing to combine unrelated process reports. Inspect retained artifacts.",
                    source="Unitrace",
                )
            if not report_files and not metrics_files:
                return ProfileResult(
                    error="Unitrace produced no report "
                    f"(stdout: {proc.stdout[-300:]!r}, stderr: {proc.stderr[-300:]!r})",
                    source="Unitrace",
                )

            text = report_files[-1].read_text(errors="replace") if report_files else ""
            device_timing = self._extract_banner_section(text, "=== Device Timing Summary ===")
            properties = self._parse_kernel_properties(text)
            from xe_forge.core.compute_metrics import summarize_compute_metrics

            if self.metric_group == "EuStallSampling":
                stalls = (
                    self._parse_stall_csv(metrics_files[0].read_text(errors="replace"))
                    if metrics_files
                    else {}
                )
                timings = self._parse_device_timing(device_timing)
                raw = {
                    "device_timing": device_timing,
                    "kernel_properties": properties,
                    "timings": timings,
                    "stall_metrics": stalls,
                    "metric_group": self.metric_group,
                    "selection_basis": "total_device_time_ns",
                }
                if not timings or not stalls:
                    return ProfileResult(
                        error="No EU stall samples or device timings. Inspect sampling coverage and retained logs.",
                        source="Unitrace",
                        raw_counters=raw,
                    )
                unmatched = sorted(set(stalls) - set(timings))
                if unmatched:
                    raw["unassociated_stall_kernels"] = unmatched
                    raw["stall_temporal_isolation"] = "unverified"
                primary = max(timings, key=lambda name: timings[name]["time_ns"])
                if primary not in stalls:
                    return ProfileResult(
                        error="No EU stall samples for the hottest timed operation.",
                        source="Unitrace",
                        raw_counters=raw,
                    )
                metrics, percentages = self._build_metrics(stalls[primary])
                recommendations = self._generate_recommendations(percentages)
                if unmatched:
                    recommendations.insert(
                        0,
                        Recommendation(
                            "collection_scope",
                            "Stall samples include operations absent from forward timings. "
                            "Their events are excluded from the hotspot summary and retained in artifacts. "
                            "Even matching kernel samples may include paused work; temporal isolation is unverified.",
                        ),
                    )
                return ProfileResult(
                    primary_kernel=primary,
                    metrics=metrics,
                    recommendations=recommendations,
                    raw_counters=raw,
                    source="Unitrace",
                )
            if metrics_files:
                with metrics_files[0].open(errors="replace") as stream:
                    compute = summarize_compute_metrics(stream)
            else:
                compute = summarize_compute_metrics(text.splitlines())
            timings = self._parse_device_timing(device_timing)
            raw = {
                "device_timing": device_timing,
                "kernel_properties": properties,
                "timings": timings,
                "compute_metrics": compute,
                "metric_group": "ComputeBasic",
            }
            if not compute["records"]:
                return ProfileResult(
                    error="No ComputeBasic samples. Check group availability, counter permissions, "
                    "sampling interval and retained process logs. Unsampled work is unknown, not zero utilization.",
                    source="Unitrace",
                    raw_counters=raw,
                )
            if not timings:
                return ProfileResult(
                    error="No device timings available to select a hotspot.",
                    source="Unitrace",
                    raw_counters=raw,
                )
            primary = max(timings, key=lambda name: timings[name]["time_ns"])
            for record in compute["records"]:
                name = record["kernel"]
                if name not in timings:
                    name = re.sub(
                        r"\[SIMD\d+ \{\d+;\s*\d+;\s*\d+\} \{\d+;\s*\d+;\s*\d+\}\]$", "", name
                    )
                    if name.startswith("zeCommandListAppendMemory"):
                        name = re.sub(r"\[\d+\]$", "", name)
                record["timing_kernel"] = name if name in timings else None
                record["timed_launches"] = timings[name]["calls"] if name in timings else None
            missing_timings = sorted(
                {
                    record["kernel"]
                    for record in compute["records"]
                    if record["timing_kernel"] is None
                }
            )
            if missing_timings:
                raw["sampled_kernels_without_timings"] = missing_timings
                return ProfileResult(
                    error="Incomplete device timing coverage: sampled kernels have no timing records. "
                    "Do not infer workload hotspots from this collection.",
                    source="Unitrace",
                    raw_counters=raw,
                )
            raw["selection_basis"] = "total_device_time_ns"
            recommendations = []
            if not any(record["timing_kernel"] == primary for record in compute["records"]):
                recommendations.append(
                    Recommendation(
                        "missing_samples",
                        "No ComputeBasic samples for the hottest timed operation.",
                    )
                )
            if any(
                record["launches"] is None or record["launches"] < record["timed_launches"]
                for record in compute["records"]
            ):
                recommendations.append(
                    Recommendation(
                        "partial_samples",
                        "Some operations have fewer sampled launches than timed launches. Unsampled launches are unknown, not zero utilization.",
                    )
                )
            return ProfileResult(
                primary_kernel=primary,
                recommendations=recommendations,
                raw_counters=raw,
                source="Unitrace",
            )

        except subprocess.TimeoutExpired as exc:
            for label, output in (("stdout", exc.stdout), ("stderr", exc.stderr)):
                if isinstance(output, bytes):
                    output = output.decode(errors="replace")
                (tmpdir / f"timeout.{label}.txt").write_text(output or "")
            (tmpdir / "timeout.command.json").write_text(json.dumps(exc.cmd))
            return ProfileResult(
                error=f"Unitrace profiling timed out after {exc.timeout}s", source="Unitrace"
            )

    def _extract_banner_section(self, text: str, banner: str) -> str:
        """Return the text between ``banner`` and the next ``===`` banner or EOF."""
        lines = text.splitlines()
        try:
            start = next(i for i, ln in enumerate(lines) if banner in ln)
        except StopIteration:
            return ""
        end = next(
            (i for i in range(start + 1, len(lines)) if lines[i].strip().startswith("===")),
            len(lines),
        )
        return "\n".join(lines[start:end]).strip()

    def _parse_kernel_properties(self, text: str) -> dict[str, list[dict]]:
        """Preserve the fields reported by unitrace without inventing missing resources."""
        section = self._extract_banner_section(text, "=== Kernel Properties ===")
        properties: dict[str, list[dict]] = {}
        header = None
        for line in section.splitlines():
            if not line.strip() or line.strip().startswith("=="):
                header = None
                continue
            values = next(csv.reader([line.lstrip()], skipinitialspace=True))
            values = [value.strip() for value in values]
            if values[0] == "Kernel":
                header = values
                continue
            if header is None or len(values) != len(header):
                continue
            row = dict(zip(header, values, strict=True))
            name = row.pop("Kernel")
            if name:
                properties.setdefault(name, []).append(
                    {key: value if value else None for key, value in row.items()}
                )
        return properties

    def _parse_device_timing(self, text: str) -> dict[str, dict]:
        """Read unitrace timing tables, including padded and quoted kernel names."""
        timings: dict[str, dict] = {}
        header = None
        for line in text.splitlines():
            if not line.strip() or line.strip().startswith("=="):
                header = None
                continue
            fields = next(csv.reader([line.lstrip()], skipinitialspace=True))
            fields = [value.strip() for value in fields]
            if "Kernel" in fields and "Time (ns)" in fields:
                header = fields
                continue
            if header is None or len(fields) != len(header):
                continue
            row = dict(zip(header, fields, strict=True))
            name = row["Kernel"]
            time_ns = _safe_float(row["Time (ns)"])
            if not name or time_ns <= 0:
                continue
            timing = timings.setdefault(name, {"time_ns": 0.0, "calls": 0.0})
            timing["time_ns"] += time_ns
            timing["calls"] += _safe_float(row.get("Calls"))
        return timings

    def _parse_stall_csv(self, text: str) -> dict[str, dict]:
        """Read per-IP sampled EU events, not elapsed-time percentages.

        unitrace's ``-o`` report embeds this as a CSV table under a
        ``=== Device #N Metrics ===`` banner, headed by a row starting with
        ``Kernel,`` that includes one ``<Cause>Stall[Events]`` column per
        stall cause plus ``Active[Events]``. Each row is one sampled IP; this
        sums across every IP and instance for a kernel.
        """
        lines = text.splitlines()
        header_idx = next(
            (
                i
                for i, ln in enumerate(lines)
                if "OtherStall[Events]" in ln or ln.startswith("Kernel,")
            ),
            None,
        )
        if header_idx is None:
            return {}
        end_idx = next(
            (i for i in range(header_idx + 1, len(lines)) if lines[i].strip().startswith("===")),
            len(lines),
        )

        payload = "\n".join(ln.lstrip() for ln in lines[header_idx:end_idx] if ln.strip())
        reader = csv.DictReader(io.StringIO(payload), skipinitialspace=True)

        per_kernel: dict[str, dict] = {}
        for row in reader:
            name = (row.get("Kernel") or "").strip()
            if not name:
                continue
            agg = per_kernel.setdefault(
                name,
                {
                    "active_events": 0.0,
                    "stall_events": dict.fromkeys(_EU_STALL_CAUSES, 0.0),
                    "ip_samples": [],
                },
            )
            agg["ip_samples"].append(
                {
                    "ip": (row.get("IP[Address]") or "").strip(),
                    "active_events": _safe_float(row.get("Active[Events]")),
                    "stall_events": {
                        cause: _safe_float(row.get(f"{cause}[Events]"))
                        for cause in _EU_STALL_CAUSES
                    },
                }
            )
            agg["active_events"] += _safe_float(row.get("Active[Events]"))
            for cause in _EU_STALL_CAUSES:
                agg["stall_events"][cause] += _safe_float(row.get(f"{cause}[Events]"))

        for agg in per_kernel.values():
            agg["total_events"] = agg["active_events"] + sum(agg["stall_events"].values())

        return per_kernel

    def _build_metrics(self, agg: dict) -> tuple[ProfileMetrics, dict[str, float]]:
        total = agg["total_events"]
        if total <= 0:
            return ProfileMetrics(), {}

        active_pct = 100.0 * agg["active_events"] / total
        stall_pcts = {
            cause: 100.0 * events / total for cause, events in agg["stall_events"].items()
        }
        metrics = ProfileMetrics(
            sampled_active_pct=active_pct,
            sampled_stalled_pct=100.0 - active_pct,
        )
        return metrics, stall_pcts

    def _generate_recommendations(self, stall_pcts: dict[str, float]) -> list[Recommendation]:
        """Rank stall causes by share of recorded EU events."""
        ranked = sorted(
            ((cause, pct) for cause, pct in stall_pcts.items() if pct > 1.0),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return [
            Recommendation("stall_cause", f"{cause}: {pct:.1f}% of sampled EU events")
            for cause, pct in ranked
        ]


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
