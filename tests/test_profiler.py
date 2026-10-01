"""Tests for profiler workload preparation and unitrace report parsing."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from xe_forge.core.profiler import ProfileMetrics, ProfileResult, UnitraceXPUProfiler, XPUProfiler

# unitrace right-aligns the "Kernel" column to the width of the longest name
# in the table by padding with spaces *before* the first comma (no comma
# separates the padding from the header/short names), which breaks
# naive csv.DictReader lookups on "Kernel" and "<Cause>Stall[Events]" keys.
PADDED_METRICS_REPORT = """\
=== Device #0 Metrics ===

                                                      Kernel, IP[Address], Active[Events], PSDepStall[Events], ControlStall[Events], PipeStall[Events], SendStall[Events], DistStall[Events], SbidStall[Events], SyncStall[Events], InstrFetchStall[Events], OtherStall[Events]
                                             "gemm_kernel",           c0,              2,                  0,                    0,                 0,                 0,                 1,                 0,                 0,                       0,                 0
"at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::PowImplUnaryFunctor1<float> >",          80,             10,                  0,                    0,                 0,                 0,                 0,                 0,                 0,                      18,                 0
"""

TIMING_REPORT = """\
=== Device Timing Summary ===

    Total Execution Time (ns): 1000000
    Total Device Time for L0 backend (ns): 100000

== L0 Backend ==

      Kernel, Calls, Time (ns), Time (%), Average (ns), Min (ns), Max (ns)
 "gemm_kernel", 2, 90000, 90, 45000, 40000, 50000
 "at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::PowImplUnaryFunctor1<float> >", 1, 10000, 10, 10000, 10000, 10000

"""


RESOURCE_REPORT = """\
=== Kernel Properties ===

     Kernel, Compiled, SIMD, Number of Arguments, SLM Per Work Group, Private Memory Per Thread, Spill Memory Per Thread, Register File Size Per Thread
 "gemm_kernel", 1, 16, 3, 4096, 0, 128, 8192
 "other<float, int>", 1, 8, 2, 0, 16, ,

=== Another Section ===
"""


def test_unitrace_resource_parser_preserves_reported_fields():
    properties = UnitraceXPUProfiler()._parse_kernel_properties(RESOURCE_REPORT)
    assert properties["gemm_kernel"][0]["Spill Memory Per Thread"] == "128"
    assert properties["gemm_kernel"][0]["Register File Size Per Thread"] == "8192"
    assert properties["other<float, int>"][0]["Spill Memory Per Thread"] is None
    assert UnitraceXPUProfiler()._parse_kernel_properties("") == {}
    result = ProfileResult(
        source="Unitrace",
        primary_kernel="gemm_kernel",
        raw_counters={"kernel_properties": properties},
    )
    assert "Spill Memory Per Thread: 128" in result.format_for_llm()
    assert "knowledge_base/sycl/xpu/resource_feedback.yaml" in result.format_for_llm()


COMPUTE_REPORT = """\
=== Device #0 Metrics ===

 Kernel, GlobalInstanceId, GpuTime[ns], GPU_MEMORY_BYTE_READ[bytes], GPU_MEMORY_BYTE_WRITE[bytes], XVE_ACTIVE[%]
 "gemm_kernel", 1, 100, 200, 10, 20
 "gemm_kernel", 1, 300, 600, 30, 60
"""


@pytest.mark.parametrize("separator", ["\n", "\n\n"])
def test_compute_metrics_keeps_amount_rate_and_unknown_separate(separator):
    from xe_forge.core.compute_metrics import summarize_compute_metrics

    report = COMPUTE_REPORT.replace("\n", separator)
    summary = summarize_compute_metrics(report.splitlines())
    record = summary["records"][0]
    assert record["launches"] == 1
    assert record["memory"]["read"] == {"bytes": 800, "bytes_per_launch": 800, "gb_per_s": 2}
    assert record["counters"]["XVE_ACTIVE[%]"] == 50
    assert record["counters"].get("XVE_THREADS_OCCUPANCY_ALL[%]") is None
    broken = report.replace("1, 300, 600", "1, unknown, missing")
    record = summarize_compute_metrics(broken.splitlines())["records"][0]
    assert record["time_ns"] is None
    assert record["memory"]["read"]["bytes"] is None
    assert record["memory"]["write"]["bytes"] == 40
    assert record["memory"]["write"]["gb_per_s"] is None


def test_compute_metrics_does_not_merge_kernels_or_devices():
    from xe_forge.core.compute_metrics import summarize_compute_metrics

    report = COMPUTE_REPORT + COMPUTE_REPORT.replace("#0", "#1")
    report += COMPUTE_REPORT.replace('"gemm_kernel"', '"wrapper<gemm_kernel>"')
    records = summarize_compute_metrics(report.splitlines())["records"]
    assert len(records) == 3
    assert all(record["launches"] == 1 for record in records)


@pytest.mark.parametrize(
    "duration_column", ["GpuTime[ns]", "GPU_TIME[ns]", "Time[ns]", "GpuDuration[ns]"]
)
def test_compute_metrics_missing_launch_id_keeps_per_launch_amount_unknown(duration_column):
    from xe_forge.core.compute_metrics import summarize_compute_metrics

    report = COMPUTE_REPORT.replace("GpuTime[ns]", duration_column).replace("1, 300", ", 300")
    record = summarize_compute_metrics(report.splitlines())["records"][0]
    assert record["launches"] is None
    assert record["sampled_us_per_launch"] is None
    assert record["memory"]["read"] == {"bytes": 800, "bytes_per_launch": None, "gb_per_s": 2}


@pytest.mark.parametrize("emit_assembly", [False, True])
def test_unitrace_assembly_capture_is_opt_in_and_retained(tmp_path, monkeypatch, emit_assembly):
    import subprocess

    profiler = UnitraceXPUProfiler(capture_assembly=True)
    kernel = tmp_path / "kernel.cpp"
    kernel.write_text("source")
    monkeypatch.setattr(profiler, "_generate_runner_script", lambda *args, **kwargs: "runner")
    calls = []

    def run(command, **kwargs):
        calls.append(kwargs)
        if "-o" in command:
            environment = kwargs["env"]
            assert environment["IGC_ShaderDumpEnable"] == "1"
            assert environment["SYCL_CACHE_PERSISTENT"] == "0"
            assert environment["NEO_CACHE_PERSISTENT"] == "0"
            if emit_assembly:
                (Path(environment["IGC_DumpToCustomDir"]) / "kernel.asm").write_text(
                    "add (16) r1 r2 r3"
                )
            report = Path(command[command.index("-o") + 1])
            report.with_name(f"{report.stem}.1.txt").write_text(TIMING_REPORT + RESOURCE_REPORT)
            report.with_name(f"{report.stem}.metrics.1.txt").write_text(COMPUTE_REPORT)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    result = profiler._collect_and_parse(kernel, None, None, 1, 2)
    assert calls[0]["env"]["ZE_FLAT_DEVICE_HIERARCHY"] == "FLAT"
    assert "IGC_DumpToCustomDir" not in calls[0]["env"]
    manifest = json.loads((Path(result.artifacts_dir) / "assembly_manifest.json").read_text())
    assert (manifest["status"] == "captured") is emit_assembly
    assert "unverified" in manifest["ip_mapping"]
    if emit_assembly:
        assert len(manifest["files"][0]["sha256"]) == 64
    assert "Assembly:" in result.format_for_llm()


def test_unitrace_rejects_sampled_kernels_missing_from_timing(tmp_path, monkeypatch):
    import subprocess

    profiler = UnitraceXPUProfiler()
    kernel = tmp_path / "kernel.cpp"
    kernel.write_text("source")
    monkeypatch.setattr(profiler, "_generate_runner_script", lambda *args, **kwargs: "runner")

    def run(command, **kwargs):
        if "-o" in command:
            report = Path(command[command.index("-o") + 1])
            timing = "\n".join(
                line for line in TIMING_REPORT.splitlines() if '"gemm_kernel"' not in line
            )
            report.with_name(f"{report.stem}.1.txt").write_text(timing)
            report.with_name(f"{report.stem}.metrics.1.txt").write_text(COMPUTE_REPORT)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    result = profiler._collect_and_parse(kernel, None, None, 1, 2)
    assert "Incomplete device timing coverage" in result.error
    assert result.raw_counters["sampled_kernels_without_timings"] == ["gemm_kernel"]


@pytest.mark.parametrize("build_fails", [False, True])
def test_build_provenance_observes_without_changing_loader(tmp_path, monkeypatch, build_fails):
    import subprocess

    from torch.utils import cpp_extension

    from xe_forge.core import resource_feedback

    build_dir = tmp_path / "extension"
    build_dir.mkdir()
    (build_dir / "build.ninja").write_text("original build description")
    calls = []

    def load_inline(name, cpp_sources, extra_sycl_cflags=None):
        calls.append((name, cpp_sources, extra_sycl_cflags))
        if build_fails:
            raise RuntimeError("compiler failed")
        return SimpleNamespace(__file__=str(build_dir / "extension.so"))

    def run(command, **kwargs):
        output = (
            json.dumps([{"command": "/tools/icpx -fsycl -O2 kernel.cpp"}])
            if "compdb" in command
            else "Intel compiler test version"
        )
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(cpp_extension, "load_inline", load_inline)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    monkeypatch.setattr(resource_feedback.shutil, "which", lambda name: f"/tools/{Path(name).name}")
    monkeypatch.setattr(resource_feedback.subprocess, "run", run)
    if build_fails:
        with (
            pytest.raises(RuntimeError, match="compiler failed"),
            resource_feedback.capture_build_provenance(tmp_path),
        ):
            cpp_extension.load_inline("extension", "source", extra_sycl_cflags=["-O2"])
    else:
        with resource_feedback.capture_build_provenance(tmp_path):
            cpp_extension.load_inline("extension", "source", extra_sycl_cflags=["-O2"])
    assert calls == [("extension", "source", ["-O2"])]
    assert cpp_extension.load_inline is load_inline
    provenance = json.loads((tmp_path / "build_provenance.json").read_text())
    assert provenance["device"] is None
    build = provenance["builds"][0]
    assert build["arguments"]["extra_sycl_cflags"] == ["-O2"]
    if not build_fails:
        artifacts = tmp_path / build["artifacts"]
        assert (artifacts / "build.ninja").read_text() == "original build description"
        assert json.loads((artifacts / "compile_commands.json").read_text())[0]["command"].endswith(
            "kernel.cpp"
        )
        assert build["compile_drivers"]["/tools/icpx"]["version"] == "Intel compiler test version"


@pytest.mark.parametrize(
    "mismatch",
    [None, "collection_context", "device_identity", "primary_kernel", "missing", "ambiguous"],
)
def test_resource_comparison_requires_matching_evidence(mismatch):
    import copy

    from xe_forge.core.resource_feedback import compare_resources

    parent = {
        "collection_context": {"variant": "small", "spec_sha256": "spec", "iters": 20},
        "device_identity": {"name": "test GPU"},
        "primary_kernel": "kernel",
        "kernel_properties": {"kernel": [{"Spill Memory Per Thread": "128", "SIMD": "unknown"}]},
    }
    current = copy.deepcopy(parent)
    current["kernel_properties"]["kernel"][0] = {"Spill Memory Per Thread": "64", "SIMD": "16"}
    if mismatch == "missing":
        current.pop("device_identity")
    elif mismatch == "ambiguous":
        current["kernel_properties"]["kernel"].append({"SIMD": "8"})
    elif mismatch:
        current[mismatch] = "different"
    comparison = compare_resources(current, parent)
    if mismatch:
        assert "fields" not in comparison
    else:
        assert comparison["fields"]["Spill Memory Per Thread"]["delta"] == -64
        assert "delta" not in comparison["fields"]["SIMD"]
        result = ProfileResult(
            source="Unitrace",
            primary_kernel="kernel",
            raw_counters={"parent_comparison": comparison},
        )
        assert "128 -> 64 (delta -64)" in result.format_for_llm()


def test_parent_profile_comparison_is_persisted(tmp_path, monkeypatch):
    context = {"variant": "small", "spec_sha256": "spec", "warmup": 1, "iters": 2}
    device = {"name": "test GPU"}
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "counters.json").write_text(
        json.dumps(
            {
                "collection_context": context,
                "device_identity": device,
                "primary_kernel": "kernel",
                "kernel_properties": {"kernel": [{"Spill Memory Per Thread": "128"}]},
            }
        )
    )
    profiler = UnitraceXPUProfiler(parent_profile=parent)
    kernel = tmp_path / "kernel.cpp"
    kernel.write_text("source")

    def collect(kernel_file, spec_path, variant, warmup, iters, artifacts, reference_path):
        (artifacts / "collection.json").write_text(json.dumps({"context": context}))
        (artifacts / "build_provenance.json").write_text(json.dumps({"device": device}))
        return ProfileResult(
            source="Unitrace",
            primary_kernel="kernel",
            raw_counters={"kernel_properties": {"kernel": [{"Spill Memory Per Thread": "64"}]}},
        )

    monkeypatch.setattr(profiler, "_run_collection", collect)
    result = profiler._collect_and_parse(kernel, None, None, 1, 2)
    counters = json.loads((Path(result.artifacts_dir) / "counters.json").read_text())
    assert counters["parent_comparison"]["fields"]["Spill Memory Per Thread"]["delta"] == -64
    assert "delta -64" in result.format_for_llm()


@pytest.mark.parametrize("interval", [None, 25])
@pytest.mark.parametrize("metric_group", ["ComputeBasic", "EuStallSampling"])
def test_profile_cli_forwards_parent_profile(monkeypatch, interval, metric_group):
    import sys

    from xe_forge.core import profiler as profiler_module
    from xe_forge.skills import main

    received = []
    profile_arguments = []

    def create_profiler(**kwargs):
        received.append(kwargs)
        return SimpleNamespace(
            available=lambda: True,
            profile=lambda *args, **kwargs: (
                profile_arguments.append(kwargs) or ProfileResult(source="Unitrace")
            ),
        )

    monkeypatch.setattr(profiler_module, "UnitraceXPUProfiler", create_profiler)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "xe-forge-skill",
            "profile",
            "kernel.cpp",
            "--tool",
            "unitrace",
            "--parent-profile",
            "parent-artifacts",
            "--reference",
            "reference.py",
            "--spec",
            "spec.yaml",
            "--metric-group",
            metric_group,
            *(["--sampling-interval-us", str(interval)] if interval is not None else []),
        ],
    )
    main()
    assert received == [
        {
            "unitrace_bin": "unitrace",
            "parent_profile": "parent-artifacts",
            "capture_assembly": False,
            "sampling_interval_us": interval,
            "metric_group": metric_group,
        }
    ]
    assert profile_arguments[0]["reference_path"] == "reference.py"
    assert profile_arguments[0]["spec_path"] == "spec.yaml"


@pytest.mark.parametrize(
    "listing,returncode,available",
    [
        ("ComputeBasic", 0, True),
        ("EuStallSampling", 0, False),
        ("ComputeBasic: permission denied", 1, False),
    ],
)
def test_unitrace_discovers_compute_basic_before_workload(
    tmp_path, monkeypatch, listing, returncode, available
):
    import subprocess

    kernel = tmp_path / "kernel.py"
    kernel.write_text("pass\n")
    profiler = UnitraceXPUProfiler()
    monkeypatch.setattr(profiler, "available", lambda: True)
    collected = []

    def discover(command, **kwargs):
        assert command == ["unitrace", "--metric-list"]
        assert kwargs["env"]["ZE_FLAT_DEVICE_HIERARCHY"] == "FLAT"
        return SimpleNamespace(stdout=listing, stderr="", returncode=returncode)

    def collect(*args):
        collected.append(args)
        return ProfileResult(source="Unitrace")

    monkeypatch.setattr(subprocess, "run", discover)
    monkeypatch.setattr(profiler, "_run_collection", collect)
    result = profiler.profile(kernel)
    assert bool(collected) is available
    discovery = json.loads((Path(result.artifacts_dir) / "metric-list.json").read_text())
    assert discovery["stdout"] == listing
    if not available:
        assert "ComputeBasic unavailable" in result.error
        assert "Metrics Library" in result.error


@pytest.mark.parametrize("available", [True, False])
def test_profile_cli_exits_nonzero_on_failure(monkeypatch, available):
    from xe_forge.skills.profile import run

    monkeypatch.setattr(UnitraceXPUProfiler, "available", lambda self: available)
    monkeypatch.setattr(
        UnitraceXPUProfiler,
        "profile",
        lambda *args, **kwargs: ProfileResult(error="ComputeBasic unavailable"),
    )
    args = SimpleNamespace(
        tool="unitrace",
        unitrace_bin="unitrace",
        kernel_file="kernel.py",
        spec=None,
        variant=None,
        warmup=1,
        iters=2,
    )
    with pytest.raises(SystemExit) as error:
        run(args)
    assert error.value.code == 1


def test_resource_knowledge_base_reaches_generated_workspace(tmp_path):
    import yaml

    from xe_forge.claude.generator import generate_workspace
    from xe_forge.config import Config

    config = Config()
    config.device_config.dsl = "sycl"
    config.profiler.unitrace_enabled = True
    config.profiler.unitrace_metric_group = "all"
    workspace = tmp_path / "workspace"
    spec = tmp_path / "kernel.yaml"
    spec.write_text("name: kernel\n")
    generate_workspace(
        workspace, config, "kernel", "source", reference_code="reference", spec_path=str(spec)
    )
    guidance = workspace / "knowledge_base/sycl/xpu/resource_feedback.yaml"
    entries = yaml.safe_load(guidance.read_text())["patterns"]
    assert len({entry["id"] for entry in entries}) == len(entries)
    assert all(entry.get("description") and entry.get("applies_to") for entry in entries)
    compute = yaml.safe_load((guidance.parent / "compute_basic.yaml").read_text())["patterns"]
    assert len(compute) == 3
    for path in (workspace / "CLAUDE.md", workspace / ".claude/agents/tool-runner.md"):
        text = path.read_text()
        assert "resource_feedback.yaml" in text
        assert "compute_basic.yaml" in text
        assert "--metric-group EuStallSampling" in text
        assert "--metric-group all" in text
        assert "--parent-profile" in text
    optimizer = (workspace / "CLAUDE.md").read_text()
    assert (
        "Observation -> hypothesis -> proposed change -> expected counter change -> benchmark result"
        in optimizer
    )
    assert "High `SbidStall`" in optimizer
    assert "reset-plus-forward benchmark" in optimizer
    runner = (workspace / ".claude/agents/tool-runner.md").read_text()
    assert "leave bottleneck hypotheses and code changes to the optimizer" in runner
    assert "--require-profiles ComputeBasic EuStallSampling" in optimizer
    assert "--kernel-name kernel" in optimizer and "--trial-id <trial_id>" in optimizer
    assert "--require-profiles" in (workspace / ".claude/commands/optimize-kernel.md").read_text()


@pytest.mark.parametrize("available", [True, False])
def test_profile_all_records_trial_attempts_once(tmp_path, monkeypatch, available):
    from xe_forge.core.trial_manager import TrialManager
    from xe_forge.skills.profile import run

    manager = TrialManager(tmp_path / "trials")
    source = tmp_path / "kernel.cpp"
    source.write_text("baseline")
    manager.init("k", source, required_profile_groups=("ComputeBasic", "EuStallSampling"))
    manager.save_trial("k", source)
    source.write_text("candidate")
    trial = manager.save_trial("k", source)
    manager.record_result("k", trial, correctness="pass", speedup=1.1)

    collected = []
    monkeypatch.setattr(UnitraceXPUProfiler, "available", lambda self: available)

    def profile(self, *args, **kwargs):
        collected.append(self.metric_group)
        failed = self.metric_group == "EuStallSampling"
        return ProfileResult(
            source="Unitrace",
            artifacts_dir=f"art-{self.metric_group}",
            error="No EU stall samples" if failed else None,
        )

    monkeypatch.setattr(UnitraceXPUProfiler, "profile", profile)
    args = SimpleNamespace(
        tool="unitrace",
        unitrace_bin="unitrace",
        kernel_file=str(source),
        metric_group="all",
        spec=None,
        variant=None,
        warmup=1,
        iters=2,
        kernel_name="k",
        trial_id=trial,
        trials_dir=str(tmp_path / "trials"),
    )
    for _ in range(2):
        with pytest.raises(SystemExit) as error:
            run(args)
        assert error.value.code == 1
    assert collected == (["ComputeBasic", "EuStallSampling"] if available else [])
    profiles = manager._load_state("k")["trials"][trial]["profiles"]
    assert profiles["EuStallSampling"]["status"] == "failed"
    assert profiles["ComputeBasic"]["status"] == ("collected" if available else "failed")
    if available:
        assert profiles["ComputeBasic"]["artifacts_dir"] == "art-ComputeBasic"
    with pytest.raises(ValueError, match="--strategy"):
        manager.save_trial("k", source)
    manager.save_trial("k", source, strategy="kernel EuStallSampling SbidStall 53% (t1)")


@pytest.mark.parametrize("failed_group", [None, "ComputeBasic", "EuStallSampling"])
def test_profile_all_collects_both_groups(monkeypatch, capsys, failed_group):
    from xe_forge.skills.profile import run

    collected = []
    monkeypatch.setattr(UnitraceXPUProfiler, "available", lambda self: True)

    def profile(self, *args, **kwargs):
        collected.append(self.metric_group)
        return ProfileResult(
            source="Unitrace",
            primary_kernel="kernel",
            error="collection failed" if self.metric_group == failed_group else None,
        )

    monkeypatch.setattr(UnitraceXPUProfiler, "profile", profile)
    args = SimpleNamespace(
        tool="auto",
        unitrace_bin="unitrace",
        kernel_file="kernel.py",
        metric_group="all",
        spec=None,
        variant=None,
        warmup=1,
        iters=2,
    )
    if failed_group:
        with pytest.raises(SystemExit) as error:
            run(args)
        assert error.value.code == 1
    else:
        run(args)
    assert collected == ["ComputeBasic", "EuStallSampling"]
    output = capsys.readouterr().out
    assert "Collection: ComputeBasic" in output
    assert "Collection: EuStallSampling" in output


def test_unitrace_device_timing_parser():
    profiler = UnitraceXPUProfiler()
    timings = profiler._parse_device_timing(TIMING_REPORT)
    assert len(timings) == 2
    assert timings["gemm_kernel"] == {"time_ns": 90000, "calls": 2}
    assert sum(timing["time_ns"] for timing in timings.values()) == 100000


@pytest.mark.parametrize("missing_hot_samples", [False, True])
@pytest.mark.parametrize("interval", [None, 25])
def test_unitrace_selects_hotspot_by_time(tmp_path, monkeypatch, missing_hot_samples, interval):
    import subprocess

    profiler = UnitraceXPUProfiler(sampling_interval_us=interval)
    kernel_file = tmp_path / "kernel.cpp"
    kernel_file.write_text("source")
    reference_file = tmp_path / "reference.py"
    reference_file.write_text("reference")
    monkeypatch.setattr(profiler, "_generate_runner_script", lambda *args, **kwargs: "runner")

    def collect(command, **kwargs):
        if "-o" in command:
            report = Path(command[command.index("-o") + 1])
            assert command[command.index("--group") + 1] == "ComputeBasic"
            assert "--start-paused" in command
            assert kwargs["env"]["PTI_ENABLE_COLLECTION"] == "0"
            assert "EuStallSampling" not in command
            if interval is None:
                assert "--sampling-interval" not in command
            else:
                assert command[command.index("--sampling-interval") + 1] == str(interval)
            metrics = COMPUTE_REPORT + ' "other", 2, 50, 100, 0, 10\n'
            if missing_hot_samples:
                metrics = "\n".join(
                    line for line in metrics.splitlines() if '"gemm_kernel"' not in line
                )
            else:
                metrics = metrics.replace(
                    '"gemm_kernel"', '"gemm_kernel[SIMD16 {1; 1; 1} {256; 1; 1}]"'
                )
            report.with_name(f"{report.stem}.123.txt").write_text(
                TIMING_REPORT.replace(
                    ' "gemm_kernel",', ' "other", 1, 50, 0, 50, 50, 50\n "gemm_kernel",'
                )
                + RESOURCE_REPORT
            )
            report.with_name(f"{report.stem}.metrics.123.txt").write_text(metrics)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", collect)
    result = profiler._collect_and_parse(
        kernel_file, None, None, 1, 2, reference_path=reference_file
    )
    assert result.error is None
    assert result.primary_kernel == "gemm_kernel"
    assert result.raw_counters["selection_basis"] == "total_device_time_ns"
    assert result.raw_counters["timings"]["gemm_kernel"]["time_ns"] == 90000
    artifacts = Path(result.artifacts_dir)
    assert artifacts.parent == tmp_path / "profiles"
    assert "Time (ns)" in (artifacts / "unitrace_report.123.txt").read_text()
    assert result.raw_counters["kernel_properties"]["gemm_kernel"][0]["SIMD"] == "16"
    assert "GpuTime[ns]" in (artifacts / "unitrace_report.metrics.123.txt").read_text()
    assert (artifacts / "kernel_mod.py").read_text() == "source"
    assert (artifacts / "runner.py").read_text() == "runner"
    assert (artifacts / "reference.py").read_text() == "reference"
    assert result.raw_counters["collection_context"]["reference_sha256"]
    assert result.raw_counters["collection_context"]["scope"] == "forward_only"
    assert "Collection scope: forward only" in result.format_for_llm()
    assert str(artifacts) in (artifacts / "summary.txt").read_text()
    assert json.loads((artifacts / "command.json").read_text())[0] == "unitrace"
    assert "Top device operations by total time" in result.format_for_llm()
    if missing_hot_samples:
        assert result.recommendations[0].category == "missing_samples"
    else:
        saved = json.loads((artifacts / "counters.json").read_text())
        assert saved["compute_metrics"]["records"][0]["memory"]["read"]["bytes"] == 800
        assert saved["compute_metrics"]["records"][0]["timing_kernel"] == "gemm_kernel"
        assert "SIMD16" in saved["compute_metrics"]["records"][0]["kernel"]
        assert "includes setup/warmup" not in result.format_for_llm()


@pytest.mark.parametrize(
    "failure", ["preflight", "collection", "timeout", "ambiguous", "no_timing"]
)
def test_unitrace_retains_failed_collection_artifacts(tmp_path, monkeypatch, failure):
    import subprocess

    profiler = UnitraceXPUProfiler(artifacts_dir=tmp_path / "retained")
    kernel_file = tmp_path / "kernel.cpp"
    kernel_file.write_text("source")
    monkeypatch.setattr(profiler, "_generate_runner_script", lambda *args, **kwargs: "runner")
    commands = []

    def collect(command, **kwargs):
        commands.append(command)
        if "-o" not in command:
            return subprocess.CompletedProcess(
                command,
                1 if failure == "preflight" else 0,
                stdout="preflight",
                stderr="build details",
            )
        report = Path(command[command.index("-o") + 1])
        report.with_name(f"{report.stem}.metrics.123.txt").write_text(COMPUTE_REPORT)
        if failure != "no_timing":
            report.with_name(f"{report.stem}.123.txt").write_text(TIMING_REPORT)
        if failure == "ambiguous":
            report.with_name(f"{report.stem}.456.txt").write_text(TIMING_REPORT)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(
                command, 300, output=b"partial output", stderr=b"timeout details"
            )
        return subprocess.CompletedProcess(
            command,
            1 if failure == "collection" else 0,
            stdout="collector",
            stderr="collector details",
        )

    monkeypatch.setattr(subprocess, "run", collect)
    result = profiler._collect_and_parse(kernel_file, None, None, 1, 2)
    assert result.error is not None
    artifacts = Path(result.artifacts_dir)
    assert artifacts.parent == tmp_path / "retained"
    assert (artifacts / "kernel_mod.py").exists()
    assert (artifacts / "preflight.stderr.txt").read_text() == "build details"
    assert result.error in (artifacts / "summary.txt").read_text()
    assert str(artifacts) in result.format_for_llm()
    if failure == "preflight":
        assert len(commands) == 1
    elif failure == "timeout":
        assert (artifacts / "timeout.stderr.txt").read_text() == "timeout details"
        assert (artifacts / "unitrace_report.123.txt").exists()
    elif failure == "ambiguous":
        assert "Ambiguous" in result.error
        assert (artifacts / "unitrace_report.456.txt").exists()
    elif failure == "no_timing":
        assert "No device timings" in result.error
    else:
        assert (artifacts / "unitrace.stderr.txt").read_text() == "collector details"


def test_unitrace_labels_sample_shares_not_utilization():
    profiler = UnitraceXPUProfiler()
    aggregate = profiler._parse_stall_csv(PADDED_METRICS_REPORT)["gemm_kernel"]
    metrics, percentages = profiler._build_metrics(aggregate)
    assert metrics.sampled_active_pct == pytest.approx(200 / 3)
    assert metrics.sampled_stalled_pct == pytest.approx(100 / 3)
    assert metrics.xve_active_pct is None
    assert metrics.peak_occupancy_pct is None
    result = ProfileResult(
        source="Unitrace",
        primary_kernel="gemm_kernel",
        metrics=metrics,
        recommendations=profiler._generate_recommendations(percentages),
    )
    output = result.format_for_llm()
    assert "Active samples:" in output
    assert "sampled EU events" in output
    assert "not elapsed-time utilization or occupancy" in output
    assert "XVE Active:" not in output
    assert "sampled EU time" not in output
    vtune = ProfileResult(primary_kernel="kernel", metrics=ProfileMetrics(xve_active_pct=50))
    assert "XVE Active:" in vtune.format_for_llm()


def test_parse_stall_csv_handles_kernel_column_padding(tmp_path, monkeypatch):
    import subprocess

    p = UnitraceXPUProfiler()
    per_kernel = p._parse_stall_csv(PADDED_METRICS_REPORT)

    assert set(per_kernel) == {
        "gemm_kernel",
        "at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::PowImplUnaryFunctor1<float> >",
    }
    gemm = per_kernel["gemm_kernel"]
    assert gemm["active_events"] == 2.0
    assert gemm["stall_events"]["DistStall"] == 1.0
    assert gemm["total_events"] == 3.0

    kernel = tmp_path / "kernel.py"
    kernel.write_text("pass\n")
    profiler = UnitraceXPUProfiler(metric_group="EuStallSampling")

    def collect(command, **kwargs):
        if "-o" in command:
            assert "--stall-sampling" in command
            assert "--metric-sampling" not in command
            report = Path(command[command.index("-o") + 1])
            report.with_name(f"{report.stem}.1.txt").write_text(TIMING_REPORT)
            metrics = PADDED_METRICS_REPORT.replace("\n", "\n\n")
            metrics = "\n".join(
                line for line in metrics.splitlines() if "PowImplUnaryFunctor1" not in line
            )
            report.with_name(f"{report.stem}.metrics.1.txt").write_text(metrics)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", collect)
    result = profiler._collect_and_parse(kernel, None, None, 1, 2)
    assert result.error is None
    assert result.raw_counters["metric_group"] == "EuStallSampling"
    assert "DistStall: 1 events" in result.format_for_llm()
    assert "not elapsed-time utilization" in result.format_for_llm()


def test_unitrace_drains_completions_while_paused(monkeypatch):
    import os

    from xe_forge.core import profile_runner
    from xe_forge.core.reference_workload import PreparedCall

    events = []

    def record_event(action):
        events.append((action, os.environ.get("PTI_ENABLE_COLLECTION")))

    monkeypatch.setattr(
        profile_runner,
        "prepare_call",
        lambda *args: PreparedCall(lambda: record_event("forward"), []),
    )
    monkeypatch.setattr(torch.xpu, "synchronize", lambda: None)
    monkeypatch.setattr(
        torch.xpu,
        "Event",
        lambda: SimpleNamespace(
            record=lambda: record_event("record"),
            synchronize=lambda: record_event("drain"),
        ),
    )
    monkeypatch.setenv("PTI_ENABLE_COLLECTION", "0")
    profile_runner.run_profile("kernel.py", tool="unitrace", warmup=0, iters=2)
    assert events == [("forward", "1"), ("forward", "1"), ("record", "0"), ("drain", "0")]


def test_shared_preparation_preserves_model_inputs_and_init_args():
    from xe_forge.core.executor import KernelBenchExecutor

    class Model(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.width = width
            self.weight = torch.nn.Parameter(torch.ones(width))

        def get_example_inputs(self, shapes, device):
            assert shapes == [(3,)]
            return [
                torch.ones(self.width, dtype=torch.float32),
                torch.tensor([1 + 2j], dtype=torch.complex64),
                torch.tensor([2], dtype=torch.int64),
                torch.tensor([True]),
                7,
            ]

        def forward(self, values, freqs, indices, mask, position):
            return values + self.weight

    module = SimpleNamespace(Model=Model, get_init_inputs=lambda: [99])
    executor = KernelBenchExecutor(device="cpu")
    forward, model, inputs = executor.prepare_workload(
        module, input_shapes=[(3,)], init_args=[3], dtype=torch.float64
    )

    assert model.width == 3
    assert model.weight.dtype == torch.float64
    assert not model.training
    assert [value.dtype for value in inputs[:4]] == [
        torch.float32,
        torch.complex64,
        torch.int64,
        torch.bool,
    ]
    assert inputs[4] == 7
    torch.testing.assert_close(forward(*inputs), torch.full((3,), 2.0, dtype=torch.float64))


def test_execute_uses_shared_preparation(monkeypatch):
    from xe_forge.core.executor import KernelBenchExecutor

    executor = KernelBenchExecutor(device="cpu")
    module = SimpleNamespace()
    calls = []

    def prepare_workload(actual_module, **kwargs):
        assert not torch.is_grad_enabled()
        assert actual_module is module
        calls.append(kwargs)
        return lambda values: values + 1, None, [torch.ones(3)]

    monkeypatch.setattr(executor, "_compile_module", lambda code: module)
    monkeypatch.setattr(executor, "prepare_workload", prepare_workload)
    monkeypatch.setattr(executor, "time", lambda fn, inputs: 5.0)
    result = executor.execute("source", input_shapes=[(3,)], dtype=torch.float32, flop=3)

    assert result.success
    assert result.execution_time_ms == 0.005
    assert calls == [
        {
            "kernel_name": None,
            "input_shapes": [(3,)],
            "inputs": None,
            "dtype": torch.float32,
            "init_args": None,
            "input_dtypes": None,
        }
    ]


@pytest.mark.parametrize("profiler_type", [UnitraceXPUProfiler, XPUProfiler])
@pytest.mark.parametrize("variant,width", [(None, 3), ("large", 7)])
@pytest.mark.parametrize("model_inputs", [False, True])
def test_profiler_runner_matches_benchmark_preparation(
    tmp_path, monkeypatch, profiler_type, variant, width, model_inputs
):
    import subprocess

    from xe_forge.core.executor import KernelBenchExecutor
    from xe_forge.core.spec_loader import load_spec

    source = """\
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.calls = 0

    def forward(self, values, indices, mask, *extra):
        assert not self.training
        assert not torch.is_grad_enabled()
        self.calls += 1
        return values + self.weight
"""
    if model_inputs:
        source += """\

    def get_example_inputs(self, shapes, device):
        width = shapes[0][0]
        return [torch.ones(width), torch.zeros(width, dtype=torch.int64),
                torch.ones(width, dtype=torch.bool), torch.tensor([1 + 2j]), 9]
"""
    kernel_path = tmp_path / "trial's kernel.cpp"
    kernel_path.write_text(source)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text("""\
inputs:
  values: {shape: [N], dtype: float64}
  indices: {shape: [N], dtype: int64}
  mask: {shape: [N], dtype: bool}
default_variant: small
small:
  - params: [values, indices, mask]
    dims: {N: 3}
large:
  - params: [values, indices, mask]
    dims: {N: 7}
""")
    profiler = profiler_type()
    extra = {"result_dir": tmp_path / "results"} if profiler_type is XPUProfiler else {}
    script = profiler._generate_runner_script(
        kernel_path, 1, 2, spec_path=spec_path, variant=variant, device="cpu", **extra
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)
    namespace = {}
    torch.manual_seed(123)
    exec(compile(script, "profiler_runner", "exec"), namespace)

    spec = load_spec(spec_path)
    resolved = spec.resolve_variant(variant)
    executor = KernelBenchExecutor(device="cpu")
    module = executor._compile_module(source)
    torch.manual_seed(123)
    _, model, inputs = executor.prepare_workload(
        module,
        input_shapes=spec.get_input_shapes(resolved),
        dtype=spec.get_dtype(resolved),
        input_dtypes=spec.get_input_dtypes(resolved),
        init_args=spec.get_init_args(resolved),
    )
    call = namespace["call"]
    assert call.model.calls == 3
    assert call.model.weight.dtype == model.weight.dtype == torch.float64
    assert call.inputs[0].shape == (width,)
    for actual, expected in zip(call.inputs, inputs, strict=True):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected)
        else:
            assert actual == expected


def test_profiler_rejects_unknown_variant(tmp_path):
    from xe_forge.core.profile_runner import prepare_call

    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text("inputs: {}\n")
    with pytest.raises(ValueError, match="Unknown profiling variant"):
        prepare_call(tmp_path / "kernel.cpp", spec_path=spec_path, variant="missing", device="cpu")


def test_profiler_builds_the_reference_workload_from_the_spec_variant(tmp_path):
    from xe_forge.core.profile_runner import prepare_call

    source = """\
import torch

class Model(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, values):
        return values * 2

def get_init_inputs(WIDTH=4):
    return [WIDTH]

def get_inputs(WIDTH=4):
    return [torch.ones(WIDTH)]
"""
    for name in ("kernel.py", "reference.py"):
        (tmp_path / name).write_text(source)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text("ci:\n  - dims: {WIDTH: 3}\nbench-gpu:\n  - dims: {WIDTH: 6}\n")
    for variant, width in (("ci", 3), (None, 6)):
        call = prepare_call(
            tmp_path / "kernel.py",
            spec_path=spec_path,
            variant=variant,
            reference_path=tmp_path / "reference.py",
            device="cpu",
        )
        assert call.model.width == width
        assert call.inputs[0].shape == (width,)


@pytest.mark.parametrize("profiler_type", [UnitraceXPUProfiler, XPUProfiler])
@pytest.mark.parametrize("reference_mode", ["none", "valid", "invalid", "raises"])
def test_spec_free_profiling_preserves_input_types(
    tmp_path, monkeypatch, profiler_type, reference_mode
):
    import os
    import subprocess

    kernel_path = tmp_path / "kernel.cpp"
    kernel_path.write_text("""\
import os
import torch

class Model(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        assert not torch.is_grad_enabled()
        self.scale = scale
        self.register_buffer("cache", torch.zeros(1), persistent=False)
        self.collected = []

    def forward(self, values, freqs, indices, mask, position):
        assert not self.training
        assert not torch.is_grad_enabled()
        assert self.scale == 2
        self.cache += 1
        self.collected.append(os.environ.get("PTI_ENABLE_COLLECTION") == "1")
        return values * self.scale

def get_init_inputs():
    return [2]

def get_inputs():
    return [torch.ones(3), torch.tensor([1 + 2j]),
            torch.tensor([1]), torch.tensor([True]), 7]
""")
    reference_path = None
    if reference_mode != "none":
        reference_path = tmp_path / "reference.py"
        reference_path.write_text(kernel_path.read_text())
        if reference_mode == "invalid":
            kernel_path.write_text(
                kernel_path.read_text().replace("return values * self.scale", "return values * 3")
            )
        elif reference_mode == "raises":
            kernel_path.write_text(
                kernel_path.read_text().replace(
                    "return values * self.scale",
                    'if self.collected[-1]:\n            raise RuntimeError("forward failed")\n        return values * self.scale',
                )
            )
    extra = {"result_dir": tmp_path / "results"} if profiler_type is XPUProfiler else {}
    script = profiler_type()._generate_runner_script(
        kernel_path, 1, 2, device="cpu", reference_path=reference_path, **extra
    )
    transitions = []

    def control(command, **kwargs):
        if "-command" in command:
            assert kwargs["check"]
            enabled = command[command.index("-command") + 1] == "resume"
            transitions.append(enabled)
            os.environ["PTI_ENABLE_COLLECTION"] = "1" if enabled else "0"

    monkeypatch.setattr(subprocess, "run", control)
    monkeypatch.setenv("PTI_ENABLE_COLLECTION", "0")
    namespace = {}
    if reference_mode in ("invalid", "raises"):
        message = "Output differs" if reference_mode == "invalid" else "forward failed"
        with pytest.raises((ValueError, RuntimeError), match=message):
            exec(compile(script, "profiler_runner", "exec"), namespace)
        assert os.environ["PTI_ENABLE_COLLECTION"] == "0"
        if reference_mode == "invalid":
            assert True not in transitions
        return
    exec(compile(script, "profiler_runner", "exec"), namespace)
    call = namespace["call"]
    assert [value.dtype for value in call.inputs[:4]] == [
        torch.float32,
        torch.complex64,
        torch.int64,
        torch.bool,
    ]
    assert call.inputs[4] == 7
    assert call.model.collected == ([False, False] if reference_mode == "valid" else []) + [
        False,
        True,
        True,
    ]
    assert call.model.cache.item() == (1 if reference_mode == "valid" else 3)
    assert os.environ["PTI_ENABLE_COLLECTION"] == "0"
    if profiler_type is XPUProfiler:
        assert transitions == [False, True, False, True, False]


@pytest.mark.parametrize("skill", ["benchmark", "profile"])
@pytest.mark.parametrize("variant", [None, "large"])
def test_cli_defers_variant_default_to_spec(monkeypatch, skill, variant):
    import importlib
    import sys

    from xe_forge.skills import main

    arguments = ["xe-forge-skill", skill]
    if skill == "benchmark":
        arguments.append("baseline.py")
    arguments.extend(["trial.py", "--spec", "spec.yaml"])
    if variant is not None:
        arguments.extend(["--variant", variant])
    received = []
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(importlib.import_module(f"xe_forge.skills.{skill}"), "run", received.append)
    main()
    assert len(received) == 1
    assert received[0].variant == variant
