"""The host as a device: the placement a run inherited, and calibration tied to it."""

from __future__ import annotations

import pytest

from xe_forge.core import topology


def _host(root, sockets=2, cores=2, smt=2, llc="1024K"):
    """A sysfs/procfs tree: ``sockets`` x ``cores`` x ``smt`` CPUs, one NUMA node and one
    shared last-level cache per socket; CPUs numbered core-major, siblings after."""
    per_socket = cores * smt
    n = sockets * per_socket
    sysfs = root / "sys/devices/system"
    for c in range(n):
        socket = (c % (sockets * cores)) // cores
        core = c % (sockets * cores)
        siblings = [core + k * sockets * cores for k in range(smt)]
        same_socket = [x for x in range(n) if (x % (sockets * cores)) // cores == socket]
        topo = sysfs / f"cpu/cpu{c}/topology"
        topo.mkdir(parents=True)
        (topo / "physical_package_id").write_text(f"{socket}\n")
        (topo / "thread_siblings_list").write_text(topology.format_list(siblings) + "\n")
        for idx, (level, kind, size, shared) in enumerate(
            [
                (1, "Data", "48K", siblings),
                (1, "Instruction", "32K", siblings),
                (3, "Unified", llc, same_socket),
            ]
        ):
            d = sysfs / f"cpu/cpu{c}/cache/index{idx}"
            d.mkdir(parents=True)
            (d / "level").write_text(f"{level}\n")
            (d / "type").write_text(f"{kind}\n")
            (d / "size").write_text(f"{size}\n")
            (d / "shared_cpu_list").write_text(topology.format_list(shared) + "\n")
    for s in range(sockets):
        node = sysfs / f"node/node{s}"
        node.mkdir(parents=True)
        mine = [x for x in range(n) if (x % (sockets * cores)) // cores == s]
        (node / "cpulist").write_text(topology.format_list(mine) + "\n")
    (root / "proc/self").mkdir(parents=True)
    (root / "proc/self/status").write_text("Name:\tpython\nMems_allowed_list:\t0\n")
    (root / "proc/cpuinfo").write_text(
        "processor\t: 0\nmodel name\t: Test CPU 9000\nflags\t\t: fpu sse2 avx2\n"
    )
    return n


@pytest.mark.parametrize(
    "text,ids", [("0-3,8,10-11", [0, 1, 2, 3, 8, 10, 11]), ("5", [5]), ("", [])]
)
def test_cpu_lists_read_and_write_back(text, ids):
    assert topology.parse_list(text) == ids
    assert topology.format_list(ids) == text


def test_a_placement_reports_the_sockets_caches_and_siblings_it_spans(tmp_path):
    n = _host(tmp_path)  # 2 sockets x 2 cores x 2 threads: CPUs 0-1 and 4-5 on socket 0
    one = topology.placement(tmp_path, cpus=[0, 1, 4, 5], env={})
    assert one["sockets"] == [0] and one["numa_nodes"] == {0: "0-1,4-5"}
    assert one["smt_siblings"] == 2 and one["llc_bytes"] == 1024 << 10
    assert one["memory_nodes"] == "0"

    both = topology.placement(tmp_path, cpus=range(n), env={})
    assert both["sockets"] == [0, 1] and both["cpu_count"] == 8
    assert both["llc_bytes"] == 2 * (1024 << 10)  # one slice per socket, counted once
    assert topology.placement_tag(one) != topology.placement_tag(both)
    assert topology.model_name(tmp_path) == "Test CPU 9000"
    assert topology.cpu_flags(tmp_path) == ["fpu", "sse2", "avx2"]


def test_a_memory_policy_is_part_of_the_placement(tmp_path):
    _host(tmp_path)
    default = topology.placement(tmp_path, cpus=[0], env={})
    assert default["memory_policy"] is None  # no numa_maps: unknown, not "default"
    maps = tmp_path / "proc/self/numa_maps"
    maps.write_text("55d0c0000000 default file=/usr/bin/python3 mapped=1\n")
    default = topology.placement(tmp_path, cpus=[0], env={})
    maps.write_text("55d0c0000000 bind:0 file=/usr/bin/python3 mapped=1\n")
    bound = topology.placement(tmp_path, cpus=[0], env={})
    assert (default["memory_policy"], bound["memory_policy"]) == ("default", "bind:0")
    assert topology.placement_tag(default) != topology.placement_tag(bound)
    assert ":" not in topology.placement_tag(bound)


def test_a_placement_keeps_only_threading_settings_from_the_environment(tmp_path):
    _host(tmp_path)
    env = {"OMP_NUM_THREADS": "4", "KMP_AFFINITY": "compact", "HOME": "/h", "TOKEN": "x"}
    got = topology.placement(tmp_path, cpus=[0], env=env)["thread_env"]
    assert got == {"KMP_AFFINITY": "compact", "OMP_NUM_THREADS": "4"}


def _cpu_device(cpus="0-3", mems="0", llc=1 << 20):
    return {"type": "cpu", "name": "Test CPU 9000",
            "placement": {"cpus": cpus, "memory_nodes": mems, "llc_bytes": llc}}  # fmt: skip


def test_a_calibration_from_another_placement_is_named_and_not_merged():
    from xe_forge.frontend import calibrate

    one, two = _cpu_device("0-3", "0"), _cpu_device("0-7", "0-1")
    assert calibrate._identity(one) != calibrate._identity(two)
    assert calibrate.default_path({"device": one}) != calibrate.default_path({"device": two})
    note = calibrate.other_device({"device": one}, {"device": two})
    assert note and "0-3" in note and "0-7" in note
    assert calibrate.other_device({"device": one}, {"device": dict(one)}) is None

    calib = {"device": one, "matmul_tflops": {"float32": 1.0}}
    rec = {"device": two, "matmul_tflops": {"bfloat16": 2.0}}
    assert calibrate.complete(calib, {"bfloat16"}, lambda ds: rec) is calib


def test_host_calibration_measures_both_first_touches_and_refuses_a_cache_sized_read(monkeypatch):
    import torch

    from xe_forge.frontend import calibrate, profile

    monkeypatch.setenv("XE_FORGE_DEVICE", "cpu")
    monkeypatch.setattr(profile, "device_info", lambda: _cpu_device(llc=1 << 20))
    monkeypatch.setattr(torch, "get_num_threads", lambda: 2)
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)

    rec = calibrate.measure(read_bytes=4 << 20, n=8, iters=1, dtypes=("float32",))
    assert rec["bandwidth_gbs"] > 0 and rec["bandwidth_gbs_one_writer"] > 0
    # A tiny matmul is slow, and is recorded as the small rate it is, not as 0.
    assert rec["matmul_tflops"]["float32"] > 0 and not rec["unmeasured"]

    small = calibrate.measure(read_bytes=1 << 20, n=8, iters=1, dtypes=("float32",))
    assert small["bandwidth_gbs"] is None
    assert "last-level cache" in small["unmeasured"]["bandwidth_gbs"]


SUMMARY_TSV = (
    "Hierarchy Level\tMetric Name\tMetric Value\n"
    "0\tElapsed Time\t2.5\n"
    "1\tMemory Bound\t40.0\n"
    "2\tDRAM Bound\t\n"
    "0\tBandwidth Domain\tPlatform Maximum\tObserved Maximum\n"
)
HOTSPOTS_TSV = "Function\tCPU Time\tModule\nmy_kernel\t1.5\text.so\nomp_wait\t0.2\tlibomp.so\n"


def test_vtune_reports_are_read_as_vtune_printed_them():
    from xe_forge.core.profiler import parse_vtune_summary, parse_vtune_table

    rows = parse_vtune_summary(SUMMARY_TSV)
    assert rows[:3] == [
        {"level": 0, "metric": "Elapsed Time", "value": "2.5"},
        {"level": 1, "metric": "Memory Bound", "value": "40.0"},
        {"level": 2, "metric": "DRAM Bound", "value": ""},
    ]
    assert rows[3] == {"cells": ["0", "Bandwidth Domain", "Platform Maximum", "Observed Maximum"]}
    table = parse_vtune_table(HOTSPOTS_TSV)
    assert table[0] == {"Function": "my_kernel", "CPU Time": "1.5", "Module": "ext.so"}


def _host_report(sections, window=None, **kw):
    placement = {"cpus": "0-3", "sockets": [0], "memory_nodes": "0", "memory_policy": "bind:0",
                 "llc_bytes": 4096, "threads": 4}  # fmt: skip
    return {"tool": "VTune", "analysis": "hotspots", "placement": placement, "window": window,
            "sections": sections, **kw}  # fmt: skip


def test_a_host_profile_reports_every_metric_and_no_recommendation():
    from xe_forge.core.profiler import ProfileResult, parse_vtune_summary, parse_vtune_table

    window = {"window": "all", "iters": 50, "forward_s": 0.03, "window_s": 0.04,
              "window_max_s": 0.05}  # fmt: skip
    functions = parse_vtune_table(HOTSPOTS_TSV)
    result = ProfileResult(primary_kernel="my_kernel", source="VTune hotspots", raw_counters={
        "host": _host_report([{"title": "summary", "rows": parse_vtune_summary(SUMMARY_TSV)},
                       {"title": "Hotspots by function", "rows": functions[:1], "total": 2}],
                      window)})  # fmt: skip
    text = result.format_for_llm()
    assert "CPUs 0-3 on sockets [0]" in text and "memory policy bind:0" in text
    assert "4 torch threads" in text
    assert "one window over 50 timed calls" in text and "took 0.03 s" in text
    assert "between 0.04 s and 0.05 s" in text
    assert "No cache flush runs between profiled calls" in text and "(4096 bytes)" in text
    assert "    Memory Bound: 40.0" in text and "DRAM Bound: unknown" in text
    assert "(top 1 of 2)" in text and "my_kernel; CPU Time=1.5" in text
    assert "Recommendations" not in text and "Metrics:" not in text


def test_a_failed_section_and_a_missing_window_are_unknown_not_empty():
    from xe_forge.core.profiler import ProfileResult

    host = _host_report([{"title": "Hotspots by function", "error": "the hotspots report failed: x"}],
                 {"iters": 3}, unmeasured={"cycles": "<not supported>"})  # fmt: skip
    text = ProfileResult(primary_kernel="k", raw_counters={"host": host}).format_for_llm()
    assert "unknown: the hotspots report failed: x" in text and "top 0" not in text
    assert "Collection window: unknown" in text  # an incomplete record is not read
    assert "cycles: unknown (<not supported>)" in text


@pytest.mark.parametrize("window", ["each", "all"])
def test_the_window_brackets_the_timed_calls(window):
    from xe_forge.core.profile_runner import _timed_calls

    events = []

    class Call:
        inputs = ()

        def reset(self):
            events.append("reset")

        def forward(self):
            events.append("forward")

    record = _timed_calls(Call(), 3, window, lambda on: events.append(on), lambda: None)
    if window == "all":
        assert events == [True, *["reset", "forward"] * 3, False]  # one start, one stop
    else:
        assert events == ["reset", True, "forward", False] * 3  # resets outside
    assert record["window"] == window and record["iters"] == 3
    assert 0 <= record["forward_s"] <= record["window_s"] <= record["window_max_s"]


def test_perf_collection_is_switched_over_its_control_descriptors():
    import os
    import threading

    from xe_forge.core.profile_runner import Collection

    control_r, control_w = os.pipe()
    ack_r, ack_w = os.pipe()
    seen = []

    def perf():  # what perf does with --control fd:control_r,ack_w
        for _ in range(2):
            seen.append(os.read(control_r, 16))
            os.write(ack_w, b"ack\n\x00")

    thread = threading.Thread(target=perf)
    thread.start()
    collection = Collection("perf", control_fds=(control_w, ack_r))
    collection.set(True)
    collection.set(False)
    thread.join(5)
    assert seen == [b"enable\n", b"disable\n"]
    for fd in (control_r, control_w, ack_r, ack_w):
        os.close(fd)

    # The runner closes perf's ends, so perf exiting is an end of file, not a hang.
    control_r, control_w = os.pipe()
    ack_r, ack_w = os.pipe()
    dead = Collection("perf", control_fds=(control_w, ack_r, control_r, ack_w))
    with pytest.raises(OSError):
        os.fstat(ack_w)  # closed by the runner
    with pytest.raises((RuntimeError, BrokenPipeError)):
        dead.set(True)
    os.close(control_w)
    os.close(ack_r)
    with pytest.raises(ValueError, match="descriptors"):
        Collection("perf")


PERF_STAT = "\n".join([
    '{"counter-value" : "100.000000", "unit" : "", "event" : "cycles", "pcnt-running" : 100.00,'
    ' "metric-value" : "1.0", "metric-unit" : "GHz"}',
    '{"counter-value" : "<not supported>", "unit" : "", "event" : "made-up", "pcnt-running" : 0}',
    '{"metric-value" : "41.4", "metric-unit" : "%  tma_frontend_bound", "metric-threshold" : "bad"}',
    "not json",
])  # fmt: skip

PERF_REPORT = """\
# Samples: 380  of event 'cycles:P'
# Event count (approx.): 567798957
#
# Overhead|     Samples|      Period|Shared Object       |Symbol                      |IPC
 2.10%  | 8          | 11898781   |ld.so               |[.] small                   |-
            |
            ---small
               |
 26.75% | 101        | 151906451  |ext.so              |[.] my_kernel               |-
# Samples: 0  of event 'instructions'
"""


def test_perf_output_is_read_as_perf_printed_it():
    from xe_forge.core.profiler import parse_perf_report, parse_perf_stat

    rows, unmeasured = parse_perf_stat(PERF_STAT)
    assert rows[0]["event"] == "cycles" and rows[0]["counter-value"] == "100.000000"
    assert rows[1] == {"metric-value": "41.4", "metric-unit": "%  tma_frontend_bound"}
    assert unmeasured == {"made-up": "<not supported>"}  # not a zero count
    sections = parse_perf_report(PERF_REPORT)
    assert [s["event"] for s in sections] == ["cycles:P", "instructions"]
    assert sections[0]["samples"] == "380" and sections[1]["rows"] == []
    assert sections[0]["rows"][0]["Symbol"] == "[.] my_kernel"  # highest overhead first
    assert len(sections[0]["rows"]) == 2  # call-chain lines are not rows


def test_perf_runs_bounded_and_drops_only_the_events_it_refuses(tmp_path, monkeypatch):
    from xe_forge.core import profiler

    kernel = tmp_path / "k.py"
    kernel.write_text("x = 1\n")
    commands = []

    def run(self, artifacts, command, name="collect", pass_fds=()):
        commands.append((name, command, pass_fds))
        if name.startswith("probe"):
            ok = "bogus" not in command
            return profiler.subprocess.CompletedProcess(
                command, 0 if ok else 129, "", "" if ok else "Bad event name\nmore"
            )
        (artifacts / "stat.json").write_text(PERF_STAT)
        (artifacts / "window.json").write_text(
            '{"window": "each", "iters": 2, "forward_s": 1, "window_s": 1, "window_max_s": 1}'
        )
        return profiler.subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(profiler._HostProfiler, "_run", run)
    monkeypatch.setattr(profiler.shutil, "which", lambda b: "/usr/bin/perf")
    perf = profiler.PerfCPUProfiler(events=["cycles", "bogus"], artifacts_dir=tmp_path / "p")
    result = perf.profile(kernel, iters=2)
    assert result.error is None and result.primary_kernel == "k"
    _, command, fds = commands[-1]
    assert command[:2] == ["perf", "stat"] and "--metric-no-threshold" in command
    assert command[command.index("-D") + 1] == "-1" and len(fds) == 4
    assert command[command.index("-e") + 1] == "cycles"  # the refused one is dropped
    probes = [c for n, c, _ in commands if n.startswith("probe")]
    assert [c[:2] for c in probes] == [["perf", "stat"]] * 2  # probed as it will be collected
    host = result.raw_counters["host"]
    assert host["unmeasured"]["bogus"].startswith("perf refused it: Bad event name")
    runner = next((tmp_path / "p").glob("k_perf_stat_*/runner.py")).read_text()
    assert "'tool': 'perf'" in runner and "'control_fds'" in runner
    assert "'window': 'all'" in runner  # one window unless asked

    refused = profiler.PerfCPUProfiler(events=["bogus"], artifacts_dir=tmp_path / "q")
    assert "refused every requested" in refused.profile(kernel, iters=2).error


def _profile_args(**kw):
    import argparse

    fields = {"kernel_file": "k.py", "spec": None, "variant": None, "warmup": 1, "iters": 1,
              "tool": "auto", "vtune_bin": "vtune", "unitrace_bin": "unitrace",
              "metric_group": "ComputeBasic", "device": "cpu", "analysis": None,
              "perf_bin": "perf"}  # fmt: skip
    return argparse.Namespace(**{**fields, **kw})


def test_the_host_is_profiled_with_perf_first_and_vtune_on_request(monkeypatch):
    from xe_forge.core import profiler
    from xe_forge.skills import profile

    seen = []

    def fake(self, kernel_file, **kw):
        seen.append((self.tool, self.analysis))
        return profiler.ProfileResult(primary_kernel="k", source=self.metric_group)

    for cls in (profiler.VTuneCPUProfiler, profiler.PerfCPUProfiler):
        monkeypatch.setattr(cls, "available", lambda self: True)
        monkeypatch.setattr(cls, "profile", fake)
    monkeypatch.setattr("shutil.which", lambda b: "/usr/bin/" + b)
    profile.run(_profile_args())
    profile.run(_profile_args(analysis="all"))
    profile.run(_profile_args(tool="vtune", analysis="memory-access"))
    assert seen == [("perf", "stat"), ("perf", "stat"), ("perf", "record"),
                    ("VTune", "memory-access")]  # fmt: skip
    monkeypatch.setattr("shutil.which", lambda b: None)
    profile.run(_profile_args())  # no perf here: VTune
    assert seen[-1] == ("VTune", "hotspots")

    with pytest.raises(SystemExit, match="unitrace profiles a GPU"):
        profile.run(_profile_args(tool="unitrace"))
    with pytest.raises(SystemExit, match="perf option"):
        profile.run(_profile_args(tool="vtune", events=["cycles"]))
    with pytest.raises(SystemExit, match="host profiling option"):
        profile.run(_profile_args(device="xpu", analysis="hotspots"))
    with pytest.raises(SystemExit, match="perf profiles the host"):
        profile.run(_profile_args(device="xpu", tool="perf"))
    with pytest.raises(SystemExit, match="stat, record or all"):
        profile.run(_profile_args(analysis="hotspots", tool="perf"))


@pytest.mark.parametrize("dsl", ["cpp", "sycl"])
def test_a_host_workspace_is_seeded_and_told_the_host_rules(tmp_path, dsl):
    import ast

    from xe_forge.claude.generator import generate_workspace
    from xe_forge.config import Config
    from xe_forge.models import DSL

    config = Config()
    config.device_config.dsl, config.device_config.device = dsl, "cpu"
    ws = tmp_path / "ws"
    generate_workspace(ws, config, "scale", "", reference_code="x = 1\n")
    seed = (ws / "test_kernels" / f"scale{DSL(dsl).kernel_ext}").read_text()
    ast.parse(seed)  # a Python module either way
    assert "load_inline" in seed and "torch.xpu" not in seed and "c10/xpu" not in seed
    claude_md = (ws / "CLAUDE.md").read_text()
    assert f"knowledge_base/{dsl}/cpu/" in claude_md and "placement" in claude_md
    assert "Use the caller's queue" not in claude_md and "grf_size" not in claude_md
    if dsl == "sycl":
        assert "waited on before returning" in claude_md


def test_host_knowledge_is_loaded_for_each_host_language():
    from pathlib import Path

    from xe_forge.knowledge.loader import _collect_yaml_files

    kb = Path(__file__).resolve().parents[1] / "knowledge_base"
    for dsl in ("cpp", "sycl"):
        names = {p.name for p in _collect_yaml_files(kb, dsl, "cpu")}
        assert "host_execution.yaml" in names
        assert ("cpp_patterns.yaml" if dsl == "cpp" else "sycl_cpu.yaml") in names
    assert "host_execution.yaml" not in {p.name for p in _collect_yaml_files(kb, "sycl", "xpu")}


def test_a_second_hardware_thread_counts_only_when_its_core_is_already_placed(tmp_path):
    _host(tmp_path)  # CPUs 4 and 5 are the second threads of cores 0 and 1
    assert topology.placement(tmp_path, cpus=[4, 5], env={})["smt_siblings"] == 0
    assert topology.placement(tmp_path, cpus=[0, 4], env={})["smt_siblings"] == 1


def test_the_thread_count_is_part_of_a_host_calibrations_identity():
    from xe_forge.frontend import calibrate

    eight, one = {**_cpu_device(), "threads": 8}, {**_cpu_device(), "threads": 1}
    assert calibrate._identity(eight) != calibrate._identity(one)
    assert calibrate.default_path({"device": eight}) != calibrate.default_path({"device": one})


def test_one_thread_has_no_second_first_touch_to_measure(monkeypatch):
    import torch

    from xe_forge.frontend import calibrate, profile

    monkeypatch.setenv("XE_FORGE_DEVICE", "cpu")
    monkeypatch.setattr(profile, "device_info", lambda: _cpu_device(llc=1 << 20))
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)
    rec = calibrate.measure(read_bytes=4 << 20, n=8, iters=1, dtypes=("float32",))
    assert rec["bandwidth_gbs"] > 0 and rec["bandwidth_gbs_one_writer"] is None
    assert "one thread" in rec["unmeasured"]["bandwidth_gbs_one_writer"]


def test_an_unknown_cache_size_is_said_in_the_result(monkeypatch):
    import xe_forge.core.topology as topo
    from xe_forge.core.executor import KernelBenchExecutor

    class Call:
        def reset(self):
            pass

        def forward(self, *args):
            pass

    monkeypatch.setattr(topo, "placement", lambda: {"llc_bytes": None})
    executor = KernelBenchExecutor(device="cpu")
    executor._host_samples(Call(), (), 1)
    assert "cache size of these CPUs is unknown" in executor.flush_note
    assert "no cache flush ran" in executor.flush_note
    assert executor._host_flush is None  # no size guessed in its place


def test_a_host_workspace_profiles_the_host(tmp_path):
    from xe_forge.claude.generator import generate_workspace
    from xe_forge.config import Config

    config = Config()
    config.device_config.dsl, config.device_config.device = "cpp", "cpu"
    config.profiler.unitrace_enabled, config.profiler.vtune_enabled = True, True

    def profile_line(name):
        ws = tmp_path / name
        generate_workspace(ws, config, "scale", "", reference_code="x = 1\n")
        return next(
            line for line in (ws / "CLAUDE.md").read_text().splitlines()
            if "xe-forge-skill profile" in line
        )  # fmt: skip

    line = profile_line("both")  # the required collection is perf's, so perf is named
    assert "--device cpu" in line and "--tool perf" in line and "--perf-bin perf" in line
    assert "--metric-group" not in line and "--unitrace-bin" not in line
    config.profiler.perf_enabled = False
    assert "--tool vtune" in profile_line("vtune")


def test_a_unitrace_group_is_refused_on_the_host():
    from xe_forge.skills import profile

    with pytest.raises(SystemExit, match="--analysis"):
        profile.run(_profile_args(metric_group="all"))


def test_a_calibration_note_names_the_threads_and_a_long_cpu_list_fits_a_file_name():
    from xe_forge.frontend import calibrate

    def device(threads, cpus="0-3"):
        return {"type": "cpu", "name": "Test CPU", "threads": threads,
                "placement": {"cpus": cpus, "memory_nodes": "0"}}  # fmt: skip

    note = calibrate.other_device({"device": device(4)}, {"device": device(2)})
    assert "with 4 threads" in note and "with 2 threads" in note
    long = ",".join(str(c) for c in range(0, 4000, 2))
    path = calibrate.default_path({"device": device(4, long)})
    assert len(path.name) < 100 and path.name.startswith("Test_CPU_")
    assert path != calibrate.default_path({"device": device(4)})


def test_an_unknown_cache_slice_makes_the_cache_size_unknown(tmp_path):
    _host(tmp_path)
    (tmp_path / "sys/devices/system/cpu/cpu4/cache/index2/size").write_text("\n")
    assert topology.placement(tmp_path, cpus=[0, 4], env={})["llc_bytes"] is None
    env = {"DPCPP_CPU_NUM_CUS": "8", "SYCL_PI_TRACE": "1", "BLIS_NUM_THREADS": "2"}
    assert topology.placement(tmp_path, cpus=[0], env=env)["thread_env"] == env


def test_the_benchmark_says_where_the_flush_ran(monkeypatch):
    import xe_forge.core.topology as topo
    from xe_forge.core.executor import KernelBenchExecutor

    class Call:
        def reset(self):
            pass

        def forward(self, *args):
            pass

    monkeypatch.setattr(topo, "placement", lambda: {"llc_bytes": 512})
    executor = KernelBenchExecutor(device="cpu")
    executor._host_samples(Call(), (), 1)
    assert "1024-byte flush ran on torch's thread pool" in executor.flush_note


def test_an_arm_that_writes_its_inputs_is_refused_even_within_tolerance():
    from xe_forge.core.executor import KernelBenchExecutor

    reference = (
        "import torch\n"
        "class Model(torch.nn.Module):\n"
        "    def forward(self, x):\n"
        "        return x * 2\n"
        "def get_inputs():\n"
        "    return [torch.ones(4)]\n"
        "def get_init_inputs():\n"
        "    return []\n"
    )
    nudges = reference.replace("return x * 2", "y = x * 2\n        x.add_(1e-7)\n        return y")
    result = KernelBenchExecutor(device="cpu").compare_reference_workload(
        reference, reference, nudges
    )
    assert not result.optimized_correct and "Mutating inputs" in result.feedback_message


def test_perf_record_reports_its_samples_and_says_when_there_are_none(tmp_path, monkeypatch):
    from xe_forge.core import profiler

    kernel = tmp_path / "k.py"
    kernel.write_text("x = 1\n")
    report = {"text": PERF_REPORT}
    commands = []

    def run(self, artifacts, command, name="collect", pass_fds=()):
        commands.append((name, command))
        return profiler.subprocess.CompletedProcess(
            command, 0, report["text"] if name == "report" else "", ""
        )

    monkeypatch.setattr(profiler._HostProfiler, "_run", run)
    monkeypatch.setattr(profiler.shutil, "which", lambda b: "/usr/bin/perf")
    perf = profiler.PerfCPUProfiler(analysis="record", artifacts_dir=tmp_path / "p")
    text = perf.profile(kernel).format_for_llm()
    assert "Samples of cycles:P by shared object and symbol (380 samples) (top 2 of 2)" in text
    assert "Symbol=[.] my_kernel" in text
    report_cmd = next(c for n, c in commands if n == "report")
    assert report_cmd[report_cmd.index("-g") + 1] == "none"
    report["text"] = ""
    assert "no samples in the collection window" in perf.profile(kernel).format_for_llm()
    with pytest.raises(ValueError, match="metrics are counted by perf stat"):
        profiler.PerfCPUProfiler(analysis="record", metrics=["TopdownL1"])


def test_an_event_spec_with_commas_is_passed_whole(monkeypatch):
    from xe_forge.core import profiler
    from xe_forge.skills import profile

    seen = []

    def fake(self, kernel_file, **kw):
        seen.append(self.events)
        return profiler.ProfileResult(primary_kernel="k", source=self.metric_group)

    monkeypatch.setattr(profiler.PerfCPUProfiler, "available", lambda self: True)
    monkeypatch.setattr(profiler.PerfCPUProfiler, "profile", fake)
    specs = ["{cycles,instructions}", "cpu/event=0xc0,umask=0x0/"]
    profile.run(_profile_args(tool="perf", events=specs))
    assert seen == [specs]


def test_a_perf_only_host_workspace_is_told_to_profile(tmp_path):
    from xe_forge.claude.generator import generate_workspace
    from xe_forge.config import Config

    config = Config()
    config.device_config.dsl, config.device_config.device = "cpp", "cpu"
    config.profiler.unitrace_enabled = config.profiler.vtune_enabled = False
    generate_workspace(tmp_path / "ws", config, "scale", "", reference_code="x = 1\n")
    text = (tmp_path / "ws" / "CLAUDE.md").read_text()
    assert "`perf_enabled: true` or `vtune_enabled: true`" in text
    assert "`unitrace_enabled: true` or" not in text
    line = next(x for x in text.splitlines() if "xe-forge-skill profile" in x)
    assert "--tool perf" in line
    # Every saved trial needs its perf counts, recorded against the trial.
    assert "--require-profiles 'perf stat'" in text and "--kernel-name scale" in line
    assert "needs its perf counts" in text and "stall-reason" not in text
    config.profiler.perf_enabled = False
    generate_workspace(tmp_path / "off", config, "scale", "", reference_code="x = 1\n")
    assert "--require-profiles" not in (tmp_path / "off" / "CLAUDE.md").read_text()


def test_a_host_calibration_measures_float8_only_when_asked(monkeypatch):
    from xe_forge.frontend import calibrate

    monkeypatch.setenv("XE_FORGE_DEVICE", "cpu")
    rec = calibrate.measure(read_bytes=4 << 20, n=8, iters=1)
    assert "float8_e4m3fn" not in rec["matmul_tflops"]  # not a value, not an unmeasured
    assert "--dtypes float8_e4m3fn" in rec["not_requested"]["matmul_tflops.float8_e4m3fn"]
    asked = calibrate.measure(read_bytes=4 << 20, n=8, iters=1, dtypes=("float8_e4m3fn",))
    assert "float8_e4m3fn" in asked["matmul_tflops"] and "not_requested" not in asked


def test_a_trial_tree_can_require_perf_counts(tmp_path):
    from xe_forge.core.trial_manager import TrialManager

    baseline = tmp_path / "base.py"
    baseline.write_text("x = 1\n")
    manager = TrialManager(tmp_path / "trials")
    manager.init("k", str(baseline), required_profile_groups=("perf stat",))
    with pytest.raises(ValueError, match="Unsupported"):
        TrialManager(tmp_path / "other").init(
            "k", str(baseline), required_profile_groups=("perf everything",)
        )
