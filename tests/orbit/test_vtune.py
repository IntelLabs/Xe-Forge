"""
GPU hardware counters via VTune (plan §5.2, §9.5).

unitrace says what a kernel is; VTune says what the device did with it, and — the part
that matters — which limit was binding. An agent told only "occupancy is 40%" proposed
larger blocks and more warps and measured 2x slower twice. "Limited by work size, not SLM
and not barriers" forecloses both.
"""

from __future__ import annotations

from xe_forge.orbit.profiling.vtune import (
    KernelOccupancy,
    VTuneResult,
    parse_computing_tasks,
    resolve_metrics_discovery,
    unavailable_reason,
)

# The real header and gemm row captured from this machine.
REAL_CSV = (
    "Computing Task;Work Size:Global;Work Size:Local;Computing Task:Total Time;"
    "Computing Task:Average Time;Computing Task:Instance Count;Computing Task:SIMD Width;"
    "Computing Task:SVM Usage Type;Computing Task:Spill Memory Size;Transfer Size;"
    "Transfer Size:Host-to-Device;Transfer Size:Device-to-Host;"
    "Peak XVE Threads Occupancy(%);Peak XVE Threads Occupancy:Work Size Limit(%);"
    "Peak XVE Threads Occupancy:SLM Use Limit(%);"
    "Peak XVE Threads Occupancy:Barriers Use Limit(%)\n"
    "gemm_kernel;128 x 8;64 x 8;0.026427;0.000529;50;16;;0;0;0;0;40.0;40.0;100.0;100.0\n"
)


class TestParsingRealOutput:
    def test_the_captured_report_parses(self):
        kernels = parse_computing_tasks(REAL_CSV)
        assert len(kernels) == 1
        assert kernels[0].name == "gemm_kernel"

    def test_the_measured_fields_survive(self):
        k = parse_computing_tasks(REAL_CSV)[0]
        assert k.simd_width == 16
        assert k.spill_bytes == 0
        assert k.instances == 50
        assert k.global_size == "128 x 8"
        assert k.occupancy_percent == 40.0

    def test_columns_are_matched_by_name_not_position(self):
        """VTune's column set varies with hardware and collection type.

        A positional parser would read occupancy out of the transfer-size column on a
        machine slightly unlike this one, and report it confidently.
        """
        reordered = (
            "Computing Task;Peak XVE Threads Occupancy(%);Computing Task:SIMD Width\nk;40.0;32\n"
        )
        k = parse_computing_tasks(reordered)[0]
        assert k.occupancy_percent == 40.0
        assert k.simd_width == 32

    def test_an_empty_report_yields_nothing(self):
        assert parse_computing_tasks("") == []
        assert parse_computing_tasks("Computing Task;X\n") == []

    def test_a_row_without_a_task_name_is_skipped(self):
        assert parse_computing_tasks("Computing Task;X\n;5\n") == []

    def test_unparseable_numbers_do_not_lose_the_row(self):
        k = parse_computing_tasks("Computing Task;Computing Task:SIMD Width\nk;n/a\n")[0]
        assert k.name == "k"
        assert k.simd_width == 0


class TestTheLimiterIsNamed:
    """The measurement's whole value: which lever to pull."""

    def test_the_smallest_limit_binds(self):
        k = KernelOccupancy(
            "k", occupancy_percent=40.0, work_size_limit=40.0, slm_limit=100.0, barrier_limit=100.0
        )
        assert k.limiter == "work size"

    def test_slm_can_be_the_binding_limit(self):
        k = KernelOccupancy(
            "k", occupancy_percent=25.0, work_size_limit=100.0, slm_limit=25.0, barrier_limit=100.0
        )
        assert k.limiter == "SLM use"

    def test_nothing_binding_is_said_plainly(self):
        """Full occupancy must not be reported as 'limited by' anything."""
        k = KernelOccupancy(
            "k",
            occupancy_percent=100.0,
            work_size_limit=100.0,
            slm_limit=100.0,
            barrier_limit=100.0,
        )
        assert k.limiter == "none of the measured limits"

    def test_an_unmeasured_limiter_is_unknown_not_guessed(self):
        assert KernelOccupancy("k", occupancy_percent=40.0).limiter == "unknown"

    def test_low_occupancy_is_flagged(self):
        assert KernelOccupancy("k", occupancy_percent=40.0).low_occupancy
        assert not KernelOccupancy("k", occupancy_percent=95.0).low_occupancy

    def test_unmeasured_occupancy_is_not_flagged_as_low(self):
        """Absent is not zero."""
        assert not KernelOccupancy("k").low_occupancy

    def test_the_description_names_the_limiter_and_the_non_limiters(self):
        k = parse_computing_tasks(REAL_CSV)[0]
        text = k.describe()
        assert "limited by work size" in text
        assert "(not limiting)" in text


class TestAbsenceIsReported:
    def test_a_missing_result_says_which_half_is_absent(self):
        """VTune and Metrics Discovery need different fixes."""
        reason = unavailable_reason()
        if reason:
            assert "vtune not found" in reason or "Metrics Discovery" in reason

    def test_the_libmd_confusion_is_called_out(self):
        """/usr/lib/libmd.so is BSD's message-digest library, not Intel's."""
        from xe_forge.orbit.profiling import vtune

        assert "message-digest" in vtune.__doc__

    def test_an_unavailable_result_describes_itself(self):
        text = VTuneResult(available=False, reason="nope").describe()
        assert "no VTune GPU counters" in text
        assert "nope" in text

    def test_metrics_discovery_resolution_returns_a_path_or_none(self):
        found = resolve_metrics_discovery()
        assert found is None or "metrics_discovery" in found


class TestChildProcessHandling:
    """Following children breaks collection for anything that imports a framework.

    Importing vLLM spawns helpers (ldconfig among them). Under the usual
    ptrace_scope=1 VTune cannot attach to them and the whole collection fails, citing a
    kernel setting that needs root — when the actual fix is a flag. A pure-torch script
    profiles fine and a vLLM one does not, which makes it look like the framework's fault.
    """

    def test_the_collect_command_disables_child_following(self):
        import inspect

        from xe_forge.orbit.profiling import vtune

        source = inspect.getsource(vtune.collect)
        assert "-no-follow-child" in source

    def test_the_reason_is_documented_where_someone_will_find_it(self):
        from xe_forge.orbit.profiling import vtune

        assert "-no-follow-child" in vtune.__doc__
        assert "ptrace" in vtune.__doc__
