"""
Device counters in the optimizer's context (plan §9.5, §11.7).

Orbit collected unitrace's per-kernel metrics and told the agent none of them. The agent
then reasoned about occupancy and register pressure from first principles and got both
wrong, by roughly 2x, in a way the measurement gate caught and the context should have
prevented.
"""

from __future__ import annotations

from xe_forge.orbit.knowledge import profiler_context
from xe_forge.orbit.profiling.unitrace import KernelProperties


def _spilling():
    return KernelProperties(
        name="_gumbel_sample_kernel",
        simd=32,
        grf_per_thread=128,
        spill_per_thread=96,
        compiled="JIT",
    )


class TestUnmeasuredIsStated:
    def test_absent_counters_are_declared_as_inference(self):
        """Silence would let the agent's guesses read as evidence."""
        text = profiler_context([])
        assert "no device counters available" in text
        assert "inference rather than evidence" in text


class TestRegisterPressure:
    def test_spills_are_flagged_against_widening_the_tile(self):
        """The exact proposal that failed: BLOCK_SIZE 1024 -> 8192, measured -108%."""
        text = profiler_context([_spilling()])
        assert "SPILLS: 96 bytes per thread" in text
        assert "widening the tile makes this worse" in text

    def test_absence_of_spills_is_stated_rather_than_omitted(self):
        clean = _spilling()
        clean.spill_per_thread = 0
        assert "no spills at this configuration" in profiler_context([clean])


class TestCompiledWidth:
    def test_the_compiled_simd_width_is_distinguished_from_the_isa(self):
        """The agent asserted 'Intel XPU uses a 16-wide sub-group' as a vendor fact."""
        text = profiler_context([_spilling()])
        assert "SIMD32" in text
        assert "not the ISA maximum" in text

    def test_the_grf_axis_is_reported(self):
        assert "GRF: 128 per thread" in profiler_context([_spilling()])

    def test_aot_versus_jit_is_carried(self):
        assert "JIT-compiled" in profiler_context([_spilling()])


class TestLaunchBoundWorkloads:
    def test_a_mostly_idle_device_inverts_what_is_worth_proposing(self):
        """Measured 7.25% busy against an 81% trace-span estimate earlier this session."""
        text = profiler_context([], gpu_busy_us=7250.0, total_time_us=100000.0)
        assert "7.2% of wall clock" in text
        assert "launch- or host-bound" in text
        assert "cannot show up end to end" in text

    def test_a_busy_device_gets_no_such_warning(self):
        text = profiler_context([], gpu_busy_us=95000.0, total_time_us=100000.0)
        assert "95.0% of wall clock" in text
        assert "launch- or host-bound" not in text

    def test_busy_is_labelled_measured_to_separate_it_from_the_estimate(self):
        """A trace-span estimate and a counter disagreed by more than 10x here."""
        assert "measured, not estimated" in profiler_context(
            [], gpu_busy_us=5.0, total_time_us=10.0
        )

    def test_launch_gap_is_surfaced_as_dead_time(self):
        assert "dead time between kernels" in profiler_context([], launch_gap_us=42000.0)


class TestUnitraceResolution:
    """A resolver that cannot find an installed tool does not degrade gracefully here.

    unitrace was built and working at `~/.cache/orbit-dev/pti-gpu/tools/unitrace/build`
    while `available()` returned False, so the pipeline silently fell back to estimating
    GPU busy from the trace span — 81% where the counter measured 7.25%.
    """

    def test_a_source_build_under_the_home_cache_is_found(self, tmp_path, monkeypatch):
        from xe_forge.orbit.profiling import unitrace

        build = tmp_path / "pti-gpu" / "tools" / "unitrace" / "build"
        build.mkdir(parents=True)
        (build / "unitrace").write_text("#!/bin/sh\n")
        monkeypatch.setattr(unitrace, "UNITRACE_SEARCH_DIRS", (str(build),))
        monkeypatch.setattr(unitrace.shutil, "which", lambda _: None)
        assert unitrace.resolve_binary() == str(build / "unitrace")

    def test_the_environment_override_wins(self, tmp_path, monkeypatch):
        from xe_forge.orbit.profiling import unitrace

        explicit = tmp_path / "unitrace"
        explicit.write_text("#!/bin/sh\n")
        monkeypatch.setenv(unitrace.UNITRACE_SEARCH_ENV, str(explicit))
        assert unitrace.resolve_binary() == str(explicit)

    def test_an_override_naming_a_directory_also_works(self, tmp_path, monkeypatch):
        from xe_forge.orbit.profiling import unitrace

        (tmp_path / "unitrace").write_text("#!/bin/sh\n")
        monkeypatch.setenv(unitrace.UNITRACE_SEARCH_ENV, str(tmp_path))
        assert unitrace.resolve_binary() == str(tmp_path / "unitrace")

    def test_a_genuinely_absent_tool_still_reports_absent(self, monkeypatch):
        """The fallback must stay honest when the tool really is missing."""
        from xe_forge.orbit.profiling import unitrace

        monkeypatch.delenv(unitrace.UNITRACE_SEARCH_ENV, raising=False)
        monkeypatch.setattr(unitrace, "UNITRACE_SEARCH_DIRS", ())
        monkeypatch.setattr(unitrace.shutil, "which", lambda _: None)
        assert unitrace.resolve_binary() is None
        assert "not found on PATH" in unitrace.unavailable_result().reason


class TestOccupancyContext:
    """The limiter leads, because it is the only actionable part."""

    def _kernel(self, occ, work=100.0, slm=100.0, barrier=100.0):
        from xe_forge.orbit.profiling.vtune import KernelOccupancy

        return KernelOccupancy(
            "gemm_kernel",
            simd_width=16,
            spill_bytes=0,
            global_size="128 x 8",
            local_size="64 x 8",
            occupancy_percent=occ,
            work_size_limit=work,
            slm_limit=slm,
            barrier_limit=barrier,
        )

    def _result(self, kernels):
        from xe_forge.orbit.profiling.vtune import VTuneResult

        return VTuneResult(available=True, kernels=kernels)

    def test_a_low_occupancy_kernel_names_its_limiter(self):
        from xe_forge.orbit.knowledge import occupancy_context

        text = occupancy_context(self._result([self._kernel(40.0, work=40.0)]))
        assert "occupancy 40%" in text
        assert "limited by work size" in text

    def test_a_saturated_kernel_is_stated_not_omitted(self):
        """Silence would let an agent propose an occupancy fix for a non-problem."""
        from xe_forge.orbit.knowledge import occupancy_context

        text = occupancy_context(self._result([self._kernel(100.0)]))
        assert "occupancy is not this kernel's problem" in text

    def test_work_size_and_spills_accompany_a_low_reading(self):
        from xe_forge.orbit.knowledge import occupancy_context

        text = occupancy_context(self._result([self._kernel(40.0, work=40.0)]))
        assert "global 128 x 8" in text
        assert "SIMD16" in text

    def test_absent_counters_are_declared_as_inference(self):
        from xe_forge.orbit.knowledge import occupancy_context

        text = occupancy_context(None)
        assert "unmeasured" in text
        assert "inference rather than evidence" in text
