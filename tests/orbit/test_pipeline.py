"""
Trace ingest, provenance and the catalog's gating and ranking (plan §12.5, §16.4, §18).

Rows 4, 5 and 13 of the plan's stage test matrix. All of it runs from a committed
trace fixture with no GPU and no profiler, which is the point of §16.3.
"""

from __future__ import annotations

import pytest

from xe_forge.orbit.analysis.catalog import build_catalog, format_catalog
from xe_forge.orbit.models import ActionType, ExtractionLevel, KernelLanguage, Provider
from xe_forge.orbit.profiling import trace as trace_mod
from xe_forge.orbit.provenance import resolvers


@pytest.fixture
def events(decode_trace_path):
    return trace_mod.ingest_trace_file(decode_trace_path)


@pytest.fixture
def catalog(events):
    return build_catalog(events, run_id="test", minimum_detectable_effect=1.0)


class TestTraceIngest:
    def test_golden_fixture_normalizes_stably(self, events):
        assert len(events.kernels) == 6
        assert events.total_gpu_time_us == pytest.approx(10060.0)
        assert any(k.name.startswith("gemm_kernel_onednn") for k in events.kernels)

    def test_host_ops_are_separated_from_device_work(self, events):
        names = {op.name for op in events.host_ops}
        assert "aten::mm" in names
        assert all("aten::" not in k.name for k in events.kernels)

    def test_bare_event_list_is_accepted(self):
        payload = [{"ph": "X", "name": "k", "cat": "kernel", "ts": 0, "dur": 10}]
        assert len(trace_mod.ingest_chrome_trace(payload).kernels) == 1

    def test_malformed_events_are_skipped_with_a_warning(self):
        payload = {
            "traceEvents": [
                {"ph": "X", "name": "good", "cat": "kernel", "ts": 0, "dur": 5},
                {"ph": "X", "cat": "kernel", "ts": 0, "dur": 5},
                "not a dict",
                {"ph": "X", "name": "bad_ts", "cat": "kernel", "ts": "x", "dur": 5},
            ]
        }
        result = trace_mod.ingest_chrome_trace(payload)
        assert len(result.kernels) == 1
        assert any("malformed" in w for w in result.warnings)

    def test_a_trace_with_no_device_work_says_so(self):
        """Silence must be reported, never mistaken for a clean GPU-bound result."""
        payload = {"traceEvents": [{"ph": "X", "name": "op", "cat": "cpu_op", "ts": 0, "dur": 9}]}
        result = trace_mod.ingest_chrome_trace(payload)
        assert result.kernels == []
        assert any("no device-side kernel" in w for w in result.warnings)

    def test_a_non_trace_payload_raises(self):
        """Returning zero kernels here would look exactly like a host-bound workload."""
        with pytest.raises(ValueError):
            trace_mod.ingest_chrome_trace({"something": "else"})

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            trace_mod.ingest_trace_file(tmp_path / "nope.json")


class TestProvenance:
    @pytest.mark.parametrize(
        ("name", "provider", "level"),
        [
            ("triton_poi_fused_rms_norm_mul_0", Provider.INDUCTOR, ExtractionLevel.E2),
            ("gemm_kernel_onednn_jit_bf16", Provider.ONEDNN, ExtractionLevel.E4),
            ("my_custom_triton_kernel", Provider.TRITON, ExtractionLevel.E2),
        ],
    )
    def test_known_providers_resolve(self, name, provider, level):
        result = resolvers.resolve(name)
        assert result.provider is provider
        assert result.default_extraction is level
        assert result.confidence > 0.5

    def test_opaque_library_kernel_is_still_actionable(self):
        """The §7.2 point: no editable source does not mean nothing can be done."""
        result = resolvers.resolve("gemm_kernel_onednn_jit_bf16")
        assert ActionType.REGION_FUSION in result.actions
        assert ActionType.BACKEND_CHANGE in result.actions
        assert ActionType.KERNEL_REWRITE not in result.actions

    def test_sycl_kernel_gets_compiler_option_actions(self):
        """Compiler flags are the correct first move on a SYCL kernel (§11.7)."""
        result = resolvers.resolve(
            "_ZTSN4sycl3_V16detail18RoundedRangeKernelIN2at6native3xpu16UnifiedAttentionIfEEEE"
        )
        assert result.language in (KernelLanguage.SYCL, KernelLanguage.SYCL_TLA)
        assert ActionType.COMPILER_OPTION in result.actions

    def test_ambiguous_template_lowers_confidence_rather_than_guessing(self):
        """A name matching several instantiations must not be pinned to one (§11.4)."""
        ambiguous = resolvers.resolve("_ZTS7cutlass4gemm<float,128><bf16,256>")
        unique = resolvers.resolve("_ZTS7sycl_kernel_simple")
        assert ambiguous.confidence < unique.confidence
        assert ambiguous.source.candidates

    def test_unknown_kernel_never_gets_an_optimization_action(self):
        """An unattributed kernel is a finding to report, not a target to guess at."""
        result = resolvers.resolve("unknown_opaque_kernel_7f3a")
        assert result.provider is Provider.UNKNOWN
        assert result.actions == [ActionType.PROFILE_MORE]

    def test_empty_name_is_unknown_not_a_crash(self):
        assert resolvers.resolve("").provider is Provider.UNKNOWN

    def test_specific_patterns_beat_generic_ones(self):
        """Inductor before generic Triton; libraries before anything naming 'gemm'."""
        assert resolvers.resolve("triton_poi_fused_add_0").provider is Provider.INDUCTOR
        assert resolvers.resolve("gemm_kernel_onednn_jit_bf16").provider is Provider.ONEDNN

    def test_memory_transfers_are_not_unattributed_kernels(self):
        """A `Memcpy` has no source because it is not a kernel, which is not a gap.

        Level Zero reports transfers in the same stream as kernels. Left to the unknown
        fallback they read as "no provenance; needs more profiling" — advice that can
        never be acted on, because no amount of profiling produces a source file for a
        host/device copy.
        """
        for name in (
            "Memcpy D2H (DEVICE -> HOST)",
            "Memcpy H2D (HOST -> DEVICE)",
            "Memset (DEVICE)",
            "zeCommandListAppendMemoryCopy",
        ):
            result = resolvers.resolve(name)
            assert result.provider is Provider.RUNTIME, name
            assert ActionType.PROFILE_MORE not in result.actions, name

    def test_a_transfer_gets_host_side_actions_not_kernel_ones(self):
        """The action space for a copy is host-side; proposing a rewrite is nonsense."""
        result = resolvers.resolve("Memcpy D2H (DEVICE -> HOST)")
        assert ActionType.HOST_OPTIMIZATION in result.actions
        assert ActionType.KERNEL_REWRITE not in result.actions
        assert ActionType.KERNEL_AUTOTUNE not in result.actions

    def test_a_transfer_is_still_distinct_from_an_opaque_library_call(self):
        """Both are E4, but conflating them misdirects the reader.

        oneDNN has a library to reconfigure and a backend to swap; a memcpy has
        neither. The provider is what keeps the two recommendations apart.
        """
        transfer = resolvers.resolve("Memcpy D2H (DEVICE -> HOST)")
        library = resolvers.resolve("gemm_kernel_onednn_jit_bf16")
        assert transfer.provider is not library.provider
        assert ActionType.LIBRARY_CONFIG in library.actions
        assert ActionType.LIBRARY_CONFIG not in transfer.actions

    def test_a_kernel_merely_mentioning_copy_is_not_a_transfer(self):
        """`copy_kernel` is a real ATen kernel with real source; it must not be swept up."""
        assert (
            resolvers.resolve("_ZTSN2at6native3xpu16CopyKernelFunctorE").provider is Provider.SYCL
        )


class TestCatalogRanking:
    def test_kernels_are_aggregated_and_shared(self, catalog):
        assert len(catalog.kernels) == 6
        assert sum(k.gpu_time_share for k in catalog.kernels) == pytest.approx(1.0, abs=1e-6)

    def test_language_bias_guard_holds(self, catalog):
        """The §11.10 guard, as an executable assertion.

        The oneDNN GEMM owns 40% of GPU time but is E4 (expensive, no editable
        source); the Inductor Triton kernel owns 25% and extracts cleanly. Left
        unbounded, tractability would promote the Triton kernel and the report would
        call that optimization. It must not.
        """
        ranked = catalog.kernels
        gemm = next(k for k in ranked if k.provider is Provider.ONEDNN)
        triton = next(k for k in ranked if k.provider is Provider.INDUCTOR)
        assert gemm.gpu_time_share > triton.gpu_time_share
        assert ranked.index(gemm) < ranked.index(triton)

    def test_ranking_is_deterministic(self, events):
        first = build_catalog(events, run_id="a", minimum_detectable_effect=1.0)
        second = build_catalog(events, run_id="b", minimum_detectable_effect=1.0)
        assert [k.id for k in first.kernels] == [k.id for k in second.kernels]

    def test_unattributed_kernels_are_reported_as_skipped(self, catalog):
        """Every run says what it did not attempt, and why (§11.10, §18)."""
        skipped = catalog.considered_but_not_attempted
        assert skipped
        assert any("unknown" in item["kernel"] for item in skipped)
        assert all(item["reason"] for item in skipped)

    def test_amdahl_ceiling_is_recorded_per_kernel(self, catalog):
        gemm = next(k for k in catalog.kernels if k.provider is Provider.ONEDNN)
        assert 0 < gemm.max_e2e_gain < gemm.gpu_time_share * 100


class TestGating:
    def test_gpu_bound_workload_proceeds(self, catalog):
        assert catalog.gpu_busy_percent > 50
        assert catalog.gating_action is ActionType.KERNEL_REWRITE

    def test_host_bound_workload_is_gated_off(self, events):
        """Ask whether the workload is GPU-bound before ranking anything (§18)."""
        catalog = build_catalog(
            events, run_id="hostbound", gpu_busy_percent=12.0, minimum_detectable_effect=1.0
        )
        assert catalog.gating_action is ActionType.HOST_OPTIMIZATION
        assert "host-bound" in catalog.gating_reason

    def test_gain_below_the_noise_floor_yields_no_action(self, events):
        """A ceiling under the MDE is unmeasurable, so optimizing it is waste."""
        catalog = build_catalog(
            events, run_id="noisy", gpu_busy_percent=95.0, minimum_detectable_effect=80.0
        )
        assert catalog.gating_action is ActionType.NO_ACTION
        assert "noise floor" in catalog.gating_reason

    def test_empty_trace_asks_for_more_profiling(self):
        catalog = build_catalog(trace_mod.TraceEvents(), run_id="empty")
        assert catalog.gating_action is ActionType.PROFILE_MORE
        assert catalog.kernels == []

    def test_estimated_gpu_busy_is_labelled_as_such(self, catalog):
        """Without unitrace the estimate is weaker, and the report must admit it."""
        assert "estimated" in catalog.gating_reason

    def test_measured_gpu_busy_is_not_labelled_estimated(self, events):
        catalog = build_catalog(events, run_id="m", gpu_busy_percent=88.0)
        assert "estimated" not in catalog.gating_reason


class TestFormatting:
    def test_table_shows_the_gate_and_the_skips(self, catalog):
        rendered = format_catalog(catalog)
        assert "GPU busy" in rendered
        assert "Gate:" in rendered
        assert "Considered but not attempted" in rendered
