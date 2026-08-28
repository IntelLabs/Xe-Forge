"""
Region detection and the Xe-Fuse handoff (plan §7.3, §9.6, §12.11, §21).

The reference decode trace is deliberately awkward for this stage: a SYCL attention
kernel runs *between* the RMSNorm and the SwiGLU while sharing no tensor with either.
A detector built on temporal adjacency alone swallows it and reports a four-kernel
"region" that no fusion could ever produce. Several tests below exist only to pin that
behaviour down.

All of it runs from the committed fixture with no GPU and no profiler (§16.3).
"""

from __future__ import annotations

import pytest

from xe_forge.orbit.analysis import xe_fuse
from xe_forge.orbit.analysis.catalog import build_catalog
from xe_forge.orbit.analysis.regions import (
    FUSION_PATTERNS,
    MIN_UNNAMED_REGION_SHARE,
    FusionPattern,
    detect_regions,
    format_regions,
)
from xe_forge.orbit.models import ActionType, RegionRecord, TensorInfo
from xe_forge.orbit.profiling import trace as trace_mod

# Kernel ids the catalog assigns to the fixture, by descending total GPU time.
K_GEMM, K_RMSNORM, K_ATTENTION, K_UNKNOWN, K_SILU = "k0", "k1", "k2", "k3", "k4"


# ---------------------------------------------------------------------------
# Synthetic trace helpers, for the cases the golden fixture does not cover.
# ---------------------------------------------------------------------------


def gpu_event(
    name: str,
    ts: float,
    dur: float,
    correlation: int,
    dims: list[list[int]] | None = None,
    dtypes: list[str] | None = None,
    stream: str = "0",
) -> dict:
    args: dict = {"correlation": correlation, "stream": stream, "device": 0}
    if dims is not None:
        args["Input Dims"] = dims
        args["Input type"] = dtypes or ["bfloat16"] * len(dims)
    return {"ph": "X", "name": name, "cat": "kernel", "ts": ts, "dur": dur, "args": args}


def synthetic(*raw_events: dict):
    """Ingest a hand-built trace and catalog it, returning (events, records)."""
    events = trace_mod.ingest_chrome_trace({"traceEvents": list(raw_events)})
    records = build_catalog(events, run_id="synthetic", with_regions=False).kernels
    return events, records


@pytest.fixture
def events(decode_trace_path):
    return trace_mod.ingest_trace_file(decode_trace_path)


@pytest.fixture
def records(events):
    return build_catalog(events, run_id="test", with_regions=False).kernels


@pytest.fixture
def regions(events, records):
    return detect_regions(events, records)


class TestChainDetection:
    def test_the_gemm_rmsnorm_swiglu_chain_is_found(self, regions):
        """The MLP block, which is the region Xe-Fuse exists to collapse (§7.3)."""
        assert len(regions) == 1
        region = regions[0]
        assert region.id == "r0"
        assert region.kernel_ids == [K_GEMM, K_RMSNORM, K_SILU]
        assert region.fusion_pattern == "gemm+rmsnorm+swiglu"

    def test_an_interleaved_unrelated_kernel_does_not_join_the_region(self, regions):
        """Attention runs between the RMSNorm and the SwiGLU and shares no tensor.

        Temporal adjacency alone would pull it in. The shape link is what steps over it.
        """
        assert all(K_ATTENTION not in region.kernel_ids for region in regions)

    def test_single_kernels_are_not_regions(self, regions):
        """A one-kernel 'region' is a kernel; the catalog already ranks those."""
        assert all(len(set(region.kernel_ids)) >= 2 for region in regions)
        assert all(len(region.kernel_ids) >= 2 for region in regions)

    def test_edges_run_producer_to_consumer_in_chain_order(self, regions):
        assert regions[0].producer_consumer_edges == [
            (K_GEMM, K_RMSNORM),
            (K_RMSNORM, K_SILU),
        ]

    def test_detection_is_deterministic(self, events, records):
        first = detect_regions(events, records)
        second = detect_regions(events, records)
        assert [r.model_dump() for r in first] == [r.model_dump() for r in second]

    def test_an_empty_trace_yields_no_regions(self):
        assert detect_regions(trace_mod.TraceEvents(), []) == []

    def test_a_trace_with_one_kernel_yields_no_regions(self):
        events, records = synthetic(gpu_event("lonely_gemm_kernel", 0, 100, 1, [[8, 8], [8, 8]]))
        assert detect_regions(events, records) == []


class TestEdgeInference:
    def test_a_host_gap_breaks_the_chain(self):
        """Device idle between two kernels means the host got involved (§12.11)."""
        events, records = synthetic(
            gpu_event("gemm_kernel_a", 0, 1000, 1, [[64, 64], [64, 64]]),
            gpu_event("rms_norm_kernel_b", 6000, 1000, 2, [[64, 64]]),
        )
        assert detect_regions(events, records) == []

    def test_mismatched_shapes_do_not_create_an_edge(self):
        """Adjacent in time but on different data paths is not a producer-consumer pair."""
        events, records = synthetic(
            gpu_event("gemm_kernel_a", 0, 1000, 1, [[64, 64], [64, 64]]),
            gpu_event("rms_norm_kernel_b", 1050, 1000, 2, [[7, 13]]),
        )
        assert detect_regions(events, records) == []

    def test_a_consumer_launched_first_cannot_consume(self):
        """Correlation ids order the launches; a later kernel with an earlier host op
        was already in flight and cannot be reading this producer's output."""
        events, records = synthetic(
            gpu_event("gemm_kernel_a", 0, 1000, 9, [[64, 64], [64, 64]]),
            gpu_event("rms_norm_kernel_b", 1050, 1000, 2, [[64, 64]]),
        )
        assert detect_regions(events, records) == []

    def test_streams_are_not_chained_across(self):
        """Two streams are independent queues; the trace records no sync between them."""
        events, records = synthetic(
            gpu_event("gemm_kernel_a", 0, 1000, 1, [[64, 64], [64, 64]], stream="0"),
            gpu_event("rms_norm_kernel_b", 1050, 1000, 2, [[64, 64]], stream="1"),
        )
        assert detect_regions(events, records) == []

    def test_repeated_occurrences_are_summed(self):
        """A decode loop runs the same chain every layer; the region owns all of it."""
        layer = [
            gpu_event("gemm_kernel_a", 0, 1000, 1, [[64, 64], [64, 64]]),
            gpu_event("rms_norm_kernel_b", 1050, 500, 2, [[64, 64]]),
            gpu_event("silu_mul_kernel_c", 1600, 200, 3, [[64, 64]]),
        ]
        second = [
            gpu_event("gemm_kernel_a", 10000, 1000, 4, [[64, 64], [64, 64]]),
            gpu_event("rms_norm_kernel_b", 11050, 500, 5, [[64, 64]]),
            gpu_event("silu_mul_kernel_c", 11600, 200, 6, [[64, 64]]),
        ]
        events, records = synthetic(*layer, *second)
        regions = detect_regions(events, records)
        assert len(regions) == 1
        assert regions[0].combined_time_us == pytest.approx(3400.0)
        assert regions[0].gpu_time_share == pytest.approx(1.0)


class TestThresholds:
    def test_region_share_is_the_sum_of_its_members(self, regions, records):
        by_id = {record.id: record for record in records}
        region = regions[0]
        expected_us = sum(by_id[k].total_time_us for k in region.kernel_ids)
        expected_share = sum(by_id[k].gpu_time_share for k in region.kernel_ids)
        assert region.combined_time_us == pytest.approx(expected_us)
        assert region.gpu_time_share == pytest.approx(expected_share)
        assert region.gpu_time_share == pytest.approx(0.7435, abs=1e-4)

    def test_a_region_below_the_share_threshold_is_dropped(self, events, records):
        regions = detect_regions(events, records, min_share=0.99)
        assert regions == []

    def test_an_unnamed_chain_must_beat_its_largest_member(self):
        """One dominant kernel plus a rounding error is that kernel, not a region.

        This applies to chains we have no name for, where nothing suggests fusing them
        helps. See the named-pattern case below for why it must NOT apply there.
        """
        events, records = synthetic(
            gpu_event("mystery_kernel_a", 0, 10000, 1, [[64, 64], [64, 64]]),
            gpu_event("other_kernel_b", 10050, 1, 2, [[64, 64]]),
        )
        assert detect_regions(events, records) == []

    def test_a_named_pattern_survives_a_dominant_member(self):
        """Epilogue fusion is worth doing BECAUSE the GEMM dominates.

        The value is eliminating the intermediate write between the two kernels, not the
        epilogue's own cost. Requiring the region to beat its largest member asks the
        epilogue to be expensive, which is the opposite of the condition that makes it
        fusable — and it is not a theoretical concern. On a real Qwen decode trace the
        rule dropped `gemm+activation` at 37.62% of GPU time and `gemm+rmsnorm` at
        25.88%: the two largest fusion opportunities in the run, rejected for being
        dominated by the GEMM they exist to fuse into.

        A pattern in FUSION_PATTERNS is an assertion that the shape is fusable. Whether
        it is worth doing is `min_share`, which it has already cleared.
        """
        events, records = synthetic(
            gpu_event("gemm_kernel_a", 0, 10000, 1, [[64, 64], [64, 64]]),
            gpu_event("rms_norm_kernel_b", 10050, 1, 2, [[64, 64]]),
        )
        regions = detect_regions(events, records)
        assert len(regions) == 1
        assert regions[0].fusion_pattern == "gemm+rmsnorm"

    def test_an_unnamed_chain_is_reported_when_it_is_large_enough(self):
        """'These kernels chain and we have no name for it' is a finding, not noise."""
        events, records = synthetic(
            gpu_event("opaque_stage_alpha", 0, 500, 1, [[64, 64]]),
            gpu_event("opaque_stage_beta", 550, 500, 2, [[64, 64]]),
        )
        regions = detect_regions(events, records)
        assert len(regions) == 1
        assert regions[0].fusion_pattern is None
        assert regions[0].gpu_time_share >= MIN_UNNAMED_REGION_SHARE

    def test_a_small_unnamed_chain_is_not_reported(self):
        events, records = synthetic(
            gpu_event("opaque_stage_alpha", 0, 10, 1, [[64, 64]]),
            gpu_event("opaque_stage_beta", 20, 10, 2, [[64, 64]]),
            gpu_event("big_unrelated_gemm_kernel", 50000, 5000, 3, [[8, 8], [8, 8]]),
        )
        assert detect_regions(events, records) == []


class TestPatternTable:
    def test_the_pattern_table_is_module_level_data(self):
        names = {pattern.name for pattern in FUSION_PATTERNS}
        assert {"gemm+rmsnorm+swiglu", "gemm+bias+activation", "attention+softmax"} <= names
        assert all(pattern.steps for pattern in FUSION_PATTERNS)

    def test_a_caller_supplied_pattern_table_is_honoured(self, events, records):
        """Extending the table is appending a row, not editing the algorithm."""
        custom = FusionPattern(name="norm+swiglu", steps=(("rmsnorm",), ("swiglu",)))
        regions = detect_regions(events, records, patterns=(custom,))
        assert len(regions) == 1
        assert regions[0].fusion_pattern == "norm+swiglu"
        assert regions[0].kernel_ids == [K_RMSNORM, K_SILU]

    def test_the_longest_matching_pattern_wins(self, events, records):
        """gemm+rmsnorm also matches the chain; the three-kernel pattern must win."""
        assert detect_regions(events, records)[0].fusion_pattern == "gemm+rmsnorm+swiglu"

    def test_every_region_offers_region_fusion(self, regions):
        assert all(ActionType.REGION_FUSION in r.actions_available for r in regions)


class TestIntermediateTensors:
    def test_the_tensors_fusion_would_eliminate_are_recorded(self, regions):
        tensors = regions[0].intermediate_tensors
        assert [t.name for t in tensors] == [f"{K_GEMM}->{K_RMSNORM}", f"{K_RMSNORM}->{K_SILU}"]
        assert all(t.shape == [4096, 8192] for t in tensors)
        assert all(t.dtype == "bfloat16" for t in tensors)
        # 4096 * 8192 elements at 2 bytes each, twice: what fusion stops writing.
        assert sum(t.bytes for t in tensors) == 2 * 4096 * 8192 * 2

    def test_no_tensor_is_claimed_when_the_trace_records_no_shapes(self):
        """Nothing is invented from a shape we do not have."""
        events, records = synthetic(
            gpu_event("opaque_stage_alpha", 0, 500, 1),
            gpu_event("opaque_stage_beta", 520, 500, 2),
        )
        regions = detect_regions(events, records)
        assert len(regions) == 1
        assert regions[0].intermediate_tensors == []

    def test_an_unknown_dtype_yields_no_byte_count_rather_than_a_guess(self):
        events, records = synthetic(
            gpu_event("gemm_kernel_a", 0, 500, 1, [[64, 64], [64, 64]], ["mystery", "mystery"]),
            gpu_event("rms_norm_kernel_b", 550, 500, 2, [[64, 64]], ["mystery"]),
        )
        region = detect_regions(events, records)[0]
        assert region.intermediate_tensors[0].bytes == 0
        assert region.intermediate_tensors[0].shape == [64, 64]


class TestOperatorAttribution:
    def test_aten_ops_come_from_host_op_correlation(self, regions):
        assert regions[0].aten_ops == ["aten::mm", "aten::rms_norm"]


class TestFormatting:
    def test_the_table_matches_the_plan_row(self, regions):
        rendered = format_regions(regions)
        assert "r0" in rendered
        assert "k0+k1+k4" in rendered
        assert "gemm+rmsnorm+swiglu" in rendered
        assert "E3" in rendered
        assert "Xe-Fuse" in rendered

    def test_the_table_names_an_executor_that_always_exists(self, regions, monkeypatch):
        """A region must be actionable without an external project installed.

        The table used to print "-> Xe-Fuse (not installed)", which made an absent
        sibling project look like a dead end for the only path that reaches an opaque
        GEMM. Authoring is the fallback that always exists (§13.8); Xe-Fuse wins when
        present, and its absence costs an option rather than the path.

        Absence is simulated explicitly — the env overrides are authoritative in
        both directions — because on a machine with a real Xe-Fuse checkout the
        route correctly prefers it, and this test is about the fallback.
        """
        monkeypatch.setenv("ORBIT_XE_FUSE_DIR", "/nonexistent-for-this-test")
        monkeypatch.setenv("SYCL_TLA_DIR", "/nonexistent-for-this-test")
        rendered = format_regions(regions)
        assert "author" in rendered
        assert "no external project is required" in rendered

    def test_an_empty_table_says_so_rather_than_printing_a_bare_header(self):
        assert "no fusable regions" in format_regions([])

    def test_the_catalog_carries_the_regions(self, events):
        catalog = build_catalog(events, run_id="t", minimum_detectable_effect=1.0)
        assert [r.id for r in catalog.regions] == ["r0"]
        assert catalog.regions[0].fusion_pattern == "gemm+rmsnorm+swiglu"


class TestXeFuseRouting:
    @pytest.fixture
    def route(self, regions):
        return xe_fuse.route_region(regions[0])

    def test_absence_is_reported_explicitly_not_papered_over(self, route):
        if xe_fuse.xe_fuse_available():  # pragma: no cover - not installed in CI
            pytest.skip("xe_fuse is installed in this environment")
        assert route["xe_fuse_available"] is False
        assert route["status"] == "blocked"
        assert "external sibling project" in route["reason"]
        assert "nothing was executed" in route["reason"]

    def test_the_route_never_claims_a_result(self, route):
        assert "speedup" not in {k.lower() for k in route}
        assert route["external"] is True
        assert route["action"] == ActionType.REGION_FUSION.value

    def test_the_route_names_what_xe_fuse_would_need(self, route):
        requires = route["requires"]
        assert requires
        joined = " ".join(requires)
        assert "E3" in joined
        assert "intermediate tensor" in joined
        assert "preset" in joined
        assert "k0->k1" in joined

    def test_the_route_carries_the_region_identity(self, route, regions):
        assert route["region_id"] == regions[0].id
        assert route["kernel_ids"] == regions[0].kernel_ids
        assert route["extraction_level"] == "E3"
        assert route["eliminated_bytes"] == 2 * 4096 * 8192 * 2

    def test_a_matching_hidden_size_finds_a_preset(self):
        region = RegionRecord(
            id="r0",
            kernel_ids=["k0", "k1"],
            fusion_pattern="gemm+rmsnorm+swiglu",
            intermediate_tensors=[
                TensorInfo(name="t", shape=[512, 3584], dtype="bfloat16", bytes=0),
                TensorInfo(name="u", shape=[512, 18944], dtype="bfloat16", bytes=0),
            ],
        )
        candidates = xe_fuse.match_model_preset(region)
        assert candidates
        assert candidates[0].key == "qwen2.5-7b"

    def test_an_ambiguous_match_is_reported_as_ambiguous_not_pinned(self, route):
        """Several architectures share a hidden size; picking one would be a guess."""
        assert route["preset_confidence"] == "ambiguous"
        assert route["preset_match"] is None
        assert len(route["preset_candidates"]) > 1

    def test_no_match_is_reported_as_none(self):
        region = RegionRecord(
            id="r0",
            kernel_ids=["k0", "k1"],
            intermediate_tensors=[TensorInfo(name="t", shape=[7, 13], dtype="bfloat16")],
        )
        route = xe_fuse.route_region(region)
        assert route["preset_confidence"] == "none"
        assert route["preset_candidates"] == []

    def test_a_region_with_no_tensors_matches_nothing(self):
        region = RegionRecord(id="r0", kernel_ids=["k0", "k1"])
        assert xe_fuse.match_model_preset(region) == []

    def test_the_route_renders(self, route):
        rendered = xe_fuse.format_route(route)
        assert "r0" in rendered
        assert "requires:" in rendered


class TestIntermediateSizing:
    """A fusion's value is the traffic it eliminates, so the byte count must be real.

    Three separate breaks stood between a recorded shape and a byte count, and each one
    produced "0.0 MB" — a number that reads as "measured and negligible" when it means
    "not recognized".
    """

    def test_cpp_dtype_spellings_are_recognized(self):
        """The profiler writes C++ type names, not torch names."""
        from xe_forge.orbit.analysis.regions import _tensor_bytes

        for spelling in ("c10::Half", "at::Half", "Half", "c10::BFloat16"):
            assert _tensor_bytes([172, 9728], spelling) == 172 * 9728 * 2, spelling

    def test_a_plain_torch_name_still_works(self):
        from xe_forge.orbit.analysis.regions import _tensor_bytes

        assert _tensor_bytes([4, 8], "float32") == 4 * 8 * 4

    def test_an_unknown_dtype_yields_zero_rather_than_a_guess(self):
        """Zero here means 'cannot size it'; inventing a width would be worse."""
        from xe_forge.orbit.analysis.regions import _tensor_bytes

        assert _tensor_bytes([4, 8], "some_future_type") == 0

    def test_dtypes_are_read_from_the_joined_field(self):
        """Types live on the host op, like shapes; reading only args returned nothing."""
        from xe_forge.orbit.analysis.regions import _input_dtypes
        from xe_forge.orbit.profiling.trace import RuntimeKernelEvent

        event = RuntimeKernelEvent(name="k", input_dtypes=["c10::Half", "c10::Half"])
        assert _input_dtypes(event) == ["c10::Half", "c10::Half"]

    def test_shapes_and_dtypes_come_from_the_same_op(self):
        """Pairing them from different sources yields plausible, mismatched tensors."""
        from xe_forge.orbit.profiling.trace import _as_dtypes, _as_shapes

        dims = _as_shapes([[16], [], [], []])
        types = _as_dtypes(["float", "Scalar", "Scalar", "Scalar"])
        assert len(dims) == len(types) == 1
