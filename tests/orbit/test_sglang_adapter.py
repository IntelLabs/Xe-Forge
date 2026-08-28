"""
The Tier 1 SGLang adapter (plan §10.9) — the scheduled v0.2 portability test.

Every test here runs on a machine with **no SGLang and no GPU**, because that is where
CI runs and because a Tier 1 adapter that can only be exercised on a serving node is a
Tier 1 adapter nobody will maintain.

The load-bearing assertions mirror the vLLM suite's, because the lies they catch are
framework-independent: an adapter that claims a metric it did not parse, a determinism
profile that omits the source that actually broke the comparison, a config pin that
was requested and silently dropped, a quality gate that reports a pass it never ran.
The SGLang-specific ones catch the checks that would run cleanly and answer something
else: a `--no-` spelling SGLang's store_true flags do not have, a "Last generation
throughput" line mistaken for the run's throughput, decoded text compared where token
ids were required.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types

import pytest
import yaml

from xe_forge.orbit.adapters import AdapterError, SGLangAdapter, describe_adapters, get_adapter
from xe_forge.orbit.adapters.base import LoadSpec
from xe_forge.orbit.adapters.sglang import (
    _CLI_FLAGS,
    _METRIC_PATTERNS,
    _PRESENCE_AXES,
    MIN_QUALITY_PROMPTS,
    QUALITY_MAX_TOKENS,
    QUALITY_SEED,
    knowledge_path,
    load_knowledge,
    parse_sglang_metrics,
    sglang_available,
)
from xe_forge.orbit.executor import LocalExecutor
from xe_forge.orbit.models import KernelRecord, WorkloadSpec

# The six sources §10.5 names. An adapter must account for all of them, one way or the
# other; silence about one is how a comparison gets confounded without a reason.
NONDETERMINISM_SOURCES = {
    "prefix_cache_reuse",
    "continuous_batching_order",
    "chunked_prefill_boundaries",
    "speculative_decoding",
    "graph_capture_warmup",
    "request_arrival_jitter",
}

# An `sglang.bench_serving` result table — the same table shape as vLLM's benchmark,
# which it is derived from, with SGLang's lower-case "Total token throughput" and the
# extra E2E-latency section.
SERVE_OUTPUT = """\
[2026-08-25 11:02:14] INFO:     Automatically detected platform xpu.
Starting initial single prompt test run...
Traffic request rate: inf
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Successful requests:                     1000
Benchmark duration (s):                  53.31
Total input tokens:                      224442
Total generated tokens:                  191337
Total generated tokens (retokenized):    191204
Request throughput (req/s):              18.76
Input token throughput (tok/s):          4210.55
Output token throughput (tok/s):         3589.44
Total token throughput (tok/s):          7799.99
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   19836.42
Median E2E Latency (ms):                 19547.71
---------------Time to First Token----------------
Mean TTFT (ms):                          2115.73
Median TTFT (ms):                        2035.66
P99 TTFT (ms):                           4322.11
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          91.20
Median TPOT (ms):                        90.85
P99 TPOT (ms):                           147.34
---------------Inter-token Latency----------------
Mean ITL (ms):                           90.61
Median ITL (ms):                         86.15
P99 ITL (ms):                            292.83
==================================================
"""

# `sglang.bench_offline_throughput` prints a shorter table: throughput only, plus a
# "Last generation throughput" line that must NOT be mistaken for the run's rate.
OFFLINE_OUTPUT = """\
====== Offline Throughput Benchmark Result =======
Backend:                                 engine
Successful requests:                     1000
Benchmark duration (s):                  62.10
Total input tokens:                      224442
Total generated tokens:                  191337
Last generation throughput (tok/s):      3012.22
Request throughput (req/s):              16.10
Input token throughput (tok/s):          3614.20
Output token throughput (tok/s):         3081.11
Total token throughput (tok/s):          6695.31
==================================================
"""

# A scheduler decode-batch log line: the only rate an offline engine run emits.
ENGINE_LOG_OUTPUT = """\
[2026-08-25 11:04:31] Decode batch. #running-req: 8, #token: 4096, \
token usage: 0.10, gen throughput (token/s): 1274.62, #queue-req: 0
"""


@pytest.fixture
def adapter() -> SGLangAdapter:
    return SGLangAdapter()


@pytest.fixture
def sleep_spec() -> WorkloadSpec:
    return WorkloadSpec(
        command=[sys.executable, "-c", "import time; time.sleep(0.01)"],
        repetitions=2,
        warmup_iterations=0,
    )


@pytest.fixture
def serve_spec() -> WorkloadSpec:
    return WorkloadSpec(
        command=[
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            "meta-llama/Llama-3.1-8B",
            "--port",
            "30001",
        ],
        repetitions=2,
        warmup_iterations=0,
    )


@pytest.fixture
def fake_sglang(monkeypatch):
    """A stub `sglang` module, so the L3 gate's logic is testable without SGLang.

    It carries a real `__spec__` because `sglang_available()` asks `find_spec`, which
    consults `sys.modules` first and rejects a module without one.
    """
    module = types.ModuleType("sglang")
    module.__spec__ = importlib.machinery.ModuleSpec("sglang", loader=None)
    monkeypatch.setitem(sys.modules, "sglang", module)
    return module


class _FakeEngine:
    """Returns canned generate outputs, and records the sampling params it was given."""

    def __init__(self, outputs: list[dict]) -> None:
        self.outputs = outputs
        self.last_params = None
        self.flushed = 0

    def generate(self, prompts, params):
        self.last_params = params
        return self.outputs[: len(prompts)]

    def flush_cache(self):
        self.flushed += 1


def _token_outputs(token_ids: list[list[int]]) -> list[dict]:
    return [{"text": "x", "output_ids": list(ids), "meta_info": {}} for ids in token_ids]


class TestRegistration:
    def test_registered_at_tier_one(self):
        rows = {row["name"]: row for row in describe_adapters()}
        assert rows["sglang"]["tier"] == 1
        assert set(rows["sglang"]["metrics"]) == {"wall_time", "throughput", "ttft", "tpot"}

    def test_resolvable_by_name(self):
        assert get_adapter("sglang").name == "sglang"


class TestDetect:
    @pytest.mark.skipif(sglang_available(), reason="this machine has sglang installed")
    def test_absent_sglang_yields_the_tier_zero_fallback(self, adapter, sleep_spec):
        """detect() must return False, not crash, so Tier 0 wins resolution (§10.2)."""
        from xe_forge.orbit.adapters import resolve_adapter

        assert adapter.detect(sleep_spec) is False
        assert resolve_adapter(sleep_spec).name == "generic_torch"

    def test_command_evidence_is_enough(self, adapter, serve_spec):
        assert adapter.detect(serve_spec) is True

    def test_benchmark_script_is_recognised(self, adapter):
        spec = WorkloadSpec(
            command=["python", "bench_offline_throughput.py", "--num-prompts", "64"]
        )
        assert adapter.detect(spec) is True

    def test_a_framework_named_virtualenv_is_not_evidence(self, adapter):
        """The interpreter's path is a fact about the machine, not the workload."""
        spec = WorkloadSpec(command=["/home/u/.cache/sglangenv/bin/python", "bench.py"])
        assert adapter.detect(spec) is False

    def test_a_directory_named_sglang_is_not_evidence(self, adapter):
        spec = WorkloadSpec(command=["python", "/opt/sglang/tools/other_script.py"])
        assert adapter.detect(spec) is False

    def test_another_framework_is_never_hijacked(self, adapter, sleep_spec):
        """A box with SGLang on it must not claim a vLLM workload."""
        spec = sleep_spec.model_copy(update={"framework": "vllm"})
        assert adapter.detect(spec) is False

    def test_versions_reports_only_what_is_installed(self, adapter):
        versions = adapter.versions()
        assert isinstance(versions, dict)
        assert all(isinstance(v, str) for v in versions.values())
        if not sglang_available():
            assert "sglang" not in versions


class TestCapabilities:
    def test_declared_metrics_match_the_schema(self, adapter):
        """Conformance test 3 compares these two sets; they must not drift."""
        assert {m.name for m in adapter.metrics_schema()} == adapter.capabilities.metrics

    def test_throughput_is_the_only_higher_is_better_metric(self, adapter):
        schema = {m.name: m for m in adapter.metrics_schema()}
        assert schema["throughput"].lower_is_better is False
        assert schema["ttft"].lower_is_better is True
        assert schema["tpot"].lower_is_better is True

    def test_single_layer_is_not_declared_so_e3_is_refused(self, adapter):
        """An unverified harness constructor mis-ranks extraction tractability (§18)."""
        assert adapter.capabilities.can_construct_single_layer is False
        with pytest.raises(AdapterError, match="single layer"):
            adapter.build_in_situ_harness(KernelRecord(id="k0", runtime_name="x"), None)

    def test_profiling_is_declared_out_of_process_with_a_hook(self, adapter):
        """SGLang's scheduler lives in its own processes; an in-process profiler
        would observe zero device kernels and report a confident empty catalog."""
        assert adapter.capabilities.profiles_in_process is False
        assert "SGLANG_TORCH_PROFILER_DIR" in adapter.capabilities.profile_hook

    def test_reset_state_is_declared_and_names_its_mechanism(self, adapter, sleep_spec):
        handle = adapter.launch(sleep_spec, LocalExecutor())
        adapter.reset_state(handle)
        assert "process-per-repetition" in handle.state["state_reset"]

    def test_reset_state_uses_the_engine_flush_when_one_is_present(self, adapter, sleep_spec):
        handle = adapter.launch(sleep_spec, LocalExecutor())
        engine = _FakeEngine([])
        handle.state["engine"] = engine
        adapter.reset_state(handle)
        assert engine.flushed == 1
        assert handle.state["state_reset"] == "Engine.flush_cache()"

    def test_an_engine_without_flush_cache_is_refused_not_faked(self, adapter, sleep_spec):
        """Claiming state was reset when it was not is a silently wrong measurement."""
        handle = adapter.launch(sleep_spec, LocalExecutor())
        handle.state["engine"] = object()
        with pytest.raises(AdapterError, match="flush_cache"):
            adapter.reset_state(handle)

    def test_patch_points_never_name_an_undeclared_layer(self, adapter):
        kernels = [
            KernelRecord(id="k0", runtime_name="decode_attention_fwd_kernel"),
            KernelRecord(id="k1", runtime_name="fused_moe_kernel"),
            KernelRecord(id="k2", runtime_name="extend_attention_kernel"),
        ]
        targets = {p.target for k in kernels for p in adapter.patch_points(k) if p.rung == "P3"}
        assert targets
        assert targets <= adapter.capabilities.patchable_layers


class TestDeterminism:
    def test_every_source_is_accounted_for(self, adapter):
        profile = adapter.determinism_profile()
        assert profile.pinnable | profile.non_pinnable == NONDETERMINISM_SOURCES
        assert not profile.pinnable & profile.non_pinnable

    def test_the_pinnable_ones_are_the_ones_with_flags(self, adapter):
        profile = adapter.determinism_profile()
        assert profile.pinnable == {
            "prefix_cache_reuse",
            "continuous_batching_order",
            "chunked_prefill_boundaries",
            "graph_capture_warmup",
        }
        # Accept rate and arrival timing are not server settings; claiming to pin them
        # would let §17 emit ACCEPT on a comparison that was never controlled.
        assert profile.non_pinnable == {"speculative_decoding", "request_arrival_jitter"}

    def test_notes_name_the_actual_knobs(self, adapter):
        notes = adapter.determinism_profile().notes
        assert "--disable-radix-cache" in notes
        assert "--max-running-requests 1" in notes
        assert "--chunked-prefill-size" in notes
        assert "--disable-cuda-graph" in notes

    def test_active_sources_come_from_the_command(self, adapter):
        spec = WorkloadSpec(
            command=[
                "python",
                "-m",
                "sglang.bench_serving",
                "--request-rate",
                "8",
                "--speculative-algorithm",
                "EAGLE",
            ]
        )
        active = adapter.determinism_profile(spec).active_non_pinnable
        assert active == {"speculative_decoding", "request_arrival_jitter"}

    def test_nothing_is_active_by_default(self, adapter, serve_spec):
        assert adapter.determinism_profile(serve_spec).active_non_pinnable == set()

    def test_prepare_warns_about_the_unpinned_seed(self, adapter, serve_spec):
        """SGLang draws a fresh seed per launch when --random-seed is unset (§10.5)."""
        notes = " ".join(adapter.prepare(serve_spec).notes)
        assert "--random-seed" in notes

    def test_a_pinned_seed_is_not_warned_about(self, adapter, serve_spec):
        pinned = serve_spec.model_copy(
            update={"command": [*serve_spec.command, "--random-seed", "0"]}
        )
        notes = " ".join(adapter.prepare(pinned).notes)
        assert "unpinned" not in notes


class TestConfigAxes:
    def test_the_declared_axes_are_present(self, adapter):
        names = {axis.name for axis in adapter.config_axes()}
        assert {
            "attention_backend",
            "disable_radix_cache",
            "max_running_requests",
            "chunked_prefill_size",
            "schedule_policy",
            "mem_fraction_static",
            "tp_size",
            "disable_cuda_graph",
            "random_seed",
        } <= names

    def test_cli_axes_become_server_arguments(self, adapter, serve_spec):
        updated = adapter.apply_config(
            serve_spec,
            {"max_running_requests": 1, "schedule_policy": "fcfs", "chunked_prefill_size": -1},
        )
        assert updated.command[updated.command.index("--max-running-requests") + 1] == "1"
        assert updated.command[updated.command.index("--schedule-policy") + 1] == "fcfs"
        assert updated.command[updated.command.index("--chunked-prefill-size") + 1] == "-1"
        assert serve_spec.command.count("--max-running-requests") == 0  # original untouched

    def test_presence_flags_have_no_negated_spelling(self, adapter, serve_spec):
        """SGLang's booleans are argparse store_true switches. True appends the bare
        flag; False removes it. Emitting a vLLM-style `--no-` form would hand the
        server an argument it rejects — a pin that breaks the run instead of pinning it.
        """
        pinned = adapter.apply_config(serve_spec, {"disable_radix_cache": True})
        assert "--disable-radix-cache" in pinned.command
        assert not any(t.startswith("--no-") for t in pinned.command)

        unpinned = adapter.apply_config(pinned, {"disable_radix_cache": False})
        assert "--disable-radix-cache" not in unpinned.command
        assert not any(t.startswith("--no-") for t in unpinned.command)

    def test_a_presence_flag_is_not_duplicated(self, adapter, serve_spec):
        spec = serve_spec.model_copy(
            update={"command": [*serve_spec.command, "--disable-radix-cache"]}
        )
        updated = adapter.apply_config(spec, {"disable_radix_cache": True})
        assert updated.command.count("--disable-radix-cache") == 1

    def test_an_existing_value_flag_is_replaced_not_duplicated(self, adapter, serve_spec):
        spec = serve_spec.model_copy(
            update={"command": [*serve_spec.command, "--max-running-requests", "256"]}
        )
        updated = adapter.apply_config(spec, {"max_running_requests": 1})
        assert updated.command.count("--max-running-requests") == 1
        assert "256" not in updated.command

    def test_a_pin_that_cannot_be_applied_is_refused_not_dropped(self, adapter, sleep_spec):
        """Silently dropping a determinism pin produces a run that looks controlled."""
        with pytest.raises(AdapterError, match="not an SGLang entrypoint"):
            adapter.apply_config(sleep_spec, {"max_running_requests": 1})

    def test_unknown_axis_is_refused(self, adapter, serve_spec):
        with pytest.raises(AdapterError, match="unknown SGLang config axis"):
            adapter.apply_config(serve_spec, {"turbo_mode": True})


class TestMetricParsing:
    def test_serving_table(self):
        metrics = parse_sglang_metrics(SERVE_OUTPUT)
        assert metrics["ttft_ms"] == pytest.approx(2115.73)
        assert metrics["tpot_ms"] == pytest.approx(91.20)
        # Output token throughput, not total: total moves with input length, which
        # makes it useless for an A/B on a decode kernel.
        assert metrics["throughput"] == pytest.approx(3589.44)

    def test_offline_throughput_table(self):
        metrics = parse_sglang_metrics(OFFLINE_OUTPUT)
        assert metrics["throughput"] == pytest.approx(3081.11)
        assert "ttft_ms" not in metrics
        assert "tpot_ms" not in metrics

    def test_last_generation_throughput_is_not_the_answer(self):
        """`Last generation throughput` is the final batch's instantaneous rate — a
        different quantity that parses cleanly if the pattern is sloppy. The wrong
        question, answered confidently, is this codebase's recurring bug class."""
        without_output_rate = "\n".join(
            line
            for line in OFFLINE_OUTPUT.splitlines()
            if not line.startswith(("Output token throughput", "Total token throughput"))
        )
        assert "throughput" not in parse_sglang_metrics(without_output_rate)

    def test_engine_log_line(self):
        assert parse_sglang_metrics(ENGINE_LOG_OUTPUT)["throughput"] == pytest.approx(1274.62)

    def test_absent_metrics_are_absent_not_zero(self):
        """A zero TTFT would sail through the statistics as a real measurement."""
        assert parse_sglang_metrics("nothing to see here") == {}

    def test_median_is_the_documented_fallback(self):
        text = SERVE_OUTPUT.replace("Mean TTFT (ms):", "P50 TTFT (ms):")
        assert parse_sglang_metrics(text)["ttft_ms"] == pytest.approx(2035.66)

    def test_total_token_throughput_is_the_documented_fallback(self):
        text = SERVE_OUTPUT.replace("Output token throughput", "Removed throughput")
        assert parse_sglang_metrics(text)["throughput"] == pytest.approx(7799.99)


class TestBenchmark:
    def test_a_non_sglang_run_degrades_to_wall_time_and_says_so(self, adapter, sleep_spec):
        """§10.4: never substitute one metric for another, and never do it quietly."""
        handle = adapter.launch(sleep_spec, LocalExecutor())
        measurement = adapter.benchmark(handle, LoadSpec(repetitions=2))

        assert measurement.metrics_available == ["wall_time"]
        assert measurement.ttft_ms is None
        assert measurement.tpot_ms is None
        assert measurement.throughput is None

        fallback = adapter.metric_fallback(handle)
        assert fallback and "did not appear" in fallback

    def test_parsed_metrics_are_reported_under_their_declared_names(self, adapter, sleep_spec):
        """`ttft_ms` is the field; `ttft` is the declared capability. They must line up."""
        script = f"import sys; sys.stdout.write({SERVE_OUTPUT!r})"
        spec = sleep_spec.model_copy(
            update={"command": [sys.executable, "-c", script], "framework": "sglang"}
        )
        handle = adapter.launch(spec, LocalExecutor())
        measurement = adapter.benchmark(handle, LoadSpec(repetitions=2))

        assert set(measurement.metrics_available) == adapter.capabilities.metrics
        assert measurement.ttft_ms.mean == pytest.approx(2115.73)
        assert measurement.tpot_ms.mean == pytest.approx(91.20)
        assert adapter.metric_fallback(handle) is None


class TestPatchPoints:
    def test_dispatcher_op_gets_p1_first(self, adapter):
        """P1 touches nothing in the framework, so it is offered ahead of P3 (§13)."""
        kernel = KernelRecord(
            id="k0",
            runtime_name="decode_attention_fwd_grouped_kernel",
            framework_op="sgl_kernel::decode_attention",
        )
        points = adapter.patch_points(kernel)
        assert points[0].rung == "P1"
        assert points[0].target == "sgl_kernel::decode_attention"
        assert "dispatch" in points[0].mechanism.lower()

    def test_attention_kernel_gets_the_backend_registry(self, adapter):
        kernel = KernelRecord(id="k1", runtime_name="radix_attention_extend_kernel")
        p3 = [p for p in adapter.patch_points(kernel) if p.rung == "P3"]
        assert [p.target for p in p3] == ["attention_backend"]
        assert "--attention-backend" in p3[0].mechanism

    def test_moe_kernel_gets_the_fused_moe_layer(self, adapter):
        kernel = KernelRecord(
            id="k2", runtime_name="fused_moe_kernel", framework_op="sgl_kernel::topk_softmax"
        )
        rungs = {(p.rung, p.target) for p in adapter.patch_points(kernel)}
        assert ("P1", "sgl_kernel::topk_softmax") in rungs
        assert ("P3", "fused_moe") in rungs

    def test_a_norm_is_p1_only_because_sglang_has_no_custom_op_registry(self, adapter):
        """vLLM routes norms through a CustomOp registry; SGLang dispatches them
        straight through torch.ops.sgl_kernel. Offering a vLLM-shaped custom_op rung
        here would name a substitution point the framework does not have."""
        kernel = KernelRecord(
            id="k3", runtime_name="rmsnorm_kernel", framework_op="sgl_kernel::rmsnorm"
        )
        points = adapter.patch_points(kernel)
        assert [p.rung for p in points] == ["P1"]

    def test_an_unattributable_kernel_gets_nothing(self, adapter):
        """No op, no recognised layer: naming a patch point here would be a guess."""
        assert adapter.patch_points(KernelRecord(id="k4", runtime_name="mystery_0000")) == []

    def test_dispatch_roots_cover_the_native_library(self, adapter):
        roots = adapter.dispatch_roots()
        assert {"torch.ops.sgl_kernel", "sgl_kernel"} <= set(roots)
        assert "sglang" in " ".join(adapter.provenance_hints())


class TestQualityGate:
    def test_unavailable_without_sglang_raises_rather_than_failing(self, adapter, sleep_spec):
        """An unavailable gate and a failed gate are different facts (§19)."""
        if sglang_available():
            pytest.skip("this machine has sglang installed")
        handle = adapter.launch(sleep_spec, LocalExecutor())
        with pytest.raises(AdapterError, match="sglang is not importable"):
            adapter.quality_gate(handle, [f"prompt {i}" for i in range(MIN_QUALITY_PROMPTS)])

    def test_too_few_prompts_is_a_failed_gate(self, adapter, sleep_spec, fake_sglang):
        result = adapter.quality_gate(adapter.launch(sleep_spec, LocalExecutor()), ["one", "two"])
        assert result.passed is False
        assert str(MIN_QUALITY_PROMPTS) in result.detail

    def test_no_baseline_reference_raises(self, adapter, sleep_spec, fake_sglang):
        handle = adapter.launch(sleep_spec, LocalExecutor())
        handle.state["engine"] = _FakeEngine(_token_outputs([[1, 2, 3]] * MIN_QUALITY_PROMPTS))
        with pytest.raises(AdapterError, match="capture_quality_reference"):
            adapter.quality_gate(handle, [f"prompt {i}" for i in range(MIN_QUALITY_PROMPTS)])

    def test_token_exact_match_passes_with_greedy_decode(self, adapter, sleep_spec, fake_sglang):
        prompts = [f"prompt {i}" for i in range(MIN_QUALITY_PROMPTS)]
        handle = adapter.launch(sleep_spec, LocalExecutor())
        engine = _FakeEngine(_token_outputs([[7, 8, 9]] * MIN_QUALITY_PROMPTS))
        handle.state["engine"] = engine

        adapter.capture_quality_reference(handle, prompts)
        result = adapter.quality_gate(handle, prompts)

        assert result.passed is True
        assert result.token_exact is True
        assert engine.last_params["temperature"] == 0.0
        assert engine.last_params["max_new_tokens"] == QUALITY_MAX_TOKENS

    def test_a_single_diverged_completion_fails_the_gate(self, adapter, sleep_spec, fake_sglang):
        prompts = [f"prompt {i}" for i in range(MIN_QUALITY_PROMPTS)]
        handle = adapter.launch(sleep_spec, LocalExecutor())
        engine = _FakeEngine(_token_outputs([[7, 8, 9]] * MIN_QUALITY_PROMPTS))
        handle.state["engine"] = engine

        adapter.capture_quality_reference(handle, prompts)
        engine.outputs = _token_outputs([[7, 8, 9]] * (MIN_QUALITY_PROMPTS - 1) + [[7, 8, 10]])
        result = adapter.quality_gate(handle, prompts)

        assert result.passed is False
        assert result.token_exact is False
        assert "diverged" in result.detail

    def test_token_ids_fall_back_to_the_logprob_records(self, adapter, sleep_spec, fake_sglang):
        """Older engines report ids only inside meta_info.output_token_logprobs."""
        prompts = [f"prompt {i}" for i in range(MIN_QUALITY_PROMPTS)]
        outputs = [
            {"text": "x", "meta_info": {"output_token_logprobs": [(-0.1, 7, "a"), (-0.2, 8, "b")]}}
        ] * MIN_QUALITY_PROMPTS
        handle = adapter.launch(sleep_spec, LocalExecutor())
        handle.state["engine"] = _FakeEngine(outputs)

        reference = adapter.capture_quality_reference(handle, prompts)
        assert reference[0] == [7, 8]
        assert adapter.quality_gate(handle, prompts).passed is True

    def test_text_only_output_is_refused_never_compared(self, adapter, sleep_spec, fake_sglang):
        """Comparing decoded text would pass a gate that token comparison fails:
        retokenization can round-trip differing ids to identical strings (§19)."""
        handle = adapter.launch(sleep_spec, LocalExecutor())
        handle.state["engine"] = _FakeEngine([{"text": "hello", "meta_info": {}}])
        with pytest.raises(AdapterError, match="no token ids"):
            adapter.capture_quality_reference(handle, ["p"])

    def test_a_process_handle_has_nothing_to_generate_with(self, adapter, sleep_spec, fake_sglang):
        handle = adapter.launch(sleep_spec, LocalExecutor())
        with pytest.raises(AdapterError, match="in-process SGLang engine"):
            adapter.capture_quality_reference(handle, ["p"])


class TestKnowledgeFile:
    """§10.6: knowledge is data, and data that disagrees with the code is worse than none."""

    @pytest.fixture
    def knowledge(self) -> dict:
        path = knowledge_path()
        assert path is not None, "knowledge_base/common/framework_sglang.yaml not found"
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_lives_where_the_loader_will_find_it(self):
        path = knowledge_path()
        assert path is not None
        # The loader collects common/ -> <dsl>/common/ -> <dsl>/<device>/; a flat file
        # at the knowledge-base root is silently ignored.
        assert path.parent.name == "common"

    def test_load_knowledge_is_optional(self, tmp_path):
        assert load_knowledge(tmp_path / "nope.yaml") == {}

    def test_names_this_adapter_at_tier_one(self, knowledge):
        assert knowledge["framework"] == "sglang"
        assert knowledge["tier"] == 1
        assert knowledge["adapter"] == "xe_forge.orbit.adapters.sglang:SGLangAdapter"

    def test_axes_agree_with_the_adapter(self, adapter, knowledge):
        code = {axis.name: axis for axis in adapter.config_axes()}
        for raw in knowledge["config_axes"]:
            axis = code[raw["name"]]
            assert axis.env_var == raw.get("env_var")
            assert list(axis.values) == list(raw.get("values") or [])
            if raw.get("cli_flag"):
                assert _CLI_FLAGS[raw["name"]] == raw["cli_flag"]
            assert (raw["name"] in _PRESENCE_AXES) == bool(raw.get("presence"))

    def test_metric_patterns_agree_with_the_adapter(self, knowledge):
        declared: dict[str, list[str]] = {}
        for metric, pattern in _METRIC_PATTERNS:
            declared.setdefault(metric, []).append(pattern.pattern)
        assert knowledge["metrics"]["patterns"] == declared

    def test_unverified_facts_are_marked_not_parsed(self, knowledge):
        """A parse rule that is not certain is documented, never shipped: a wrong
        rule that runs cleanly reports a confident wrong number."""
        for entry in knowledge["metrics"].get("unverified", []):
            assert entry["confidence"] == "unverified"
            for patterns in knowledge["metrics"]["patterns"].values():
                assert not any(entry["source"].split(".")[-1] in p for p in patterns)

    def test_determinism_sources_agree_with_the_adapter(self, adapter, knowledge):
        profile = adapter.determinism_profile()
        assert set(knowledge["determinism"]["pinnable"]) == profile.pinnable
        assert set(knowledge["determinism"]["non_pinnable"]) == profile.non_pinnable

    def test_declares_what_section_10_6_requires(self, knowledge):
        assert {
            "dispatch_roots",
            "kernel_module_globs",
            "backend_selection_env",
            "config_axes",
            "metrics",
            "determinism",
            "opaque_providers",
            "patch_points",
            "version_compatibility",
            "kernel_sources",
        } <= set(knowledge)

    def test_patch_layers_agree_with_the_declared_capabilities(self, adapter, knowledge):
        layers = {p["layer"] for p in knowledge["patch_points"] if p["rung"] == "P3"}
        assert layers == adapter.capabilities.patchable_layers

    def test_quality_gate_constants_agree_with_the_adapter(self, knowledge):
        gate = knowledge["quality_gate"]
        assert gate["min_prompts"] == MIN_QUALITY_PROMPTS
        assert gate["seed"] == QUALITY_SEED
        assert gate["max_tokens"] == QUALITY_MAX_TOKENS
        assert gate["comparison"] == "token_exact"

    def test_the_knowledge_base_loader_skips_it_visibly(self):
        """It is not a pattern file; the legacy loader must record the skip, not
        absorb it as a silent no-op indistinguishable from "loaded fine" (§9.5)."""
        from xe_forge.knowledge.loader import load_knowledge_base

        path = knowledge_path()
        assert path is not None
        kb = load_knowledge_base(path.parents[1])
        framework_skips = [
            s for s in kb.skipped if "orbit framework knowledge" in s.get("name", "")
        ]
        assert any(s["file"] == "framework_sglang.yaml" for s in framework_skips)


class TestDetectionRequiresWorkloadEvidence:
    """An installed SGLang says nothing about the workload in front of us.

    The same regression guard the vLLM adapter carries, because the failure is
    structural, not framework-specific: `resolve_adapter` prefers the highest tier,
    so installability-as-evidence makes a Tier 1 adapter claim every workload on the
    machine, handing each one declared TTFT/TPOT and a determinism profile
    describing radix caching it does not have.
    """

    def _adapter(self):
        return get_adapter("sglang")

    def test_an_sglang_command_is_claimed(self):
        spec = WorkloadSpec(
            command=["python", "-m", "sglang.launch_server", "--model-path", "Qwen/Qwen3-0.6B"]
        )
        assert self._adapter().detect(spec) is True

    def test_a_plain_torch_workload_is_not_claimed(self):
        """Must degrade to Tier 0, which is the design (§10.2), not a loss."""
        spec = WorkloadSpec(command=["python", "train.py"])
        assert self._adapter().detect(spec) is False

    def test_another_framework_is_not_hijacked(self):
        spec = WorkloadSpec(command=["vllm", "serve", "Qwen/Qwen3-0.6B"])
        assert self._adapter().detect(spec) is False

    def test_sglang_environment_variables_count_as_evidence(self):
        spec = WorkloadSpec(
            command=["python", "bench.py"], env={"SGLANG_TORCH_PROFILER_DIR": "/tmp/prof"}
        )
        assert self._adapter().detect(spec) is True

    def test_an_explicit_declaration_is_honoured(self):
        """The escape hatch for a script that uses SGLang without naming it."""
        spec = WorkloadSpec(command=["python", "serve.py"], framework="sglang")
        assert self._adapter().detect(spec) is True

    def test_resolution_picks_tier_zero_for_a_plain_workload(self):
        from xe_forge.orbit.adapters import resolve_adapter

        spec = WorkloadSpec(command=["python", "train.py"])
        assert resolve_adapter(spec).name == "generic_torch"

    def test_the_two_tier_one_adapters_never_claim_each_other(self):
        """Both Tier 1 adapters on one machine must partition, not race."""
        from xe_forge.orbit.adapters import get_adapter as _get

        sglang_spec = WorkloadSpec(command=["python", "-m", "sglang.launch_server"])
        vllm_spec = WorkloadSpec(command=["vllm", "serve", "m"])
        assert _get("sglang").detect(sglang_spec) is True
        assert _get("vllm").detect(sglang_spec) is False
        assert _get("vllm").detect(vllm_spec) is True
        assert _get("sglang").detect(vllm_spec) is False


class TestConformance:
    """§10.7: the same suite every adapter passes — the portability proof itself."""

    def test_loadable_and_passes_the_quick_suite_without_sglang(self, adapter):
        """The adapter must be runnable by the conformance harness on a machine with
        no SGLang: detect/versions round-trip, the full lifecycle on a synthetic
        workload, metrics consistent with declared capabilities (missing serving
        metrics *explained*, per §10.4's honest-degradation rule), and reset_state
        honouring its declared capability."""
        from xe_forge.orbit.adapters.conformance import run_conformance

        report = run_conformance(adapter, repetitions=2, quick=True)
        failures = [c.name for c in report.checks if not c.passed and not c.skipped]
        assert report.passed, f"conformance failures: {failures}"

    def test_what_needs_a_live_engine_fails_by_naming_the_package(self, adapter, sleep_spec):
        """Anything that genuinely needs SGLang must raise a clean AdapterError
        naming the missing package — never a bare ImportError traceback, and never a
        fabricated result."""
        if sglang_available():
            pytest.skip("this machine has sglang installed")
        handle = adapter.launch(sleep_spec, LocalExecutor())
        with pytest.raises(AdapterError, match="sglang"):
            adapter.quality_gate(handle, [f"p{i}" for i in range(MIN_QUALITY_PROMPTS)])
        with pytest.raises(AdapterError, match="sglang"):
            adapter.capture_quality_reference(handle, ["p"])
