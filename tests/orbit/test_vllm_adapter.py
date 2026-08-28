"""
The Tier 1 vLLM adapter (plan §10, §13, §19).

Every test here runs on a machine with **no vLLM and no GPU**, because that is where
CI runs and because a Tier 1 adapter that can only be exercised on a serving node is a
Tier 1 adapter nobody will maintain.

The load-bearing assertions are the ones that catch a plausible lie: an adapter that
claims a metric it did not parse, a determinism profile that omits the source that
actually broke the comparison, a config pin that was requested and silently dropped, a
quality gate that reports a pass it never ran.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types

import pytest
import yaml

from xe_forge.orbit.adapters import AdapterError, VLLMAdapter, describe_adapters, get_adapter
from xe_forge.orbit.adapters.base import LoadSpec
from xe_forge.orbit.adapters.vllm import (
    _CLI_FLAGS,
    _METRIC_PATTERNS,
    MIN_QUALITY_PROMPTS,
    knowledge_path,
    load_knowledge,
    parse_vllm_metrics,
    vllm_available,
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

# A captured `vllm bench serve` result table. Kept verbatim, including the trailing
# column padding, because the parser has to survive the real thing.
SERVE_OUTPUT = """\
INFO 08-25 11:02:14 [__init__.py:216] Automatically detected platform xpu.
Starting initial single prompt test run...
Traffic request rate: inf
============ Serving Benchmark Result ============
Successful requests:                     1000
Benchmark duration (s):                  56.78
Total input tokens:                      215196
Total generated tokens:                  198343
Request throughput (req/s):              17.61
Output token throughput (tok/s):         3493.28
Total Token throughput (tok/s):          7283.53
---------------Time to First Token----------------
Mean TTFT (ms):                          2456.23
Median TTFT (ms):                        2345.67
P99 TTFT (ms):                           5678.90
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          45.67
Median TPOT (ms):                        43.21
P99 TPOT (ms):                           89.01
---------------Inter-token Latency----------------
Mean ITL (ms):                           43.12
Median ITL (ms):                         40.00
P99 ITL (ms):                            120.55
==================================================
"""

# `vllm bench throughput` reports a single line instead.
THROUGHPUT_OUTPUT = """\
Processed prompts: 100%|##########| 1000/1000 [00:47<00:00, 21.2it/s]
Throughput: 21.16 requests/s, 10842.36 total tokens/s, 5417.92 output tokens/s
"""

# An offline engine log: throughput only, no TTFT or TPOT anywhere.
ENGINE_LOG_OUTPUT = """\
INFO 08-25 11:04:31 metrics.py:396] Avg prompt throughput: 0.0 tokens/s, \
Avg generation throughput: 1274.6 tokens/s, Running: 8 reqs, Pending: 0 reqs, \
GPU KV cache usage: 12.3%, CPU KV cache usage: 0.0%
"""


@pytest.fixture
def adapter() -> VLLMAdapter:
    return VLLMAdapter()


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
        command=["vllm", "serve", "meta-llama/Llama-3.1-8B", "--port", "8123"],
        repetitions=2,
        warmup_iterations=0,
    )


@pytest.fixture
def fake_vllm(monkeypatch):
    """A stub `vllm` module, so the L3 gate's logic is testable without vLLM.

    It carries a real `__spec__` because `vllm_available()` asks `find_spec`, which
    consults `sys.modules` first and rejects a module without one.
    """
    module = types.ModuleType("vllm")
    module.__spec__ = importlib.machinery.ModuleSpec("vllm", loader=None)

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module.SamplingParams = SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", module)
    return module


class _FakeLLM:
    """Returns canned token ids, and records the sampling parameters it was given."""

    def __init__(self, token_ids: list[list[int]]) -> None:
        self.token_ids = token_ids
        self.last_params = None

    def generate(self, prompts, params):
        self.last_params = params
        completion = types.SimpleNamespace
        return [
            completion(outputs=[completion(token_ids=ids)])
            for ids in self.token_ids[: len(prompts)]
        ]


class TestRegistration:
    def test_registered_at_tier_one(self):
        rows = {row["name"]: row for row in describe_adapters()}
        assert rows["vllm"]["tier"] == 1
        assert set(rows["vllm"]["metrics"]) == {"wall_time", "throughput", "ttft", "tpot"}

    def test_resolvable_by_name(self):
        assert get_adapter("vllm").name == "vllm"


class TestDetect:
    @pytest.mark.skipif(vllm_available(), reason="this machine has vllm installed")
    def test_absent_vllm_yields_the_tier_zero_fallback(self, adapter, sleep_spec):
        """detect() must return False, not crash, so Tier 0 wins resolution (§10.2)."""
        from xe_forge.orbit.adapters import resolve_adapter

        assert adapter.detect(sleep_spec) is False
        assert resolve_adapter(sleep_spec).name == "generic_torch"

    def test_command_evidence_is_enough(self, adapter, serve_spec):
        assert adapter.detect(serve_spec) is True

    def test_benchmark_script_is_recognised(self, adapter):
        spec = WorkloadSpec(command=["python", "benchmark_serving.py", "--num-prompts", "64"])
        assert adapter.detect(spec) is True

    def test_another_framework_is_never_hijacked(self, adapter, sleep_spec):
        """A box with vLLM on it must not claim an SGLang workload."""
        spec = sleep_spec.model_copy(update={"framework": "sglang"})
        assert adapter.detect(spec) is False

    def test_versions_reports_only_what_is_installed(self, adapter):
        versions = adapter.versions()
        assert isinstance(versions, dict)
        assert all(isinstance(v, str) for v in versions.values())
        if not vllm_available():
            assert "vllm" not in versions


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

    def test_reset_state_is_declared_and_names_its_mechanism(self, adapter, sleep_spec):
        handle = adapter.launch(sleep_spec, LocalExecutor())
        adapter.reset_state(handle)
        assert "process-per-repetition" in handle.state["state_reset"]

    def test_patch_points_never_name_an_undeclared_layer(self, adapter):
        kernels = [
            KernelRecord(id="k0", runtime_name="vllm::unified_attention"),
            KernelRecord(id="k1", runtime_name="fused_moe_kernel"),
            KernelRecord(id="k2", runtime_name="rms_norm_kernel"),
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
        # Accept rate and arrival timing are not engine settings; claiming to pin them
        # would let §17 emit ACCEPT on a comparison that was never controlled.
        assert profile.non_pinnable == {"speculative_decoding", "request_arrival_jitter"}

    def test_notes_name_the_actual_knobs(self, adapter):
        notes = adapter.determinism_profile().notes
        assert "--no-enable-prefix-caching" in notes
        assert "--max-num-seqs 1" in notes
        assert "--enforce-eager" in notes

    def test_active_sources_come_from_the_command(self, adapter):
        spec = WorkloadSpec(
            command=[
                "python",
                "benchmark_serving.py",
                "--request-rate",
                "8",
                "--num-speculative-tokens",
                "4",
            ]
        )
        active = adapter.determinism_profile(spec).active_non_pinnable
        assert active == {"speculative_decoding", "request_arrival_jitter"}

    def test_nothing_is_active_by_default(self, adapter, serve_spec):
        assert adapter.determinism_profile(serve_spec).active_non_pinnable == set()

    def test_prepare_warns_about_the_unpinned_engine_version(
        self, adapter, serve_spec, monkeypatch
    ):
        monkeypatch.delenv("VLLM_USE_V1", raising=False)
        notes = " ".join(adapter.prepare(serve_spec).notes)
        assert "VLLM_USE_V1" in notes


class TestConfigAxes:
    def test_the_declared_axes_are_present(self, adapter):
        names = {axis.name for axis in adapter.config_axes()}
        assert {
            "attention_backend",
            "use_v1",
            "enable_prefix_caching",
            "max_num_seqs",
            "max_num_batched_tokens",
            "enforce_eager",
            "gpu_memory_utilization",
        } <= names

    def test_env_axes_map_to_environment_variables(self, adapter, serve_spec):
        env_vars = {axis.name: axis.env_var for axis in adapter.config_axes() if axis.env_var}
        assert env_vars["attention_backend"] == "VLLM_ATTENTION_BACKEND"
        assert env_vars["use_v1"] == "VLLM_USE_V1"

        updated = adapter.apply_config(
            serve_spec, {"attention_backend": "TRITON_ATTN", "use_v1": 1}
        )
        assert updated.env["VLLM_ATTENTION_BACKEND"] == "TRITON_ATTN"
        assert updated.env["VLLM_USE_V1"] == "1"
        assert serve_spec.env == {}  # the original is untouched

    def test_cli_axes_become_engine_flags(self, adapter, serve_spec):
        updated = adapter.apply_config(
            serve_spec, {"max_num_seqs": 1, "enable_prefix_caching": False, "enforce_eager": True}
        )
        assert "--max-num-seqs" in updated.command
        assert updated.command[updated.command.index("--max-num-seqs") + 1] == "1"
        assert "--no-enable-prefix-caching" in updated.command
        assert "--enforce-eager" in updated.command

    def test_an_existing_flag_is_replaced_not_duplicated(self, adapter):
        spec = WorkloadSpec(command=["vllm", "serve", "m", "--max-num-seqs", "256"])
        updated = adapter.apply_config(spec, {"max_num_seqs": 1})
        assert updated.command.count("--max-num-seqs") == 1
        assert "256" not in updated.command

    def test_a_pin_that_cannot_be_applied_is_refused_not_dropped(self, adapter, sleep_spec):
        """Silently dropping a determinism pin produces a run that looks controlled."""
        with pytest.raises(AdapterError, match="not a vLLM entrypoint"):
            adapter.apply_config(sleep_spec, {"max_num_seqs": 1})

    def test_unknown_axis_is_refused(self, adapter, serve_spec):
        with pytest.raises(AdapterError, match="unknown vLLM config axis"):
            adapter.apply_config(serve_spec, {"turbo_mode": True})


class TestMetricParsing:
    def test_serving_table(self):
        metrics = parse_vllm_metrics(SERVE_OUTPUT)
        assert metrics["ttft_ms"] == pytest.approx(2456.23)
        assert metrics["tpot_ms"] == pytest.approx(45.67)
        # Output token throughput, not total: total moves with input length, which
        # makes it useless for an A/B on a decode kernel.
        assert metrics["throughput"] == pytest.approx(3493.28)

    def test_throughput_command_line(self):
        metrics = parse_vllm_metrics(THROUGHPUT_OUTPUT)
        assert metrics["throughput"] == pytest.approx(5417.92)
        assert "ttft_ms" not in metrics
        assert "tpot_ms" not in metrics

    def test_engine_log_line(self):
        assert parse_vllm_metrics(ENGINE_LOG_OUTPUT)["throughput"] == pytest.approx(1274.6)

    def test_absent_metrics_are_absent_not_zero(self):
        """A zero TTFT would sail through the statistics as a real measurement."""
        assert parse_vllm_metrics("nothing to see here") == {}

    def test_median_is_the_documented_fallback(self):
        text = SERVE_OUTPUT.replace("Mean TTFT (ms):", "P50 TTFT (ms):")
        assert parse_vllm_metrics(text)["ttft_ms"] == pytest.approx(2345.67)


class TestBenchmark:
    def test_a_non_vllm_run_degrades_to_wall_time_and_says_so(self, adapter, sleep_spec):
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
            update={"command": [sys.executable, "-c", script], "framework": "vllm"}
        )
        handle = adapter.launch(spec, LocalExecutor())
        measurement = adapter.benchmark(handle, LoadSpec(repetitions=2))

        assert set(measurement.metrics_available) == adapter.capabilities.metrics
        assert measurement.ttft_ms.mean == pytest.approx(2456.23)
        assert measurement.tpot_ms.mean == pytest.approx(45.67)
        assert adapter.metric_fallback(handle) is None


class TestPatchPoints:
    def test_dispatcher_op_gets_p1_first(self, adapter):
        """P1 touches nothing in the framework, so it is offered ahead of P3 (§13)."""
        kernel = KernelRecord(
            id="k0",
            runtime_name="paged_attention_v1_kernel",
            framework_op="_C::paged_attention_v1",
        )
        points = adapter.patch_points(kernel)
        assert points[0].rung == "P1"
        assert points[0].target == "_C::paged_attention_v1"
        assert "dispatch" in points[0].mechanism.lower()

    def test_attention_kernel_gets_the_backend_registry(self, adapter):
        kernel = KernelRecord(id="k1", runtime_name="vllm_unified_attention_kernel")
        p3 = [p for p in adapter.patch_points(kernel) if p.rung == "P3"]
        assert [p.target for p in p3] == ["attention_backend"]
        assert "VLLM_ATTENTION_BACKEND" in p3[0].mechanism

    def test_moe_kernel_gets_the_fused_moe_layer(self, adapter):
        kernel = KernelRecord(id="k2", runtime_name="fused_moe_kernel", framework_op="_moe_C::topk")
        rungs = {(p.rung, p.target) for p in adapter.patch_points(kernel)}
        assert ("P1", "_moe_C::topk") in rungs
        assert ("P3", "fused_moe") in rungs

    def test_custom_op_layer_is_offered_for_a_norm(self, adapter):
        kernel = KernelRecord(id="k3", runtime_name="rms_norm_kernel")
        assert [p.target for p in adapter.patch_points(kernel)] == ["custom_op"]

    def test_an_unattributable_kernel_gets_nothing(self, adapter):
        """No op, no recognised layer: naming a patch point here would be a guess."""
        assert adapter.patch_points(KernelRecord(id="k4", runtime_name="mystery_0000")) == []

    def test_dispatch_roots_cover_the_native_libraries(self, adapter):
        roots = adapter.dispatch_roots()
        assert {"torch.ops._C", "torch.ops._moe_C"} <= set(roots)
        assert "vllm" in " ".join(adapter.provenance_hints())


class TestQualityGate:
    def test_unavailable_without_vllm_raises_rather_than_failing(self, adapter, sleep_spec):
        """An unavailable gate and a failed gate are different facts (§19)."""
        if vllm_available():
            pytest.skip("this machine has vllm installed")
        handle = adapter.launch(sleep_spec, LocalExecutor())
        with pytest.raises(AdapterError, match="vllm is not importable"):
            adapter.quality_gate(handle, [f"prompt {i}" for i in range(MIN_QUALITY_PROMPTS)])

    def test_too_few_prompts_is_a_failed_gate(self, adapter, sleep_spec, fake_vllm):
        result = adapter.quality_gate(adapter.launch(sleep_spec, LocalExecutor()), ["one", "two"])
        assert result.passed is False
        assert str(MIN_QUALITY_PROMPTS) in result.detail

    def test_no_baseline_reference_raises(self, adapter, sleep_spec, fake_vllm):
        handle = adapter.launch(sleep_spec, LocalExecutor())
        handle.state["llm"] = _FakeLLM([[1, 2, 3]] * MIN_QUALITY_PROMPTS)
        with pytest.raises(AdapterError, match="capture_quality_reference"):
            adapter.quality_gate(handle, [f"prompt {i}" for i in range(MIN_QUALITY_PROMPTS)])

    def test_token_exact_match_passes_with_greedy_seeded_decode(
        self, adapter, sleep_spec, fake_vllm
    ):
        prompts = [f"prompt {i}" for i in range(MIN_QUALITY_PROMPTS)]
        handle = adapter.launch(sleep_spec, LocalExecutor())
        handle.state["llm"] = _FakeLLM([[7, 8, 9]] * MIN_QUALITY_PROMPTS)

        adapter.capture_quality_reference(handle, prompts)
        result = adapter.quality_gate(handle, prompts)

        assert result.passed is True
        assert result.token_exact is True
        params = handle.state["llm"].last_params.kwargs
        assert params["temperature"] == 0.0 and params["seed"] == 0

    def test_a_single_diverged_completion_fails_the_gate(self, adapter, sleep_spec, fake_vllm):
        prompts = [f"prompt {i}" for i in range(MIN_QUALITY_PROMPTS)]
        handle = adapter.launch(sleep_spec, LocalExecutor())
        llm = _FakeLLM([[7, 8, 9]] * MIN_QUALITY_PROMPTS)
        handle.state["llm"] = llm

        adapter.capture_quality_reference(handle, prompts)
        llm.token_ids = [[7, 8, 9]] * (MIN_QUALITY_PROMPTS - 1) + [[7, 8, 10]]
        result = adapter.quality_gate(handle, prompts)

        assert result.passed is False
        assert result.token_exact is False
        assert "diverged" in result.detail

    def test_a_process_handle_has_nothing_to_generate_with(self, adapter, sleep_spec, fake_vllm):
        handle = adapter.launch(sleep_spec, LocalExecutor())
        with pytest.raises(AdapterError, match="in-process vLLM engine"):
            adapter.capture_quality_reference(handle, ["p"])


class TestKnowledgeFile:
    """§10.6: knowledge is data, and data that disagrees with the code is worse than none."""

    @pytest.fixture
    def knowledge(self) -> dict:
        path = knowledge_path()
        assert path is not None, "knowledge_base/common/framework_vllm.yaml not found"
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_lives_where_the_loader_will_find_it(self):
        path = knowledge_path()
        assert path is not None
        # The loader collects common/ -> <dsl>/common/ -> <dsl>/<device>/; a flat file
        # at the knowledge-base root is silently ignored.
        assert path.parent.name == "common"

    def test_load_knowledge_is_optional(self, tmp_path):
        assert load_knowledge(tmp_path / "nope.yaml") == {}

    def test_axes_agree_with_the_adapter(self, adapter, knowledge):
        code = {axis.name: axis for axis in adapter.config_axes()}
        for raw in knowledge["config_axes"]:
            axis = code[raw["name"]]
            assert axis.env_var == raw.get("env_var")
            assert list(axis.values) == list(raw.get("values") or [])
            if raw.get("cli_flag"):
                assert _CLI_FLAGS[raw["name"]] == raw["cli_flag"]

    def test_metric_patterns_agree_with_the_adapter(self, knowledge):
        declared: dict[str, list[str]] = {}
        for metric, pattern in _METRIC_PATTERNS:
            declared.setdefault(metric, []).append(pattern.pattern)
        assert knowledge["metrics"]["patterns"] == declared

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
        } <= set(knowledge)

    def test_patch_layers_agree_with_the_declared_capabilities(self, adapter, knowledge):
        layers = {p["layer"] for p in knowledge["patch_points"] if p["rung"] == "P3"}
        assert layers == adapter.capabilities.patchable_layers

    def test_the_knowledge_base_loader_skips_it_visibly(self):
        """It is not a pattern file; the legacy loader must record the skip, not absorb it.

        The previous behaviour — loading it as a silent no-op, absent even from
        kb.skipped — was indistinguishable from "loaded fine" (§9.5). Now every
        framework knowledge file appears in kb.skipped with a reason naming its
        actual consumer, and contributes no patterns or constraints.
        """
        from xe_forge.knowledge.loader import load_knowledge_base

        path = knowledge_path()
        assert path is not None
        kb = load_knowledge_base(path.parents[1])
        framework_skips = [
            s for s in kb.skipped if "orbit framework knowledge" in s.get("name", "")
        ]
        assert any(s["file"] == "framework_vllm.yaml" for s in framework_skips)
        assert all("xe_forge.orbit" in s["name"] for s in framework_skips)


class TestDetectionRequiresWorkloadEvidence:
    """An installed vLLM says nothing about the workload in front of us.

    Regression guard for a bug that only a real vLLM install exposed. `detect()` once
    returned True whenever vLLM was importable; because `resolve_adapter` prefers the
    highest tier, that made this adapter claim every workload on the machine — plain
    PyTorch training included — and §10.4 has the decision layer read `capabilities` to
    choose which actions exist. The workload would inherit declared TTFT/TPOT and a
    determinism profile for prefix caching it does not have.
    """

    def _adapter(self):
        from xe_forge.orbit.adapters import get_adapter

        return get_adapter("vllm")

    def test_a_vllm_command_is_claimed(self):
        spec = WorkloadSpec(command=["vllm", "serve", "Qwen/Qwen3-0.6B"])
        assert self._adapter().detect(spec) is True

    def test_a_plain_torch_workload_is_not_claimed(self):
        """Must degrade to Tier 0, which is the design (§10.2), not a loss."""
        spec = WorkloadSpec(command=["python", "train.py"])
        assert self._adapter().detect(spec) is False

    def test_another_framework_is_not_hijacked(self):
        spec = WorkloadSpec(command=["python", "-m", "sglang.launch_server"])
        assert self._adapter().detect(spec) is False

    def test_vllm_environment_variables_count_as_evidence(self):
        spec = WorkloadSpec(command=["python", "bench.py"], env={"VLLM_USE_V1": "1"})
        assert self._adapter().detect(spec) is True

    def test_an_explicit_declaration_is_honoured(self):
        """The escape hatch for a script that uses vLLM without naming it."""
        spec = WorkloadSpec(command=["python", "serve.py"], framework="vllm")
        assert self._adapter().detect(spec) is True

    def test_resolution_picks_tier_zero_for_a_plain_workload(self):
        from xe_forge.orbit.adapters import resolve_adapter

        spec = WorkloadSpec(command=["python", "train.py"])
        assert resolve_adapter(spec).name == "generic_torch"
