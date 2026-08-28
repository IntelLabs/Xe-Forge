"""
Tier 1 vLLM adapter: serving metrics, determinism pins, config axes, patch points and
the L3 model-level quality gate. `vllm` is imported lazily so the module works with no
vLLM installed. Design rationale: docs/DESIGN.md.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any

from xe_forge.orbit.adapters.base import (
    AdapterError,
    BaseAdapter,
    Handle,
    LoadSpec,
    MetricSpec,
    PreparedWorkload,
)
from xe_forge.orbit.bench.core import BenchRunner
from xe_forge.orbit.executor import Executor, LocalExecutor, RunResult
from xe_forge.orbit.models import (
    ConfigAxis,
    DeterminismProfile,
    FrameworkCapabilities,
    KernelRecord,
    PatchPoint,
    QualityResult,
    WorkloadMeasurement,
    WorkloadSpec,
)

logger = logging.getLogger(__name__)

# The L3 gate is fixed-shape: greedy decode, fixed seed, at least this many prompts.
MIN_QUALITY_PROMPTS = 32
QUALITY_SEED = 0
QUALITY_MAX_TOKENS = 64

# Versions whose change invalidates stored artifacts and accepted candidates.
# Reported as found; an absent package is simply not listed.
_TRACKED_PACKAGES = (
    "vllm",
    "torch",
    "intel-extension-for-pytorch",
    "pytorch-triton-xpu",
    "triton",
)

# Tokens that identify a vLLM entrypoint on a command line. Used both by `detect()`
# and by `apply_config()`, which refuses to append engine flags to a command that will
# not understand them.
# Whole-token markers in a command's arguments. Matched exactly, never as substrings.
_VLLM_COMMAND_TOKENS = (
    "vllm",
    "benchmark_serving",
    "benchmark_throughput",
    "benchmark_latency",
)

# Executable basenames that are vLLM entrypoints in their own right.
_VLLM_EXECUTABLES = frozenset({"vllm"})

# ---------------------------------------------------------------------------
# Metrics parsing
# ---------------------------------------------------------------------------

# Ordered preference per metric: the first pattern that matches wins. Total token
# throughput moves with input length, so it is a fallback, never the default.
_METRIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ttft_ms", re.compile(r"^\s*Mean TTFT \(ms\):\s*([0-9.]+)", re.M)),
    ("ttft_ms", re.compile(r"^\s*Median TTFT \(ms\):\s*([0-9.]+)", re.M)),
    ("tpot_ms", re.compile(r"^\s*Mean TPOT \(ms\):\s*([0-9.]+)", re.M)),
    ("tpot_ms", re.compile(r"^\s*Median TPOT \(ms\):\s*([0-9.]+)", re.M)),
    ("tpot_ms", re.compile(r"^\s*Mean ITL \(ms\):\s*([0-9.]+)", re.M)),
    ("throughput", re.compile(r"^\s*Output token throughput \(tok/s\):\s*([0-9.]+)", re.M)),
    ("throughput", re.compile(r"^\s*Total Token throughput \(tok/s\):\s*([0-9.]+)", re.M)),
    (
        "throughput",
        re.compile(
            r"Throughput:\s*[0-9.]+\s*requests/s,\s*[0-9.]+\s*total tokens/s,"
            r"\s*([0-9.]+)\s*output tokens/s"
        ),
    ),
    ("throughput", re.compile(r"Throughput:\s*[0-9.]+\s*requests/s,\s*([0-9.]+)\s*total tokens/s")),
    ("throughput", re.compile(r"Avg generation throughput:\s*([0-9.]+)\s*tokens/s")),
)

# BenchRunner field names on the left, capability names declared in
# `FrameworkCapabilities.metrics` on the right.
_METRIC_CAPABILITY_NAMES = {
    "throughput": "throughput",
    "ttft_ms": "ttft",
    "tpot_ms": "tpot",
}


def parse_vllm_metrics(
    text: str,
    extra_patterns: tuple[tuple[str, re.Pattern[str]], ...] = (),
) -> dict[str, float]:
    """Pull serving metrics out of vLLM benchmark output.

    A metric that did not appear is absent from the result rather than zero.
    """
    found: dict[str, float] = {}
    for key, pattern in (*_METRIC_PATTERNS, *extra_patterns):
        if key in found:
            continue
        match = pattern.search(text)
        if match is None:
            continue
        try:
            found[key] = float(match.group(1))
        except (TypeError, ValueError):  # a pattern whose group is not a number
            continue
    return found


# ---------------------------------------------------------------------------
# Config axes
# ---------------------------------------------------------------------------

# Axes steered by environment variables. These apply to any vLLM process, including
# one launched by a harness Orbit does not control.
_ENV_AXES = (
    ConfigAxis(
        name="attention_backend",
        env_var="VLLM_ATTENTION_BACKEND",
        values=[
            "FLASH_ATTN",
            "FLASHINFER",
            "TRITON_ATTN",
            "FLEX_ATTENTION",
            "XFORMERS",
            "TORCH_SDPA",
            "IPEX",
        ],
        description=(
            "Attention backend selection. The legal set is version- and platform-bound "
            "(XPU builds accept far fewer); the knowledge file carries the per-version list"
        ),
    ),
    ConfigAxis(
        name="use_v1",
        env_var="VLLM_USE_V1",
        values=["0", "1"],
        description=(
            "V0 vs V1 engine. Changes the scheduler, the chunked-prefill policy and the "
            "kernel mix, so it must be pinned before any comparison, never left to default"
        ),
    ),
    ConfigAxis(
        name="custom_ops",
        env_var="VLLM_CUSTOM_OPS",
        description=(
            "Enable or disable individual CustomOp implementations, e.g. '+rms_norm' or "
            "'-silu_and_mul'. This is also the revert path for a P3 CustomOp substitution"
        ),
    ),
)

# Axes that are engine CLI flags. `ConfigAxis` has no flag field, so the
# framework-specific flag mapping lives here rather than in the shared model.
_CLI_AXES = (
    ConfigAxis(
        name="enable_prefix_caching",
        values=[True, False],
        description="Prefix/radix KV reuse across requests — the largest single source of "
        "run-to-run nondeterminism in a serving benchmark",
    ),
    ConfigAxis(
        name="max_num_seqs",
        values=[1, 8, 32, 64, 128, 256],
        description="Scheduler batch width. 1 pins continuous-batching order at the cost of "
        "measuring a batch size the deployment does not run",
    ),
    ConfigAxis(
        name="max_num_batched_tokens",
        values=[512, 2048, 8192, 32768],
        description="Chunked-prefill boundary. Setting it at or above max_model_len fixes "
        "where prefill splits instead of letting the scheduler choose per step",
    ),
    ConfigAxis(
        name="enforce_eager",
        values=[True, False],
        description="Skip graph capture. Removes capture/warmup state from the measurement, "
        "and removes graph replay from the kernel mix — a different workload, honestly named",
    ),
    ConfigAxis(
        name="gpu_memory_utilization",
        values=[0.80, 0.90, 0.95],
        description="KV-cache size. Changes how many blocks exist and therefore when "
        "preemption and recompute occur",
    ),
)

_CLI_FLAGS: dict[str, str] = {
    "enable_prefix_caching": "--enable-prefix-caching",
    "max_num_seqs": "--max-num-seqs",
    "max_num_batched_tokens": "--max-num-batched-tokens",
    "enforce_eager": "--enforce-eager",
    "gpu_memory_utilization": "--gpu-memory-utilization",
}

# Flags vLLM exposes in `--flag` / `--no-flag` form.
_BOOLEAN_AXES = frozenset({"enable_prefix_caching", "enforce_eager"})

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

# The source names are the ones Tier 0 uses, so a report can compare what a Tier 1
# adapter pins against what the generic path cannot pin at all.
_PINNABLE_KNOBS: dict[str, str] = {
    "prefix_cache_reuse": "--no-enable-prefix-caching (V1) / omit --enable-prefix-caching (V0)",
    "continuous_batching_order": "--max-num-seqs 1 serialises the scheduler",
    "chunked_prefill_boundaries": (
        "--max-num-batched-tokens >= max_model_len fixes the split point; V1 always chunks, "
        "so the boundary is pinned rather than removed"
    ),
    "graph_capture_warmup": "--enforce-eager skips capture, or discard the first N runs",
}

_NON_PINNABLE_KNOBS: dict[str, str] = {
    "speculative_decoding": (
        "the accept rate is a property of draft/target agreement at run time. It can be "
        "removed (no speculative config) but not pinned, and removing it measures a "
        "different workload"
    ),
    "request_arrival_jitter": (
        "arrival times come from the load generator, not the engine. A fixed --request-rate "
        "narrows the distribution; the OS scheduler still moves the batch boundaries"
    ),
}

# Command-line evidence that a non-pinnable source is actually active in this run;
# the measurement layer refuses ACCEPT when one is live and variance exceeds the MDE.
_ACTIVE_MARKERS: dict[str, tuple[str, ...]] = {
    "speculative_decoding": (
        "--speculative",
        "--num-speculative-tokens",
        "speculative_config",
        "--ngram",
    ),
    "request_arrival_jitter": ("--request-rate", "--num-prompts", "benchmark_serving"),
}

# ---------------------------------------------------------------------------
# Patch points
# ---------------------------------------------------------------------------

_ATTENTION_HINTS = (
    "paged_attention",
    "unified_attention",
    "flash_attn",
    "flashinfer",
    "chunked_prefill",
    "reshape_and_cache",
    "attn",
)
_MOE_HINTS = ("fused_moe", "moe_align", "topk_softmax", "moe_sum", "grouped_gemm")
_CUSTOM_OP_HINTS = (
    "rms_norm",
    "silu_and_mul",
    "gelu_and_mul",
    "act_and_mul",
    "rotary_embedding",
)


def _mentions(kernel: KernelRecord, hints: tuple[str, ...]) -> bool:
    haystack = " ".join(
        part
        for part in (
            kernel.runtime_name,
            kernel.demangled_name,
            kernel.framework_op,
            kernel.source_symbol,
        )
        if part
    ).lower()
    return any(hint in haystack for hint in hints)


# ---------------------------------------------------------------------------
# Knowledge file
# ---------------------------------------------------------------------------

KNOWLEDGE_FILENAME = "framework_vllm.yaml"


def knowledge_path() -> Path | None:
    """Locate `common/framework_vllm.yaml`, or None.

    The loader is DSL/device-scoped — `common/` then `<dsl>/common/` then
    `<dsl>/<device>/` — so the file lives under `common/`; a flat file at the
    knowledge-base root is silently ignored.
    """
    roots: list[Path] = []
    for env_var in ("XE_ORBIT_KNOWLEDGE_DIR", "KNOWLEDGE_DIR"):
        value = os.environ.get(env_var)
        if value:
            roots.append(Path(value))
    roots.append(Path.cwd() / "knowledge_base")
    # src/xe_forge/orbit/adapters/vllm.py -> repository root
    roots.append(Path(__file__).resolve().parents[4] / "knowledge_base")

    for root in roots:
        candidate = root / "common" / KNOWLEDGE_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load_knowledge(path: Path | None = None) -> dict[str, Any]:
    """Read the knowledge file if it can be found; `{}` otherwise.

    Never required. The adapter's defaults are complete on their own, so a wheel
    install with no knowledge base still works; the file lets a site extend the action
    space or the metric patterns without patching code.
    """
    target = path or knowledge_path()
    if target is None:
        return {}
    try:
        import yaml

        with open(target, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except Exception as exc:  # a broken knowledge file must not break measurement
        logger.warning("could not read vLLM knowledge file %s: %s", target, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _knowledge_metric_patterns(
    knowledge: dict[str, Any],
) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Extra `metric -> regex` rules declared in the knowledge file."""
    rules = (knowledge.get("metrics") or {}).get("extra_patterns") or []
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        metric, pattern = rule.get("metric"), rule.get("pattern")
        if metric not in _METRIC_CAPABILITY_NAMES or not pattern:
            continue
        try:
            compiled.append((metric, re.compile(pattern, re.M)))
        except re.error as exc:
            logger.warning("ignoring invalid metric pattern for %s: %s", metric, exc)
    return tuple(compiled)


def _knowledge_axes(knowledge: dict[str, Any]) -> list[ConfigAxis]:
    """Config axes declared in the knowledge file, as `ConfigAxis` records."""
    axes: list[ConfigAxis] = []
    for raw in knowledge.get("config_axes") or []:
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        axes.append(
            ConfigAxis(
                name=str(raw["name"]),
                values=list(raw.get("values") or []),
                env_var=raw.get("env_var"),
                description=str(raw.get("description", "")),
            )
        )
    return axes


# ---------------------------------------------------------------------------
# Lazy vLLM access
# ---------------------------------------------------------------------------


def vllm_available() -> bool:
    """True when `import vllm` would find something. Does not import it."""
    try:
        return importlib.util.find_spec("vllm") is not None
    except (ImportError, ValueError):  # a broken or partially removed install
        return False


def _import_vllm() -> Any:
    try:
        import vllm
    except Exception as exc:  # ImportError, or a native-extension load failure
        raise AdapterError(
            f"vllm is not importable, so this operation cannot be performed: {exc}"
        ) from exc
    return vllm


def _mentions_vllm(command: list[str]) -> bool:
    """True when the *command* invokes vLLM, not merely when a path contains "vllm".

    The executable is matched on its basename and the remaining arguments as whole
    dotted tokens, so `vllm serve ...` and `python -m vllm.entrypoints...` count while
    an interpreter path that merely contains "vllm" does not.
    """
    if not command:
        return False

    executable = PurePosixPath(command[0]).name.lower()
    if executable in _VLLM_EXECUTABLES:
        return True

    for argument in command[1:]:
        lowered = argument.lower()
        # `-m vllm.entrypoints.openai.api_server` and bare `vllm`, but not a path
        # component that merely contains the substring.
        head = lowered.split(".", 1)[0]
        if head == "vllm" and "/" not in lowered:
            return True
        # A script argument is matched on its filename stem, so `benchmark_serving.py`
        # and `/opt/vllm/benchmarks/benchmark_serving.py` both count while a directory
        # named after vLLM on the way to some other script does not.
        if PurePosixPath(lowered).stem in _VLLM_COMMAND_TOKENS:
            return True
    return False


def _env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _negated(flag: str) -> str:
    return f"--no-{flag.removeprefix('--')}"


def _apply_cli_flag(command: list[str], flag: str, value: Any, boolean: bool) -> list[str]:
    """Set one engine flag on the workload command, replacing any existing value."""
    negative = _negated(flag)
    out: list[str] = []
    skip_next = False
    for token in command:
        if skip_next:
            skip_next = False
            continue
        if token == flag or token == negative:
            skip_next = not boolean
            continue
        if token.startswith(f"{flag}=") or token.startswith(f"{negative}="):
            continue
        out.append(token)

    if boolean:
        out.append(flag if value else negative)
    else:
        out.extend([flag, str(value)])
    return out


class VLLMAdapter(BaseAdapter):
    """Tier 1 vLLM adapter. Declares only what it can deliver without guessing."""

    name = "vllm"
    tier = 1
    capabilities = FrameworkCapabilities(
        # Parsed from vLLM's own benchmark output; when they do not parse, `benchmark()`
        # reports wall_time alone and records why.
        metrics={"wall_time", "throughput", "ttft", "tpot"},
        # Prefix cache, captured graphs and scheduler queues are reset by
        # POST /reset_prefix_cache against a live server, by
        # `llm_engine.reset_prefix_cache()` in-process, or — for the process-launched
        # workloads Orbit actually runs — by the next repetition being a fresh process.
        can_reset_state=True,
        # --max-num-seqs 1 plus a fixed --max-num-batched-tokens.
        can_pin_batching=True,
        # vLLM v1 runs its engine in a separate process, so an in-process
        # torch.profiler in the caller observes zero device kernels.
        profiles_in_process=False,
        profile_hook=(
            'ProfilerConfig(profiler="torch", torch_profiler_dir=...) + '
            "LLM.start_profile()/stop_profile()"
        ),
        # --no-enable-prefix-caching.
        can_disable_prefix_cache=True,
        # Not declared: an in-situ layer harness cannot be verified without vLLM
        # installed, and an unverified constructor mis-ranks extraction tractability.
        can_construct_single_layer=False,
        patchable_layers={"attention_backend", "fused_moe", "custom_op"},
    )

    def __init__(self, executor: Executor | None = None) -> None:
        self.executor = executor or LocalExecutor()
        self.knowledge = load_knowledge()
        self._extra_patterns = _knowledge_metric_patterns(self.knowledge)

    # --- identity and lifecycle ------------------------------------------

    def detect(self, spec: WorkloadSpec) -> bool:
        """Claim only workloads with positive evidence of being vLLM ones.

        Evidence must be about the workload — the command names vLLM, the environment
        carries VLLM_* configuration, or the caller declared the framework — never the
        mere presence of the package on the machine. Anything else degrades to Tier 0.
        A script that uses vLLM without saying so is reached with `--framework vllm`.
        """
        declared = (spec.framework or "").strip().lower()
        if declared:
            return declared == self.name
        if _mentions_vllm(spec.command):
            return True
        # vLLM's own environment variables are evidence about this workload, unlike the
        # mere presence of the package on the machine.
        return any(key.startswith("VLLM_") for key in spec.env)

    def versions(self) -> dict[str, str]:
        """vLLM, torch and IPEX, as installed. Absent packages are simply not listed."""
        from xe_forge.orbit.runtime import environment

        return environment.package_versions(_TRACKED_PACKAGES)

    def prepare(self, spec: WorkloadSpec) -> PreparedWorkload:
        notes: list[str] = []
        if not vllm_available():
            notes.append(
                "vllm is not importable here: serving metrics will not parse, the L3 "
                "quality gate is unavailable, and this run degrades to wall-clock only."
            )
        if "VLLM_USE_V1" not in spec.env and "VLLM_USE_V1" not in os.environ:
            notes.append(
                "VLLM_USE_V1 is unpinned. V0 and V1 differ in scheduler, chunked-prefill "
                "policy and kernel mix; pin it before comparing anything."
            )
        profile = self.determinism_profile(spec)
        if profile.active_non_pinnable:
            notes.append(
                "active non-pinnable nondeterminism: "
                + ", ".join(sorted(profile.active_non_pinnable))
                + " — the verdict must be INCONCLUSIVE rather than ACCEPT if variance "
                "exceeds the MDE."
            )
        notes.append(
            "determinism pins available: --no-enable-prefix-caching, --max-num-seqs 1, "
            "--max-num-batched-tokens >= max_model_len, --enforce-eager."
        )
        return PreparedWorkload(spec=spec, notes=notes)

    def launch(self, spec: WorkloadSpec, executor: Executor) -> Handle:
        state: dict[str, Any] = {"executor": executor}
        base_url = _server_base_url(spec)
        if base_url:
            state["base_url"] = base_url
        return Handle(spec=spec, adapter=self.name, state=state)

    # --- measurement ------------------------------------------------------

    def metric_extractor(self, result: RunResult) -> dict[str, float]:
        """Serving metrics from one run. vLLM logs to both streams, so read both."""
        return parse_vllm_metrics(
            f"{result.stdout}\n{result.stderr}", extra_patterns=self._extra_patterns
        )

    def benchmark(self, handle: Handle, load: LoadSpec) -> WorkloadMeasurement:
        """Measure, reporting exactly the metrics that were actually parsed.

        A serving metric that did not appear in the output is absent, not zero, and the
        fallback to wall-clock is recorded on the handle so the report can say so.
        """
        executor = handle.state.get("executor") or self.executor
        runner = BenchRunner(executor=executor, metric_extractor=self.metric_extractor)
        measurement = runner.measure(
            handle.spec,
            repetitions=load.repetitions,
            profile_id=load.profile_id,
        )

        available = ["wall_time"]
        available.extend(
            declared
            for field, declared in _METRIC_CAPABILITY_NAMES.items()
            if getattr(measurement, field) is not None
        )
        measurement.metrics_available = available

        missing = sorted(set(self.capabilities.metrics) - set(available))
        if missing:
            reason = (
                f"vLLM serving metrics {missing} did not appear in the workload output; "
                f"reporting {available} only. Run a `vllm bench serve` / `bench throughput` "
                "command, or add a parsing rule to the knowledge file."
            )
            handle.state["metric_fallback"] = reason
            logger.warning("%s", reason)
        else:
            handle.state.pop("metric_fallback", None)
        return measurement

    def metric_fallback(self, handle: Handle) -> str | None:
        """Why the last `benchmark()` reported fewer metrics than declared, or None."""
        return handle.state.get("metric_fallback")

    def metrics_schema(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                name="wall_time",
                unit="s",
                lower_is_better=True,
                description="End-to-end process wall time",
            ),
            MetricSpec(
                name="throughput",
                unit="tok/s",
                lower_is_better=False,
                description="Output token throughput; total token throughput is the fallback",
            ),
            MetricSpec(
                name="ttft",
                unit="ms",
                lower_is_better=True,
                description="Mean time to first token (prefill-dominated)",
            ),
            MetricSpec(
                name="tpot",
                unit="ms",
                lower_is_better=True,
                description="Mean time per output token (decode-dominated)",
            ),
        ]

    # --- reproducibility --------------------------------------------------

    def determinism_profile(self, spec: WorkloadSpec | None = None) -> DeterminismProfile:
        """Which nondeterminism sources vLLM can pin, and which it cannot.

        Passing a spec is optional and additive: it fills `active_non_pinnable` from
        command-line evidence, which is what lets the verdict layer refuse ACCEPT and
        name the reason.
        """
        active: set[str] = set()
        if spec is not None:
            joined = " ".join(spec.command).lower()
            for source, markers in _ACTIVE_MARKERS.items():
                if any(marker in joined for marker in markers):
                    active.add(source)

        notes = "; ".join(
            [
                *(f"{name}: {knob}" for name, knob in _PINNABLE_KNOBS.items()),
                *(f"{name} (not pinnable): {why}" for name, why in _NON_PINNABLE_KNOBS.items()),
            ]
        )
        return DeterminismProfile(
            pinnable=set(_PINNABLE_KNOBS),
            non_pinnable=set(_NON_PINNABLE_KNOBS),
            active_non_pinnable=active,
            notes=notes,
        )

    def reset_state(self, handle: Handle) -> None:
        """Drop prefix-cache and engine state between measurements.

        Three mechanisms, in decreasing directness. The last one is not a no-op dressed
        up as a reset: Orbit runs each repetition as its own process, so no prefix
        cache, captured graph or scheduler queue survives into the next one.
        """
        super().reset_state(handle)

        base_url = handle.state.get("base_url")
        if base_url:
            self._post(f"{base_url}/reset_prefix_cache")
            handle.state["state_reset"] = f"POST {base_url}/reset_prefix_cache"
            return

        llm = handle.state.get("llm")
        if llm is not None:
            engine = getattr(llm, "llm_engine", llm)
            reset = getattr(engine, "reset_prefix_cache", None)
            if reset is None:
                raise AdapterError(
                    "the in-process vLLM engine on this handle exposes no "
                    "reset_prefix_cache(); refusing to claim state was reset"
                )
            reset()
            handle.state["state_reset"] = "llm_engine.reset_prefix_cache()"
            return

        handle.state["state_reset"] = (
            "process-per-repetition: the next run starts a fresh engine, so prefix "
            "cache, captured graphs and scheduler queues do not carry over"
        )

    @staticmethod
    def _post(url: str, timeout: float = 10.0) -> None:
        import urllib.error
        import urllib.request

        request = urllib.request.Request(url, data=b"", method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status >= 400:
                    raise AdapterError(f"{url} returned HTTP {response.status}")
        except urllib.error.URLError as exc:
            raise AdapterError(f"could not reach the vLLM server at {url}: {exc}") from exc

    # --- discovery and provenance ----------------------------------------

    def dispatch_roots(self) -> list[str]:
        """Where a vLLM kernel is reached from. `_C` and `_moe_C` are its native libs."""
        return [
            "aten",
            "torch.ops",
            "torch.ops._C",
            "torch.ops._C_cache_ops",
            "torch.ops._moe_C",
            "vllm.attention.ops",
            "vllm.model_executor.layers",
        ]

    def provenance_hints(self) -> list[str]:
        return [
            "vllm",
            "paged_attention",
            "unified_attention",
            "flash_attn",
            "fused_moe",
            "inductor",
            "triton",
            "onednn",
            "onemkl",
            "sycl",
            "ipex",
        ]

    # --- extraction -------------------------------------------------------

    def patch_points(self, kernel: KernelRecord) -> list[PatchPoint]:
        """Where an optimized kernel can be reinserted, highest rung first.

        P1 first because it touches nothing in the framework and reverts by not
        importing a module; P3 only for the layers this adapter actually declares in
        `patchable_layers`, so patch_points can never name a substitution point the
        capabilities deny.
        """
        points: list[PatchPoint] = []

        if kernel.framework_op:
            points.append(
                PatchPoint(
                    rung="P1",
                    target=kernel.framework_op,
                    mechanism="torch.library operator override on the XPU dispatch key",
                    description=(
                        "vLLM's native ops are registered through TORCH_LIBRARY "
                        "(torch.ops._C, torch.ops._moe_C), so an optimized kernel ships as "
                        "a small out-of-tree extension that shadows the default. vLLM is "
                        "untouched; revert is not importing the module. The dispatch "
                        "assertion must follow: the new kernel must appear in the "
                        "re-profile and the old one must not."
                    ),
                )
            )

        if _mentions(kernel, _ATTENTION_HINTS):
            points.append(
                PatchPoint(
                    rung="P3",
                    target="attention_backend",
                    mechanism=(
                        "register an alternative AttentionBackend and select it with "
                        "VLLM_ATTENTION_BACKEND (or the platform's get_attn_backend_cls hook)"
                    ),
                    description=(
                        "Framework registry substitution: vLLM chooses the attention "
                        "implementation through its own selection point, so the backend can "
                        "be replaced by config rather than by patching source. Revert is "
                        "restoring the environment variable."
                    ),
                )
            )

        if _mentions(kernel, _MOE_HINTS):
            points.append(
                PatchPoint(
                    rung="P3",
                    target="fused_moe",
                    mechanism=(
                        "substitute the FusedMoE method in "
                        "vllm.model_executor.layers.fused_moe, or override the fused_moe "
                        "custom op it dispatches to"
                    ),
                    description=(
                        "MoE runs through a selectable method object plus a tuned-config "
                        "JSON. The config is data and must be copied, never regenerated; "
                        "a mismatched config silently benchmarks a different tile."
                    ),
                )
            )

        if _mentions(kernel, _CUSTOM_OP_HINTS):
            points.append(
                PatchPoint(
                    rung="P3",
                    target="custom_op",
                    mechanism=(
                        "CustomOp.register(<name>) substitution in "
                        "vllm.model_executor.custom_op, selected with VLLM_CUSTOM_OPS"
                    ),
                    description=(
                        "vLLM routes elementwise and normalization layers through its "
                        "CustomOp registry, which is a supported selection point and "
                        "reverts by clearing VLLM_CUSTOM_OPS."
                    ),
                )
            )

        return points

    # --- action space -----------------------------------------------------

    def config_axes(self) -> list[ConfigAxis]:
        """Declared axes, extended (never overridden) by the knowledge file."""
        axes = [*_ENV_AXES, *_CLI_AXES]
        known = {axis.name for axis in axes}
        axes.extend(axis for axis in _knowledge_axes(self.knowledge) if axis.name not in known)
        return axes

    def apply_config(self, spec: WorkloadSpec, config: dict[str, Any]) -> WorkloadSpec:
        """Apply a config point to a workload, as env vars and as engine flags.

        Refuses rather than drops: a pin that was requested and silently not applied
        produces a measurement that looks controlled and is not.
        """
        axes = {axis.name: axis for axis in self.config_axes()}
        env = dict(spec.env)
        command = list(spec.command)

        for name, value in config.items():
            axis = axes.get(name)
            if axis is None:
                raise AdapterError(
                    f"unknown vLLM config axis {name!r}; declared axes: {sorted(axes)}"
                )
            if axis.env_var:
                env[axis.env_var] = _env_value(value)
                continue

            flag = _CLI_FLAGS.get(name)
            if flag is None:
                raise AdapterError(
                    f"config axis {name!r} declares neither an environment variable nor a "
                    "CLI flag, so it cannot be applied"
                )
            if not _mentions_vllm(command):
                raise AdapterError(
                    f"{flag} is a vLLM engine flag but the workload command is not a vLLM "
                    f"entrypoint ({command[0]!r}); appending it would break the command and "
                    "dropping it would leave the run unpinned"
                )
            command = _apply_cli_flag(command, flag, value, boolean=name in _BOOLEAN_AXES)

        return spec.model_copy(update={"env": env, "command": command})

    # --- correctness ------------------------------------------------------

    def capture_quality_reference(self, handle: Handle, prompts: list[str]) -> list[list[int]]:
        """Record the baseline token ids the L3 gate will compare against."""
        reference = self._generate_token_ids(handle, prompts)
        handle.state["quality_reference"] = reference
        return reference

    def quality_gate(self, handle: Handle, prompts: list[str]) -> QualityResult:
        """L3 gate: greedy decode, fixed seed, >= 32 prompts, token-exact comparison.

        Raises rather than returning a failure when the gate cannot be *run*: an
        unavailable gate and a failed gate are different facts.
        """
        if not vllm_available():
            raise AdapterError(
                "vllm is not importable, so the L3 model-level gate cannot be run; "
                "refusing to report a quality result that was never measured"
            )
        if len(prompts) < MIN_QUALITY_PROMPTS:
            return QualityResult(
                passed=False,
                detail=(
                    f"L3 requires at least {MIN_QUALITY_PROMPTS} prompts; got "
                    f"{len(prompts)}. Fewer prompts is a different gate, not a weaker one"
                ),
            )
        reference = handle.state.get("quality_reference")
        if reference is None:
            raise AdapterError(
                "no baseline token reference on this handle: call "
                "capture_quality_reference() against the unmodified workload first. A gate "
                "with nothing to compare against can neither pass nor fail"
            )

        candidate = self._generate_token_ids(handle, prompts)
        if len(candidate) != len(reference):
            return QualityResult(
                passed=False,
                token_exact=False,
                detail=(
                    f"generated {len(candidate)} completions against a reference of "
                    f"{len(reference)}; the prompt set changed between runs"
                ),
            )

        # strict=True is safe: the length mismatch is caught and reported above.
        pairs = zip(reference, candidate, strict=True)
        mismatched = [i for i, (ref, cand) in enumerate(pairs) if ref != cand]
        return QualityResult(
            passed=not mismatched,
            token_exact=not mismatched,
            detail=(
                f"token-exact over {len(prompts)} prompts "
                f"(greedy, seed={QUALITY_SEED}, max_tokens={QUALITY_MAX_TOKENS})"
                if not mismatched
                else f"{len(mismatched)} of {len(prompts)} completions diverged "
                f"(first at prompt index {mismatched[0]})"
            ),
        )

    def _generate_token_ids(self, handle: Handle, prompts: list[str]) -> list[list[int]]:
        """Greedy, seeded decode through an in-process engine on the handle."""
        vllm = _import_vllm()
        llm = handle.state.get("llm")
        if llm is None:
            raise AdapterError(
                "this handle has no in-process vLLM engine (handle.state['llm']), so the "
                "L3 gate has nothing to generate with; a process-launched workload must "
                "expose one for quality gating"
            )
        params = vllm.SamplingParams(
            temperature=0.0,
            top_p=1.0,
            n=1,
            seed=QUALITY_SEED,
            max_tokens=QUALITY_MAX_TOKENS,
        )
        outputs = llm.generate(prompts, params)
        return [list(output.outputs[0].token_ids) for output in outputs]


def _server_base_url(spec: WorkloadSpec) -> str | None:
    """`http://host:port` when the command launches a vLLM API server, else None."""
    joined = " ".join(spec.command)
    if not re.search(r"vllm\s+serve|api_server|openai\.api_server", joined):
        return None
    host_match = re.search(r"--host[= ]([^\s]+)", joined)
    port_match = re.search(r"--port[= ](\d+)", joined)
    host = host_match.group(1) if host_match else "localhost"
    if host in ("0.0.0.0", "::"):
        host = "localhost"
    port = port_match.group(1) if port_match else "8000"
    return f"http://{host}:{port}"
