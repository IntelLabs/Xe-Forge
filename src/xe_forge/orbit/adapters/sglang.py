"""
Tier 1: the SGLang adapter (plan §10.9).

This is the scheduled v0.2 portability test: the second Tier 1 adapter, built
deliberately before any planner work, because its cost in lines changed outside
`adapters/` is the evidence that the §10 boundary holds. If SGLang had required a
change to the core, the abstraction would have been wrong (§10.8) — it did not.

`sglang` itself is imported lazily, inside the few methods that genuinely need a live
engine, for the same reasons the vLLM adapter does it (§10.2, §10.4):

* the module imports, and most of it tests, on a machine with no SGLang and no GPU —
  which is where CI runs;
* `detect()` returns False when the *workload* shows no SGLang evidence, so the Tier 0
  adapter wins resolution instead of a Tier 1 adapter claiming a workload that is not
  its own — the same regression the vLLM adapter guards against;
* anything that needs a live engine — the §19 L3 quality gate, a radix-cache flush
  against a running server — raises `AdapterError` naming exactly what is missing.
  Never a fabricated number.

What Tier 1 buys over Tier 0: serving metrics parsed from SGLang's own benchmark
output (`sglang.bench_serving` result table, `bench_offline_throughput`, scheduler
decode logs), the determinism knobs that make an A/B mean something (§10.5) — chiefly
`--disable-radix-cache`, which pins RadixAttention's prefix reuse — a config action
space of real server arguments, framework-aware patch points (§13 rungs P1 and P3)
and the §19 L3 quality gate.

Like the vLLM adapter, it does **not** declare `can_construct_single_layer`: building
and *verifying* an in-situ SGLang layer harness needs SGLang installed, and an
unverified harness constructor mis-ranks extraction tractability (§18).

Per §10.6 the framework's knowledge lives in
`knowledge_base/common/framework_sglang.yaml`, not here. This module carries
operational defaults so it works from a wheel with no knowledge base on disk, and the
YAML may extend them (extra config axes, extra metric patterns) without a code change.
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

# §19 L3 fixes the shape of the model-level gate: greedy decode, fixed seed, at least
# this many prompts. Identical to the vLLM adapter because it is the same gate — the
# framework changes, the correctness bar does not.
MIN_QUALITY_PROMPTS = 32
QUALITY_SEED = 0
QUALITY_MAX_TOKENS = 64

# Versions whose change invalidates every stored artifact and accepted candidate
# (§12.9, §13). Reported as found; a package that is absent is simply not listed.
_TRACKED_PACKAGES = (
    "sglang",
    "sgl-kernel",
    "torch",
    "intel-extension-for-pytorch",
    "pytorch-triton-xpu",
    "triton",
)

# Whole-token markers in a command's arguments, matched exactly, never as substrings —
# the vLLM adapter learned the substring lesson on a real machine (a virtualenv named
# after the framework claimed every workload run from it). SGLang's benchmark scripts
# are `bench_serving.py` / `bench_offline_throughput.py` / `bench_one_batch.py`;
# vLLM's is `benchmark_serving.py`, so the stems do not collide.
_SGLANG_COMMAND_TOKENS = (
    "bench_serving",
    "bench_offline_throughput",
    "bench_one_batch",
)

# Executable basenames that are SGLang entrypoints in their own right.
_SGLANG_EXECUTABLES = frozenset({"sglang"})

# ---------------------------------------------------------------------------
# Metrics parsing (§10.3 `benchmark`, §10.6 "metrics parsing rules")
# ---------------------------------------------------------------------------

# Ordered preference per metric: first match wins, so the mean from SGLang's own
# result table beats a median, and a decode-side token rate beats a total token rate
# (total moves with input length, which makes it useless for an A/B on a decode
# kernel). `sglang.bench_serving` prints the same result-table shape as vLLM's
# benchmark — it is derived from it — with `Total token throughput` in lower case.
# The last pattern is the scheduler's own decode-batch log line
# (`gen throughput (token/s)`), the only rate an offline engine run emits.
_METRIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ttft_ms", re.compile(r"^\s*Mean TTFT \(ms\):\s*([0-9.]+)", re.M)),
    ("ttft_ms", re.compile(r"^\s*Median TTFT \(ms\):\s*([0-9.]+)", re.M)),
    ("tpot_ms", re.compile(r"^\s*Mean TPOT \(ms\):\s*([0-9.]+)", re.M)),
    ("tpot_ms", re.compile(r"^\s*Median TPOT \(ms\):\s*([0-9.]+)", re.M)),
    ("tpot_ms", re.compile(r"^\s*Mean ITL \(ms\):\s*([0-9.]+)", re.M)),
    ("throughput", re.compile(r"^\s*Output token throughput \(tok/s\):\s*([0-9.]+)", re.M)),
    (
        "throughput",
        re.compile(
            r"^\s*Total token throughput \(tok/s\):\s*([0-9.]+)",
            re.M,
        ),
    ),
    ("throughput", re.compile(r"gen throughput \(token/s\):\s*([0-9.]+)")),
)

# BenchRunner's field names on the left, the capability names declared in
# `FrameworkCapabilities.metrics` on the right — the same translation the vLLM
# adapter performs, so conformance rule 3 compares like with like.
_METRIC_CAPABILITY_NAMES = {
    "throughput": "throughput",
    "ttft_ms": "ttft",
    "tpot_ms": "tpot",
}


def parse_sglang_metrics(
    text: str,
    extra_patterns: tuple[tuple[str, re.Pattern[str]], ...] = (),
) -> dict[str, float]:
    """Pull serving metrics out of SGLang benchmark output.

    Returns only what was actually present. A metric that did not appear is absent
    from the result rather than zero — a zero TTFT would sail through the statistics
    and produce a confident wrong comparison.
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
# Config axes (§10.3 `config_axes`, §10.6 "config axes and their legal values")
# ---------------------------------------------------------------------------

# Every axis here is a real SGLang server argument. There is deliberately no
# environment-variable axis: SGLang steers its engine through `ServerArgs`, and
# declaring an env knob the engine does not read would be a pin that runs cleanly and
# pins nothing — this codebase's documented recurring bug class.
_CLI_AXES = (
    ConfigAxis(
        name="attention_backend",
        values=["flashinfer", "triton", "torch_native", "fa3"],
        description=(
            "Attention backend selection (--attention-backend). The legal set is "
            "version- and platform-bound (an XPU build accepts fewer); the knowledge "
            "file carries the per-version list"
        ),
    ),
    ConfigAxis(
        name="disable_radix_cache",
        values=[True, False],
        description=(
            "Turn off RadixAttention prefix/KV reuse across requests — the largest "
            "single source of run-to-run nondeterminism in a serving benchmark (§10.5)"
        ),
    ),
    ConfigAxis(
        name="max_running_requests",
        values=[1, 8, 32, 64, 128, 256],
        description=(
            "Scheduler batch width. 1 pins continuous-batching order at the cost of "
            "measuring a batch size the deployment does not run"
        ),
    ),
    ConfigAxis(
        name="chunked_prefill_size",
        values=[-1, 512, 2048, 8192],
        description=(
            "Chunked-prefill budget per batch; -1 disables chunking entirely, a fixed "
            "value pins where prefill splits instead of letting the scheduler choose"
        ),
    ),
    ConfigAxis(
        name="schedule_policy",
        values=["fcfs", "lpm", "random", "dfs-weight"],
        description=(
            "Request scheduling policy. fcfs pins queue order; lpm (the default) "
            "reorders by prefix match, which couples ordering to cache state"
        ),
    ),
    ConfigAxis(
        name="mem_fraction_static",
        values=[0.7, 0.8, 0.88],
        description=(
            "Fraction of GPU memory for weights plus KV cache. Changes how many "
            "tokens fit and therefore when retraction/recompute occurs"
        ),
    ),
    ConfigAxis(
        name="tp_size",
        values=[1, 2, 4, 8],
        description="Tensor-parallel width. Changes the kernel mix and the collectives",
    ),
    ConfigAxis(
        name="disable_cuda_graph",
        values=[True, False],
        description=(
            "Skip graph capture. Removes capture/warmup state from the measurement, "
            "and removes graph replay from the kernel mix — a different workload, "
            "honestly named"
        ),
    ),
    ConfigAxis(
        name="random_seed",
        description=(
            "Engine seed (--random-seed). SGLang draws a fresh seed per launch when "
            "it is unset, so pin it before comparing anything"
        ),
    ),
)

_CLI_FLAGS: dict[str, str] = {
    "attention_backend": "--attention-backend",
    "disable_radix_cache": "--disable-radix-cache",
    "max_running_requests": "--max-running-requests",
    "chunked_prefill_size": "--chunked-prefill-size",
    "schedule_policy": "--schedule-policy",
    "mem_fraction_static": "--mem-fraction-static",
    "tp_size": "--tp-size",
    "disable_cuda_graph": "--disable-cuda-graph",
    "random_seed": "--random-seed",
}

# SGLang's boolean server args are argparse store_true flags: present means on,
# absent means the default. There is no `--no-` form, unlike vLLM — representing
# these as negatable would emit flags the server rejects.
_PRESENCE_AXES = frozenset({"disable_radix_cache", "disable_cuda_graph"})

# ---------------------------------------------------------------------------
# Determinism (§10.5)
# ---------------------------------------------------------------------------

# The source names are shared with Tier 0 and the vLLM adapter, so a report can
# compare what each adapter pins against the same vocabulary.
_PINNABLE_KNOBS: dict[str, str] = {
    "prefix_cache_reuse": (
        "--disable-radix-cache turns off RadixAttention reuse; POST /flush_cache "
        "resets the tree between runs on a live server"
    ),
    "continuous_batching_order": (
        "--max-running-requests 1 serialises the scheduler; --schedule-policy fcfs "
        "pins the queue order (the default, lpm, reorders by prefix match)"
    ),
    "chunked_prefill_boundaries": (
        "--chunked-prefill-size -1 disables chunking; a fixed value pins the split "
        "point instead of letting the scheduler choose per step"
    ),
    "graph_capture_warmup": "--disable-cuda-graph skips capture, or discard the first N runs",
}

_NON_PINNABLE_KNOBS: dict[str, str] = {
    "speculative_decoding": (
        "the accept rate is a property of draft/target agreement at run time "
        "(EAGLE and friends). It can be removed (no speculative config) but not "
        "pinned, and removing it measures a different workload"
    ),
    "request_arrival_jitter": (
        "arrival times come from the load generator, not the engine. A fixed "
        "--request-rate narrows the distribution; the OS scheduler still moves the "
        "batch boundaries"
    ),
}

# Command-line evidence that a non-pinnable source is actually active in this run,
# so §17 can refuse ACCEPT and name the reason (§10.5).
_ACTIVE_MARKERS: dict[str, tuple[str, ...]] = {
    "speculative_decoding": (
        "--speculative-algorithm",
        "--speculative-draft-model-path",
        "--speculative-num-draft-tokens",
        "--speculative-num-steps",
    ),
    "request_arrival_jitter": ("--request-rate", "--num-prompts", "bench_serving"),
}

# ---------------------------------------------------------------------------
# Patch points (§13)
# ---------------------------------------------------------------------------

_ATTENTION_HINTS = (
    "radix_attention",
    "decode_attention",
    "extend_attention",
    "flashinfer",
    "flash_attn",
    "reshape_and_cache",
    "attn",
)
_MOE_HINTS = ("fused_moe", "moe_align", "topk_softmax", "moe_sum", "grouped_gemm", "ep_moe")


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
# Knowledge file (§10.6)
# ---------------------------------------------------------------------------

KNOWLEDGE_FILENAME = "framework_sglang.yaml"


def knowledge_path() -> Path | None:
    """Locate `common/framework_sglang.yaml`, or None.

    The loader is DSL/device-scoped — `common/` then `<dsl>/common/` then
    `<dsl>/<device>/` — so the file lives under `common/`; a flat file at the
    knowledge-base root is silently ignored (§10.6).
    """
    roots: list[Path] = []
    for env_var in ("XE_ORBIT_KNOWLEDGE_DIR", "KNOWLEDGE_DIR"):
        value = os.environ.get(env_var)
        if value:
            roots.append(Path(value))
    roots.append(Path.cwd() / "knowledge_base")
    # src/xe_forge/orbit/adapters/sglang.py -> repository root
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
        logger.warning("could not read SGLang knowledge file %s: %s", target, exc)
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
# Lazy SGLang access
# ---------------------------------------------------------------------------


def sglang_available() -> bool:
    """True when `import sglang` would find something. Does not import it."""
    try:
        return importlib.util.find_spec("sglang") is not None
    except (ImportError, ValueError):  # a broken or partially removed install
        return False


def _import_sglang() -> Any:
    try:
        import sglang
    except Exception as exc:  # ImportError, or a native-extension load failure
        raise AdapterError(
            f"sglang is not importable, so this operation cannot be performed: {exc}"
        ) from exc
    return sglang


def _mentions_sglang(command: list[str]) -> bool:
    """True when the *command* invokes SGLang, not merely when a path contains it.

    Same matching discipline as the vLLM adapter, for the same reason: substring
    matching over the joined command made an interpreter path inside a framework-named
    virtualenv count as workload evidence. The executable is matched on its basename
    and the remaining arguments as whole dotted tokens, which is what distinguishes
    `python -m sglang.launch_server` and `python -m sglang.bench_serving` from a
    directory that merely has "sglang" in its name.
    """
    if not command:
        return False

    executable = PurePosixPath(command[0]).name.lower()
    if executable in _SGLANG_EXECUTABLES:
        return True

    for argument in command[1:]:
        lowered = argument.lower()
        # `-m sglang.launch_server`, `-m sglang.bench_serving` and bare `sglang`,
        # but not a path component that merely contains the substring.
        head = lowered.split(".", 1)[0]
        if head == "sglang" and "/" not in lowered:
            return True
        # A script argument is matched on its filename stem, so `bench_serving.py`
        # and `/opt/sglang/python/sglang/bench_serving.py` both count while a
        # directory named after SGLang on the way to some other script does not.
        if PurePosixPath(lowered).stem in _SGLANG_COMMAND_TOKENS:
            return True
    return False


def _apply_cli_flag(command: list[str], flag: str, value: Any, presence: bool) -> list[str]:
    """Set one server argument on the workload command, replacing any existing value.

    `presence` flags are argparse store_true switches (`--disable-radix-cache`):
    True appends the bare flag, False removes it — there is no `--no-` spelling to
    emit, and emitting one would break the command.
    """
    out: list[str] = []
    skip_next = False
    for token in command:
        if skip_next:
            skip_next = False
            continue
        if token == flag:
            skip_next = not presence
            continue
        if token.startswith(f"{flag}="):
            continue
        out.append(token)

    if presence:
        if value:
            out.append(flag)
    else:
        out.extend([flag, str(value)])
    return out


class SGLangAdapter(BaseAdapter):
    """Tier 1 SGLang adapter. Declares only what it can deliver without guessing."""

    name = "sglang"
    tier = 1
    capabilities = FrameworkCapabilities(
        # Parsed from SGLang's own benchmark output; when they do not parse,
        # `benchmark()` reports wall_time alone and records why (§10.4).
        metrics={"wall_time", "throughput", "ttft", "tpot"},
        # The radix tree is flushed by POST /flush_cache against a live server, by
        # Engine.flush_cache() in-process, or — for the process-launched workloads
        # Orbit actually runs — by the next repetition being a fresh process.
        can_reset_state=True,
        # --max-running-requests 1 plus --schedule-policy fcfs.
        can_pin_batching=True,
        # SGLang's scheduler runs in its own processes (launch_server and Engine both
        # spawn them), so an in-process torch.profiler in the caller observes zero
        # device kernels — the same measured fact as vLLM v1. Device attribution must
        # go through SGLang's own hook.
        profiles_in_process=False,
        profile_hook=(
            "SGLANG_TORCH_PROFILER_DIR + POST /start_profile / /stop_profile "
            "(Engine.start_profile()/stop_profile() in-process)"
        ),
        # --disable-radix-cache.
        can_disable_prefix_cache=True,
        # Not declared: constructing an SGLang attention/MoE layer in-process and
        # proving it reproduces the reference output cannot be verified without
        # SGLang installed; an unverified harness constructor mis-ranks extraction
        # tractability (§18). Over-declaring is worse than missing (§10.4).
        can_construct_single_layer=False,
        patchable_layers={"attention_backend", "fused_moe"},
    )

    def __init__(self, executor: Executor | None = None) -> None:
        self.executor = executor or LocalExecutor()
        self.knowledge = load_knowledge()
        self._extra_patterns = _knowledge_metric_patterns(self.knowledge)

    # --- identity and lifecycle ------------------------------------------

    def detect(self, spec: WorkloadSpec) -> bool:
        """Claim only workloads with positive evidence of being SGLang ones.

        The evidence rule is the one the vLLM adapter converged on after a real
        install exposed the alternative: an importable package is evidence about the
        *machine*, and `resolve_adapter` prefers the highest tier, so installability
        as evidence makes a Tier 1 adapter claim every workload on the box — which
        hands each of them declared TTFT/TPOT and a determinism profile describing
        radix caching they do not have. Evidence must be about the workload: the
        command names SGLang, the environment carries SGLang's own configuration, or
        the caller declared the framework. Anything else correctly degrades to
        Tier 0 (§10.2).

        A script that uses SGLang without saying so is reached with
        `--framework sglang` — one explicit word from the caller, instead of silently
        wrong capabilities for everyone else.
        """
        declared = (spec.framework or "").strip().lower()
        if declared:
            return declared == self.name
        if _mentions_sglang(spec.command):
            return True
        # SGLang's own environment variables are evidence about this workload, unlike
        # the mere presence of the package on the machine.
        return any(key.startswith("SGLANG_") for key in spec.env)

    def versions(self) -> dict[str, str]:
        """SGLang, sgl-kernel, torch and IPEX, as installed. Absent ones not listed."""
        from xe_forge.orbit.runtime import environment

        return environment.package_versions(_TRACKED_PACKAGES)

    def prepare(self, spec: WorkloadSpec) -> PreparedWorkload:
        notes: list[str] = []
        if not sglang_available():
            notes.append(
                "sglang is not importable here: serving metrics will not parse, the "
                "§19 L3 quality gate is unavailable, and this run degrades to "
                "wall-clock only."
            )
        if _mentions_sglang(spec.command) and not any(
            token == "--random-seed" or token.startswith("--random-seed=") for token in spec.command
        ):
            notes.append(
                "--random-seed is unpinned. SGLang draws a fresh seed per launch when "
                "it is unset; pin it before comparing anything (§10.5)."
            )
        profile = self.determinism_profile(spec)
        if profile.active_non_pinnable:
            notes.append(
                "active non-pinnable nondeterminism: "
                + ", ".join(sorted(profile.active_non_pinnable))
                + " — §17 must emit INCONCLUSIVE rather than ACCEPT if variance "
                "exceeds the MDE."
            )
        notes.append(
            "determinism pins available: --disable-radix-cache, "
            "--max-running-requests 1, --schedule-policy fcfs, "
            "--chunked-prefill-size -1, --disable-cuda-graph."
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
        """Serving metrics from one run. SGLang logs to both streams, so read both."""
        return parse_sglang_metrics(
            f"{result.stdout}\n{result.stderr}", extra_patterns=self._extra_patterns
        )

    def benchmark(self, handle: Handle, load: LoadSpec) -> WorkloadMeasurement:
        """Measure, reporting exactly the metrics that were actually parsed.

        A serving metric that did not appear in the output is absent, not zero, and
        the fallback to wall-clock is recorded on the handle so the report can say so
        instead of leaving a reader to assume TTFT was measured (§10.4).
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
                f"SGLang serving metrics {missing} did not appear in the workload "
                f"output; reporting {available} only. Run `python -m "
                "sglang.bench_serving` / `sglang.bench_offline_throughput`, or add a "
                "parsing rule to the knowledge file."
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
        """Which of §10.5's sources SGLang can pin, and which it cannot.

        Passing a spec is optional and additive: it fills `active_non_pinnable` from
        command-line evidence, which is what lets §17 refuse ACCEPT and name the
        reason rather than returning a number with a caveat nobody reads.
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
        """Drop radix-cache and scheduler state between measurements.

        Three mechanisms, in decreasing directness. The last one is not a no-op
        dressed up as a reset: Orbit runs each repetition as its own process, so no
        radix tree, captured graph or scheduler queue survives into the next one.
        """
        super().reset_state(handle)

        base_url = handle.state.get("base_url")
        if base_url:
            self._post(f"{base_url}/flush_cache")
            handle.state["state_reset"] = f"POST {base_url}/flush_cache"
            return

        engine = handle.state.get("engine")
        if engine is not None:
            flush = getattr(engine, "flush_cache", None)
            if flush is None:
                raise AdapterError(
                    "the in-process SGLang engine on this handle exposes no "
                    "flush_cache(); refusing to claim state was reset"
                )
            flush()
            handle.state["state_reset"] = "Engine.flush_cache()"
            return

        handle.state["state_reset"] = (
            "process-per-repetition: the next run starts a fresh engine, so the radix "
            "tree, captured graphs and scheduler queues do not carry over"
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
            raise AdapterError(f"could not reach the SGLang server at {url}: {exc}") from exc

    # --- discovery and provenance ----------------------------------------

    def dispatch_roots(self) -> list[str]:
        """Where an SGLang kernel is reached from. `sgl_kernel` is its native lib."""
        return [
            "aten",
            "torch.ops",
            "torch.ops.sgl_kernel",
            "sgl_kernel",
            "sglang.srt.layers",
        ]

    def provenance_hints(self) -> list[str]:
        return [
            "sglang",
            "sgl_kernel",
            "radix_attention",
            "decode_attention",
            "extend_attention",
            "fused_moe",
            "flashinfer",
            "inductor",
            "triton",
            "onednn",
            "onemkl",
            "sycl",
            "ipex",
        ]

    # --- extraction (§13) -------------------------------------------------

    def patch_points(self, kernel: KernelRecord) -> list[PatchPoint]:
        """Where an optimized kernel can be reinserted, highest rung first (§13).

        P1 first because it touches nothing in the framework and reverts by not
        importing a module; P3 only for the layers this adapter actually declares in
        `patchable_layers`, so patch_points can never name a substitution point the
        capabilities deny. SGLang has no CustomOp registry like vLLM's — norms and
        activations dispatch straight through `torch.ops.sgl_kernel.*`, which is
        exactly the P1 case.
        """
        points: list[PatchPoint] = []

        if kernel.framework_op:
            points.append(
                PatchPoint(
                    rung="P1",
                    target=kernel.framework_op,
                    mechanism="torch.library operator override on the XPU dispatch key",
                    description=(
                        "sgl-kernel registers its native ops through TORCH_LIBRARY "
                        "(torch.ops.sgl_kernel), so an optimized kernel ships as a "
                        "small out-of-tree extension that shadows the default. SGLang "
                        "is untouched; revert is not importing the module. §13 "
                        "requires the dispatch assertion afterwards: the new kernel "
                        "must appear in the re-profile and the old one must not."
                    ),
                )
            )

        if _mentions(kernel, _ATTENTION_HINTS):
            points.append(
                PatchPoint(
                    rung="P3",
                    target="attention_backend",
                    mechanism=(
                        "register an attention backend in sglang.srt.layers.attention "
                        "and select it with --attention-backend"
                    ),
                    description=(
                        "Framework registry substitution: SGLang chooses the "
                        "attention implementation through its own selection point, so "
                        "the backend can be replaced by config rather than by "
                        "patching source. Revert is restoring the server argument."
                    ),
                )
            )

        if _mentions(kernel, _MOE_HINTS):
            points.append(
                PatchPoint(
                    rung="P3",
                    target="fused_moe",
                    mechanism=(
                        "substitute the fused MoE implementation in "
                        "sglang.srt.layers.moe, or override the sgl_kernel op it "
                        "dispatches to"
                    ),
                    description=(
                        "MoE runs through a selectable implementation plus a tuned-"
                        "config JSON. The config is data and must be copied, never "
                        "regenerated (§12.8); a mismatched config silently benchmarks "
                        "a different tile."
                    ),
                )
            )

        return points

    # --- action space -----------------------------------------------------

    def config_axes(self) -> list[ConfigAxis]:
        """Declared axes, extended (never overridden) by the knowledge file (§10.6)."""
        axes = list(_CLI_AXES)
        known = {axis.name for axis in axes}
        axes.extend(axis for axis in _knowledge_axes(self.knowledge) if axis.name not in known)
        return axes

    def apply_config(self, spec: WorkloadSpec, config: dict[str, Any]) -> WorkloadSpec:
        """Apply a config point to a workload, as server arguments.

        Refuses rather than drops. A determinism pin that was requested and silently
        not applied produces a measurement that looks controlled and is not — which
        is exactly the failure §10.5 exists to prevent.
        """
        axes = {axis.name: axis for axis in self.config_axes()}
        env = dict(spec.env)
        command = list(spec.command)

        for name, value in config.items():
            axis = axes.get(name)
            if axis is None:
                raise AdapterError(
                    f"unknown SGLang config axis {name!r}; declared axes: {sorted(axes)}"
                )
            if axis.env_var:  # knowledge-file additions may be env-steered
                env[axis.env_var] = str(value)
                continue

            flag = _CLI_FLAGS.get(name)
            if flag is None:
                raise AdapterError(
                    f"config axis {name!r} declares neither an environment variable "
                    "nor a CLI flag, so it cannot be applied"
                )
            if not _mentions_sglang(command):
                raise AdapterError(
                    f"{flag} is an SGLang server argument but the workload command is "
                    f"not an SGLang entrypoint ({command[0]!r}); appending it would "
                    "break the command and dropping it would leave the run unpinned"
                )
            command = _apply_cli_flag(command, flag, value, presence=name in _PRESENCE_AXES)

        return spec.model_copy(update={"env": env, "command": command})

    # --- correctness (§19 L3) --------------------------------------------

    def capture_quality_reference(self, handle: Handle, prompts: list[str]) -> list[list[int]]:
        """Record the baseline token ids the L3 gate will compare against."""
        reference = self._generate_token_ids(handle, prompts)
        handle.state["quality_reference"] = reference
        return reference

    def quality_gate(self, handle: Handle, prompts: list[str]) -> QualityResult:
        """§19 L3: greedy decode, fixed seed, >= 32 prompts, token-exact comparison.

        Raises rather than returning a failure when the gate cannot be *run*: an
        unavailable gate and a failed gate are different facts, and collapsing them
        into `passed=False` is how a candidate gets rejected for the wrong reason —
        or, worse, how an unrun gate gets read as a pass elsewhere.
        """
        if not sglang_available():
            raise AdapterError(
                "sglang is not importable, so the §19 L3 model-level gate cannot be "
                "run; refusing to report a quality result that was never measured"
            )
        if len(prompts) < MIN_QUALITY_PROMPTS:
            return QualityResult(
                passed=False,
                detail=(
                    f"§19 L3 requires at least {MIN_QUALITY_PROMPTS} prompts; got "
                    f"{len(prompts)}. Fewer prompts is a different gate, not a weaker one"
                ),
            )
        reference = handle.state.get("quality_reference")
        if reference is None:
            raise AdapterError(
                "no baseline token reference on this handle: call "
                "capture_quality_reference() against the unmodified workload first. A "
                "gate with nothing to compare against can neither pass nor fail"
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
                f"(greedy, seed={QUALITY_SEED}, max_new_tokens={QUALITY_MAX_TOKENS})"
                if not mismatched
                else f"{len(mismatched)} of {len(prompts)} completions diverged "
                f"(first at prompt index {mismatched[0]})"
            ),
        )

    def _generate_token_ids(self, handle: Handle, prompts: list[str]) -> list[list[int]]:
        """Greedy decode through an in-process engine on the handle.

        The engine seed is pinned at launch (`--random-seed`, noted by `prepare()`);
        greedy decode makes per-request sampling seed moot.
        """
        _import_sglang()  # raises a clean AdapterError when the package is missing
        engine = handle.state.get("engine")
        if engine is None:
            raise AdapterError(
                "this handle has no in-process SGLang engine (handle.state['engine']), "
                "so the L3 gate has nothing to generate with; a process-launched "
                "workload must expose one for quality gating"
            )
        params = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": QUALITY_MAX_TOKENS,
        }
        outputs = engine.generate(prompts, params)
        return [_completion_token_ids(output) for output in outputs]


def _completion_token_ids(output: Any) -> list[int]:
    """Token ids from one SGLang completion, or a refusal — never a text fallback.

    SGLang's generate output carries `output_ids` directly on recent versions, and
    token ids inside `meta_info.output_token_logprobs` when logprobs were requested.
    Comparing decoded *text* instead would pass a gate that token comparison fails
    (retokenization can round-trip differing ids to identical strings), so when
    neither field is present the gate refuses rather than degrades (§19).
    """
    if isinstance(output, dict):
        ids = output.get("output_ids")
        if ids is not None:
            return [int(i) for i in ids]
        meta = output.get("meta_info") or {}
        logprobs = meta.get("output_token_logprobs")
        if logprobs:
            # Each entry is (logprob, token_id, text).
            return [int(entry[1]) for entry in logprobs]
    raise AdapterError(
        "SGLang generate output carries no token ids (neither 'output_ids' nor "
        "meta_info['output_token_logprobs']); run the engine with return_logprob=True "
        "or a version that reports output_ids. The §19 gate compares token ids, "
        "never decoded text"
    )


def _server_base_url(spec: WorkloadSpec) -> str | None:
    """`http://host:port` when the command launches an SGLang server, else None."""
    joined = " ".join(spec.command)
    if not re.search(r"sglang\.launch_server|sglang\s+serve|sglang_router", joined):
        return None
    host_match = re.search(r"--host[= ]([^\s]+)", joined)
    port_match = re.search(r"--port[= ](\d+)", joined)
    host = host_match.group(1) if host_match else "localhost"
    if host in ("0.0.0.0", "::"):
        host = "localhost"
    port = port_match.group(1) if port_match else "30000"
    return f"http://{host}:{port}"
