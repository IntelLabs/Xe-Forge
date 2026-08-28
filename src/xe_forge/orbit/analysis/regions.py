"""
Region detection: the multi-kernel optimization unit (plan §7.3, §12.11, §18).

**The optimization unit is not always a kernel.** The largest wins in inference come
from *eliminating* kernels, not from speeding them up: three kernels that write an
intermediate tensor to HBM and read it straight back become one sycl-tla kernel with a
fused epilogue, and the memory traffic between them stops existing. A `KernelRecord`-only
model cannot express a many-to-one replacement, which is exactly why v1's fusion action
had no executor behind it (§7.3).

How a region is found
---------------------
1. **Temporal adjacency bounds the search.** Kernels are ordered per stream. A consumer
   is only considered within `MAX_LOOKAHEAD` positions of its producer, and only when
   the *device-idle* time between them is small — idle meaning the gap minus whatever
   else ran on that stream in between. An idle gap means the host got involved, and two
   kernels separated by host work are not one region.

2. **A tensor link is what actually creates the edge.** Where the trace records input
   shapes, an edge exists when one of the consumer's inputs matches the producer's
   inferred output. This is the part that keeps the chains honest: in the reference
   decode trace an attention kernel runs *between* the RMSNorm and the SwiGLU, and it
   shares no tensor with either — a different data path that happened to be interleaved.
   Temporal adjacency alone would swallow it; the shape link steps over it, and leaves
   attention as its own single-kernel chain, which is not a region.

   When shapes are absent for either kernel (a trace taken without `record_shapes`) the
   fallback is strict immediate adjacency, and no intermediate tensor is claimed for
   that edge. Nothing is invented from a shape we do not have.

3. **Correlation ids order the launches.** Where both kernels carry one, the consumer's
   host op must have been issued after the producer's; a kernel launched first cannot
   consume a later kernel's output.

4. **A pattern names the region.** `FUSION_PATTERNS` is module-level data, matched
   against the chain's op classes longest-first. Adding a fusable shape is appending a
   row, not editing the algorithm.

Each edge is one-in/one-out, which keeps regions linear. A true fusion DAG is out of
scope for v0.1 — the presets Xe-Fuse actually ships are chains (§9.6).

What survives
-------------
A region must be worth more than its parts, so it is emitted only when it holds at
least two *distinct* kernels, clears `MIN_REGION_SHARE` of GPU time, and beats its own
largest member by `MIN_REGION_UPLIFT`. A "region" that is one dominant kernel with a
rounding error attached is a kernel, and the catalog already ranks it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from xe_forge.orbit.analysis.xe_fuse import REGION_EXTRACTION_LEVEL, xe_fuse_available
from xe_forge.orbit.models import ActionType, KernelRecord, RegionRecord, TensorInfo
from xe_forge.orbit.profiling.trace import RuntimeKernelEvent, TraceEvents

# How far ahead on a stream a consumer may sit. Three covers one interleaved side
# branch (the attention path in a decode step) without letting a chain leap across an
# entire layer.
MAX_LOOKAHEAD = 3

# Device-idle microseconds tolerated between producer and consumer. Idle is the gap
# minus the work that ran in between, so an interleaved kernel does not break a chain
# but a host round-trip does. A quarter of a millisecond of a *silent* device between
# two kernels means the framework went away and came back; they are not one region.
MAX_IDLE_GAP_US = 250.0

# A region below this share of GPU time cannot repay the cost of a fused rewrite.
MIN_REGION_SHARE = 0.05

# ...and it must beat its own largest member by this factor, or it is that member.
MIN_REGION_UPLIFT = 1.10

# An adjacent chain with no recognized pattern is still a finding, but only when it is
# large enough that "we have no name for this" is the headline rather than noise.
MIN_UNNAMED_REGION_SHARE = 0.20


@dataclass(frozen=True)
class FusionPattern:
    """One fusable shape, expressed as a sequence of acceptable op classes.

    `steps[i]` is the set of op classes the i-th kernel of the chain may have, so a
    pattern can accept `rmsnorm` or `layernorm` in the same slot without duplicating the
    row. Data, not code: extending the table is how new fusions are supported.
    """

    name: str
    steps: tuple[tuple[str, ...], ...]
    actions: tuple[ActionType, ...] = (ActionType.REGION_FUSION,)
    note: str = ""


# The pattern table. Order is irrelevant — matching is longest-first — but keeping the
# longest rows at the top makes the table readable as a specificity ladder.
FUSION_PATTERNS: tuple[FusionPattern, ...] = (
    FusionPattern(
        name="gemm+rmsnorm+swiglu",
        steps=(("gemm",), ("rmsnorm", "layernorm"), ("swiglu",)),
        actions=(ActionType.REGION_FUSION, ActionType.LAYOUT_CHANGE),
        note="the transformer MLP block: one sycl-tla GEMM with a normalize+gate epilogue",
    ),
    FusionPattern(
        name="gemm+bias+activation",
        steps=(("gemm",), ("bias",), ("activation", "swiglu")),
        actions=(ActionType.REGION_FUSION, ActionType.LAYOUT_CHANGE),
        note="classic GEMM epilogue fusion; the bias and activation never reach HBM",
    ),
    FusionPattern(
        name="gemm+rmsnorm",
        steps=(("gemm",), ("rmsnorm", "layernorm")),
        note="normalization folded into the producing GEMM's epilogue",
    ),
    FusionPattern(
        name="attention+softmax",
        steps=(("attention",), ("softmax",)),
        note="an unfused attention that still materializes its scores matrix",
    ),
    FusionPattern(
        name="gemm+activation",
        steps=(("gemm",), ("activation", "swiglu")),
        note="the minimal epilogue fusion, worth naming so it is not left unmatched",
    ),
)


# Op-class rules, most specific first. The text matched is the runtime kernel name plus
# the aten op it was attributed to, so `triton_poi_fused_silu_mul_1` and
# `aten::silu` both land on the same class.
_OP_CLASS_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("memcpy", re.compile(r"(memcpy|memset|copy_)", re.I)),
    ("swiglu", re.compile(r"(swiglu|geglu|silu\w*_?mul|mul\w*_?silu|gelu\w*_?mul)", re.I)),
    ("rmsnorm", re.compile(r"rms_?norm", re.I)),
    ("layernorm", re.compile(r"(layer_?norm|group_?norm)", re.I)),
    ("softmax", re.compile(r"soft_?max", re.I)),
    ("attention", re.compile(r"(attention|attn|flash|paged)", re.I)),
    ("gemm", re.compile(r"(gemm|matmul|addmm|bmm|linear|\bmm\b|\bdot\b)", re.I)),
    ("bias", re.compile(r"(add_?bias|bias_?add|_bias\b)", re.I)),
    ("activation", re.compile(r"(silu|gelu|relu|swish|sigmoid|tanh)", re.I)),
    ("elementwise", re.compile(r"(elementwise|poi_fused|_add|_mul|_div|_sub|scale)", re.I)),
)

# Bytes per element, for sizing the intermediate tensors fusion would eliminate. An
# unrecognized dtype yields zero bytes rather than a guessed width — the tensor is still
# reported, its size just is not claimed.
_DTYPE_BYTES: dict[str, int] = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "float8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "bfloat16": 2,
    "bf16": 2,
    "float16": 2,
    "half": 2,
    "fp16": 2,
    "int16": 2,
    "float": 4,
    "float32": 4,
    "fp32": 4,
    "f32": 4,
    "int32": 4,
    "int": 4,
    "float64": 8,
    "double": 8,
    "int64": 8,
    "long": 8,
}


@dataclass
class _Occurrence:
    """One matched span of one chain: a region as it appeared once in the trace."""

    kernel_ids: tuple[str, ...]
    pattern: FusionPattern | None
    edges: list[tuple[str, str]] = field(default_factory=list)
    tensors: list[TensorInfo] = field(default_factory=list)
    aten_ops: list[str] = field(default_factory=list)
    time_us: float = 0.0
    # Time each member contributed *within this region*. The uplift test needs this
    # rather than the member's run-wide share: a kernel used both inside and outside a
    # region has a global share that the region can never exceed, which made every
    # region containing a hot kernel structurally unreachable.
    member_time_us: dict[str, float] = field(default_factory=dict)


def detect_regions(
    events: TraceEvents,
    kernels: Sequence[KernelRecord],
    *,
    total_gpu_time_us: float | None = None,
    patterns: Sequence[FusionPattern] = FUSION_PATTERNS,
    max_lookahead: int = MAX_LOOKAHEAD,
    max_idle_gap_us: float = MAX_IDLE_GAP_US,
    min_share: float = MIN_REGION_SHARE,
) -> list[RegionRecord]:
    """Find fusable producer-consumer regions in a trace (§7.3, §12.11).

    `kernels` is the catalog's records, which supply the `k<n>` ids a `RegionRecord`
    refers to and the operator attribution used for op classification. Events whose
    kernel is not in the catalog are ignored — a region cannot cite an id that does not
    exist. Returns regions ranked by combined GPU-time share, ids assigned `r0`, `r1`, …
    in that order.

    Total ordering, no LLM, no randomness: two runs over the same trace produce the same
    regions (§3).
    """
    if not events.kernels or not kernels:
        return []

    total_us = total_gpu_time_us if total_gpu_time_us is not None else events.total_gpu_time_us
    if total_us <= 0:
        return []

    by_name = {record.runtime_name: record for record in kernels}
    host_ops = {
        op.correlation_id: op.name for op in events.host_ops if op.correlation_id is not None
    }

    # Only events whose kernel the catalog knows can take part in a region.
    usable = [event for event in events.kernels if event.name in by_name]
    if len(usable) < 2:
        return []

    classes = {id(event): _op_class(event, by_name[event.name], host_ops) for event in usable}

    occurrences: list[_Occurrence] = []
    for stream_events in _by_stream(usable):
        for chain in _chains(stream_events, classes, max_lookahead, max_idle_gap_us):
            if len(chain) < 2:
                continue
            occurrences.extend(_occurrences_for_chain(chain, classes, by_name, host_ops, patterns))

    member_shares = {record.id: record.gpu_time_share for record in kernels}
    return _merge_and_rank(occurrences, total_us, min_share, member_shares)


# ---------------------------------------------------------------------------
# Chain construction
# ---------------------------------------------------------------------------


def _by_stream(events: Sequence[RuntimeKernelEvent]) -> list[list[RuntimeKernelEvent]]:
    """Group events by stream and order each group by start time.

    Streams are independent execution queues, so a producer-consumer chain cannot span
    two of them without an explicit sync — which the trace does not record, so we do not
    infer one. Missing stream ids collapse to a single implicit stream.
    """
    grouped: dict[str, list[RuntimeKernelEvent]] = {}
    for event in events:
        grouped.setdefault(event.stream or "?", []).append(event)
    ordered = []
    for key in sorted(grouped):
        ordered.append(sorted(grouped[key], key=lambda e: (e.start_us, e.name)))
    return ordered


def _chains(
    events: Sequence[RuntimeKernelEvent],
    classes: dict[int, str],
    max_lookahead: int,
    max_idle_gap_us: float,
) -> list[list[RuntimeKernelEvent]]:
    """Link one stream's events into linear producer-consumer chains."""
    count = len(events)
    successor: dict[int, int] = {}
    consumed: set[int] = set()

    for i in range(count):
        for j in range(i + 1, min(i + 1 + max_lookahead, count)):
            if j in consumed:
                continue
            if _edge(events, classes, i, j, max_idle_gap_us)[0]:
                successor[i] = j
                consumed.add(j)
                break

    chains: list[list[RuntimeKernelEvent]] = []
    for i in range(count):
        if i in consumed:
            continue
        indices = [i]
        node = i
        while node in successor:
            node = successor[node]
            indices.append(node)
        chains.append([events[index] for index in indices])
    return chains


def _edge(
    events: Sequence[RuntimeKernelEvent],
    classes: dict[int, str],
    i: int,
    j: int,
    max_idle_gap_us: float,
) -> tuple[bool, TensorInfo | None]:
    """Is `events[j]` a consumer of `events[i]`, and if so, over which tensor?"""
    producer, consumer = events[i], events[j]

    # The device must not have gone quiet in between: gap minus interleaved work.
    idle = (consumer.start_us - producer.end_us) - sum(
        events[m].duration_us for m in range(i + 1, j)
    )
    if idle > max_idle_gap_us:
        return False, None

    # A kernel launched before the producer cannot consume its output.
    if (
        producer.correlation_id is not None
        and consumer.correlation_id is not None
        and consumer.correlation_id < producer.correlation_id
    ):
        return False, None

    link = _shape_link(producer, classes.get(id(producer), "unknown"), consumer)
    if link is not None:
        return True, link

    # Shapes present on both sides and no match: this is a genuine negative, the two
    # kernels are on different data paths.
    if _input_dims(producer) and _input_dims(consumer):
        return False, None

    # Shapes unavailable: fall back to strict immediate adjacency and claim no tensor.
    return (j == i + 1), None


def _shape_link(
    producer: RuntimeKernelEvent, producer_class: str, consumer: RuntimeKernelEvent
) -> TensorInfo | None:
    """The tensor the producer writes and the consumer reads, if the trace shows one."""
    output = _output_shape(producer, producer_class)
    if output is None:
        return None
    dims = _input_dims(consumer)
    dtypes = _input_dtypes(consumer)
    for position, candidate in enumerate(dims):
        if candidate == output:
            dtype = dtypes[position] if position < len(dtypes) else ""
            return TensorInfo(
                name=f"{producer.name}->{consumer.name}",
                shape=list(candidate),
                dtype=dtype,
                bytes=_tensor_bytes(candidate, dtype),
            )
    return None


def _output_shape(event: RuntimeKernelEvent, op_class: str) -> list[int] | None:
    """Infer a kernel's output shape from its recorded input shapes.

    Two rules, both conservative: a GEMM of `[..., M, K] x [K, N]` writes `[..., M, N]`,
    and everything else is treated as shape-preserving in its first input. A kernel with
    no recorded shapes has no inferable output, and returns None rather than a guess.
    """
    dims = _input_dims(event)
    if not dims:
        return None
    if op_class == "gemm" and len(dims) >= 2 and len(dims[0]) >= 2 and len(dims[1]) >= 2:
        return [*dims[0][:-1], dims[1][-1]]
    return list(dims[0])


# ---------------------------------------------------------------------------
# Pattern matching and region assembly
# ---------------------------------------------------------------------------


def _occurrences_for_chain(
    chain: Sequence[RuntimeKernelEvent],
    classes: dict[int, str],
    by_name: dict[str, KernelRecord],
    host_ops: dict[int, str],
    patterns: Sequence[FusionPattern],
) -> list[_Occurrence]:
    """Carve one chain into the named regions it contains."""
    events = list(chain)
    class_list = [classes.get(id(event), "unknown") for event in events]
    spans = _match_patterns(class_list, patterns)

    if not spans:
        return [_build_occurrence(events, None, by_name, host_ops, class_list)]

    occurrences = []
    for start, stop, pattern in spans:
        occurrences.append(
            _build_occurrence(
                events[start:stop], pattern, by_name, host_ops, class_list[start:stop]
            )
        )
    return occurrences


def _match_patterns(
    class_list: Sequence[str], patterns: Sequence[FusionPattern]
) -> list[tuple[int, int, FusionPattern]]:
    """Non-overlapping longest-first matches of the pattern table over a chain."""
    ordered = sorted(patterns, key=lambda p: len(p.steps), reverse=True)
    spans: list[tuple[int, int, FusionPattern]] = []
    index = 0
    while index < len(class_list):
        for pattern in ordered:
            length = len(pattern.steps)
            if index + length > len(class_list):
                continue
            if all(class_list[index + step] in pattern.steps[step] for step in range(length)):
                spans.append((index, index + length, pattern))
                index += length
                break
        else:
            index += 1
    return spans


def _build_occurrence(
    events: Sequence[RuntimeKernelEvent],
    pattern: FusionPattern | None,
    by_name: dict[str, KernelRecord],
    host_ops: dict[int, str],
    class_list: Sequence[str],
) -> _Occurrence:
    kernel_ids = tuple(by_name[event.name].id for event in events)

    edges: list[tuple[str, str]] = []
    tensors: list[TensorInfo] = []
    for position in range(len(events) - 1):
        producer, consumer = events[position], events[position + 1]
        producer_id = by_name[producer.name].id
        consumer_id = by_name[consumer.name].id
        edges.append((producer_id, consumer_id))
        link = _shape_link(producer, class_list[position], consumer)
        if link is not None:
            tensors.append(
                TensorInfo(
                    name=f"{producer_id}->{consumer_id}",
                    shape=link.shape,
                    dtype=link.dtype,
                    bytes=link.bytes,
                )
            )

    aten_ops: list[str] = []
    for event in events:
        op = host_ops.get(event.correlation_id) if event.correlation_id is not None else None
        if op is None:
            op = by_name[event.name].framework_op
        if op and op not in aten_ops:
            aten_ops.append(op)

    return _Occurrence(
        kernel_ids=kernel_ids,
        pattern=pattern,
        edges=edges,
        tensors=tensors,
        aten_ops=aten_ops,
        time_us=sum(event.duration_us for event in events),
        member_time_us=_member_times(events, kernel_ids),
    )


def _merge_and_rank(
    occurrences: Sequence[_Occurrence],
    total_gpu_time_us: float,
    min_share: float,
    member_shares: dict[str, float],
) -> list[RegionRecord]:
    """Fold repeated occurrences of the same region together and apply the thresholds.

    A decode loop runs the same chain every layer, so the same `(kernel_ids, pattern)`
    shows up many times. Summing the occurrences is what makes a region's share
    comparable with a kernel's share — both are then "fraction of measured GPU time".
    """
    merged: dict[tuple[tuple[str, ...], str], _Occurrence] = {}
    order: list[tuple[tuple[str, ...], str]] = []
    for occurrence in occurrences:
        key = (occurrence.kernel_ids, occurrence.pattern.name if occurrence.pattern else "")
        existing = merged.get(key)
        if existing is None:
            merged[key] = occurrence
            order.append(key)
        else:
            existing.time_us += occurrence.time_us
            for member, value in occurrence.member_time_us.items():
                existing.member_time_us[member] = existing.member_time_us.get(member, 0.0) + value

    candidates: list[tuple[float, _Occurrence]] = []
    for key in order:
        occurrence = merged[key]
        distinct = set(occurrence.kernel_ids)
        if len(distinct) < 2:
            # A single kernel repeating is a loop, not a many-to-one replacement.
            continue
        share = occurrence.time_us / total_gpu_time_us
        floor = min_share if occurrence.pattern else max(min_share, MIN_UNNAMED_REGION_SHARE)
        if share < floor:
            continue
        # Worth more than its parts, or it is its largest part. A "region" that adds a
        # rounding error to one dominant kernel is that kernel, and the catalog already
        # ranks it — emitting it twice would double-count the same GPU time.
        # The uplift test applies to UNNAMED chains only, and getting that wrong hid
        # the single largest fusion opportunity in the run.
        #
        # Its intent is sound for an unnamed chain: "one dominant kernel with a rounding
        # error attached is that kernel, and the catalog already ranks it". For a *named*
        # pattern the reasoning inverts. `gemm+activation` is worth fusing precisely
        # BECAUSE the GEMM dominates — the value is eliminating the intermediate write
        # between them, not the activation's own cost. Requiring the region to be 10%
        # larger than its biggest member asks the epilogue to be expensive, which is the
        # opposite of the condition that makes it fusable.
        #
        # Measured on a Qwen decode trace: `gemm+activation` held 37.62% of GPU time and
        # `gemm+rmsnorm` 25.88%, and both were dropped — the first against a run-wide
        # share it could never exceed, and after that was fixed, against a within-region
        # share it still could not exceed. A pattern in FUSION_PATTERNS is an assertion
        # that the shape is fusable; the threshold that decides whether it is worth doing
        # is `min_share`, which it has already cleared.
        if occurrence.pattern is None:
            largest_member_us = max(occurrence.member_time_us.values(), default=0.0)
            if largest_member_us > 0:
                if occurrence.time_us < largest_member_us * MIN_REGION_UPLIFT:
                    continue
            else:
                largest = max((member_shares.get(k, 0.0) for k in distinct), default=0.0)
                if largest > 0 and share < largest * MIN_REGION_UPLIFT:
                    continue
        candidates.append((share, occurrence))

    candidates.sort(key=lambda item: (-item[0], item[1].kernel_ids))

    regions: list[RegionRecord] = []
    for index, (share, occurrence) in enumerate(candidates):
        actions = list(occurrence.pattern.actions) if occurrence.pattern else []
        if ActionType.REGION_FUSION not in actions:
            actions.insert(0, ActionType.REGION_FUSION)
        regions.append(
            RegionRecord(
                id=f"r{index}",
                kernel_ids=list(occurrence.kernel_ids),
                aten_ops=list(occurrence.aten_ops),
                producer_consumer_edges=list(occurrence.edges),
                intermediate_tensors=list(occurrence.tensors),
                combined_time_us=occurrence.time_us,
                fusion_pattern=occurrence.pattern.name if occurrence.pattern else None,
                actions_available=actions,
                gpu_time_share=share,
            )
        )
    return regions


# ---------------------------------------------------------------------------
# Classification and trace-argument helpers
# ---------------------------------------------------------------------------


def _op_class(event: RuntimeKernelEvent, record: KernelRecord, host_ops: dict[int, str]) -> str:
    """Classify a kernel by name plus the aten op it was attributed to."""
    parts = [event.name, record.framework_op or "", record.source_symbol or ""]
    if event.correlation_id is not None and event.correlation_id in host_ops:
        parts.append(host_ops[event.correlation_id])
    text = " ".join(part for part in parts if part)
    for name, pattern in _OP_CLASS_RULES:
        if pattern.search(text):
            return name
    return "unknown"


def _input_dims(event: RuntimeKernelEvent) -> list[list[int]]:
    """The shapes torch records with `record_shapes=True`, normalized.

    Prefers the joined `input_shapes` field over the raw args. The profiler puts
    `Input Dims` on the *host* op, never on the device kernel, so a kernel's own args
    carry nothing — reading only args meant this returned `[]` for every kernel and no
    dataflow edge could ever form, whatever the shapes said. `trace.attach_shapes`
    performs the kernel -> runtime-event -> enclosing-cpu_op join; this reads its result.
    """
    joined = getattr(event, "input_shapes", None)
    if joined:
        return [list(shape) for shape in joined if shape]
    raw = event.args.get("Input Dims") or event.args.get("input_dims")
    if not isinstance(raw, list):
        return []
    dims: list[list[int]] = []
    for entry in raw:
        if not isinstance(entry, list):
            continue
        try:
            shape = [int(value) for value in entry]
        except (TypeError, ValueError):
            continue
        if shape:
            dims.append(shape)
    return dims


def _input_dtypes(event: RuntimeKernelEvent) -> list[str]:
    """Dtypes for the same operands `_input_dims` returns.

    Reads the joined field first, for the same reason as `_input_dims`: the profiler
    records types on the host op, not the device kernel. Without this the shapes were
    joined and the dtypes were not, so every intermediate tensor came back as 0 bytes
    and the region table reported "traffic saved: 0.0 MB" for a fusion eliminating a
    172x9728 tensor.
    """
    joined = getattr(event, "input_dtypes", None)
    if joined:
        return [str(entry) for entry in joined]
    raw = event.args.get("Input type") or event.args.get("input_type")
    if not isinstance(raw, list):
        return []
    return [entry if isinstance(entry, str) else "" for entry in raw]


def _tensor_bytes(shape: Sequence[int], dtype: str) -> int:
    # The profiler spells dtypes as C++ type names — `c10::Half`, `c10::BFloat16`,
    # `at::Half` — not as the torch names the table is keyed on. Looking up the raw
    # string missed every one of them and returned 0 bytes, so a fusion eliminating a
    # 172x9728 half tensor reported "0.0 MB saved": a number that reads as "measured and
    # negligible" when it means "not recognized".
    name = dtype.strip().lower().rsplit("::", 1)[-1]
    width = _DTYPE_BYTES.get(name)
    if width is None:
        return 0
    total = 1
    for dim in shape:
        total *= max(1, dim)
    return total * width


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def format_regions(regions: Sequence[RegionRecord], limit: int = 20) -> str:
    """Render the region table the CLI prints (§21).

    Mirrors `format_catalog`: a header, one row per region, then the facts a reader
    needs to judge the table. The route column names Xe-Fuse and says whether it is
    actually installed, because a row that reads `-> Xe-Fuse` next to an absent project
    is the kind of thing a reader takes for a result.
    """
    header = f"{'ID':<5} {'GPU%':>6}  {'Kernels':<16} {'Pattern':<24} {'Extract':<8} {'Route'}"
    lines = [header, "-" * 100]

    if not regions:
        lines.append("(no fusable regions: no producer-consumer chain cleared the thresholds)")
        return "\n".join(lines)

    # Xe-Fuse is one executor among several, not the route. When it is absent the
    # region is authored instead (§13.8) — an external project must not be a hard
    # dependency for the only path that reaches an opaque GEMM.
    available = xe_fuse_available()
    route = "Xe-Fuse" if available else "author"

    for region in regions[:limit]:
        kernels = "+".join(region.kernel_ids)
        pattern = region.fusion_pattern or "(unnamed chain)"
        lines.append(
            f"{region.id:<5} {region.gpu_time_share * 100:>6.1f}  {kernels[:16]:<16} "
            f"{pattern[:24]:<24} {REGION_EXTRACTION_LEVEL.value:<8} {route}"
        )

    covered = sum(region.gpu_time_share for region in regions)
    eliminated = sum(t.bytes for region in regions for t in region.intermediate_tensors)
    lines.append("")
    traffic = (
        f"; fusion would eliminate {eliminated / 1e6:.1f} MB of intermediate traffic per pass"
        if eliminated
        else "; the intermediate sizes were not recorded, so the traffic saved is unmeasured"
    )
    lines.append(
        f"Regions cover {covered * 100:.1f}% of GPU time across {len(regions)} candidate(s)"
        f"{traffic}."
    )
    if not available:
        lines.append(
            "Executor 'author' means Orbit writes the fused kernel for these "
            "shapes and registers it as an operator override (\u00a713.8); 'Xe-Fuse' "
            "means its autotuned template matched. Either way the result goes through "
            "the same correctness and measurement gates, and no external project is "
            "required to act on a region."
        )
    return "\n".join(lines)


def _member_times(
    events: Sequence[RuntimeKernelEvent], kernel_ids: Sequence[str]
) -> dict[str, float]:
    """Time each member contributed inside this occurrence.

    Keyed by catalog id and positionally aligned with `kernel_ids`, which the caller
    builds from the same event sequence. A member appearing twice in one chain — the
    GEMM in `gemm -> rmsnorm -> gemm` — accumulates both, because the region replaces
    both instances.
    """
    times: dict[str, float] = {}
    for event, kernel_id in zip(events, kernel_ids, strict=False):
        times[kernel_id] = times.get(kernel_id, 0.0) + event.duration_us
    return times
