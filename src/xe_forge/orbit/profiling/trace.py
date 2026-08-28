"""
Trace ingestion: normalizes raw profiler output (torch.profiler Chrome traces,
unitrace) into the `TraceEvents` every downstream stage reads. Torch is imported
lazily so committed fixtures ingest with no torch installed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field

from xe_forge.orbit.models import Artifact

# Chrome-trace categories that denote work executing on the device, as opposed to
# host-side operator bookkeeping.
_GPU_CATEGORIES = {"kernel", "gpu_op", "gpu_user_annotation", "xpu_op", "cuda_runtime_kernel"}
_HOST_CATEGORIES = {"cpu_op", "operator", "user_annotation", "python_function"}
_LAUNCH_CATEGORIES = {"cuda_runtime", "xpu_runtime", "runtime"}


class RuntimeKernelEvent(BaseModel):
    """One device-side execution, normalized away from its source format."""

    name: str
    start_us: float = 0.0
    duration_us: float = 0.0
    category: str = "kernel"
    stream: str | None = None
    correlation_id: int | None = None
    device: int | None = None
    # Inherited from the launching host op via correlation id: the profiler records
    # shapes against `cpu_op` events, never against device kernels.
    input_shapes: list[list[int]] = Field(default_factory=list)
    input_dtypes: list[str] = Field(default_factory=list)
    args: dict[str, object] = Field(default_factory=dict)

    @property
    def end_us(self) -> float:
        return self.start_us + self.duration_us


class HostEvent(BaseModel):
    """One host-side operator span, used to attribute kernels to framework ops."""

    name: str
    start_us: float = 0.0
    duration_us: float = 0.0
    category: str = "cpu_op"
    correlation_id: int | None = None
    # Containment is per-thread: a cpu_op only encloses launches made on its own thread.
    thread_id: int | None = None
    # `record_shapes` puts "Input Dims" on the *host* op, never on the device kernel.
    # Carrying it here is what lets the kernel inherit it by correlation below.
    input_shapes: list[list[int]] = Field(default_factory=list)
    # Recorded beside "Input Dims" as "Input type"; a shape without a dtype cannot be
    # turned into bytes.
    input_dtypes: list[str] = Field(default_factory=list)

    @property
    def end_us(self) -> float:
        return self.start_us + self.duration_us


class TraceEvents(Artifact):
    """Normalized trace: what ran on the device, what ran on the host, and when."""

    source: str = "torch.profiler"
    device_type: str = "cpu"
    kernels: list[RuntimeKernelEvent] = Field(default_factory=list)
    host_ops: list[HostEvent] = Field(default_factory=list)
    trace_start_us: float = 0.0
    trace_end_us: float = 0.0
    warnings: list[str] = Field(default_factory=list)

    @property
    def wall_span_us(self) -> float:
        return max(0.0, self.trace_end_us - self.trace_start_us)

    @property
    def total_gpu_time_us(self) -> float:
        return sum(k.duration_us for k in self.kernels)


def ingest_chrome_trace(payload: object, source: str = "torch.profiler") -> TraceEvents:
    """Normalize a Chrome-trace JSON object into `TraceEvents`.

    Accepts either the `{"traceEvents": [...]}` wrapper torch emits or a bare list.
    Malformed individual events are skipped with a warning rather than aborting the
    ingest — a truncated trace is common and still partially useful — but a payload
    that is not a trace at all raises, because silently returning zero kernels would
    look exactly like a host-bound workload.
    """
    if isinstance(payload, dict):
        raw_events = payload.get("traceEvents")
        if raw_events is None:
            raise ValueError("trace object has no 'traceEvents' key")
    elif isinstance(payload, list):
        raw_events = payload
    else:
        raise ValueError(f"unsupported trace payload type: {type(payload).__name__}")

    if not isinstance(raw_events, list):
        raise ValueError("'traceEvents' must be a list")

    kernels: list[RuntimeKernelEvent] = []
    host_ops: list[HostEvent] = []
    warnings: list[str] = []
    starts: list[float] = []
    ends: list[float] = []
    skipped = 0

    for raw in raw_events:
        if not isinstance(raw, dict):
            skipped += 1
            continue
        # Only complete duration events describe elapsed work.
        if raw.get("ph") not in (None, "X"):
            continue
        name = raw.get("name")
        if not isinstance(name, str):
            skipped += 1
            continue
        try:
            start = float(raw.get("ts", 0.0))
            duration = float(raw.get("dur", 0.0))
        except (TypeError, ValueError):
            skipped += 1
            continue

        category = str(raw.get("cat", "") or "").lower()
        args = raw.get("args") if isinstance(raw.get("args"), dict) else {}
        correlation = _as_int(args.get("correlation") if args else None)

        if _is_gpu_category(category):
            kernels.append(
                RuntimeKernelEvent(
                    name=name,
                    start_us=start,
                    duration_us=duration,
                    category=category or "kernel",
                    stream=_as_str(args.get("stream") if args else None),
                    correlation_id=correlation,
                    device=_as_int(args.get("device") if args else None),
                    args=dict(args) if args else {},
                )
            )
        elif category in _HOST_CATEGORIES or category in _LAUNCH_CATEGORIES:
            host_ops.append(
                HostEvent(
                    name=name,
                    start_us=start,
                    duration_us=duration,
                    category=category,
                    correlation_id=correlation,
                    thread_id=_as_int(raw.get("tid")),
                    input_shapes=_as_shapes(args.get("Input Dims") if args else None),
                    input_dtypes=_as_dtypes(args.get("Input type") if args else None),
                )
            )
        else:
            continue

        starts.append(start)
        ends.append(start + duration)

    if skipped:
        warnings.append(f"skipped {skipped} malformed trace events")
    if not kernels:
        warnings.append(
            "no device-side kernel events found; the workload may be host-bound, "
            "or the profiler may not have had a device activity source enabled"
        )

    # Join shapes onto kernels here rather than in each consumer, so region detection,
    # roofline and the shape matrix all see the same data without re-deriving it.
    return attach_shapes(
        TraceEvents(
            source=source,
            kernels=kernels,
            host_ops=host_ops,
            trace_start_us=min(starts) if starts else 0.0,
            trace_end_us=max(ends) if ends else 0.0,
            warnings=warnings,
        )
    )


def ingest_trace_file(path: Path, source: str = "torch.profiler") -> TraceEvents:
    """Read and normalize a trace file from disk.

    Handles gzip transparently: `torch.profiler` writes `*.pt.trace.json.gz` by
    default, so plain-JSON-only reading would fail on the most common real input.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"no trace file at {target}")

    try:
        if target.suffix == ".gz" or target.name.endswith(".json.gz"):
            import gzip

            with gzip.open(target, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        else:
            payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"trace file {target} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"trace file {target} could not be read: {exc}") from exc

    events = ingest_chrome_trace(payload, source=source)
    return events


def _is_gpu_category(category: str) -> bool:
    if not category:
        return False
    if category in _GPU_CATEGORIES:
        return True
    # torch labels XPU/CUDA device activity variously across versions; treat any
    # category that names a device backend and is not a host runtime call as GPU.
    return ("kernel" in category) or category.startswith(("gpu", "xpu_kernel", "cuda_kernel"))


def _as_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_str(value: object) -> str | None:
    return str(value) if value is not None else None


def profile_workload(
    fn,
    device_type: str = "cpu",
    output: Path | None = None,
) -> TraceEvents:
    """Profile a callable with `torch.profiler` and return normalized events.

    This is the live-hardware path. It needs torch, and it needs a device activity
    source to see kernels — on CPU it will correctly report that no device work
    occurred rather than inventing any.
    """
    try:
        import torch
        from torch.profiler import ProfilerActivity, profile
    except ImportError as exc:  # pragma: no cover - depends on install
        raise RuntimeError("torch is required to profile a live workload") from exc

    activities = [ProfilerActivity.CPU]
    device_attr = getattr(ProfilerActivity, device_type.upper(), None)
    if device_type != "cpu" and device_attr is not None:
        activities.append(device_attr)

    with profile(activities=activities, record_shapes=True) as prof:
        fn()
        if device_type != "cpu":
            backend = getattr(torch, device_type, None)
            if backend is not None and hasattr(backend, "synchronize"):
                backend.synchronize()

    target = Path(output) if output else None
    if target is None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            target = Path(tmp.name)

    target.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(target))
    events = ingest_trace_file(target)
    events.device_type = device_type
    return events


def merge_unitrace(events: TraceEvents, launch_gaps_us: float, gpu_busy_us: float) -> TraceEvents:
    """Fold Level Zero timing into an existing trace (see `profiling.unitrace`)."""
    events.warnings.append(
        f"unitrace merged: gpu_busy={gpu_busy_us:.1f}us launch_gaps={launch_gaps_us:.1f}us"
    )
    return events


def iter_kernel_names(events: Iterable[RuntimeKernelEvent]) -> list[str]:
    return [e.name for e in events]


def _as_shapes(value: object) -> list[list[int]]:
    """Normalize a profiler "Input Dims" entry into a list of integer shapes.

    The profiler emits ragged data — scalars arrive as `[]`, absent arguments as `[]`,
    and nested lists for tensor lists. Empty entries are dropped rather than kept as
    zero-length shapes, because a region edge keyed on `[]` would match anything.
    """
    if not isinstance(value, list):
        return []
    shapes: list[list[int]] = []
    for entry in value:
        if not isinstance(entry, list) or not entry:
            continue
        dims = [int(d) for d in entry if isinstance(d, (int, float)) and not isinstance(d, bool)]
        if dims:
            shapes.append(dims)
    return shapes


def attach_shapes(events: TraceEvents) -> TraceEvents:
    """Give each device kernel the shapes recorded against the op that launched it.

    The `cpu_op` carrying `Input Dims` has no correlation id, so the join is
    kernel -> correlation -> runtime event -> innermost enclosing `cpu_op` on the same
    thread. Innermost matters: ops nest (`aten::linear` contains `aten::addmm`), and
    an outer op's shapes describe a different call than the one that launched the
    kernel.
    """
    from bisect import bisect_right

    launches = {
        op.correlation_id: op
        for op in events.host_ops
        if op.correlation_id is not None and op.category in _LAUNCH_CATEGORIES
    }
    if not launches:
        return events

    # cpu_ops that actually carry shapes, indexed by thread and sorted by start.
    by_thread: dict[int, list[HostEvent]] = {}
    for op in events.host_ops:
        if op.input_shapes and op.thread_id is not None:
            by_thread.setdefault(op.thread_id, []).append(op)
    if not by_thread:
        return events
    for ops in by_thread.values():
        ops.sort(key=lambda o: o.start_us)
    starts = {tid: [o.start_us for o in ops] for tid, ops in by_thread.items()}

    for kernel in events.kernels:
        if kernel.input_shapes or kernel.correlation_id is None:
            continue
        launch = launches.get(kernel.correlation_id)
        if launch is None or launch.thread_id is None:
            continue
        ops = by_thread.get(launch.thread_id)
        if not ops:
            continue

        # Walk back from the last op starting at or before the launch, taking the first
        # that still encloses it. Because ops nest, that is the innermost container.
        index = bisect_right(starts[launch.thread_id], launch.start_us)
        for candidate in reversed(ops[:index]):
            if candidate.end_us >= launch.start_us:
                kernel.input_shapes = candidate.input_shapes
                # Taken from the same op, so shapes and dtypes describe the same call.
                # Pairing them from different sources would produce a plausible,
                # mismatched tensor description.
                kernel.input_dtypes = candidate.input_dtypes
                break
    return events


def _as_dtypes(value: object) -> list[str]:
    """Normalize a profiler "Input type" entry, dropping the non-tensor slots.

    `_as_shapes` drops empty entries (scalars, absent arguments), so this must drop the
    same positions or the two lists stop describing the same operands — and a shape
    paired with the wrong dtype yields confidently wrong byte counts.
    """
    if not isinstance(value, list):
        return []
    return [str(entry) for entry in value if isinstance(entry, str) and entry != "Scalar"]
