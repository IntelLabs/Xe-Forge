"""Read a torch.profiler chrome trace into ``CapturedOp`` rows.

One parser for every PyTorch framework (vLLM, SGLang, a bare module): they all export the
same trace. A device kernel names the host op that launched it through ``External id``;
that op is charged to its *outermost* enclosing dispatched op, so an op whose work happens
in nested calls -- how serving stacks register their heaviest ops -- collects its
children's time and the charges never overlap (the rule FlashInfer-Bench's
``device/attribution.py`` applies to in-memory profiler events). Host ops nest by
``ts``/``dur`` on their thread; ``nn.Module`` frames (``with_stack=True``) enclosing the
owner give the call site, with instance numbers dropped so every layer shares one.

Requires the trace to have been recorded with ``record_shapes=True``; without it every
row has an empty signature and dedup collapses all calls of an op into one.
"""

from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xe_forge.frontend.ir import CapturedOp, TensorArg

# Runtime-API frames interleaved with ops; never owners (FIB attribution._NOT_OPS).
_NOT_OPS = ("cudaLaunch", "cudaMemcpy", "ze", "ur", "cl", "Runtime Triggered")
_DEVICE_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
_MODULE_PREFIX = "nn.Module: "

_DTYPES = {
    "c10::BFloat16": "bfloat16",
    "c10::Half": "float16",
    "float": "float32",
    "double": "float64",
    "long int": "int64",
    "int": "int32",
    "short int": "int16",
    "signed char": "int8",
    "unsigned char": "uint8",
    "bool": "bool",
    "c10::Float8_e4m3fn": "float8_e4m3fn",
    "c10::Float8_e5m2": "float8_e5m2",
    "c10::Float8_e4m3fnuz": "float8_e4m3fnuz",
    "c10::complex<float>": "complex64",
}


def is_op_name(name: str) -> bool:
    """A dispatched op (``namespace::op``), not a runtime frame."""
    return "::" in name and not name.startswith(_NOT_OPS)


@dataclass
class _Node:
    name: str
    ts: float
    end: float
    args: dict
    parent: _Node | None = None
    modules: tuple[str, ...] = ()


@dataclass
class TraceSummary:
    ops: list[CapturedOp]
    total_us: float
    attributed_us: float
    kernels: dict[str, float] = field(default_factory=dict)
    launches: int = 0

    @property
    def attributed_pct(self) -> float:
        return 100.0 * self.attributed_us / self.total_us if self.total_us else 0.0


def load_events(path: str | Path) -> list[dict]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        data = json.load(f)
    return data["traceEvents"] if isinstance(data, dict) else data


def _module_name(name: str) -> str:
    # "nn.Module: Qwen2DecoderLayer_3" -> "Qwen2DecoderLayer"
    return re.sub(r"_\d+$", "", name[len(_MODULE_PREFIX) :])


def _nest(events: list[dict]) -> dict[int, _Node]:
    """Host ops by External id, each with its parent op and enclosing module frames."""
    by_thread: dict[tuple, list[tuple]] = defaultdict(list)
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        cat = e.get("cat")
        if cat == "cpu_op":
            kind = 1
        elif cat == "python_function" and e.get("name", "").startswith(_MODULE_PREFIX):
            kind = 0
        else:
            continue
        ts, dur = float(e["ts"]), float(e["dur"])
        by_thread[(e.get("pid"), e.get("tid"))].append((ts, -dur, kind, e))

    nodes: dict[int, _Node] = {}
    for items in by_thread.values():
        items.sort(key=lambda t: (t[0], t[1], t[2]))
        stack: list[tuple[str, Any]] = []  # ("op", _Node) | ("mod", (name, end))
        for ts, neg_dur, kind, e in items:
            end = ts - neg_dur
            while stack and _end(stack[-1]) <= ts:
                stack.pop()
            if kind == 0:
                stack.append(("mod", (_module_name(e["name"]), end)))
                continue
            parent = next((n for k, n in reversed(stack) if k == "op"), None)
            mods = tuple(n[0] for k, n in stack if k == "mod")
            node = _Node(e["name"], ts, end, e.get("args") or {}, parent, mods)
            ext = node.args.get("External id")
            if ext is not None:
                nodes[ext] = node
            stack.append(("op", node))
    return nodes


def _end(item: tuple[str, Any]) -> float:
    kind, n = item
    return n.end if kind == "op" else n[1]


def _owner(node: _Node) -> _Node | None:
    owner = None
    while node is not None:
        if is_op_name(node.name):
            owner = node
        node = node.parent
    return owner


def _launcher(node: _Node, owner: _Node) -> _Node | None:
    """Innermost non-ATen dispatched op between the launching host op and its owner."""
    while node is not None and node is not owner:
        if is_op_name(node.name) and not node.name.startswith("aten::"):
            return node
        node = node.parent
    return None


def _parse_args(args: dict) -> tuple[list[TensorArg | None], list[Any]]:
    dims = args.get("Input Dims") or []
    types = args.get("Input type") or []
    strides = args.get("Input Strides") or []
    concrete = args.get("Concrete Inputs") or []
    tensors: list[TensorArg | None] = []
    scalars: list[Any] = []
    for i, typ in enumerate(types):
        value = concrete[i] if i < len(concrete) else ""
        shape = dims[i] if i < len(dims) else []
        # A concrete value means a scalar, whatever its type name ("double" is both).
        if (
            value == ""
            and typ in _DTYPES
            and isinstance(shape, list)
            and all(isinstance(d, int) for d in shape)
        ):
            stride = strides[i] if i < len(strides) and isinstance(strides[i], list) else None
            tensors.append(TensorArg(shape=list(shape), dtype=_DTYPES[typ], stride=stride))
            scalars.append(None)
        else:
            tensors.append(None)
            scalars.append(value if value != "" else None)
    return tensors, scalars


def _signature(tensors, scalars) -> str:
    return json.dumps(
        [[t.shape, t.dtype, t.stride] if t else None for t in tensors] + [scalars],
        separators=(",", ":"),
    )


def parse_events(events: list[dict]) -> TraceSummary:
    nodes = _nest(events)
    rows: dict[tuple, CapturedOp] = {}
    counted: set[int] = set()
    kernels: dict[str, float] = defaultdict(float)
    total = attributed = 0.0
    launches = 0
    unowned: dict[str, CapturedOp] = {}

    for e in events:
        if e.get("ph") != "X" or e.get("cat") not in _DEVICE_CATS:
            continue
        kname = e.get("name", "")[:240]
        us = float(e.get("dur", 0.0))
        total += us
        launches += 1
        kernels[kname] += us
        node = nodes.get((e.get("args") or {}).get("External id"))
        owner = _owner(node) if node is not None else None
        if owner is None:
            # Launched outside any dispatched op (e.g. a Triton launcher called directly):
            # kept as its own row, named by the kernel, with no signature to read.
            row = unowned.setdefault(
                kname,
                CapturedOp(id="", framework_op=f"kernel::{kname}", args=[], call_site=""),
            )
            row.calls += 1
            row.device_us += us
            row.kernel_symbols[kname] = row.kernel_symbols.get(kname, 0.0) + us
            continue
        attributed += us
        # Rows are keyed on the op that implements the work: the owner, or -- when the
        # owner is registered in Python -- the provider op inside it that launched the
        # kernel, with that op's own arguments. Each kernel still lands in exactly one row.
        impl = _launcher(node, owner) or owner
        tensors, scalars = _parse_args(impl.args)
        site = "/".join(owner.modules)
        key = (impl.name, _signature(tensors, scalars), site)
        row = rows.get(key)
        if row is None:
            row = rows[key] = CapturedOp(
                id="",
                framework_op=impl.name,
                args=tensors,
                scalars=scalars,
                call_site=site,
            )
            if impl is not owner:
                row.framework_meta["owner"] = owner.name
        if id(impl) not in counted:
            counted.add(id(impl))
            row.calls += 1
        row.device_us += us
        row.kernel_symbols[kname] = row.kernel_symbols.get(kname, 0.0) + us

    ops = sorted([*rows.values(), *unowned.values()], key=lambda r: -r.device_us)
    for i, row in enumerate(ops):
        row.id = f"op-{i:04d}"
        row.device_us = round(row.device_us, 3)
        row.kernel_symbols = {
            k: round(v, 3) for k, v in sorted(row.kernel_symbols.items(), key=lambda kv: -kv[1])
        }
    return TraceSummary(
        ops=ops, total_us=total, attributed_us=attributed, kernels=dict(kernels), launches=launches
    )


def parse_trace(path: str | Path) -> TraceSummary:
    return parse_events(load_events(path))
