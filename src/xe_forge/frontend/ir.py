"""The capture file: what a framework ran, independent of which framework ran it.

Two levels. ``CapturedOp`` is one row per (framework op, argument signature, call site) as
the trace showed it, strides and all. ``Workload`` is the canonical form several ops can
share: a family, role-named dims, dtypes, a layout class, and a histogram over the axes
that varied. Analysis reads workloads; ops are kept so a decision can be traced back to
what was observed. Whatever only one framework can say goes in ``framework_meta`` and is
never read by analysis.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

SCHEMA = "xe_forge.capture/v1"


@dataclass
class TensorArg:
    shape: list[int]
    dtype: str
    stride: list[int] | None = None


@dataclass
class CapturedOp:
    id: str
    framework_op: str
    args: list[TensorArg | None]
    scalars: list[Any] = field(default_factory=list)
    kernel_symbols: dict[str, float] = field(default_factory=dict)  # kernel -> device us
    calls: int = 0
    device_us: float = 0.0
    call_site: str = ""
    impl_class: str = "unknown"  # onednn | provider | triton | aten | python | unknown
    framework_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Naming:
    """What the kernel-locator is told to look for. ``route`` says whether Xe-Forge can
    take it: ``kernel`` (a named kernel in a repository), ``library`` (a library call --
    change the call, not the kernel) or ``unnamed`` (no rule matched)."""

    route: str
    name: str | None = None
    repo: str | None = None
    dsl: str | None = None
    via: str = ""


@dataclass
class Workload:
    workload_id: str
    family: str
    dims: dict[str, Any]  # role -> int, or "var"
    dtypes: list[str]
    layout: list[str]
    attrs: dict[str, Any]
    var_axes: dict[str, dict[str, dict[str, float]]]  # axis -> value -> {calls, device_us}
    calls: int
    device_us: float
    share: float
    ops: list[str]
    call_sites: list[str]
    naming: Naming
    framework_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaptureRun:
    run: dict[str, Any]
    ops: list[CapturedOp]
    workloads: list[Workload] = field(default_factory=list)
    schema: str = SCHEMA


def _build(cls, data: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


def to_dict(capture: CaptureRun) -> dict:
    return asdict(capture)


def from_dict(data: dict) -> CaptureRun:
    if data.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} capture (schema={data.get('schema')!r})")
    ops = []
    for o in data.get("ops", []):
        o = dict(o)
        o["args"] = [None if a is None else _build(TensorArg, a) for a in o.get("args", [])]
        ops.append(_build(CapturedOp, o))
    workloads = []
    for w in data.get("workloads", []):
        w = dict(w)
        w["naming"] = _build(Naming, w.get("naming") or {"route": "unnamed"})
        workloads.append(_build(Workload, w))
    return CaptureRun(run=data.get("run", {}), ops=ops, workloads=workloads)


def dump(capture: CaptureRun, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(capture), indent=1) + "\n")
    return path


def load(path: str | Path) -> CaptureRun:
    return from_dict(json.loads(Path(path).read_text()))
