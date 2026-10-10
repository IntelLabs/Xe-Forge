"""Captured ops -> canonical workloads.

A workload is what two calls share when one kernel could serve both: the op, the dtypes,
the layout class of every tensor, the non-tensor arguments, and every dimension that does
not follow the token count. Dimensions are named after the op's own schema
(``INPUT_D1`` is dim 1 of the argument called ``input``), so the spec Xe-Forge receives
speaks the kernel's vocabulary; GEMM-shaped ATen ops get ``M``/``N``/``K`` instead, so
``aten::linear``, ``aten::mm`` and ``aten::addmm`` at one shape are one workload.

Which dimensions are *var* is measured, not declared. The token count of a call is the
leading dim of its first tensor (``M`` for a GEMM), and a dimension is var when it equals
the token count in every call of the group and the token count itself varied. Calls whose
remaining dimensions differ are different workloads.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict

from xe_forge.frontend.ir import CapturedOp, Naming, TensorArg, Workload

_FAMILIES = (
    ("gemm", r"^(linear|mm|addmm|bmm|matmul|baddbmm)$|gemm|_mm$|scaled_mm"),
    ("attention", r"attention|attn|flash|paged|mla"),
    ("norm", r"norm"),
    ("rope", r"rotary|rope"),
    ("activation", r"silu|gelu|relu|swiglu|_and_mul|sigmoid|tanh"),
    ("moe", r"moe|expert|topk|router|grouped"),
    ("cache", r"cache"),
    ("sampling", r"sampl|softmax|argmax|top_?p|multinomial"),
    ("copy", r"^(copy_|to|_to_copy|contiguous|clone|cat|index|index_select|embedding|fill_)"),
)

_ATEN_GEMM = {"linear", "mm", "addmm", "bmm", "matmul", "baddbmm"}


def base_name(framework_op: str) -> str:
    return framework_op.split("::", 1)[-1].split(".", 1)[0]


def family_of(framework_op: str) -> str:
    name = base_name(framework_op).lower()
    for fam, pattern in _FAMILIES:
        if re.search(pattern, name):
            return fam
    return "elementwise" if framework_op.startswith("aten::") else "other"


def _layout(t: TensorArg) -> str:
    """Layout class from the innermost dimension only. Outer pitches (a row-strided view
    into a fused QKV output, say) depend on the token count -- a one-row view reads as
    contiguous -- so they stay in the op's raw strides rather than split a workload."""
    if not t.stride or len(t.stride) != len(t.shape) or not t.shape:
        return "c"
    if t.stride[-1] == 1 or t.shape[-1] == 1:
        return "c"
    if len(t.shape) >= 2 and t.stride[-2] == 1:
        return "t"  # last two dims transposed
    return "s"


def _gemm_dims(op: CapturedOp) -> dict[str, int] | None:
    name = base_name(op.framework_op)
    if not op.framework_op.startswith("aten::") or name not in _ATEN_GEMM:
        return None
    ts = [a for a in op.args if a is not None]
    try:
        if name == "linear":
            x, w = ts[0], ts[1]
            return {"M": math.prod(x.shape[:-1]), "N": w.shape[0], "K": x.shape[-1]}
        if name == "addmm":
            a, b = ts[1], ts[2]
        elif name == "baddbmm":
            a, b = ts[1], ts[2]
        else:
            a, b = ts[0], ts[1]
        dims = {
            "M": math.prod(a.shape[:-1]) if len(a.shape) > 2 and name == "matmul" else a.shape[-2]
        }
        dims.update(N=b.shape[-1], K=a.shape[-1])
        if len(a.shape) == 3 and name in ("bmm", "baddbmm"):
            dims["B"] = a.shape[0]
        return dims
    except (IndexError, AttributeError):
        return None


def _int(value) -> int | None:
    return int(value) if isinstance(value, str) and re.fullmatch(r"-?\d+", value) else None


def _named_dims(op: CapturedOp) -> dict[str, int]:
    """Every tensor dimension, plus every integer scalar argument (``max_seqlen_q``,
    ``head_size``): an integer argument is a size, and one that follows the token count
    must be var like the tensor dims that do."""
    names = op.framework_meta.get("arg_names") or []
    dims: dict[str, int] = {}
    for i, t in enumerate(op.args):
        label = (names[i] if i < len(names) else f"arg{i}").upper()
        if t is not None:
            for j, size in enumerate(t.shape):
                dims[f"{label}_D{j}"] = size
        elif i < len(op.scalars) and _int(op.scalars[i]) is not None:
            dims[label] = _int(op.scalars[i])
    return dims


def _token(op: CapturedOp, dims: dict[str, int]) -> tuple[str, int] | None:
    if "M" in dims and op.framework_op.startswith("aten::"):
        return "M", dims["M"]
    first = next((a for a in op.args if a is not None and a.shape), None)
    if first is None:
        return None
    key = next(k for k, v in dims.items() if k.endswith("_D0") and v == first.shape[0])
    return key, first.shape[0]


def _attrs(op: CapturedOp) -> dict:
    names = op.framework_meta.get("arg_names") or []
    out = {}
    for i, value in enumerate(op.scalars):
        if value is not None and _int(value) is None:
            out[names[i] if i < len(names) else f"arg{i}"] = value
    if base_name(op.framework_op) in ("linear", "addmm", "baddbmm"):
        out["bias"] = (
            len([a for a in op.args if a is not None]) >= 3
            if base_name(op.framework_op) == "linear"
            else True
        )
    return out


def _digest(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


def workloads(ops: list[CapturedOp], total_us: float, naming_fn=None) -> list[Workload]:
    """Group ops into workloads, costliest first."""
    if naming_fn is None:
        from xe_forge.frontend.naming import name_op as naming_fn

    rows = []
    for op in ops:
        gemm = _gemm_dims(op)
        dims = gemm or _named_dims(op)
        scalar_dims = set() if gemm else {k for k in dims if not re.search(r"_D\d+$", k)}
        tensors = [a for a in op.args if a is not None]
        family = family_of(op.framework_op)
        if family == "other" and op.framework_meta.get("owner"):
            family = family_of(op.framework_meta["owner"])
        base = {
            "family": family,
            "op": op.framework_op,
            "dtypes": [t.dtype for t in tensors],
            "layout": [_layout(t) for t in tensors],
            "attrs": _attrs(op),
            "dim_names": list(dims),
        }
        rows.append((op, dims, _token(op, dims), base, scalar_dims))

    groups: dict[str, list] = defaultdict(list)
    for row in rows:
        groups[_digest(row[3])].append(row)

    out: list[Workload] = []
    for members in groups.values():
        tokens = {m[2][1] for m in members if m[2]}
        var: set[str] = set()
        if len(tokens) > 1:
            names = members[0][1].keys()
            var = {n for n in names if all(m[2] and m[1][n] == m[2][1] for m in members)}
        sub: dict[str, list] = defaultdict(list)
        for m in members:
            const = {k: v for k, v in m[1].items() if k not in var and k not in m[4]}
            sub[_digest([m[3], const])].append(m)
        for rows_ in sub.values():
            # Integer arguments that still vary once the tensor shapes agree are runtime
            # sizes (a context length growing every decode step), not distinct workloads.
            moving = {n for n in rows_[0][4] if len({r[1][n] for r in rows_}) > 1} - var
            out.append(_workload(rows_, var | moving, moving, total_us, naming_fn))
    out.sort(key=lambda w: -w.device_us)
    return out


def _workload(members, var: set[str], moving: set[str], total_us: float, naming_fn) -> Workload:
    """``var``: dims marked var; ``moving``: the var dims that do not follow the token
    count, whose value each token count was captured with is kept beside it."""
    _op0, dims0, token0, base, _ = members[0]
    top = max(members, key=lambda m: m[0].device_us)[0]
    dims = {k: ("var" if k in var else v) for k, v in dims0.items()}
    axis = token0[0] if token0 and (token0[0] in var or moving) else None
    hist: dict[str, dict] = {}
    heaviest: dict[str, float] = {}
    calls = 0
    device_us = 0.0
    for op, op_dims, token, _, _ in members:
        calls += op.calls
        device_us += op.device_us
        if axis:
            value = str(token[1])
            h = hist.setdefault(value, {"calls": 0, "device_us": 0.0})
            h["calls"] += op.calls
            h["device_us"] = round(h["device_us"] + op.device_us, 3)
            if moving and op.device_us > heaviest.get(value, -1.0):
                heaviest[value] = op.device_us
                h["with"] = {k: op_dims[k] for k in sorted(moving)}
    ident = {k: v for k, v in base.items() if k != "dim_names"} | {"dims": dims}
    for m in members:
        if m[0] is not top:
            naming_fn(m[0])  # stamps impl_class on every op, not only the costliest
    naming: Naming = naming_fn(top)
    return Workload(
        workload_id=f"{base['family']}/{_digest(ident)}",
        family=base["family"],
        dims=dims,
        dtypes=base["dtypes"],
        layout=base["layout"],
        attrs=base["attrs"],
        var_axes={axis: hist} if axis else {},
        calls=calls,
        device_us=round(device_us, 3),
        share=round(device_us / total_us, 6) if total_us else 0.0,
        ops=[m[0].id for m in members],
        call_sites=sorted({m[0].call_site for m in members if m[0].call_site}),
        naming=naming,
        framework_meta={"framework_op": base["op"], "kernel_symbols": list(top.kernel_symbols)[:3]},
    )
