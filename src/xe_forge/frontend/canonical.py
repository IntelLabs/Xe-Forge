"""Group captured ops into canonical workloads: same op, dtypes, layout and non-token dims.

A dim is var when it equals the token count in every call and the token count varied.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict

from xe_forge.frontend.dtypes import storage_bytes, tensor_bytes
from xe_forge.frontend.ir import CapturedOp, Naming, TensorArg, Workload


def _layout(t: TensorArg) -> str:
    """Layout class from the innermost dims only: outer pitches vary with the token count."""
    if not t.stride or len(t.stride) != len(t.shape) or not t.shape:
        return "c"
    if t.stride[-1] == 1 or t.shape[-1] == 1:
        return "c"
    if len(t.shape) >= 2 and t.stride[-2] == 1:
        return "t"
    return "s"


def arg_label(op: CapturedOp, i: int) -> str:
    """Argument ``i``'s schema name, else ``arg<i>``."""
    names = op.framework_meta.get("arg_names") or []
    return names[i] if i < len(names) else f"arg{i}"


def _one_gemm(op: CapturedOp, spec: dict) -> dict | None:
    by = {arg_label(op, i): t for i, t in enumerate(op.args) if t is not None}
    ints = {arg_label(op, i): _int(v) for i, v in enumerate(op.scalars)}
    act, w = by.get(spec.get("activation")), by.get(spec.get("weight"))
    if act is None or w is None or not act.shape or len(w.shape) < 2:
        return None
    per = spec.get("act_per_stored") or 1
    k = act.shape[-1] * per if type(per) is int and per > 0 else 0
    m = math.prod(act.shape[:-1])
    grouped = spec.get("grouped") is True
    experts = ints.get(spec.get("experts") or "")
    if experts is None and grouped and len(w.shape) == 3:
        experts = w.shape[0]
    axis = spec.get("n_axis")
    n = w.shape[axis] if type(axis) is int and 0 <= axis < len(w.shape) else None
    rows = {"rows": spec["rows"]} if spec.get("rows") else {}
    if n is None or (grouped and experts is None) or min(m, n, k, experts or 1) <= 0:
        return {"M": m, "K": k, "grouped": grouped, **rows}
    per_matrix = experts if grouped else 1
    side_args = spec.get("weight_side") if isinstance(spec.get("weight_side"), list) else []
    side = [tensor_bytes(by[a].shape, by[a].dtype) for a in side_args if a in by]
    side = side or [tensor_bytes(w.shape, w.dtype)]
    a_total = tensor_bytes(act.shape, act.dtype)
    if None in side or a_total is None:
        return {"M": m, "N": n, "K": k, "grouped": grouped, **rows}
    out = by.get(spec.get("output") or "")
    o_total = tensor_bytes(out.shape, out.dtype) if out is not None else None
    return {
        "M": m, "N": n, "K": k, "E": per_matrix,
        "w_bytes": round(sum(side) / (per_matrix * n * k), 5),
        "a_bytes": round(a_total / (m * k), 5), "compute": act.dtype, "grouped": grouped,
        **({"o_bytes": round(o_total / (m * n), 5)} if o_total else {}), **rows,
    }  # fmt: skip


def _provider_gemm(op: CapturedOp) -> dict | None:
    """M/N/K/E and measured bytes per element of a GEMM from ``named["gemm"]``; None if unnamed."""
    spec = (op.framework_meta.get("named") or {}).get("gemm")
    specs = spec if isinstance(spec, list) else [spec] if isinstance(spec, dict) else []
    parts = [_one_gemm(op, s) for s in specs if isinstance(s, dict)]
    if not parts or None in parts:
        return None
    first = dict(parts[0])
    if len(parts) > 1:
        first["parts"] = parts
    return first


def _gemm_dims(op: CapturedOp) -> dict[str, int] | None:
    gemm = _provider_gemm(op)
    if gemm is not None:
        op.framework_meta["gemm"] = gemm
        keys = ("M", "N", "K", "E") if gemm.get("grouped") else ("M", "N", "K")
        return {k: gemm[k] for k in keys if k in gemm}
    stated = op.framework_meta.get("gemm")
    if stated and "tokens" in op.framework_meta:
        m, n, k = op.framework_meta["tokens"], stated["N"], stated["K"]
        act = next((t for t in op.args if t is not None), None)
        if stated.get("weight_bytes") and act is not None and min(n, k) > 0:
            stated.update(w_bytes=round(stated["weight_bytes"] / (n * k), 5),
                          a_bytes=storage_bytes(act.dtype), compute=act.dtype)  # fmt: skip
        return {"M": m, "N": n, "K": k}
    return None


def _int(value) -> int | None:
    return int(value) if isinstance(value, str) and re.fullmatch(r"-?\d+", value) else None


def _named_dims(op: CapturedOp) -> dict[str, int]:
    """Every tensor dimension, plus every integer scalar argument (a size)."""
    dims: dict[str, int] = {}
    for i, t in enumerate(op.args):
        label = arg_label(op, i).upper()
        if t is not None:
            for j, size in enumerate(t.shape):
                dims[f"{label}_D{j}"] = size
        elif i < len(op.scalars) and _int(op.scalars[i]) is not None:
            dims[label] = _int(op.scalars[i])
    for key in ("tokens", "context"):
        if key in op.framework_meta:
            dims[key.upper()] = op.framework_meta[key]
    return dims


def _token(op: CapturedOp, dims: dict[str, int]) -> tuple[str, int] | None:
    named = op.framework_meta.get("named") or {}
    if named.get("token_arg") is not None:
        key = f"{named['token_arg'].upper()}_D{named.get('token_axis', 0)}"
        if key in dims:
            return key, dims[key]
    if "TOKENS" in dims:
        return "TOKENS", dims["TOKENS"]
    if "M" in dims and (op.framework_op.startswith("aten::") or "gemm" in op.framework_meta):
        return "M", dims["M"]
    first = next((a for a in op.args if a is not None and a.shape), None)
    if first is None:
        return None
    key = next(k for k, v in dims.items() if k.endswith("_D0") and v == first.shape[0])
    return key, first.shape[0]


def _attrs(op: CapturedOp) -> dict:
    out = {}
    for i, value in enumerate(op.scalars):
        if value is not None and _int(value) is None:
            out[arg_label(op, i)] = value
    gemm = op.framework_meta.get("gemm") or {}
    keep = ("w_bytes", "a_bytes", "o_bytes", "compute", "grouped", "rows", "parts")
    out.update({k: gemm[k] for k in keep if k in gemm})
    named = op.framework_meta.get("named") or {}
    if isinstance(named.get("cost"), dict):
        out["cost"] = named["cost"]
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
        named = _named_dims(op)
        dims = {**named, **(gemm or {})}
        scalar_dims = {k for k in named if not re.search(r"_D\d+$", k)}
        tensors = [a for a in op.args if a is not None]
        family = "gemm" if gemm else (op.framework_meta.get("named") or {}).get("family", "other")
        named = op.framework_meta.get("named") or {}
        base = {
            "family": family,
            "op": op.framework_op,
            # One op running different kernels is that many workloads.
            "impl": [named.get("route"), named.get("repo"), named.get("name")],
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
        follows: set[str] = set()
        if len(tokens) > 1 and all(m[2] for m in members):
            names = members[0][1].keys()
            var = {n for n in names if all(m[1][n] == m[2][1] for m in members)}
            # A dim with one value per token count that changes with it follows the context.
            for n in names:
                if n in var:
                    continue
                per_token: dict[int, set] = defaultdict(set)
                for m in members:
                    per_token[m[2][1]].add(m[1][n])
                if (
                    all(len(v) == 1 for v in per_token.values())
                    and len(set().union(*per_token.values())) > 1
                ):
                    follows.add(n)
        sub: dict[str, list] = defaultdict(list)
        for m in members:
            const = {
                k: v for k, v in m[1].items() if k not in var and k not in follows and k not in m[4]
            }
            sub[_digest([m[3], const])].append(m)
        for rows_ in sub.values():
            # Integer args still varying once shapes agree are runtime sizes, not workloads.
            moving = {n for n in rows_[0][4] if len({r[1][n] for r in rows_}) > 1} - var
            moving |= {n for n in follows if len({r[1][n] for r in rows_}) > 1}
            out.append(_workload(rows_, var | moving, moving, total_us, naming_fn))
    out.sort(key=lambda w: -w.device_us)
    return out


def _workload(members, var: set[str], moving: set[str], total_us: float, naming_fn) -> Workload:
    """One workload; ``moving`` var dims keep their value at each token count in ``with``."""
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
            naming_fn(m[0])  # stamps impl_class
    naming: Naming = naming_fn(top)
    kernels: dict[str, float] = {}
    for m in members:
        for sym, us in m[0].kernel_symbols.items():
            kernels[sym] = round(kernels.get(sym, 0.0) + us, 3)
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
        framework_meta={"framework_op": base["op"], "kernels": kernels},
        tensors=_tensor_labels(top),
    )


def _tensor_labels(op: CapturedOp) -> list[list[str]]:
    return [[arg_label(op, i).upper(), t.dtype] for i, t in enumerate(op.args) if t is not None]


def resolve_dims(w: Workload, value: int | None) -> dict:
    """Concrete dims at one captured token count."""
    with_ = {}
    if value is not None and w.var_axes:
        with_ = next(iter(w.var_axes.values())).get(str(value), {}).get("with", {})
    return {k: (with_.get(k, value) if v == "var" else v) for k, v in w.dims.items()}
