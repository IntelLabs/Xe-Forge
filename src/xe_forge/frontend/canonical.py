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

from xe_forge.frontend.dtypes import tensor_bytes
from xe_forge.frontend.ir import CapturedOp, Naming, TensorArg, Workload

_FAMILIES = (
    ("gemm", r"^(linear|mm|addmm|bmm|matmul|baddbmm)$|gemm|_mm$|scaled_mm|fullyconnected"),
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


def _provider_gemm(op: CapturedOp) -> dict | None:
    """M/N/K/E and bytes per element of a GEMM a provider implements, from the naming
    agent's answer for it (``named["gemm"]``: which argument is the activation, the weight,
    what is read with the weight, the output, where N sits) -- no argument-name or
    quantization-format table here.

    Bytes are *measured* from the captured tensors: weight-side bytes (weight, scales, zero
    points, bias) over the logical E*N*K weight elements, activation bytes over M*K, output
    bytes over M*N. E is the *local* expert count the kernel was given (the argument the
    agent names, or the weight's leading dim); a grouped call without one gets no E, and its
    bound stays unknown -- never E=1. The compute dtype is the activation's; which peak that
    runs at is the calibration's to say."""
    spec = (op.framework_meta.get("named") or {}).get("gemm")
    names = op.framework_meta.get("arg_names") or []
    if not spec or not names:
        return None
    by = {n: op.args[i] for i, n in enumerate(names) if i < len(op.args) and op.args[i] is not None}
    ints = {n: _int(op.scalars[i]) for i, n in enumerate(names) if i < len(op.scalars)}
    act, w = by.get(spec.get("activation")), by.get(spec.get("weight"))
    if act is None or w is None or not act.shape or len(w.shape) < 2:
        return None
    k = act.shape[-1] * int(spec.get("act_per_stored") or 1)
    m = math.prod(act.shape[:-1])
    grouped = bool(spec.get("grouped"))
    experts = ints.get(spec.get("experts") or "")
    if experts is None and grouped and len(w.shape) == 3:
        experts = w.shape[0]  # [E, ...] weights: the kernel's own (local) count
    n = w.shape[spec["n_axis"]] if isinstance(spec.get("n_axis"), int) else None
    if n is None or (grouped and experts is None):
        return {"M": m, "K": k, "grouped": grouped}  # still a GEMM; its bound is unknown
    per_matrix = experts if grouped else 1
    side = [
        tensor_bytes(by[a].shape, by[a].dtype) for a in spec.get("weight_side") or [] if a in by
    ]
    side = side or [tensor_bytes(w.shape, w.dtype)]
    a_total = tensor_bytes(act.shape, act.dtype)
    if None in side or a_total is None:
        return {"M": m, "N": n, "K": k, "grouped": grouped}
    out = by.get(spec.get("output") or "")
    o_total = tensor_bytes(out.shape, out.dtype) if out is not None else None
    return {
        "M": m, "N": n, "K": k, "E": per_matrix,
        "w_bytes": round(sum(side) / (per_matrix * n * k), 5),
        "a_bytes": round(a_total / (m * k), 5), "compute": act.dtype, "grouped": grouped,
        **({"o_bytes": round(o_total / (m * n), 5)} if o_total else {}),
    }  # fmt: skip


def _gemm_dims(op: CapturedOp) -> dict[str, int] | None:
    provider = _provider_gemm(op)
    if provider is not None:
        op.framework_meta["gemm"] = provider
        return {k: provider[k] for k in ("M", "N", "K", "E") if k in provider}
    # An adapter whose matmul weights are not ports (OpenVINO FullyConnected) states K and N.
    stated = op.framework_meta.get("gemm")
    if stated and "tokens" in op.framework_meta:
        return {"M": op.framework_meta["tokens"], "N": stated["N"], "K": stated["K"]}
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
        elif name == "matmul" and len(b.shape) > 2:  # batched on both sides: B matmuls
            dims.update(M=a.shape[-2], B=math.prod(b.shape[:-2]))
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
    # An adapter whose trace has no per-call shapes (OpenVINO) states the call's size.
    for key in ("tokens", "context"):
        if key in op.framework_meta:
            dims[key.upper()] = op.framework_meta[key]
    return dims


def _token(op: CapturedOp, dims: dict[str, int]) -> tuple[str, int] | None:
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
    names = op.framework_meta.get("arg_names") or []
    out = {}
    for i, value in enumerate(op.scalars):
        if value is not None and _int(value) is None:
            out[names[i] if i < len(names) else f"arg{i}"] = value
    gemm = op.framework_meta.get("gemm") or {}
    out.update(
        {k: gemm[k] for k in ("w_bytes", "a_bytes", "o_bytes", "compute", "grouped") if k in gemm}
    )
    if (op.framework_meta.get("named") or {}).get("paged"):
        out["paged"] = True
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
        family = (op.framework_meta.get("named") or {}).get("family") or family_of(op.framework_op)
        if family == "other" and op.framework_meta.get("owner"):
            family = family_of(op.framework_meta["owner"])
        if gemm:
            family = "gemm"
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
        follows: set[str] = set()
        if len(tokens) > 1 and all(m[2] for m in members):
            names = members[0][1].keys()
            var = {n for n in names if all(m[1][n] == m[2][1] for m in members)}
            # A dim that takes one value per token count, and changes with it, is a size
            # the context dictates (a page table one page wider once decode passes the
            # prompt), not a different workload. Two shapes at one token count are.
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
            # Integer arguments that still vary once the tensor shapes agree are runtime
            # sizes (a context length growing every decode step), not distinct workloads.
            moving = {n for n in rows_[0][4] if len({r[1][n] for r in rows_}) > 1} - var
            moving |= {n for n in follows if len({r[1][n] for r in rows_}) > 1}
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
        tensors=_tensor_labels(top),
    )


def _tensor_labels(op: CapturedOp) -> list[list[str]]:
    names = op.framework_meta.get("arg_names") or []
    return [
        [(names[i] if i < len(names) else f"arg{i}").upper(), t.dtype]
        for i, t in enumerate(op.args)
        if t is not None
    ]


def resolve_dims(w: Workload, value: int | None) -> dict:
    """Concrete dims for one captured token count: var dims that follow the token count
    take it; the others take what the heaviest call at that count used."""
    with_ = {}
    if value is not None and w.var_axes:
        with_ = next(iter(w.var_axes.values())).get(str(value), {}).get("with", {})
    return {k: (with_.get(k, value) if v == "var" else v) for k, v in w.dims.items()}
