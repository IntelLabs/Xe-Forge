"""Turn the naming agent's ``names.json`` entry for an op into a ``Naming``, and validate it."""

from __future__ import annotations

from xe_forge.frontend import formula
from xe_forge.frontend.ir import CapturedOp, Naming

_IMPL = {"kernel": "provider", "aten": "aten"}


def key(op: CapturedOp) -> str:
    """The names.json key: the implementing op, stable across a patched build."""
    return op.framework_op


def _base(op: str) -> str:
    return op.split("::", 1)[-1].split(".", 1)[0]


def name_op(op: CapturedOp) -> Naming:
    entry = op.framework_meta.get("named")
    if entry and entry.get("route") in ROUTES:
        route = entry.get("route")
        library = entry.get("library") or "library"
        op.impl_class = library if route == "library" else _IMPL.get(route, "unknown")
        return Naming(
            entry.get("route", "unnamed"),
            entry.get("name") or _base(op.framework_op),
            entry.get("repo"),
            entry.get("dsl"),
            via=entry.get("via", "named by the naming agent"),
        )
    op.impl_class = "unknown"
    return Naming("unnamed", _base(op.framework_op), via="not named (no names.json entry)")


ROUTES = ("kernel", "library", "aten", "unnamed")
FAMILIES = ("gemm", "attention", "mla", "indexer", "linear_attention", "ssm", "norm",
            "activation", "moe", "rope", "cache", "sampling", "copy", "collective", "elementwise",
            "other")  # fmt: skip
ROWS = ("token", "token_expert")


def _pos_int(v) -> bool:
    return type(v) is int and v > 0


def validate(names, listing: dict, repos) -> list[str]:
    """Problems with a naming answer against the ops it was asked about; empty if usable."""
    from pathlib import Path

    from xe_forge.models import DSL

    dsls = {d.value for d in DSL}
    if not isinstance(names, dict) or not isinstance(names.get("ops"), dict):
        return ["not {'ops': {...}, 'model': {...}}"]
    out = []

    def entry(where: str, e, args: dict, dims: dict, kernel_level: bool = False) -> None:
        if not isinstance(e, dict):
            out.append(f"{where}: not an object")
            return
        route = e.get("route")
        if route not in ROUTES:
            out.append(f"{where}: route {route!r} not one of {ROUTES}")
        if route in ("kernel", "library"):
            name, repo = e.get("name"), e.get("repo")
            if not isinstance(name, str) or not name or "/" in name or ".." in name:
                out.append(f"{where}: name {name!r} is not a symbol")
            if not isinstance(repo, str) or not repo or Path(repo).is_absolute() or ".." in repo:
                out.append(f"{where}: repo {repo!r} is not a directory name")
            elif not (Path(repos) / repo).is_dir():
                out.append(f"{where}: repo {repo!r} not found under {repos}")
            if e.get("dsl") not in dsls:
                out.append(f"{where}: dsl {e.get('dsl')!r} not one of {sorted(dsls)}")
        if e.get("family") is not None and e["family"] not in FAMILIES:
            out.append(f"{where}: family {e['family']!r} not one of {FAMILIES}")
        cost = e.get("cost")
        if cost is not None:
            if not isinstance(cost, dict) or "bytes" not in cost:
                out.append(f"{where}.cost: needs {{'bytes': ..., 'flops': ...}}")
            else:
                for what in ("bytes", "flops"):
                    if what not in cost:
                        continue
                    used = formula.names(cost[what])
                    if used is None:
                        out.append(f"{where}.cost.{what}: not an arithmetic expression")
                    elif used - set(dims):
                        out.append(
                            f"{where}.cost.{what}: {sorted(used - set(dims))} are not dims of the op"
                        )
        if kernel_level:
            return
        if e.get("token_arg") is not None and e["token_arg"] not in args:
            out.append(f"{where}: token_arg {e['token_arg']!r} is not an argument")
        if "token_axis" in e and type(e["token_axis"]) is not int:
            out.append(f"{where}: token_axis must be an integer")
        specs = e.get("gemm")
        for i, g in enumerate(specs if isinstance(specs, list) else [specs] if specs else []):
            gw = f"{where}.gemm[{i}]"
            if not isinstance(g, dict):
                out.append(f"{gw}: not an object")
                continue
            for role in ("activation", "weight"):
                if g.get(role) not in args:
                    out.append(f"{gw}: {role} {g.get(role)!r} is not an argument")
            for role in ("output", "experts"):
                if g.get(role) is not None and g[role] not in args:
                    out.append(f"{gw}: {role} {g[role]!r} is not an argument")
            side = g.get("weight_side", [])
            if not isinstance(side, list) or any(a not in args for a in side):
                out.append(f"{gw}: weight_side must list arguments")
            rank = len(args.get(g.get("weight")) or [])
            axis = g.get("n_axis")
            if axis is not None and not (type(axis) is int and 0 <= axis < rank):
                out.append(f"{gw}: n_axis {axis!r} is not an axis of the weight (rank {rank})")
            if "act_per_stored" in g and not _pos_int(g["act_per_stored"]):
                out.append(f"{gw}: act_per_stored must be a positive integer")
            if "grouped" in g and type(g["grouped"]) is not bool:
                out.append(f"{gw}: grouped must be true or false")
            if g.get("rows") is not None and g["rows"] not in ROWS:
                out.append(f"{gw}: rows {g['rows']!r} not one of {ROWS}")
        kernels = e.get("kernels")
        if kernels is not None:
            if not isinstance(kernels, dict):
                out.append(f"{where}.kernels: not an object")
            else:
                for sym, k in kernels.items():
                    entry(f"{where}.kernels[{sym!r}]", k, args, dims, kernel_level=True)

    for key, e in names["ops"].items():
        if key not in listing:
            out.append(f"{key!r}: not a key of ops.json")
            continue
        args = {a["name"]: a.get("shape") for a in listing[key].get("args", [])}
        entry(repr(key), e, args, listing[key].get("dims") or {})
    model = names.get("model", {})
    if not isinstance(model, dict):
        out.append("model: not an object")
    else:
        for fact in ("topk", "experts"):
            if model.get(fact) is not None and not _pos_int(model[fact]):
                out.append(f"model.{fact}: {model[fact]!r} is not a positive integer")
    return out


def keep_valid(names, listing: dict, repos) -> tuple[dict | None, list[str]]:
    """The answer minus its invalid entries, and the problems; None if not an answer."""
    ops = names.get("ops") if isinstance(names, dict) else None
    whole = validate({**names, "ops": {}} if isinstance(ops, dict) else names, listing, repos)
    if whole:
        return None, whole
    problems = []
    for key in list(ops):
        wrong = validate({"ops": {key: ops[key]}}, listing, repos)
        if wrong:
            problems += wrong
            del ops[key]
    return names, problems


def refuted(names: dict, listing: dict, calib: dict | None, noise: float) -> list[tuple]:
    """(key, kernel group or None, evidence) per stated cost a measured call contradicts."""
    bw = (calib or {}).get("bandwidth_gbs")
    if not bw:
        return []
    peaks = {k: v for k, v in (calib.get("matmul_tflops") or {}).items() if v}
    out = []

    def check(key, sub, cost, shapes):
        for shape in shapes:
            us = shape["us_per_call"]
            if sub is not None:
                us = sum(t for s, t in shape.get("kernels_per_call", {}).items() if sub in s)
            moved = formula.evaluate(cost.get("bytes", 0), shape["dims"])
            flops = formula.evaluate(cost.get("flops", 0), shape["dims"])
            peak = peaks.get(shape.get("dtype")) or max(peaks.values(), default=0)
            if moved is None or flops is None:
                continue
            bound = max(moved / (bw * 1e3), flops / (peak * 1e6) if peak else 0.0)
            if us and bound > us * (1 + noise):
                dims = ", ".join(f"{k}={v}" for k, v in list(shape["dims"].items())[:4])
                out.append((key, sub, f"bound {bound:.0f} us > measured {us:.0f} us at {dims}"))
                return

    for key, e in (names.get("ops") or {}).items():
        shapes = (listing.get(key) or {}).get("shapes") or []
        if isinstance(e, dict) and isinstance(e.get("cost"), dict):
            check(key, None, e["cost"], shapes)
        for sub, k in (e.get("kernels") or {}).items() if isinstance(e, dict) else ():
            if isinstance(k, dict) and isinstance(k.get("cost"), dict):
                check(key, sub, k["cost"], shapes)
    return out


def drop_costs(names: dict, which: list[tuple]) -> None:
    """Remove refuted costs in place."""
    for key, sub, _ in which:
        entry = names["ops"][key]
        (entry["kernels"][sub] if sub is not None else entry).pop("cost", None)
