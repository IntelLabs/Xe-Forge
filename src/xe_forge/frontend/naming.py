"""Captured op -> what the kernel-locator is told to look for.

Which repository carries an op, which kernel or library call implements it, in which
language, and which of its arguments is the activation or the weight is the naming agent's
answer (``frontend-name.md.j2`` -> ``names.json``): it reads the framework and the
repositories instead of matching names against tables kept here. Each op carries its entry
as ``framework_meta["named"]``; this module only turns it into a ``Naming``.

Routes: ``kernel`` (a named kernel in a repository Xe-Forge can edit), ``library`` (a
library primitive -- change the call, not the kernel), ``aten`` (inside PyTorch, no kernel
repository) and ``unnamed`` (not established).
"""

from __future__ import annotations

from xe_forge.frontend.ir import CapturedOp, Naming

_IMPL = {"kernel": "provider", "aten": "aten"}  # a library call: the library the agent names


def key(op: CapturedOp) -> str:
    """What names.json is keyed by: the implementing op, which a patched build keeps."""
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
    op.impl_class = "unknown"  # an ATen op too: its kernel may be in an editable repository
    return Naming("unnamed", _base(op.framework_op), via="not named (no names.json entry)")


ROUTES = ("kernel", "library", "aten", "unnamed")
FAMILIES = ("gemm", "attention", "mla", "indexer", "linear_attention", "ssm", "norm",
            "activation", "moe", "rope", "cache", "sampling", "copy", "elementwise", "other")  # fmt: skip
ROWS = ("token", "token_expert")


def _pos_int(v) -> bool:
    return type(v) is int and v > 0


def validate(names, listing: dict, repos) -> list[str]:
    """What is wrong with a naming answer, checked against the ops it was asked about; an
    empty list means it can be used. The contract is the template's
    (``frontend-name.md.j2``), and this is its check at the boundary."""
    from pathlib import Path

    from xe_forge.models import DSL

    dsls = {d.value for d in DSL}
    if not isinstance(names, dict) or not isinstance(names.get("ops"), dict):
        return ["not {'ops': {...}, 'model': {...}}"]
    out = []

    def entry(where: str, e, args: dict, kernel_level: bool = False) -> None:
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
        for flag in ("paged", "args_complete"):
            if flag in e and type(e[flag]) is not bool:
                out.append(f"{where}: {flag} must be true or false")
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
                    entry(f"{where}.kernels[{sym!r}]", k, args, kernel_level=True)

    for key, e in names["ops"].items():
        if key not in listing:
            out.append(f"{key!r}: not a key of ops.json")
            continue
        args = {a["name"]: a.get("shape") for a in listing[key].get("args", [])}
        entry(repr(key), e, args)
    model = names.get("model", {})
    if not isinstance(model, dict):
        out.append("model: not an object")
    else:
        for fact in ("topk", "experts"):
            if model.get(fact) is not None and not _pos_int(model[fact]):
                out.append(f"model.{fact}: {model[fact]!r} is not a positive integer")
    return out
