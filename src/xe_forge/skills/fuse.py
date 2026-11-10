"""``fuse check``: the gate on the fusions an agent proposed after the kernels were optimized.

<fusions.json>: {"fusions": [{"name", "repo", "dsl", "ops": ["<capture op id>", ...],
"anchor": "<one of them>", "reference": "<the chain as a _pytorch.py>",
"integration_repos": [...], "why"}]}. Each op a captured one, two or more; the anchor's
workload has the shapes the fusion is specified at, and the reference's ``get_inputs`` and
``get_init_inputs`` between them take exactly its dims and ``dtype``, as a spec passes them
(``core/reference_workload``); the repositories exist (relative paths are read
from fusions.json's directory). Ranks the fusions by the time their ops took, each divided
by the speedup its kernel already won (``--results``). Exit 0 passes.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


def _params(tree: ast.Module, name: str) -> set[str] | None:
    """A top-level function's keyword parameters; None if absent, {"**"} if it takes any."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            a = node.args
            return {"**"} if a.kwarg else {p.arg for p in a.args + a.kwonlyargs}
    return None


def _reference(path: Path, dims: set[str]) -> list[str]:
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError) as exc:
        return [f"reference {path}: {exc}"]
    bad = []
    if not any(isinstance(n, ast.ClassDef) and n.name == "Model" for n in tree.body):
        bad.append(f"reference {path} defines no Model")
    got = {fn: _params(tree, fn) for fn in ("get_inputs", "get_init_inputs")}
    bad += [f"reference {path} defines no {fn}" for fn, p in got.items() if p is None]
    params = [p for p in got.values() if p is not None]
    if params and not any("**" in p for p in params):
        taken, wanted = set().union(*params), dims | {"dtype"}
        if taken != wanted:
            bad.append(f"its builders take {sorted(taken)}, not {sorted(wanted)}: the anchor's "
                       "dims and the dtype")  # fmt: skip
    return bad


def wins(summary: Path | None) -> dict[str, float]:
    """{kernel: speedup} of the kernels optimized before, from results/summary.json."""
    rows = json.loads(summary.read_text()) if summary and summary.is_file() else []
    return {
        r["kernel"]: r["speedup"]
        for r in rows
        if str(r.get("outcome", "")).startswith("win") and r.get("speedup")
    }


def check(answer: dict, capture, won: dict[str, float], base: Path) -> tuple[list[str], list[dict]]:
    """(problems, the passing fusions with their chain time, costliest first)."""
    from xe_forge.frontend.canonical import resolve_dims
    from xe_forge.models import DSL

    ops = {o.id: o for o in capture.ops}
    owner = {i: w for w in capture.workloads for i in w.ops}
    total = capture.run.get("total_device_us") or sum(o.device_us for o in ops.values()) or 1.0
    items = answer.get("fusions")
    if not isinstance(items, list) or not items:
        return ["no 'fusions' list"], []
    problems, passed, seen = [], [], set()
    for f in items:
        name = f.get("name") if isinstance(f, dict) else None
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_]\w*", name) or name in seen:
            problems.append(f"{name!r}: each fusion needs its own identifier as name")
            continue
        seen.add(name)
        chain, bad = f.get("ops") or [], []
        if len(chain) < 2:
            bad.append("a fusion has two ops or more")
        bad += [f"{i} is not a captured op" for i in chain if i not in ops]
        anchor = owner.get(f.get("anchor"))
        if f.get("anchor") not in chain or anchor is None or not anchor.dims:
            bad.append("the anchor is one of its ops, in a workload with dims")
        for r in [f.get("repo"), *(f.get("integration_repos") or [])]:
            if not r or not (base / r).is_dir():
                bad.append(f"repository {r!r} not found")
        if f.get("dsl") not in {d.value for d in DSL}:
            bad.append(f"dsl {f.get('dsl')!r} is not one of {[d.value for d in DSL]}")
        if anchor is not None and anchor.dims:
            value = next(iter(next(iter(anchor.var_axes.values()), {})), None)
            dims = set(resolve_dims(anchor, int(value) if value is not None else None))
            bad += _reference(base / (f.get("reference") or "-"), dims)
        if bad:
            problems += [f"{name}: {b}" for b in bad]
            continue
        us = sum(
            ops[i].device_us / won.get(owner[i].naming.name if i in owner else "", 1.0)
            for i in chain
        )
        passed.append({**f, "chain_us": round(us, 1), "share": round(us / total, 6)})
    passed.sort(key=lambda f: -f["chain_us"])
    return problems, passed


def run(args) -> int:
    from xe_forge.frontend import ir

    path = Path(args.fusions)
    try:
        answer = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        print(f"fuse check: PROBLEM cannot read {path}: {exc}")
        return 1
    answer = answer if isinstance(answer, dict) else {}
    won = wins(Path(args.results) if args.results else None)
    problems, passed = check(answer, ir.load(args.capture), won, path.parent)
    for f in passed:
        print(f"fuse check: {f['name']}: {len(f['ops'])} ops, {f['chain_us']} us "
              f"({f['share']:.2%} of device time) after the kernels' wins")  # fmt: skip
    for p in problems:
        print(f"fuse check: PROBLEM {p}")
    print("fuse check: " + ("FAIL" if problems else "PASS"))
    return 1 if problems else 0
