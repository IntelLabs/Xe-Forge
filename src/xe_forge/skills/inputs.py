"""``inputs check``: the gate on the trial shapes an agent chose for a kernel. A kernel
serves configurations -- the captured workloads, each a different set of parameters (a
GEMM's N and K, a dtype, a flag) -- and each ran at several token counts. Every
configuration must have a shape of its own: the kernel has to hold at all of them. Within
one configuration the agent may merge the token counts that take the same path, naming
one to stand for the others.

<inputs.json>: {"<kernel name>": {"shapes": {"<variant>": {"stands_for": ["<variant>", ...],
"weight": <w>}, ...}, "why": "..."}} for every enabled kernel of the manifest. Each shape a
captured one, in its own group; a group within one configuration; the groups disjoint and
together every shape the capture ran; weights positive, summing to 1. Prints each group's
captured share beside its weight. Exit 0 passes.
"""

from __future__ import annotations

import json
from pathlib import Path


def shares(record: dict) -> dict[str, dict]:
    """{variant: {calls, device_us, share}} of a kernel's record, from its capture."""
    cap = json.loads(Path(record["dataset"]).read_text())
    by_id = {w["workload_id"]: w for w in cap.get("workloads") or []}
    out = {}
    for name, ref in record["variants"].items():
        wid, _, point = ref.partition("@")
        w = by_id.get(wid) or {}
        if point:
            axis, _, value = point.partition("=")
            h = ((w.get("var_axes") or {}).get(axis) or {}).get(value) or {}
            calls, us = h.get("calls", 0), h.get("device_us", 0.0)
        else:
            calls, us = w.get("calls", 0), w.get("device_us", 0.0)
        out[name] = {"calls": calls, "device_us": us}
    total = sum(v["device_us"] for v in out.values()) or 1.0
    for v in out.values():
        v["share"] = v["device_us"] / total
    return out


def _number(w) -> bool:
    return isinstance(w, int | float) and not isinstance(w, bool) and w > 0


def check(answer: dict, manifest: dict) -> tuple[dict[str, list[str]], list[str]]:
    """({kernel: problems} for the kernels that fail, report lines)."""
    problems: dict[str, list[str]] = {}
    lines: list[str] = []
    for entry in manifest["kernels"]:
        if not entry.get("enabled") or not entry.get("record"):
            continue
        name = entry["name"]
        bad = problems.setdefault(name, [])
        record = json.loads(Path(entry["record"]).read_text())
        captured = shares(record)
        config = {v: ref.partition("@")[0] for v, ref in record["variants"].items()}
        got = answer.get(name) if isinstance(answer.get(name), dict) else {}
        shapes = got.get("shapes")
        if not isinstance(shapes, dict) or not shapes:
            bad.append("no shapes")
            continue
        seen: dict[str, str] = {}
        total = 0.0
        lines.append(f"{name}: {len(shapes)} shapes stand for its {len(captured)} captured")
        for rep, g in shapes.items():
            g = g if isinstance(g, dict) else {}
            group = g.get("stands_for") if isinstance(g.get("stands_for"), list) else []
            if rep not in group:
                bad.append(f"{rep!r} must be in its own stands_for")
            if len({config[v] for v in group if v in config}) > 1:
                bad.append(f"{rep!r} stands for shapes of different configurations; each "
                           "configuration needs its own")  # fmt: skip
            for v in group:
                if v not in captured:
                    bad.append(f"{v!r} is not a shape the capture ran")
                elif v in seen:
                    bad.append(f"{v!r} stands with both {seen[v]!r} and {rep!r}")
                seen.setdefault(v, rep)
            if not _number(g.get("weight")):
                bad.append(f"weight of {rep!r} must be a positive number")
            else:
                total += g["weight"]
            share = sum(captured[v]["share"] for v in group if v in captured)
            lines.append(f"  {rep:26s} weight {g.get('weight')}  captured {100 * share:5.1f}% "
                         f"over {len(group)} shapes")  # fmt: skip
        missing = [v for v in captured if v not in seen]
        if missing:
            bad.append(f"no shape stands for {', '.join(missing)}")
        if abs(total - 1) > 1e-3:
            bad.append(f"weights sum to {total:.4f}, not 1")
        if not str(got.get("why") or "").strip():
            bad.append('say why in "why"')
    return {k: v for k, v in problems.items() if v}, lines


def weights(answer: dict, kernels: list[str]) -> dict[str, dict[str, float]]:
    """{kernel: {shape: weight}} for ``kernels`` (ones that passed)."""
    return {k: {rep: float(g["weight"]) for rep, g in answer[k]["shapes"].items()} for k in kernels}


def run(args) -> int:
    from xe_forge.frontend import manifest

    try:
        answer = json.loads(Path(args.inputs).read_text())
    except (OSError, ValueError) as exc:
        print(f"inputs check: PROBLEM cannot read {args.inputs}: {exc}")
        return 1
    answer = answer if isinstance(answer, dict) else {}
    problems, lines = check(answer, manifest.load(args.manifest))
    for line in lines:
        print(line)
    for kernel, bad in problems.items():
        for p in bad:
            print(f"inputs check: PROBLEM {kernel}: {p}")
    print("inputs check: " + ("FAIL" if problems else "PASS"))
    return 1 if problems else 0
