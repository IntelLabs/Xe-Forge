"""``bound``: each workload's efficiency against the stricter known bound, computed, never typed.

The arithmetic ``analyze`` decides on, as a tool an agent can test its claims with: the GEMM
roofline from shapes and the calibrated peaks, the counters' speed-of-light, and any cost
names.json states. ``--names`` applies a draft names.json first (its costs and config_args), so
the naming agent sees what its answer makes of the capture before it answers; with ``--ops``
and ``--repos`` it is validated first as the run validates it, each part it drops printed as INVALID.
Exit 1 when the capture has no workloads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def run(args) -> int:
    from xe_forge.frontend import analyze, canonical, ir, naming

    cap = ir.load(args.capture)
    problems = []
    if args.names and Path(args.names).is_file():  # none yet: the capture as it is
        from xe_forge.skills.capture import _precision_problems, apply_names, precision_record

        # A draft applied over earlier names would keep them on every op it leaves out, and
        # their kernel-group splits cannot be undone here.
        if any("named" in o.framework_meta for o in cap.ops):
            print(f"bound: {args.capture} is already named; --names needs a capture parsed "
                  "without --names", file=sys.stderr)  # fmt: skip
            return 2
        names = json.loads(Path(args.names).read_text())
        if args.ops and args.repos:  # what the run keeps of the answer, applied as it will be
            listing = json.loads(Path(args.ops).read_text())
            names, invalid = naming.keep_valid(names, listing, args.repos)
            for p in invalid:
                print(f"bound: INVALID {p}", file=sys.stderr)
            names = names or {}
        cap.ops = apply_names(cap.ops, names)
        cap.workloads = canonical.workloads(cap.ops, cap.run["total_device_us"], naming.name_op)
        # The weights check the run applies to this answer, so the namer sees it first; the
        # model facts reach the bounds below, as they do in the run.
        cap.run = run = {**cap.run, "precision": precision_record(cap),
                         "model_facts": names.get("model") or {}}  # fmt: skip
        problems = _precision_problems(run, named=True, ops=cap.ops)
        kv = (cap.run.get("runtime") or {}).get("kv_cache")
        if kv not in (None, "auto") and not run["precision"]["kv_args"]:
            print(f"bound: NOTE the run declares KV cache {kv}, and no op names its kv_args: "
                  "it goes unchecked", file=sys.stderr)  # fmt: skip
    calib = json.loads(Path(args.calibration).read_text()) if args.calibration else None
    ctx = analyze.context(cap)
    rows = []
    for w in sorted(cap.workloads, key=lambda w: -w.share):
        if w.share < args.min_share:
            continue
        b = analyze.bound(w, calib, ctx)
        rows.append({
            "workload_id": w.workload_id,
            "op": w.framework_meta.get("framework_op"),
            "kernel": w.naming.name, "route": w.naming.route, "repo": w.naming.repo,
            "share": round(w.share, 5), "device_us": w.device_us, "calls": w.calls,
            "eff": None if b is None else round(b[1], 3),
            "recoverable_us": None if b is None else round(b[0], 1),
            "gain": round(w.share * b[0] / w.device_us, 5) if b and w.device_us else None,
        })  # fmt: skip
    if args.json:
        print(json.dumps(rows, indent=1))
    else:
        print(f"bound: {len(rows)} workloads >= {args.min_share:.1%} of device time"
              + ("" if calib else " (no calibration: only counter bounds)"))  # fmt: skip
        for r in rows:
            eff = "unknown" if r["eff"] is None else f"{r['eff']:.2f}"
            gain = "-" if r["gain"] is None else f"{r['gain']:.2%}"
            print(f"  {r['share']:6.2%}  eff {eff:>7}  gain {gain:>7}  {r['kernel']} "
                  f"({r['route']}, {r['repo'] or '-'})  {r['workload_id']}")  # fmt: skip
    if calib:
        from xe_forge.frontend.calibrate import unpeaked_note

        for n in unpeaked_note(ctx):
            print(f"bound: NOTE {n}", file=sys.stderr)
    for p in problems:  # stderr: --json stays parseable
        print(f"bound: PROBLEM {p}", file=sys.stderr)
    return 0 if cap.workloads else 1
