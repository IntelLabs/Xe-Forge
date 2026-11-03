"""``bound``: each workload's efficiency against the stricter known bound, computed, never typed.

The arithmetic ``analyze`` decides on, as a tool an agent can test its claims with: the GEMM
roofline from shapes and the calibrated peaks, the counters' speed-of-light, and any cost
names.json states. Exit 1 when the capture has no workloads.
"""

from __future__ import annotations

import json
from pathlib import Path


def run(args) -> int:
    from xe_forge.frontend import analyze, ir

    cap = ir.load(args.capture)
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
    return 0 if cap.workloads else 1
