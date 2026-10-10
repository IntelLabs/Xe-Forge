"""Capture entry point, run under the framework's own interpreter.

    python -m xe_forge.frontend.capture --framework vllm --model <id> -o run/capture.json

``xe-forge capture`` launches this in a subprocess with ``--framework-python`` (default:
the current interpreter) and this package's ``src`` on ``PYTHONPATH``, so nothing is
installed into the framework's environment. The adapter writes ``raw/`` beside the output
and returns the run metadata; ``build`` turns ``raw/`` into ``capture.json`` and can be
re-run on its own (``--reparse``) without running the framework again.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path

from xe_forge.frontend import adapters, canonical, ir
from xe_forge.frontend.naming import name_op
from xe_forge.frontend.torch_trace import parse_trace

# Environment worth recording for reproducibility. Anything that could carry a credential
# is left out by name, whatever its prefix.
_ENV_PREFIXES = ("VLLM_", "ONEAPI_", "SYCL_", "ZE_", "IGC_", "TORCH_", "PYTORCH_", "HF_HUB_")
_SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH")


def _env_subset() -> dict:
    return {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith(_ENV_PREFIXES) and not any(w in k for w in _SECRET_WORDS)
    }


def build(raw_dir: Path, out: Path) -> ir.CaptureRun:
    """``raw/`` (run.json + trace) -> capture.json."""
    run = json.loads((raw_dir / "run.json").read_text())
    adapter = adapters.get(run["framework"])
    parse = getattr(adapter, "parse", None)
    traces = sorted(raw_dir.glob("trace*.json*"))
    if parse is not None:
        summary = parse(raw_dir)
    elif traces:
        summary = parse_trace(traces[0])
    else:
        raise SystemExit(f"no trace in {raw_dir}")

    run["total_device_us"] = round(summary.total_us, 1)
    run["attributed_device_us"] = round(summary.attributed_us, 1)
    run["attributed_pct"] = round(summary.attributed_pct, 2)
    run["kernel_invocations"] = summary.launches
    run["raw"] = [os.path.relpath(p, out.parent) for p in traces]
    _annotate_schemas(summary.ops, raw_dir / "schemas.json")
    capture = ir.CaptureRun(run=run, ops=summary.ops)
    framework = run["framework"]
    capture.workloads = canonical.workloads(
        capture.ops, summary.total_us, lambda op: name_op(op, framework)
    )
    ir.dump(capture, out)
    return capture


def _schema_args(op_name: str, nargs: int) -> list[str] | None:
    """Argument names of the overload of ``ns::op`` taking ``nargs`` arguments."""
    try:
        import torch

        ns, name = op_name.split("::", 1)
        packet = getattr(getattr(torch.ops, ns), name)
        for overload in packet.overloads():
            args = getattr(packet, overload)._schema.arguments
            if len(args) == nargs:
                return [a.name for a in args]
    except Exception:
        return None
    return None


def _annotate_schemas(ops: list[ir.CapturedOp], cache: Path) -> None:
    """Name each op's arguments from its schema. Provider ops are only registered in the
    framework's interpreter, so the names found there are kept in ``raw/schemas.json`` and
    a re-parse elsewhere reads them back."""
    known = json.loads(cache.read_text()) if cache.is_file() else {}
    for op in ops:
        if "::" not in op.framework_op or op.framework_op.startswith("kernel::"):
            continue
        key = f"{op.framework_op}/{len(op.args)}"
        if key not in known:
            names = _schema_args(op.framework_op, len(op.args))
            if names is None:
                continue
            known[key] = names
        op.framework_meta["arg_names"] = known[key]
    if known:
        cache.write_text(json.dumps(known, indent=1, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--framework")
    pre.add_argument("--reparse", type=Path)
    known, _ = pre.parse_known_args(argv)

    parser = argparse.ArgumentParser(prog="xe-forge capture", description=__doc__.splitlines()[0])
    parser.add_argument(
        "--framework", required=known.reparse is None, choices=sorted(adapters.ADAPTERS)
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="capture.json to write")
    parser.add_argument("--reparse", type=Path, help="rebuild from an existing raw/ directory")
    if known.framework in adapters.ADAPTERS:
        adapters.get(known.framework).add_arguments(parser)
    args = parser.parse_args(argv)

    out = args.output.resolve()
    if args.reparse:
        capture = build(args.reparse.resolve(), out)
    else:
        raw_dir = out.parent / "raw"
        run = adapters.get(args.framework).capture(args, raw_dir)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        model = str((run.get("model") or {}).get("name", "model")).split("/")[-1]
        run.update(
            id=f"{model}-{args.framework}-{stamp}",
            command=shlex.join([sys.executable, "-m", "xe_forge.frontend.capture", *argv]),
            env=_env_subset(),
        )
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "run.json").write_text(json.dumps(run, indent=1) + "\n")
        capture = build(raw_dir, out)

    r = capture.run
    print(
        f"capture: {len(capture.ops)} ops, {len(capture.workloads)} workloads, "
        f"{r['kernel_invocations']} invocations, {r['total_device_us'] / 1e3:.1f} ms device time, "
        f"{r['attributed_pct']}% attributed -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
