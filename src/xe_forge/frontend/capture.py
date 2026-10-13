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
import copy
import importlib
import json
import os
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path

from xe_forge.frontend import adapters, canonical, ir, naming, prompt
from xe_forge.frontend.torch_trace import parse_traces

# Environment worth recording for reproducibility. Anything that could carry a credential
# is left out by name, whatever its prefix.
_ENV_PREFIXES = tuple(
    os.environ.get(
        "XE_FORGE_CAPTURE_ENV",
        "VLLM_,SGLANG_,SGL_,OV_,ONEDNN_,DNNL_,ONEAPI_,SYCL_,UR_,ZE_,IGC_,CCL_,TORCH_,PYTORCH_,HF_HUB_",
    ).split(",")
)  # environment that shapes a capture, recorded with it
_SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH")


def _env_subset() -> dict:
    return {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith(_ENV_PREFIXES) and not any(w in k for w in _SECRET_WORDS)
    }


def build(raw_dir: Path, out: Path, names: dict | None = None) -> ir.CaptureRun:
    """``raw/`` (run.json + trace) -> capture.json. ``names`` is the naming agent's answer
    (``names.json``): each op gets the entry for its key, and the run the model's facts."""
    run = json.loads((raw_dir / "run.json").read_text())
    adapter = adapters.get(run["framework"])
    parse = getattr(adapter, "parse", None)
    traces = sorted(raw_dir.glob("trace*.json*"))
    if parse is not None:
        summary = parse(raw_dir)
    elif traces:
        summary = parse_traces(traces)
    else:
        raise SystemExit(f"no trace in {raw_dir}")

    if not summary.launches:
        raise SystemExit(f"capture: no device kernel in {raw_dir}; nothing ran on the device")
    run["total_device_us"] = round(summary.total_us, 1)
    run["attributed_device_us"] = round(summary.attributed_us, 1)
    run["attributed_pct"] = round(summary.attributed_pct, 2)
    run["kernel_invocations"] = summary.launches
    run["raw"] = [
        os.path.relpath(p, out.parent) for p in sorted(raw_dir.iterdir()) if p.name != "run.json"
    ]
    _annotate_schemas(summary.ops, raw_dir / "schemas.json")
    key = getattr(adapter, "name_key", naming.key)  # stable across a patched build
    entries = (names or {}).get("ops") or {}
    ops = []
    for op in summary.ops:
        op.framework_meta["name_key"] = key(op)
        if key(op) in entries:
            op.framework_meta["named"] = entries[key(op)]
        ops.extend(_split(op))
    summary.ops = ops
    if names is not None:
        run["model_facts"] = names.get("model") or {}
    capture = ir.CaptureRun(run=run, ops=summary.ops)
    capture.workloads = canonical.workloads(capture.ops, summary.total_us, naming.name_op)
    ir.dump(capture, out)
    return capture


def _split(op) -> list:
    """An op whose kernels live in different places (a Python op running several Triton
    kernels) is one row per kernel group the naming agent names (``named["kernels"]``:
    symbol substring -> entry): same arguments and calls, each group's own time."""
    named = op.framework_meta.get("named") or {}
    groups = named.get("kernels")
    if not isinstance(groups, dict) or not groups:
        return [op]
    by: dict = {}
    for sym, us in op.kernel_symbols.items():
        sub = next((g for g in groups if g in sym), None)
        by.setdefault(sub, {})[sym] = us
    parts = []
    for sub, syms in by.items():
        part = copy.deepcopy(op)
        part.kernel_symbols = syms
        part.device_us = round(sum(syms.values()), 3)
        entry = {k: v for k, v in named.items() if k != "kernels"}
        if sub is not None and isinstance(groups[sub], dict):
            entry.update(groups[sub])
        part.framework_meta["named"] = entry
        parts.append(part)
    return parts


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


def workload_args(argv: list[str]) -> list[str]:
    """The arguments that decide what was captured: all but where it was written to."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a in ("-o", "--output", "--names", "--reparse"):
            skip = True
        elif not a.startswith(("--output=", "--names=", "--reparse=")):
            out.append(a)
    return out


_NO_FIT = re.compile(
    r"OUT_OF_RESOURCES|OUT_OF_DEVICE_MEMORY|out of memory|OutOfMemory"
    r"|not enough (KV cache|memory)|KV cache is needed|No available memory for the cache",
    re.I,
)  # what runtimes say when the model does not fit at this workload


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
    parser.add_argument("--names", type=Path, help="the naming agent's names.json")
    parser.add_argument(
        "--import-ops", nargs="*", default=[], metavar="MODULE",
        help="modules registering the providers' torch ops, so their schemas resolve",
    )  # fmt: skip
    prompt.add_arguments(parser)
    if known.framework in adapters.ADAPTERS:
        adapters.get(known.framework).add_arguments(parser)
    args = parser.parse_args(argv)

    out = args.output.resolve()
    names = json.loads(args.names.read_text()) if args.names else None
    for module in args.import_ops:  # providers register their ops on import
        importlib.import_module(module)
    if args.reparse:
        capture = build(args.reparse.resolve(), out, names)
    else:
        raw_dir = out.parent / "raw"
        for stale in raw_dir.glob("trace*"):  # an earlier capture's traces are not this one's
            stale.unlink()
        try:
            run = adapters.get(args.framework).capture(args, raw_dir)
        except Exception as exc:
            if not _NO_FIT.search(str(exc)):
                raise
            line = next((ln for ln in str(exc).splitlines() if _NO_FIT.search(ln)), str(exc))
            print(f"capture: does not fit on the device at this workload: {line.strip()}",
                  file=sys.stderr, flush=True)  # fmt: skip
            os._exit(3)  # a runtime can abort in teardown after a device OOM; exit with the reason
        runtime = run.setdefault("runtime", {})
        runtime.update(prompt=prompt.describe(args.prompt_file), inputs="text")
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        model = str((run.get("model") or {}).get("name", "model")).split("/")[-1]
        run.update(
            id=f"{model}-{args.framework}-{stamp}",
            capture_args=workload_args(argv),
            command=shlex.join([sys.executable, "-m", "xe_forge.frontend.capture", *argv]),
            env=_env_subset(),
        )
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "run.json").write_text(json.dumps(run, indent=1) + "\n")
        capture = build(raw_dir, out, names)

    r = capture.run
    print(
        f"capture: {len(capture.ops)} ops, {len(capture.workloads)} workloads, "
        f"{r['kernel_invocations']} invocations, {r['total_device_us'] / 1e3:.1f} ms device time, "
        f"{r['attributed_pct']}% attributed -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
