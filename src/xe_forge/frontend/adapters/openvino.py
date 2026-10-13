"""OpenVINO: per-primitive device time from PERF_COUNT, for an LLM run by optimum-intel.

The model (an exported IR directory, or an HF id an agent exported once per
``--weight-format``, see ``prepare``) is compiled on the GPU with ``PERF_COUNT``.
``generate`` runs one prefill and ``--output-len`` decode steps; a thin proxy around the
model's infer request reads ``get_profiling_info()`` after every step. The plugin reports
each node's *mean* time over the runs it executed in, so a step's own time is recovered as
``n * mean_n - (n - 1) * mean_(n-1)``. The compiled runtime model gives each executed node
its port shapes and precisions.

OpenVINO reports no per-call shapes -- the runtime model's are dynamic -- so each row
carries the call's token count and context length, and ``-1`` stays where a dim is
dynamic: the agent maps model-level sizes to the kernel's arguments.

Each row's kernel symbol is the implementation profiling reports, or, where it reports
``undef``, the runtime model's ``primitiveType``. Which source that is -- an OpenCL kernel
in the plugin, a oneDNN call -- is the naming agent's to establish (``name_key``).
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

from xe_forge.frontend import CACHE, prompt
from xe_forge.frontend.ir import CapturedOp, TensorArg
from xe_forge.frontend.model_meta import stored
from xe_forge.frontend.torch_trace import TraceSummary

_DTYPES = {"f16": "float16", "bf16": "bfloat16", "f32": "float32", "i64": "int64",
           "i32": "int32", "i8": "int8", "u8": "uint8", "boolean": "bool",
           "u4": "uint4", "i4": "int4", "nf4": "nf4", "f4e2m1": "float4_e2m1",
           "f8e4m3": "float8_e4m3fn", "f8e5m2": "float8_e5m2", "f8e8m0": "float8_e8m0fnu"}  # fmt: skip


def add_arguments(parser) -> None:
    parser.add_argument("--model", help="exported IR directory or HF id")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=32, help="decode steps profiled")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="GPU")
    parser.add_argument(
        "--weight-format",
        default="fp16",
        help="weight format of an HF id's export, as OpenVINO names it (int4, fp8, ...); "
        "an agent exports it once and it is cached",
    )
    parser.add_argument("--ov-xml", default="openvino_model.xml", help="the IR's model file")
    parser.add_argument(
        "--site-pattern", default=r"\.\d+(?=[./])",
        help="regex folding per-layer parts of node names into one call site",
    )  # fmt: skip


class _Recorder:
    """Stands in for the model's InferRequest; logs each step's own profiling info."""

    def __init__(self, request, steps: list, shapes: list):
        self._request, self._steps, self._total, self._runs = request, steps, {}, {}
        self._shapes = shapes

    def __getattr__(self, name):
        return getattr(self._request, name)

    def wait(self):
        self._request.wait()
        rows = []
        for p in self._request.get_profiling_info():
            node, ran = p.node_name, "EXECUTED" in str(p.status)
            # The plugin reports a node's mean over the runs it executed in, in whole us.
            # A step's own time is the change in mean * runs; rounding leaves +-n us per
            # step, which cancels in any sum over steps -- so negatives are kept, not dropped.
            self._runs[node] = self._runs.get(node, 0) + ran
            total = p.real_time.total_seconds() * 1e6 * self._runs[node]
            us = total - self._total.get(node, 0.0) if ran else 0.0
            self._total[node] = total
            rows.append({"node": node, "type": p.node_type, "exec": p.exec_type,
                         "us": us, "status": str(p.status)})  # fmt: skip
        self._steps.append(rows)
        try:  # what this step fed the model: its rows and length, measured, not assumed
            self._shapes.append(list(self._request.get_input_tensor(0).shape))
        except Exception:
            self._shapes.append(None)


def ir_dir(args) -> Path:
    """Where the IR for an HF id and weight format lives, once exported."""
    name = f"{args.model.replace('/', '--')}--{args.weight_format}"
    return CACHE / "ov-models" / name


def prepare(args) -> tuple[Path, str, dict] | None:
    """What the model needs before capture -- an exported IR, and the agent's task to make
    it -- or None when it has one. Export is per model (formats, calibration, shapes), so an
    agent does it rather than code here."""
    out = Path(args.model) if Path(args.model).is_dir() else ir_dir(args)
    if (out / args.ov_xml).is_file():
        return None
    context = {"model": args.model, "out": out, "fmt": args.weight_format}
    return out, "frontend-prepare-openvino.md.j2", context


def capture(args, raw_dir: Path) -> dict:
    if not args.model:
        raise SystemExit("--framework openvino needs --model")
    import openvino as ov
    import torch
    from optimum.intel import OVModelForCausalLM

    path = Path(args.model) if Path(args.model).is_dir() else ir_dir(args)
    if not (path / args.ov_xml).is_file():
        raise SystemExit(f"no OpenVINO IR at {path}; `xe-forge run` has an agent export it")
    model = OVModelForCausalLM.from_pretrained(
        path, device=args.device, ov_config={"PERF_COUNT": "YES"},
        trust_remote_code=args.trust_remote_code,
    )  # fmt: skip
    model.compile()
    from transformers import AutoTokenizer

    source = args.model if not Path(args.model).is_dir() else path
    tok = AutoTokenizer.from_pretrained(source, trust_remote_code=args.trust_remote_code)
    ids = prompt.token_ids(prompt.hf_vocab(tok), args.batch, args.input_len, args.prompt_file)
    tokens = torch.tensor(ids, dtype=torch.int64)
    steps: list = []
    shapes: list = []
    model.request = _Recorder(
        model.request, steps, shapes
    )  # from the first infer: means accumulate
    warm_n = args.warmup_tokens
    model.generate(tokens, max_new_tokens=warm_n, min_new_tokens=warm_n, do_sample=False)
    warm = len(steps)
    model.generate(
        tokens, max_new_tokens=args.output_len, min_new_tokens=args.output_len, do_sample=False
    )

    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / "ov_profile.jsonl", "w") as f:
        context = 0
        for i, rows in enumerate(steps[warm:]):
            shape = shapes[warm + i]
            if shape:  # [batch, tokens]: rows of a FullyConnected, context per sequence
                m, context = math.prod(shape), context + shape[-1]
            else:  # not readable: one prefill, then one token per sequence per step
                m = args.batch * (args.input_len if i == 0 else 1)
                context = args.input_len + i
            f.write(json.dumps({"tokens": m, "context": context, "rows": rows}) + "\n")

    def ports(ps):
        return [
            {"shape": [d.get_length() if d.is_static else -1 for d in p.get_partial_shape()],
             "dtype": p.get_element_type().get_type_name()}
            for p in ps
        ]  # fmt: skip

    nodes = {}
    for op in model.request.get_compiled_model().get_runtime_model().get_ordered_ops():
        rt = {k: v.astype(str) for k, v in op.get_rt_info().items()}
        nodes[op.get_friendly_name()] = {
            "rt": rt,
            "inputs": ports(op.inputs()),
            "outputs": ports(op.outputs()),
        }
    (raw_dir / "runtime_model.json").write_text(json.dumps(nodes, indent=1) + "\n")

    return {
        "framework": "openvino",
        "framework_version": ov.__version__,
        "model": {"name": args.model, "config": stored(model.config.to_dict())},
        "runtime": {"device": args.device, "batch": args.batch, "input_len": args.input_len,
                    "output_len": args.output_len, "weight_format": args.weight_format, "site_pattern": args.site_pattern, "stateful": bool(getattr(model, "stateful", False))},
        "device": {"type": args.device, "name": ov.Core().get_property(args.device, "FULL_DEVICE_NAME")},
    }  # fmt: skip


def parse(raw_dir: Path) -> TraceSummary:
    nodes = json.loads((raw_dir / "runtime_model.json").read_text())
    runtime = json.loads((raw_dir / "run.json").read_text()).get("runtime") or {}
    fold = re.compile(runtime.get("site_pattern") or r"\.\d+(?=[./])")
    rows: dict[tuple, CapturedOp] = {}
    total = 0.0
    launches = 0
    kernels: dict[str, float] = defaultdict(float)
    steps = [json.loads(line) for line in (raw_dir / "ov_profile.jsonl").read_text().splitlines()]
    # A step's time is recovered from a running mean in whole us: it may be off by about one
    # us per run, never by more. Anything below that is not a running mean for this model.
    floor = -2.0 * (len(steps) + 16)
    for step in steps:
        for r in step["rows"]:
            if "EXECUTED" not in r["status"]:
                continue  # fused away or not run: no time of its own
            # Profiling names a node "<type>:<name>"; the runtime model names it "<name>".
            r["node"] = re.sub(r"^[a-z0-9_]+:(?!:)", "", r["node"])
            if r["us"] < floor:
                raise SystemExit(
                    f"OpenVINO profiling: {r['node']} has {r['us']:.0f} us in a step; the "
                    "plugin's counters are not a running mean per run for this model"
                )
            if r["exec"] in ("", "undef"):  # the runtime model knows what profiling does not
                r["exec"] = (nodes.get(r["node"]) or {}).get("rt", {}).get(
                    "primitiveType"
                ) or "undef"
            total += r["us"]
            launches += 1
            kernels[r["exec"]] += r["us"]
            site = fold.sub(".*", r["node"])
            key = (r["type"], r["exec"], site, step["tokens"], step["context"])
            op = rows.get(key)
            if op is None:
                ports = (nodes.get(r["node"]) or {}).get("inputs", [])
                op = rows[key] = CapturedOp(
                    id="",
                    framework_op=f"ov::{r['type']}",
                    args=[
                        TensorArg(p["shape"], _DTYPES.get(p["dtype"], p["dtype"])) for p in ports
                    ],
                    call_site=site,
                    framework_meta={"tokens": step["tokens"], "context": step["context"]},
                )
                node = nodes.get(r["node"]) or {}
                outs = node.get("outputs") or []
                if r["type"] == "FullyConnected" and ports and outs:
                    # A matmul against weights the runtime model keeps internal: K from the
                    # input, N from the output, M is the call's token count.
                    op.framework_meta["gemm"] = {
                        "K": ports[0]["shape"][-1],
                        "N": outs[0]["shape"][-1],
                    }
            op.calls += 1
            op.device_us += r["us"]
            op.kernel_symbols[r["exec"]] = op.kernel_symbols.get(r["exec"], 0.0) + r["us"]
    ops = sorted(rows.values(), key=lambda o: -o.device_us)
    for i, op in enumerate(ops):
        op.id = f"op-{i:04d}"
        op.device_us = round(op.device_us, 3)
    return TraceSummary(
        ops=ops, total_us=total, attributed_us=total, kernels=dict(kernels), launches=launches
    )


def name_key(op: CapturedOp) -> str:
    """names.json key: the node type and the implementation the plugin picked for it (one
    type runs different kernels at different shapes); a patched kernel keeps its name."""
    impl = max(op.kernel_symbols, key=op.kernel_symbols.get) if op.kernel_symbols else ""
    return f"{op.framework_op}|{impl}"
