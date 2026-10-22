"""OpenVINO: per-primitive device time from PERF_COUNT for an LLM run by optimum-intel.

PERF_COUNT keeps a per-graph mean over uncounted runs, so prefill and one decode infer are
recorded once and each replayed on a freshly loaded model.
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
        "--phase-runs", type=int, default=5,
        help="replays of each phase (prefill, decode); the first, cold one is excluded",
    )  # fmt: skip
    parser.add_argument(
        "--site-pattern", default=r"\.\d+(?=[./])",
        help="regex folding per-layer parts of node names into one call site",
    )  # fmt: skip


class _Recorder:
    """InferRequest proxy recording each infer's inputs and the variable states before it."""

    def __init__(self, request, calls: list):
        self._request, self._calls = request, calls

    def __getattr__(self, name):
        return getattr(self._request, name)

    def start_async(self, inputs=None, *args, **kwargs):
        import numpy as np

        def value(state):
            try:
                return np.array(state.state.data, copy=True)
            except RuntimeError:  # no shape before a prefill; replayed as a reset
                return None

        self._calls.append({
            "inputs": {k: np.array(v, copy=True) for k, v in (inputs or {}).items()},
            "states": {s.name: value(s) for s in self._request.query_state()},
        })  # fmt: skip
        return self._request.start_async(inputs, *args, **kwargs)


def _phase(request, call: dict, runs: int, tensor=None) -> tuple[list[dict], float]:
    """Replay one infer ``runs`` times; warm us = (runs * mean - first) / (runs - 1)."""
    import statistics
    import time

    if tensor is None:
        from openvino import Tensor as tensor

    def once() -> float:
        for state in request.query_state():  # an infer advances the states: restore them
            if call["states"].get(state.name) is None:
                state.reset()
            else:
                state.state = tensor(call["states"][state.name])
        start = time.perf_counter()
        request.infer(call["inputs"])
        return (time.perf_counter() - start) * 1e6

    once()
    first = {p.node_name: p.real_time.total_seconds() * 1e6 for p in request.get_profiling_info()}
    walls = [once() for _ in range(max(1, runs - 1))]
    rows = []
    for p in request.get_profiling_info():
        mean, one = p.real_time.total_seconds() * 1e6, first.get(p.node_name, 0.0)
        us = (runs * mean - one) / (runs - 1) if runs > 1 else mean
        rows.append({"node": p.node_name, "type": p.node_type, "exec": p.exec_type,
                     "us": us, "status": str(p.status)})  # fmt: skip
    return rows, statistics.median(walls)


def const_bytes(model) -> dict[str, int]:
    """Bytes each node reads from constant-only input subgraphs (a weight and its decompression)."""
    const: dict = {}  # node name -> constant-subgraph bytes, or None
    out = {}
    for op in model.get_ordered_ops():
        name, kind = op.get_friendly_name(), op.get_type_name()
        sources = [
            op.input_value(i).get_node().get_friendly_name() for i in range(op.get_input_size())
        ]
        if kind == "Constant":
            const[name] = op.get_byte_size()
        elif kind != "Parameter" and sources and all(const.get(s) is not None for s in sources):
            const[name] = sum(const[s] for s in sources)
        else:
            const[name] = None
        read = sum(const[s] for s in sources if const.get(s))
        if read and const[name] is None:
            out[name] = read
    return out


def ir_dir(args) -> Path:
    """Cache directory of the exported IR for an HF id and weight format."""
    name = f"{args.model.replace('/', '--')}--{args.weight_format}"
    return CACHE / "ov-models" / name


def prepare(args) -> tuple[Path, str, dict] | None:
    """(IR dir, template, context) of the agent's export task; None when the IR exists."""
    out = Path(args.model) if Path(args.model).is_dir() else ir_dir(args)
    if (out / args.ov_xml).is_file():
        return None
    context = {
        "model": args.model, "out": out, "fmt": args.weight_format,
        "preflight": Path(__file__).with_name("openvino_preflight.py"),
    }  # fmt: skip
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

    def load():
        m = OVModelForCausalLM.from_pretrained(
            path, device=args.device, ov_config={"PERF_COUNT": "YES"},
            trust_remote_code=args.trust_remote_code,
        )  # fmt: skip
        m.compile()
        return m

    from transformers import AutoTokenizer

    source = args.model if not Path(args.model).is_dir() else path
    tok = AutoTokenizer.from_pretrained(source, trust_remote_code=args.trust_remote_code)
    ids = prompt.token_ids(prompt.hf_vocab(tok), args.batch, args.input_len, args.prompt_file)
    tokens = torch.tensor(ids, dtype=torch.int64)

    model = load()
    calls: list = []
    model.request = _Recorder(model.request, calls)
    model.generate(tokens, max_new_tokens=2, min_new_tokens=2, do_sample=False)
    stateful, config = bool(getattr(model, "stateful", False)), model.config
    del model
    if len(calls) < 2:
        raise SystemExit(f"OpenVINO: the reference run made {len(calls)} infer(s), not 2")

    # A freshly loaded model per phase, so its counters hold that phase only.
    raw_dir.mkdir(parents=True, exist_ok=True)
    context = 0
    with open(raw_dir / "ov_profile.jsonl", "w") as f:
        for call in calls[:2]:
            first = next(iter(call["inputs"].values()))  # [batch, tokens]
            m, context = math.prod(first.shape), context + first.shape[-1]
            model = load()
            rows, wall = _phase(model.request, call, args.phase_runs)
            f.write(
                json.dumps({"tokens": m, "context": context, "wall_us": wall, "rows": rows}) + "\n"
            )
            if call is not calls[1]:
                del model

    def ports(ps):
        return [
            {"shape": [d.get_length() if d.is_static else -1 for d in p.get_partial_shape()],
             "dtype": p.get_element_type().get_type_name()}
            for p in ps
        ]  # fmt: skip

    nodes = {}
    for op in (
        model.request.get_compiled_model().get_runtime_model().get_ordered_ops()
    ):  # last phase's
        rt = {k: v.astype(str) for k, v in op.get_rt_info().items()}
        nodes[op.get_friendly_name()] = {
            "rt": rt,
            "inputs": ports(op.inputs()),
            "outputs": ports(op.outputs()),
        }
    (raw_dir / "runtime_model.json").write_text(json.dumps(nodes, indent=1) + "\n")
    # Weights the compiled model keeps internal are measured in the original graph.
    (raw_dir / "const_bytes.json").write_text(json.dumps(const_bytes(model.model), indent=1) + "\n")

    return {
        "framework": "openvino",
        "framework_version": ov.__version__,
        "model": {"name": args.model, "config": stored(config.to_dict())},
        "runtime": {"device": args.device, "batch": args.batch, "input_len": args.input_len,
                    "output_len": args.output_len, "weight_format": args.weight_format, "site_pattern": args.site_pattern, "stateful": stateful,
                    "decode_context": context, "phase_runs": args.phase_runs},
        "device": {"type": args.device, "name": ov.Core().get_property(args.device, "FULL_DEVICE_NAME")},
    }  # fmt: skip


def parse(raw_dir: Path) -> TraceSummary:
    nodes = json.loads((raw_dir / "runtime_model.json").read_text())
    consts_path = raw_dir / "const_bytes.json"
    consts = json.loads(consts_path.read_text()) if consts_path.is_file() else {}
    runtime = json.loads((raw_dir / "run.json").read_text()).get("runtime") or {}
    fold = re.compile(runtime.get("site_pattern") or r"\.\d+(?=[./])")
    rows: dict[tuple, CapturedOp] = {}
    total = 0.0
    launches = 0
    kernels: dict[str, float] = defaultdict(float)
    steps = [json.loads(line) for line in (raw_dir / "ov_profile.jsonl").read_text().splitlines()]
    for step in steps:
        busy = sum(r["us"] for r in step["rows"] if "EXECUTED" in r["status"])
        if step.get("wall_us") and busy > 2 * step["wall_us"]:
            raise SystemExit(
                f"OpenVINO profiling: {busy:.0f} us of node time in a {step['wall_us']:.0f} us "
                f"phase ({step['tokens']} tokens); the counters do not measure this run"
            )
    # Recovery from a whole-us running mean is off by about 1 us per run; below that is wrong.
    floor = -2.0 * (len(steps) + 16)
    for step in steps:
        for r in step["rows"]:
            if "EXECUTED" not in r["status"]:
                continue
            # Profiling names a node "<type>:<name>"; the runtime model names it "<name>".
            r["node"] = re.sub(r"^[a-z0-9_]+:(?!:)", "", r["node"])
            if r["us"] < floor:
                raise SystemExit(
                    f"OpenVINO profiling: {r['node']} has {r['us']:.0f} us in a step; the "
                    "plugin's counters are not a running mean per run for this model"
                )
            if r["exec"] in ("", "undef"):
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
                    # Weights are internal: K from the input, N from the output.
                    sources = (node.get("rt") or {}).get("originalLayersNames", "").split(",")
                    # A compiled node may carry an output suffix ("MatMul.0") the original lacks.
                    weight = sum(consts.get(n, consts.get(re.sub(r"\.\d+$", "", n), 0))
                                 for n in sources if n)  # fmt: skip
                    op.framework_meta["gemm"] = {
                        "K": ports[0]["shape"][-1],
                        "N": outs[0]["shape"][-1],
                        **({"weight_bytes": weight} if weight else {}),
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
    """names.json key: node type and the implementation the plugin picked for it."""
    impl = max(op.kernel_symbols, key=op.kernel_symbols.get) if op.kernel_symbols else ""
    return f"{op.framework_op}|{impl}"
