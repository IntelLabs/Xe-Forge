"""OpenVINO: per-primitive device time from PERF_COUNT, for an LLM run by optimum-intel.

The model (an exported IR directory, or an HF id an agent exported once per
``--weight-format``, see ``prepare``) is compiled on the GPU with ``PERF_COUNT``.
``generate`` runs one prefill and ``--output-len`` decode steps; a thin proxy around the
model's infer request reads ``get_profiling_info()`` after every step. The plugin reports
each node's *mean* time
over the runs it executed in, so a step's own time is recovered as
``n * mean_n - (n - 1) * mean_(n-1)``. The compiled runtime model gives each executed node
its port shapes and precisions.

OpenVINO reports no per-call shapes -- the runtime model's are dynamic -- so each row
carries the call's token count and context length, and ``-1`` stays where a dim is
dynamic: the agent maps model-level sizes to the kernel's arguments.

Naming: a primitive implemented by oneDNN is a library call; any other GPU primitive is
an OpenCL kernel in the ``openvino`` repository, named by its implementation as the source
names it (``ocl::sdpa::opt__f16`` -> ``sdpa_opt``, ``permute_ref__f16`` -> ``permute_ref``).
Where profiling reports ``undef``, the runtime model's ``primitiveType`` is used.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from xe_forge.frontend.ir import CapturedOp, Naming, TensorArg
from xe_forge.frontend.model_meta import stored
from xe_forge.frontend.torch_trace import TraceSummary

_LIBRARY = re.compile(r"onednn|dnnl", re.I)
_DTYPES = {"f16": "float16", "bf16": "bfloat16", "f32": "float32", "i64": "int64",
           "i32": "int32", "i8": "int8", "u8": "uint8", "boolean": "bool"}  # fmt: skip


def add_arguments(parser) -> None:
    parser.add_argument("--model", help="exported IR directory or HF id")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=32, help="decode steps profiled")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="GPU")
    parser.add_argument(
        "--weight-format",
        default="fp16",
        choices=("fp16", "int8", "int4", "fp8"),
        help="weights of an HF id, exported once by an agent and cached",
    )


class _Recorder:
    """Stands in for the model's InferRequest; logs each step's own profiling info."""

    def __init__(self, request, steps: list):
        self._request, self._steps, self._total, self._runs = request, steps, {}, {}

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


PREPARE = """\
Export {model} to an OpenVINO IR directory at {out}, loadable by optimum-intel's
OVModelForCausalLM, with weights in {fmt}: fp16 is no compression, int8/int4 weight
compression, fp8 full f8e4m3 quantization calibrated on a standard text dataset. Choose
settings the model's shapes allow (e.g. a group size dividing every channel). Download
what you need (unset HF_HUB_OFFLINE for it). Check that, under the framework's interpreter,
OVModelForCausalLM.from_pretrained("{out}", device="GPU") loads it and generates 4 tokens.
Write prepared.json {{"model": "{out}"}}, or {{"error": "<why>"}}.
"""


def ir_dir(args) -> Path:
    """Where the IR for an HF id and weight format lives, once exported."""
    name = f"{args.model.replace('/', '--')}--{args.weight_format}"
    return Path.home() / ".cache/xe_forge/ov-models" / name


def prepare(args) -> tuple[Path, str] | None:
    """What the model needs before capture -- an exported IR, and the agent's task to make
    it -- or None when it has one. Export is per model (formats, calibration, shapes), so an
    agent does it rather than code here."""
    out = Path(args.model) if Path(args.model).is_dir() else ir_dir(args)
    if (out / "openvino_model.xml").is_file():
        return None
    return out, PREPARE.format(model=args.model, out=out, fmt=args.weight_format)


def capture(args, raw_dir: Path) -> dict:
    if not args.model:
        raise SystemExit("--framework openvino needs --model")
    import openvino as ov
    import torch
    from optimum.intel import OVModelForCausalLM

    path = Path(args.model) if Path(args.model).is_dir() else ir_dir(args)
    if not (path / "openvino_model.xml").is_file():
        raise SystemExit(f"no OpenVINO IR at {path}; `xe-forge run` has an agent export it")
    model = OVModelForCausalLM.from_pretrained(
        path, device=args.device, ov_config={"PERF_COUNT": "YES"}
    )
    model.compile()
    prompt = torch.tensor(
        [[100 + (i * 7919) % 20000 for i in range(args.input_len)]] * args.batch,
        dtype=torch.int64,
    )
    steps: list = []
    model.request = _Recorder(model.request, steps)  # from the first infer: means accumulate
    model.generate(prompt, max_new_tokens=4, min_new_tokens=4, do_sample=False)  # warm-up
    warm = len(steps)
    model.generate(
        prompt, max_new_tokens=args.output_len, min_new_tokens=args.output_len, do_sample=False
    )

    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / "ov_profile.jsonl", "w") as f:
        for i, rows in enumerate(steps[warm:]):
            tokens = args.batch * (args.input_len if i == 0 else 1)  # rows of a FullyConnected
            f.write(
                json.dumps({"tokens": tokens, "context": args.input_len + i, "rows": rows}) + "\n"
            )

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
                    "output_len": args.output_len, "weight_format": args.weight_format, "stateful": bool(getattr(model, "stateful", False))},
        "device": {"type": args.device, "name": ov.Core().get_property(args.device, "FULL_DEVICE_NAME")},
    }  # fmt: skip


def parse(raw_dir: Path) -> TraceSummary:
    nodes = json.loads((raw_dir / "runtime_model.json").read_text())
    rows: dict[tuple, CapturedOp] = {}
    total = 0.0
    launches = 0
    kernels: dict[str, float] = defaultdict(float)
    for line in (raw_dir / "ov_profile.jsonl").read_text().splitlines():
        step = json.loads(line)
        for r in step["rows"]:
            if "EXECUTED" not in r["status"]:
                continue  # fused away or not run: no time of its own
            if r["exec"] in ("", "undef"):  # the runtime model knows what profiling does not
                r["exec"] = (nodes.get(r["node"]) or {}).get("rt", {}).get(
                    "primitiveType"
                ) or "undef"
            total += r["us"]
            launches += 1
            kernels[r["exec"]] += r["us"]
            site = re.sub(r"\.\d+(?=[./])", ".*", r["node"])
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


def name_op(op: CapturedOp) -> Naming:
    impl = max(op.kernel_symbols, key=op.kernel_symbols.get) if op.kernel_symbols else ""
    # The GPU plugin names every OpenCL kernel it runs; it reports oneDNN primitives as
    # "undef" (FullyConnected on XMX: ONEDNN_VERBOSE shows matmul jit:gemm:any).
    if impl == "undef" or _LIBRARY.search(impl):
        op.impl_class = "onednn"
        # The call to tune is the plugin's oneDNN implementation of the op.
        name = {"FullyConnected": "fully_connected_onednn", "Gemm": "gemm_onednn"}.get(
            op.framework_op.split("::")[-1], op.framework_op.split("::")[-1]
        )
        return Naming("library", name, "openvino", "sycl", via=f"oneDNN ({impl})")
    if not impl:
        return Naming("unnamed", op.framework_op, via="no implementation reported")
    op.impl_class = "provider"
    # "ocl::sdpa::opt__f16" -> "sdpa_opt": the source file the locator greps for.
    name = re.sub(r"_*_(f16|f32|bf16|i8|u8|i32|i64)$", "", re.sub(r"^(ocl|cm|sycl)::", "", impl))
    return Naming(
        "kernel",
        name.replace("::", "_"),
        "openvino",
        "opencl",
        via=f"OpenVINO GPU primitive {impl}",
    )
