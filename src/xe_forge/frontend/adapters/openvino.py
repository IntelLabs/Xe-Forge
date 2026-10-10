"""OpenVINO: per-primitive device time from PERF_COUNT, for an LLM run by optimum-intel.

The model (an exported IR directory, or an HF id exported on the fly) is compiled on the
GPU with ``PERF_COUNT``. ``generate`` runs one prefill and ``--output-len`` decode steps;
a thin proxy around the model's infer request reads ``get_profiling_info()`` after every
step. The compiled runtime model gives each executed node its port shapes and precisions.

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
from xe_forge.frontend.torch_trace import TraceSummary

_LIBRARY = re.compile(r"onednn|dnnl", re.I)
_DTYPES = {"f16": "float16", "bf16": "bfloat16", "f32": "float32", "i64": "int64",
           "i32": "int32", "i8": "int8", "u8": "uint8", "boolean": "bool"}  # fmt: skip


def add_arguments(parser) -> None:
    parser.add_argument("--model", help="exported IR directory or HF id")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=32, help="decode steps profiled")
    parser.add_argument("--device", default="GPU")


class _Recorder:
    """Stands in for the model's InferRequest; logs profiling info after each step."""

    def __init__(self, request, steps: list):
        self._request, self._steps = request, steps

    def __getattr__(self, name):
        return getattr(self._request, name)

    def wait(self):
        self._request.wait()
        self._steps.append(
            [
                {"node": p.node_name, "type": p.node_type, "exec": p.exec_type,
                 "us": p.real_time.total_seconds() * 1e6, "status": str(p.status)}
                for p in self._request.get_profiling_info()
            ]
        )  # fmt: skip


def capture(args, raw_dir: Path) -> dict:
    if not args.model:
        raise SystemExit("--framework openvino needs --model")
    import openvino as ov
    import torch
    from optimum.intel import OVModelForCausalLM

    exported = Path(args.model).is_dir()
    model = OVModelForCausalLM.from_pretrained(
        args.model,
        export=not exported,
        load_in_8bit=False,
        device=args.device,
        ov_config={"PERF_COUNT": "YES"},
    )
    model.compile()
    prompt = torch.tensor(
        [[100 + (i * 7919) % 20000 for i in range(args.input_len)]], dtype=torch.int64
    )
    model.generate(prompt, max_new_tokens=4, min_new_tokens=4, do_sample=False)  # warm-up

    steps: list = []
    model.request = _Recorder(model.request, steps)
    model.generate(
        prompt, max_new_tokens=args.output_len, min_new_tokens=args.output_len, do_sample=False
    )

    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / "ov_profile.jsonl", "w") as f:
        for i, rows in enumerate(steps):
            tokens = args.input_len if i == 0 else 1
            f.write(
                json.dumps({"tokens": tokens, "context": args.input_len + i, "rows": rows}) + "\n"
            )

    nodes = {}
    for op in model.request.get_compiled_model().get_runtime_model().get_ordered_ops():
        rt = {k: v.astype(str) for k, v in op.get_rt_info().items()}
        nodes[op.get_friendly_name()] = {
            "rt": rt,
            "inputs": [
                {"shape": [d.get_length() if d.is_static else -1 for d in i.get_partial_shape()],
                 "dtype": i.get_element_type().get_type_name()}
                for i in op.inputs()
            ],
        }  # fmt: skip
    (raw_dir / "runtime_model.json").write_text(json.dumps(nodes, indent=1) + "\n")

    return {
        "framework": "openvino",
        "framework_version": ov.__version__,
        "model": {"name": args.model, **{k: getattr(model.config, k, None) for k in (
            "model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "vocab_size")}},
        "runtime": {"device": args.device, "input_len": args.input_len,
                    "output_len": args.output_len, "stateful": bool(getattr(model, "stateful", False))},
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
            if "EXECUTED" not in r["status"] or r["us"] <= 0:
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
        return Naming("library", op.framework_op.split("::")[-1], via=f"oneDNN ({impl})")
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
