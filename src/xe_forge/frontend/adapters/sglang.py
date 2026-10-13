"""SGLang: profile one prefill and the decode steps with ``sglang.benchmark.one_batch``, one
process per phase, graph capture off; the shared parser reads the traces.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from xe_forge.frontend import prompt
from xe_forge.frontend.adapters.pytorch import device_info, device_type
from xe_forge.frontend.model_meta import stored


def add_arguments(parser) -> None:
    parser.add_argument("--model", help="HF id or local path")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=32, help="decode steps profiled")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attention-backend", help="default: intel_xpu on XPU")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--quantization", help="e.g. fp8: SGLang quantizes the checkpoint at load")
    parser.add_argument(
        "--tp", type=int, default=1, help="tensor parallel size; rank 0 is profiled"
    )


def _command(args, dev: str, stage: str, ids_file: Path) -> list[str]:
    cmd = [
        sys.executable, "-m", "xe_forge.frontend.adapters.sglang_one_batch", str(ids_file),
        "--model-path", args.model,
        "--device", dev,
        "--dtype", args.dtype,
        "--tp-size", str(args.tp),
        "--page-size", str(args.page_size),
        "--disable-cuda-graph",
        "--batch-size", str(args.batch),
        "--input-len", str(args.input_len),
        "--output-len", str(args.output_len),
        "--profile",
        "--profile-record-shapes",
        "--profile-activities", "CPU", {"xpu": "XPU", "cuda": "GPU"}.get(dev, "CPU"),
        "--profile-stage", stage,
        "--profile-start-step", "0",
        # Decode runs output_len - 1 steps; a longer window never stops and aborts on XPU.
        "--profile-steps", str(max(args.output_len - 1, 1)),
        "--profile-prefix", "trace",
    ]  # fmt: skip
    if args.quantization:
        cmd += ["--quantization", args.quantization]
    if args.trust_remote_code:
        cmd += ["--trust-remote-code"]
    backend = args.attention_backend or ("intel_xpu" if dev == "xpu" else None)
    if backend:
        cmd += ["--attention-backend", backend]
    return cmd


def capture(args, raw_dir: Path) -> dict:
    if not args.model:
        raise SystemExit("--framework sglang needs --model")
    import sglang

    dev = device_type()
    raw_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, SGLANG_TORCH_PROFILER_DIR=str(raw_dir))
    # One process per phase: Kineto on XPU cannot open a second session in a process.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    ids = prompt.token_ids(prompt.hf_vocab(tok), args.batch, args.input_len, args.prompt_file)
    ids_file = raw_dir / "prompt_ids.json"
    ids_file.write_text(json.dumps(ids))
    for stage in ("prefill", "decode"):
        cmd = _command(args, dev, stage, ids_file)
        print(f"sglang: {' '.join(cmd)}", flush=True)
        code = subprocess.call(cmd, env=env)
        if code != 0:
            raise SystemExit(f"sglang one_batch ({stage}) failed (exit {code})")
    traces = sorted(raw_dir.glob("trace*.trace.json.gz"))
    if not traces:
        raise SystemExit(f"sglang one_batch wrote no trace into {raw_dir}")

    # Provider ops register on import; the schema lookup needs them.
    try:
        import sgl_kernel  # noqa: F401
    except ImportError:
        pass

    model_cfg = {"name": args.model}
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        model_cfg["config"] = stored(config.to_dict())
    except Exception as exc:
        model_cfg["config_error"] = repr(exc)

    return {
        "framework": "sglang",
        "framework_version": getattr(sglang, "__version__", "unknown"),
        "model": model_cfg,
        "runtime": {
            "tp": args.tp,
            "cuda_graph": False,
            "batch": args.batch,
            "dtype": args.dtype,
            "quantization": args.quantization,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "attention_backend": args.attention_backend or ("intel_xpu" if dev == "xpu" else None),
            "page_size": args.page_size,
            "one_batch": " ".join(cmd[3:]).replace(
                "--profile-stage decode", "--profile-stage <phase>"
            ),
        },
        "device": device_info(),
    }
