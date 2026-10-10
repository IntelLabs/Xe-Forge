"""SGLang: profile one batch with SGLang's own benchmark and read its traces.

``python -m sglang.benchmark.one_batch`` loads a ``ModelRunner`` in-process (the server
runs the model in a scheduler subprocess, out of reach of a profiler), warms up, and runs
one prefill and ``--output-len`` decode steps. With ``--profile`` it records them with
torch.profiler -- XPU activity, ``with_stack``, and shapes with ``--profile-record-shapes``
-- and exports one chrome trace per phase into ``SGLANG_TORCH_PROFILER_DIR``. This adapter
runs it once per phase, with every decode step profiled, and leaves the traces in ``raw/``; the
shared parser reads them. Depending on SGLang's command line rather than its internals
is deliberate: those move between releases. Tensor parallel 1; graph capture is disabled
so every op is dispatched.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from xe_forge.frontend.adapters.pytorch import device_info, device_type
from xe_forge.frontend.adapters.vllm import _CONFIG_KEYS


def add_arguments(parser) -> None:
    parser.add_argument("--model", help="HF id or local path")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=32, help="decode steps profiled")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attention-backend", help="default: intel_xpu on XPU")
    parser.add_argument("--page-size", type=int, default=64)


def _command(args, dev: str, stage: str) -> list[str]:
    cmd = [
        sys.executable, "-m", "sglang.benchmark.one_batch",
        "--model-path", args.model,
        "--device", dev,
        "--dtype", args.dtype,
        "--tp-size", "1",
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
        # one_batch decodes output_len - 1 steps (prefill yields the first token); a
        # window past the last step is never stopped, and on XPU the profiler torn down
        # at exit aborts the process.
        "--profile-steps", str(max(args.output_len - 1, 1)),
        "--profile-prefix", "trace",
    ]  # fmt: skip
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
    # One process per phase: on XPU, Kineto cannot open a second profiling session in a
    # process (clearActivities: PTI_ERROR_NOT_IMPLEMENTED), and one_batch profiles prefill
    # and decode as two sessions.
    for stage in ("prefill", "decode"):
        cmd = _command(args, dev, stage)
        print(f"sglang: {' '.join(cmd)}", flush=True)
        code = subprocess.call(cmd, env=env)
        if code != 0:
            raise SystemExit(f"sglang one_batch ({stage}) failed (exit {code})")
    traces = sorted(raw_dir.glob("trace*.trace.json.gz"))
    if not traces:
        raise SystemExit(f"sglang one_batch wrote no trace into {raw_dir}")

    # Provider ops register on import; the schema lookup after this needs them.
    try:
        import sgl_kernel  # noqa: F401
    except ImportError:
        pass

    model_cfg = {"name": args.model}
    try:
        from transformers import AutoConfig

        hf = AutoConfig.from_pretrained(args.model).to_dict()
        text = hf.get("text_config") or {}
        model_cfg.update({k: hf.get(k, text.get(k)) for k in _CONFIG_KEYS if k in hf or k in text})
    except Exception as exc:  # the config is context, not a requirement
        model_cfg["config_error"] = repr(exc)

    return {
        "framework": "sglang",
        "framework_version": getattr(sglang, "__version__", "unknown"),
        "model": model_cfg,
        "runtime": {
            "tp": 1,
            "cuda_graph": False,
            "batch": args.batch,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "attention_backend": args.attention_backend or ("intel_xpu" if dev == "xpu" else None),
            "page_size": args.page_size,
            "one_batch": " ".join(cmd[1:]).replace(
                "--profile-stage decode", "--profile-stage <phase>"
            ),
        },
        "device": device_info(),
    }
