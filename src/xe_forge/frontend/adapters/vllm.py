"""vLLM: run the model in-process for one profiled request batch.

The engine runs in this process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``, tensor parallel 1),
so a plain torch.profiler around ``LLM.generate`` sees every op and kernel. Eager mode
(``enforce_eager``) keeps graph replay from hiding the ops; the shares it measures are the
eager ones. One warm-up request runs first so compilation and autotuning stay outside the
profiled window. The prompt is synthetic token ids: shapes, not text, are what is captured.
"""

from __future__ import annotations

import os
from pathlib import Path

from xe_forge.frontend.adapters.pytorch import device_info, profile_region

# Model-config keys worth carrying into the capture; everything else is in the HF config.
_CONFIG_KEYS = (
    "architectures",
    "model_type",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "quantization_config",
)


def add_arguments(parser) -> None:
    parser.add_argument("--model", help="HF id or local path")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=32, help="decode steps profiled")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--no-stack", action="store_true", help="skip call-site recording")


def _prompts(n: int, length: int):
    from vllm.inputs import TokensPrompt

    # Deterministic ids away from the special-token ranges; only the length matters.
    return [
        TokensPrompt(
            prompt_token_ids=[100 + (i * 7919 + j * 104729) % 20000 for i in range(length)]
        )
        for j in range(n)
    ]


def capture(args, raw_dir: Path) -> dict:
    if not args.model:
        raise SystemExit("--framework vllm needs --model")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import vllm
    from vllm import LLM, SamplingParams

    max_len = args.max_model_len or args.input_len + args.output_len + 64
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=args.batch,
        max_model_len=max_len,
        dtype=args.dtype,
    )
    warm = SamplingParams(max_tokens=4, ignore_eos=True, temperature=0.0)
    llm.generate(_prompts(args.batch, min(args.input_len, 64)), warm, use_tqdm=False)

    params = SamplingParams(max_tokens=args.output_len, ignore_eos=True, temperature=0.0)
    prompts = _prompts(args.batch, args.input_len)
    profile_region(
        lambda: llm.generate(prompts, params, use_tqdm=False), raw_dir, with_stack=not args.no_stack
    )

    model_cfg = {"name": args.model}
    try:
        hf = llm.llm_engine.model_config.hf_config.to_dict()
        text = hf.get("text_config") or {}
        model_cfg.update({k: hf.get(k, text.get(k)) for k in _CONFIG_KEYS if k in hf or k in text})
        model_cfg["dtype"] = str(llm.llm_engine.model_config.dtype).replace("torch.", "")
    except Exception as exc:  # the config is context, not a requirement
        model_cfg["config_error"] = repr(exc)

    return {
        "framework": "vllm",
        "framework_version": vllm.__version__,
        "model": model_cfg,
        "runtime": {
            "tp": 1,
            "enforce_eager": True,
            "batch": args.batch,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "max_model_len": max_len,
        },
        "device": device_info(),
    }
