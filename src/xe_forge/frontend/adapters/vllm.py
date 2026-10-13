"""vLLM: run the model in-process for one profiled request batch.

The engine runs in this process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``, tensor parallel 1),
so a plain torch.profiler around ``LLM.generate`` sees every op and kernel. Eager mode
(``enforce_eager``) keeps graph replay from hiding the ops; the shares it measures are the
eager ones. One warm-up request runs first so compilation and autotuning stay outside the
profiled window. The prompt comes from ``frontend.prompt``, as for every adapter.
"""

from __future__ import annotations

import os
from pathlib import Path

from xe_forge.frontend import prompt
from xe_forge.frontend.adapters.pytorch import device_info, profile_region
from xe_forge.frontend.model_meta import stored


def add_arguments(parser) -> None:
    parser.add_argument("--model", help="HF id or local path")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=32, help="decode steps profiled")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", help="e.g. fp8: vLLM quantizes the checkpoint at load")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--no-stack", action="store_true", help="skip call-site recording")


def _prompts(llm, args, length: int):
    from vllm.inputs import TokensPrompt

    ids = prompt.token_ids(prompt.hf_vocab(llm.get_tokenizer()), args.batch, length,
                           args.prompt_file)  # fmt: skip
    return [TokensPrompt(prompt_token_ids=seq) for seq in ids]


def capture(args, raw_dir: Path) -> dict:
    if not args.model:
        raise SystemExit("--framework vllm needs --model")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # out of process, the profiler sees nothing

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
        quantization=args.quantization,
        # A cached prefix (from the warm-up, or another sequence) shortens the prefill the
        # profile sees: every call must run the input_len it says it runs.
        enable_prefix_caching=False,
        trust_remote_code=args.trust_remote_code,
    )
    warm = SamplingParams(max_tokens=args.warmup_tokens, ignore_eos=True, temperature=0.0)
    llm.generate(_prompts(llm, args, min(args.input_len, 64)), warm, use_tqdm=False)

    params = SamplingParams(max_tokens=args.output_len, ignore_eos=True, temperature=0.0)
    prompts = _prompts(llm, args, args.input_len)
    profile_region(
        lambda: llm.generate(prompts, params, use_tqdm=False), raw_dir, with_stack=not args.no_stack
    )

    model_cfg = {"name": args.model}
    try:
        model_cfg["config"] = stored(llm.llm_engine.model_config.hf_config.to_dict())
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
            "prefix_caching": False,
            "batch": args.batch,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "max_model_len": max_len,
            "quantization": args.quantization,
        },
        "device": device_info(),
    }
