"""One arm of a served e2e A/B on vLLM: engine load, declared warmup, k decode replicates.

Runnable as ``python -m xe_forge.orbit.bench.vllm_arm --model <hf-id>``; emits one
``ARM_RESULT`` JSON line for the caller to pool and compare. The arm switch is the
process boundary, carried entirely by environment (``ORBIT_FUSED_MLP=1`` guards the
patched tree's fused branch), so both arms share one tree.
"""

from __future__ import annotations

import argparse
import json
import os
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model id, served exactly as the user would")
    parser.add_argument("--replicates", type=int, default=6)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument(
        "--greedy-check",
        action="store_true",
        help="temperature=0 and emit the decoded token ids instead of timing "
        "(the caller diffs the two arms' streams; near-ties are chaotic, so "
        "divergence is reported, not gated on — the hard correctness gate is "
        "the kernel-level differential check)",
    )
    args = parser.parse_args()

    import torch

    free, _ = torch.xpu.mem_get_info()
    if free < 1200 * 2**20:
        raise SystemExit("refusing under memory pressure")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        kv_cache_memory_bytes=256 * 2**20,
        # startup gate only on unified-memory iGPUs; KV is pinned above
        gpu_memory_utilization=float(os.environ.get("ORBIT_GPU_UTIL", "0.30")),
        enforce_eager=True,
        disable_log_stats=True,
    )
    prompts = [f"Write one sentence about the number {i}." for i in range(args.batch)]

    if args.greedy_check:
        params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
        outputs = llm.generate(prompts, params)
        ids = [list(o.outputs[0].token_ids) for o in outputs]
        print("ARM_RESULT " + json.dumps({"token_ids": ids}))
        return 0

    # temperature>0 so the decode path under test is the served one; seeded so
    # both arms decode the same work.
    params = SamplingParams(temperature=0.8, seed=1234, max_tokens=args.max_tokens)
    llm.generate(prompts, SamplingParams(temperature=0.8, seed=1234, max_tokens=8))  # warmup, discarded

    replicates: list[float] = []
    rates: list[float] = []
    for _ in range(args.replicates):
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, params)
        dt = time.perf_counter() - t0
        tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        replicates.append(dt)
        rates.append(tokens / dt)
    print("ARM_RESULT " + json.dumps({"replicates_s": replicates, "tok_per_s": rates}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
