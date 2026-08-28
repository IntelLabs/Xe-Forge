"""One arm of a served e2e A/B on vLLM: engine load, declared warmup, k decode replicates.

Runnable as ``python -m xe_forge.orbit.bench.vllm_arm --model <hf-id>``. Emits one
JSON line ``ARM_RESULT {"replicates_s": [...], "tok_per_s": [...], "token_ids": ...}``
so the caller (``xe-orbit fuse-apply --e2e``) can pool replicates and compare arms
with §17 statistics. The arm switch is the process boundary — an in-place source
patch cannot be swapped inside a live interpreter (§17.5.2) — so replicates are
in-process, with slow drift accepted as the stated limitation rather than traded
for the ~9x variance of fresh-process timing.

The candidate/baseline split is carried entirely by environment: the patched tree's
fused branch is guarded by ``ORBIT_FUSED_MLP=1`` (plus ``ORBIT_FUSED_LIB``), and with
the guard off the original path runs byte-for-byte, so both arms share one tree.
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
        "the kernel-level differential check, §13.5)",
    )
    args = parser.parse_args()

    import torch

    free, _ = torch.xpu.mem_get_info()
    if free < 1200 * 2**20:
        raise SystemExit("refusing under memory pressure (§17.5)")

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
