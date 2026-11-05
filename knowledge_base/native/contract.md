# Engine contract

The tools measure an engine only through this contract.

## Command line

```
xf-run WEIGHTS_DIR --ids 1,2,3|@FILE [--new N] [--recompute] [--logits FILE] [--threads N]
```

- Greedy decoding: prefill the prompt, then N-1 single-token steps reusing the engine's state
  (KV caches, recurrent state). `--recompute` instead re-runs the whole sequence from a
  cleared state for every token.
- `--ids @FILE` reads the comma-separated ids from a file: a long prompt (tens of thousands of
  tokens) exceeds the kernel's limit on one command-line argument.
- `--logits FILE`: appends each forward's last-position logits as raw fp32.
- stdout: `TOKENS: <ids>`, `PREFILL_MS: <ms>`, `DECODE_MS_PER_TOKEN: <ms>` (when N > 1), then
  `DONE` last.

## Hooks (`runtime/xf_hooks.c`)

Call `xf_hook_step` after every step of the forward pass.

- `XF_DUMP=dir`: writes the outputs of the first forward call (`XF_DUMP_CALLS=n` for more) as
  raw fp32, plus `dir/index.jsonl`. Each line holds `block`, `layer`, `step`, `op`, `module`,
  `rows` and `numel`.
  - `module` names the reference submodule whose forward output equals this output,
    flattened per token. `{i}` in it stands for `layer`. Use `""` where no such module exists.
  - `rows` is the number of tokens, or 1 for a value holding only the last position.
- `XF_PROFILE=file`: one JSON row per (block.step, T) with summed time, which is what the
  profile tool ranks.

## Verification (`python -m xe_forge.frontend.verify`)

The tool runs the engine and the reference model on the same prompt. Every gate must pass:

- `layer`: every dumped step that has a `module` matches that module's output (relative
  error at most 1e-3).
- `logits`: the last-position logits match, and the argmax is equal.
- `decode`: incremental decoding gives the same tokens as `--recompute`.
- `tokens`: the greedy tokens equal the reference's greedy tokens.
