# Optimizing a kernel from another repo

`--kernel-repo` optimizes a kernel that already lives in a repository: vllm-xpu-kernels (SYCL,
SYCL-TLA), vLLM (Triton), or any other. The result is a patch against that repository, which
you apply yourself.

## Needs

- A local git checkout of the repository. It is read-only: nothing is written to it.
- `--engine claude`. No other engine can locate a kernel.
- `.env` with the LLM settings.
- SYCL: the oneAPI environment sourced. Triton: the XPU Triton the venv's torch uses.

## Which file is the baseline

- No `-i`: the session copies the repository's kernel into `test_kernels/<name>.<ext>`.
- `-i` with the DSL's suffix (`.cpp` for `--dsl sycl`, `.py` for `--dsl triton`): `-i` is
  the baseline. Nothing is copied from the repository; it is only profiled. Port-back diffs
  baseline against winner, so `-i` must be a copy of the repository's kernel.
- `-i` with any other suffix: `-i` is the PyTorch reference.

Pass a reference with `--reference`, never `-i`. With `--dsl triton`, `-i ref.py` has the
DSL's suffix and becomes the baseline.

## `-n`

The op's name as the repository registers or calls it: `rms_norm`, or the name of a Triton
launcher. The locator searches for that literal name.

## Spec

You write it. Without `--reference`, every trial must match the baseline copy; the spec's
form decides the inputs and what happens before the first trial:

| Spec | Inputs | Before the first trial |
|---|---|---|
| Without `inputs:` | built the way the repository's test or call site builds them, cited by file:line | the session checks the copy builds, runs and repeats deterministically |
| With `inputs:` | random tensors at the spec's shapes | the locator writes `<name>_pytorch.py` from the kernel's source for a standard op, or names the repository's test for a bespoke one; the session reads it as the op's semantics |

With `--reference ref.py`, the baseline and every trial are checked against the reference.
Example, with `inputs:`:

```yaml
inputs:
  X: {shape: [NUM_TOKENS, HIDDEN], dtype: bfloat16}
inits:
  - hidden_size: HIDDEN
bench-gpu:
  - params: [X]
    dtype: bfloat16
    dims: {NUM_TOKENS: 4096, HIDDEN: 4096}
    flop: "4*NUM_TOKENS*HIDDEN"
```

## Commands

```bash
# vllm-xpu-kernels (SYCL)
xe-forge --engine claude --dsl sycl --device xpu --kernel-repo <vllm-xpu-kernels> \
    -n rms_norm -s rms_norm.yaml --variant bench-gpu --workspace ws/rms_norm

# vLLM (Triton)
xe-forge --engine claude --dsl triton --device xpu --kernel-repo <vllm> \
    -n <op> -s <op>.yaml --variant bench-gpu --workspace ws/<op>
```

Optional: `--reference ref.py`, `--auto-launch --max-trials N`.

## What the run does

1. The `kernel-locator` agent writes `experiments/kernel_profile.md`: the commit, the file:line
   of every region, the build flags, the repository's test and benchmark.
2. The session writes the baseline copy, inlining the kernel's own device code, or the
   Triton launcher's imports from the repository.
3. Trials run. `trial finalize` writes `output/<name>_optimized.<ext>`; a regression is
   never finalized.
4. The `port-back` agent writes `output/upstream.patch` and `output/PORT.txt` from a private
   clone at the recorded commit.

## What stays with you

- Applying the patch, rebuilding, testing and serving. A SYCL patch is checked only with
  `git apply --check`. A Triton patch is `py_compile`d, and the repository's test runs when
  the interpreter can import the repository.
- Timings exist only for the spec's variants; `PORT.txt` lists what was measured. Other ops
  that call the changed code change with it.

What decides correctness in each mode: [CORRECTNESS.md](CORRECTNESS.md).
