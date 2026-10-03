# Correctness: what decides it, per mode

Every run has exactly one **oracle**. A trial that disagrees with it is wrong, however fast.
Pick the row that matches what you have; the flags decide the oracle, not the session.

## Which oracle you get

`-i` is the **baseline** when its suffix is the target DSL's (`.cpp` for `--dsl sycl`, `.py`
for `--dsl triton`), and the **reference** otherwise. With `--dsl triton` a PyTorch file given
with `-i` is therefore the baseline: pass a reference with `--reference`.

| You have | Oracle | Command (plus `--engine claude --dsl <dsl> --workspace <ws>`) |
|---|---|---|
| A PyTorch op, no kernel yet | the op (`test_kernels/<name>_pytorch.py`, immutable); Claude writes the kernel | `-i op.py -n <name> -s spec.yaml --variant <v>` |
| A kernel and a PyTorch reference | the reference: baseline and every trial must match it | `-i kernel.cpp --reference ref.py -n <name> -s spec.yaml --variant <v>` (or put `<stem>_pytorch.py` beside `-i`) |
| A kernel, no reference | **the input kernel**: trials must match its outputs | `-i kernel.cpp -n <name> -s spec.yaml --variant <v>` |
| A kernel in another repo, no reference | **the copy of the repo kernel** (`test_kernels/<name>.<ext>`); see [KERNEL_REPO.md](KERNEL_REPO.md) | `--kernel-repo <repo> -n <name> -s spec.yaml --variant <v>` |
| A kernel in another repo and a reference | the reference | `--kernel-repo <repo> --reference ref.py -n <name> -s spec.yaml --variant <v>` |
| A host project that measures | the host command's `CORRECTNESS` / `VERDICT` lines | `--external-benchmark "<cmd> {trial} {baseline} {variant}"` |

`--kernel-repo` works only with `--engine claude`.

## Rules

1. The oracle is read-only. Never edit it to make a trial pass.
2. When the oracle is a baseline kernel (the two "no reference" rows), its inputs come from:
   - **a spec without `inputs:`** — the repo's own test, and the session must first show the
     baseline builds, runs and repeats deterministically; no trial is created before that;
   - **a spec with `inputs:`** — random tensors at the spec's shapes. For a standard op the
     locator writes `<name>_pytorch.py` from the kernel's source, and the session reads it as
     the op's semantics; `experiments/kernel_profile.md` says what it wrote.
3. A host result with no correctness verdict is never a pass, and `DONE` must be the last
   stdout line of the host command.
4. `--reference` overrides the `<stem>_pytorch.py` lookup beside `-i`.
5. Always pass `--variant`, and compare timings only within one variant.
6. `--no-correctness` switches checking off for the DSPy engine only. A Claude workspace
   always checks. Results from it are timings, not correctness claims.
7. A regression is never finalized; parity is. `output/` holds the winner. With
   `--kernel-repo`, `output/upstream.patch` is the change against the repo, unbuilt:
   rebuilding and testing the repo is the owner's step.

## Integrity checks (built-in benchmark)

Matching the oracle once does not make a trial valid: the timed calls all see the same
inputs at the same addresses, so a trial that caches or skips work can match and look fast.
After a trial matches, `benchmark --builtin-benchmark` runs it a few more times against the
oracle (`src/xe_forge/core/integrity.py`). A failure prints `Correctness: FAILED`,
`VERDICT: INTEGRITY` and one `Error: <NAME>: ...` line per rule broken:

| Name | Catches |
|---|---|
| `HARNESS_ACCESS` | trial source naming the harness, the seeding or the timer (`xe_forge`, `ai_bench`, `elapsed_time`, `Event(`, ...) |
| `OUTPUT_ALIASES_INPUT` | an output sharing storage with an input |
| `OUTPUT_REUSED` | a later call overwriting an earlier call's output (a persistent buffer) |
| `STALE_RESULT` | a wrong output on new inputs: a result cached by shape or from an earlier call |
| `CACHED_BY_ADDRESS` | a wrong output after new values are written into the same input tensors |
| `UNWRITTEN_OUTPUT` | a wrong output when freshly allocated memory is poisoned first (GPU only) |
| `NONDETERMINISTIC` | one of three repeated calls disagreeing with the oracle: a race |
| `OFF_STREAM` | an output still incomplete when the caller's stream has finished: work on another queue, outside what the timer sees (GPU only) |

If the oracle itself aliases or reuses its output, the trial may too. With `--reference`, the
reference is the oracle. `UNWRITTEN_OUTPUT`, `NONDETERMINISTIC` and `OFF_STREAM` are probabilistic: a pass
means the defect did not show, not that it is absent. `OFF_STREAM` was verified against a SYCL
kernel submitting to a `sycl::queue` of its own (caught on every run); a torch op issued on
another torch stream was not caught, so do not read a pass as covering that case. A wrong value fails every value check that runs after it, so read
the names together: `UNWRITTEN_OUTPUT` among them points at unwritten elements,
`OFF_STREAM` alone at the queue. A host command
(`--external-benchmark`) owns its own checks; none of these run on that path.

## A kernel from another repo with no reference

A baseline oracle proves "the same as before", never "correct": a bug the repo kernel already
has passes every check, and every trial inherits it. A `_pytorch.py` the locator derives from
that kernel's source follows the same numerics. If the op matters, write a PyTorch
reference (a `Model` with `forward`, `get_inputs()` and `get_init_inputs()`; see the
[README](../README.md#writing-the-model-class)) and pass it with `--reference`. That takes
minutes, and nothing else catches a bug that is already there.
