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

## A kernel from another repo with no reference

A baseline oracle proves "the same as before", never "correct": a bug the repo kernel already
has passes every check, and every trial inherits it. A `_pytorch.py` the locator derives from
that kernel's source follows the same numerics. If the op matters, write a PyTorch
reference (a `Model` with `forward`, `get_inputs()` and `get_init_inputs()`; see the
[README](../README.md#writing-the-model-class)) and pass it with `--reference`. That takes
minutes, and nothing else catches a bug that is already there.
