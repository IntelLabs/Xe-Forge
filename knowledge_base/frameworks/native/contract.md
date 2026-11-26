# Engine contract

The run, `patch check` and the capture measure an engine only through this contract.

## Repository

- Only the patch step builds an engine for a run: the capture runs what a patch made and
  passed its check, never a build of its own from a repository it finds.
- When the model's repository is already under `--kernel-repos`, the patch reuses it: its
  base and branch as they are, the engine brought up to this contract in new commits on top.
- Otherwise a new git repository under the run's `--kernel-repos`, named for the model
  (`native-<model>`). Its first commit is the skeleton: the runtime from `runtime/` brought
  into the layout below, the build, and the binding with no model in it. That commit is the
  patch's base; the model's engine is committed on top of it, on a branch `x-forge/<model>`,
  so HEAD is the base plus the engine. The skeleton carries a copy of `patterns.yaml` as
  `docs/patterns.yaml`, and its README points to it: the agents that later optimize the
  engine's kernels work in this repository and read it there.
- Its conventions are this contract's: C++17, formatted consistently, no generated files
  committed, a `README.md` saying how to build and run it.
- PR.md (beside patch.json) describes the engine: its layout, its kernels per device, what it
  was checked with.

## Layout

```
include/xf.h            the core's header: context, runtime services, model API
include/xf_kernels.h    the kernel interface: plain C++ types only, no SYCL type in it
src/                    the core: safetensors, weight binding, hooks, the model (model.cpp)
src/kernels/cpu/        host kernels, C++ (intrinsics where the profile says)
src/kernels/sycl/       Intel GPU kernels, SYCL, built only for a GPU device
python/<package>/       the binding (below)
CMakeLists.txt          builds libxf.so (out of tree: build.sh gives it its directory); -DXF_SYCL=ON adds the SYCL kernels (icpx -fsycl)
```

- The core calls kernels only through `xf_kernels.h`: one function per op, taking the
  context, device pointers and sizes. A device's kernels implement that interface; the core
  never includes a SYCL header. The core holds activations and the KV cache in the device's
  memory (host memory on the host, USM device memory on a GPU) and copies only the ids in and
  the logits out.
- The engine serves the model's whole context: the KV cache, and every buffer sized by
  positions, holds the context the model's config declares (`max_position_embeddings`, or
  its equivalent), read from the checkpoint, never a constant of the engine. A sequence
  longer than that fails with an error naming both lengths. A run of several rows runs the
  launcher at each row's batch and lengths, long prompts included, and each is a workload the
  engine is optimized for.
- Host kernels are `noinline` functions named by op (`xf_cpu_matmul_q4`, ...), so a sampled
  profile attributes time to them. SYCL kernels are named by op too (the kernel name type), so
  the device trace attributes time to them.
- Plain and correct kernels first: obvious loops, long reductions in double, threads across
  independent outputs. Making each kernel fast is the run's later work, kernel by kernel.

## Structure

What no kernel change can fix later is the core's, and is part of the engine the check
gates: write it this way from the start.

- One parallel region per forward, phases separated by a spin barrier
  (`decode_single_region`), or a persistent spinning pool (`spinning_pool`): never a fork and
  join per op. At one token a step's work is microseconds; a dispatch per op costs as much.
- Ops that read the same input in one pass over a virtual concatenation of their weights
  (`virtual_concat_projections`), activations in the epilogue of the op before
  (`gemm_epilogue_activation`), norms recomputed per thread rather than shared through a
  barrier (`redundant_norm_before_gemv`).
- Separate decode (one token) and prefill paths (`decode_prefill_dispatch`).
- Per-thread state sized from the resolved thread count (`team_sized_from_resolved_threads`).

## Build

`build.sh` beside patch.json (patch.json `"build": "build.sh"`) builds any tree of the
repository into any venv: `SRC` (default: the clone) is the tree, `VENV` (default: `./venv`)
the venv layered over the framework's interpreter. It builds out of tree, in a build
directory under `$VENV`, writes nothing into `$SRC`, and then runs the repository's own kernel tests against that build, exiting nonzero
when the build or a test fails. The run builds every kernel trial with it, each in a tree of
its own, and times the trial's kernel on that build on its own, at the shapes the run's
workloads give it, against the baseline's kernel; only `--prove` runs the whole engine. Keep the tests
quick (seconds) and covering every kernel at the shapes the model serves, decode and prefill:
they are each trial's numeric check beside the tokens.

## Kernel references and checks

Every kernel the core calls has its op as PyTorch, and a check that holds the built kernel
to it. The run checks every kernel trial with them, at every shape the model serves, before
timing it; the PyTorch op is also the reference the agent optimizing that kernel reads.

- `reference/<kernel>.py`, named by the kernel's symbol (`reference/xf_cpu_linear.py`): the
  op copied from the model's code with its semantics exact (norm placement and offsets,
  rotary layout, scaling, masks, how the checkpoint packs the weight), as `class
  Model(torch.nn.Module)` with `get_init_inputs(**dims)` and `get_inputs(**dims)`. The dims are
  the spec's: each argument's sizes named by its role on the timeline and its index
  (`X_D0`, `WEIGHT_D1`), plus the named sizes (`M`, `N`, `K`). The variants of one kernel
  may name different arguments (a bias, a residual present or not): `get_inputs(**dims)`
  accepts every set the spec holds and builds what the names say. Inputs are built as the
  model builds them -- structured inputs (positions from the cache length, fused projections'
  segment sizes from the model's config) as the model calls the kernel. `--dtype` is the
  activations' dtype; weights are the dtype the engine stores them in.
- patch.json `"kernel_check": "<package>.kernel_check"`: `python -m <module> --kernel <kernel>
  --dims '<json>' --dtype <dtype> --rtol R --atol A` builds those inputs, runs the kernel of
  this build (through the library, its own thread team, on its own) and the reference, prints
  the largest absolute and relative difference, and exits 0 when the kernel is within the
  tolerance everywhere, 1 otherwise. `--help` exits 0.
- The module is also imported, under a name of its own, by X-Forge's interpreter (torch,
  numpy and ctypes; the framework's other packages are not there), with another build's
  beside it: the package and its modules import one another relatively, never by the
  package's name, and each loads the library of its own build. It offers:

  ```python
  check(kernel, dims, dtype, rtol, atol, threads=0) -> {"ok": bool, "max_abs": float, "max_rel": float}
  prepare(kernel, dims, dtype, threads=0) -> call   # call.forward(), call.reset()
  ```

  `check` is the command's check on one shape: it returns `"ok": False` for a kernel that is
  wrong and never exits the process. `prepare` builds the inputs as `check` builds them,
  seeded the same way so two builds get the same values, converted once into what the library
  takes, the outputs allocated once; `call.forward()` is the library's kernel call alone, on
  its own team, and `call.reset()` (optional when the kernel writes no input) restores what
  the kernel writes in place, so every forward does the same work. The library binds no
  symbol once per process (no `STB_GNU_UNIQUE`; GCC: `-fno-gnu-unique`): two builds of it in
  one process would share it, and the run refuses such a build. The run checks every shape of a trial with `check`, in one process, and times
  `forward` with its own timer, the baseline's and the trial's builds in one process, in
  alternating rounds; the module holds no timer.
- `build.sh`'s tests may be these checks at a few shapes; the run's check covers every shape.

## Costs

The engine chooses its kernels' arithmetic, and the checks judge it, so a cost bounds what
any engine the checks accept must do, not what this kernel does today. A matmul's products
can run in the dtype the weights are stored in: when that dtype is narrower than the
activations' and has a matmul to measure, it is the cost's `"compute"`, whatever dtype the
activations hold. Low-bit weights take the dtype their products run in (integer dot
products). The flops stay the operation's count. A kernel that splits the activations to
keep their accuracy does a multiple of that work, and its efficiency shows the multiple.

## Threads and placement

The engine takes the thread count it is given (the binding's `threads`, else the OpenMP
default) and never binds threads or memory: the job's launcher placed the process, and the
engine runs where it was put.

## Binding

`python/<package>` is imported from a venv layered over the framework's interpreter (which
holds torch and transformers), with nothing installed into that interpreter. On import it
loads the library its own build made (under `$VENV`, see Build) with ctypes, never one from
the clone, and offers:

```python
Engine(weights_dir, device="cpu" | "xpu" | "xpu:N", threads=None)
Engine.threads                                                              # the thread count it resolved
Engine.generate(ids: list[list[int]], new_tokens: int) -> list[list[int]]   # greedy, no stop at EOS
Engine.forward(ids: list[int], pos: int) -> numpy.ndarray                   # last-position logits, fp32
Engine.reset()
Engine.timeline(path: str | None)                                           # XF_TIMELINE on, to path, or off
```

`timeline` switches the timeline between calls, so a launcher times repeats with it on and
one with it off in one process.

Greedy is the argmax of the raw logits, every sequence decoded on its own state; a batch is
its sequences in turn until the engine serves a batch at once. patch.json's "package" is this
package.

## Weights and quantization

- The engine reads the checkpoint's safetensors as stored: bf16 stays bf16 in memory, packed
  formats stay packed (`weights_as_stored`, `packed_formats_as_bytes`).
- Quantization is the engine's: `python -m <package>.quantize --scheme <scheme> --model <dir>
  -o <out>` writes the quantized weights the engine loads, and `<out>/dequantized/` holding,
  for every tensor it quantized, the float32 values its quantized form decodes to, in the
  checkpoint's names and shapes (safetensors). Made once per scheme and model, reused.
- generate.sh with `X_FORGE_QUANTIZATION` set runs the engine on the quantized weights and
  writes `$OUT/dequantized.json`: `{"weights": "<out>/dequantized"}`. `patch check` loads the
  original model, puts those tensors in its place (every one must exist there with the same
  shape) and gates the quantized tokens against that model; it reports how far each tensor is
  from the checkpoint.

## Hooks

The core calls `xf_hook_step` after every step of the forward pass, from one thread between
two barriers when the step runs inside a parallel region.

- `XF_DUMP=dir`: the outputs of the first forward call (`XF_DUMP_CALLS=n` for more) as raw
  fp32, plus `dir/index.jsonl`, one line per step: `call` (the forward's index), `block`,
  `layer`, `step`, `op`, `module`, `rows`, `numel`, `file`. `module` names the reference submodule whose forward output equals
  this output, flattened per token, `{i}` standing for `layer`; `""` where none does. `rows` is
  the number of tokens, or 1 for a value holding only the last position.
- `XF_TIMELINE=file` (or `Engine.timeline`): one JSON line per step of every forward, in
  the order the steps ran, on CLOCK_MONOTONIC (on a GPU, after the step's kernels completed):
  `{"op", "block", "layer", "step", "start_ns", "end_ns", "args": [{"name", "shape",
  "dtype"}]}`. `op` is the step's op type, the same name for the same op everywhere; `args`
  is every tensor the step reads or writes, weights included, each named by its role on the
  op (the same names on every step of that op), dtype in torch's names. This is the engine's
  own runtime description of its graph: the launcher turns each line into
  `HostTimeline.node(op, start_ns, end_ns)` inside its window and writes ops.json and each
  node's launch ranges from the same lines; the capture measures what the timeline costs
  against a repeat with it off.
- `XF_PROFILE=file`: one JSON line per (block.step, T) with summed time, for the engine's own
  quick look.

With none of them set a step pays no hook cost and no synchronization.

## Switches

- `XF_NO_AMX=1` turns every tile path off: the vector kernels run and give the same tokens.

## What the engine is checked with

- `patch check`: the tokens of `generate.sh` within the tolerance of transformers' best, fed
  the same prefix; with a quantization, the quantized tokens against the original model holding
  the dequantized export.
- Before that, the agent's own checks: each kernel against the op it reproduces (ctypes),
  incremental decoding giving the tokens a full recompute gives, and a small random model of
  the same architecture (every layer kind and ratio kept, windows below the prompt) before the
  real one.
- Each kernel trial: `build.sh` (the build and the repository's kernel tests), the kernel
  against its reference at every shape (`check`), the tokens scored again on that build, and
  the kernel timed on its own against the baseline's at every weighted shape (`prepare`). It
  is profiled on its own too, at those shapes: per symbol its cycles, instructions, IPC and
  misses inside the kernel's calls.
- The whole engine against the reference model, step by step: `patch check` dumps the first
  forward (`XF_DUMP`) and reports every step's error against the module it names, and the
  worst, beside the tokens; `--prove` reports it for the merged engine.
- `--prove`: the same scoring again on the build with the run's winning kernels, and the
  whole engine timed against the one the run started from, at every workload: a kernel that
  is faster on its own and slows its neighbours shows there.
