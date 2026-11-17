# Optimizing one kernel

`xe-forge` takes one kernel, or a PyTorch op with no kernel yet, and returns an optimized
kernel or a patch against the repository the kernel came from. Every flag and environment
variable named below is in [REFERENCE.md](REFERENCE.md).

```bash
xe-forge -i kernel.py -s kernel.yaml -o kernel_optimized.py
xe-forge -i kernel.py -s kernel.yaml --engine claude --workspace ws/kernel --auto-launch
```

## Inputs

| Input | Flag | What it is |
|---|---|---|
| Kernel file | `-i` | A file with the target DSL's suffix (`.py` for `triton`, `gluon` and `opencl`, `.cpp` for `sycl`, `.cu` for `cuda`) is the **baseline**. A file with any other suffix is the **reference**, and a workspace engine writes the kernel |
| Reference | `--reference` | The semantic reference the kernel must match. Without it, `<stem>_pytorch.py` beside a baseline `-i` is used when it exists |
| Spec | `-s` | Shapes, dtypes, tolerances and FLOP of each benchmark variant |
| Name | `-n` | The kernel's name; with `--kernel-repo`, the op's name as the repository registers or calls it |

With a `.py` DSL a PyTorch file passed with `-i` is the baseline, so pass a reference with
`--reference`. `-i` is not needed with `--kernel-repo` and a workspace engine. A host project can take over
building, validation and timing ([REFERENCE.md](REFERENCE.md#host-integration)).

### The kernel file

A kernel file defines `class Model(torch.nn.Module)` whose `forward` takes the spec's
inputs in order, launches the kernel and returns the output tensor(s). Arguments of
`__init__` come from the spec's `inits`. A `Model` may define
`get_example_inputs(self, input_shapes, device)` to build inputs that random tensors of the
spec's shapes cannot be (sorted indices, rows that sum to one).

```python
class Model(torch.nn.Module):
    def __init__(self, D_HEAD: int):
        super().__init__()
        self.sm_scale = 1.0 / D_HEAD**0.5

    def forward(self, Q, K, V):
        out = torch.empty_like(Q)
        grid = (triton.cdiv(Q.shape[2], 128), Q.shape[0] * Q.shape[1])
        attn_fwd[grid](Q, K, V, out, self.sm_scale, BLOCK_M=128)
        return out
```

### The reference file

A reference is a PyTorch file with a `Model`, `get_inputs()` and optionally
`get_init_inputs()`. When the spec declares no `inputs:` (or there is no spec), these
functions build the workload, taking the variant's `dims` as keyword arguments; a dim
neither takes is an error. Baseline and trial are then built with the reference's init
arguments and state dict and compared on outputs and registered buffers. An op that writes
rows in a nondeterministic order may define `canonicalize` on the reference; it runs on
every arm's output before the comparison, untimed. When the spec declares `inputs:`, the
inputs are random tensors at its shapes.

### The spec

```yaml
inputs:
  Q: {shape: [B, H, S, D], dtype: bfloat16}
  K: {shape: [B, H, S, D], dtype: bfloat16}
  V: {shape: [B, H, S, D], dtype: bfloat16}
inits:
  - D_HEAD: D
default_variant: bench-gpu
bench-gpu:
  - params: [Q, K, V]
    dims: {B: 4, H: 32, S: 4096, D: 128}
    flop: "4*B*H*S*S*D"
    bytes: "2*4*B*H*S*D"
    rtol: 1.0e-2
    atol: 1.0e-2
bench-gpu-1:
  - params: [Q, K, V]
    dims: {B: 1, H: 32, S: 512, D: 128}
    flop: "4*B*H*S*S*D"
weights: {bench-gpu: 0.8, bench-gpu-1: 0.2}
captured_us: {bench-gpu: 812.0, bench-gpu-1: 41.5}
```

| Key | Meaning |
|---|---|
| `inputs` | Each input's `shape` (dimension names) and `dtype`. Optional when a reference builds the inputs |
| `inits` | `Model.__init__` arguments in order, each `{name: DIM}`; the value is a dim of the variant or a literal |
| `ci`, `bench-gpu`, `bench-xpu`, `bench-cpu` | Variants. Any other key holding a list is a variant too (`bench-gpu-1`, `decode`) |
| `default_variant` | The variant used when `--variant` is not given |
| `weights` | `{variant: weight}`, summing to 1: the shapes a trial is scored on together, each by its share of the kernel's time |
| `captured_us` | `{variant: us}`: the kernel's device time per call at that shape, as the model ran it. The frontend writes it; see the benchmark gate below |

A variant entry has `params` (input names), `dims` (values of the dimension names),
optional `dtype` (overrides the inputs' dtype), `flop` and `bytes` (Python expressions of
the dims) and `rtol` / `atol`. The dspy engine takes tolerances from `--rtol` / `--atol`,
then the variant, then `CORRECTNESS_RTOL` / `CORRECTNESS_ATOL`; `benchmark` from the variant.

The variant run is `--variant`, else the spec's `default_variant`, else `bench-gpu`
(`bench-xpu` for `--dsl sycl`).

## Engines

| Engine | What runs | Notes |
|---|---|---|
| `dspy` (default) | A DSPy pipeline: analysis, a stage plan, one optimizer agent per stage with compile, run and compare feedback (`AGENT_STRATEGY`, `AGENT_MAX_ITERATIONS`) | The only engine that reads `--stages` and `--no-correctness` and loads the knowledge base |
| `claude` | Generates an agent workspace and, with `--auto-launch`, runs `claude -p "/optimize-kernel <name>"` in it | Without `--auto-launch` it prints the command to run yourself |
| `codex` | The same workspace, with `AGENTS.md` and `.codex/agents/` made from it, run by `codex exec` | `--auto-launch` as for `claude` |
| `dspy-agent` | The same workspace, driven by a DSPy tool-calling agent | Always runs |

`claude`, `codex` and `dspy-agent` are the workspace engines. The LLM settings
(`LLM_MODEL`, `OPENAI_API_BASE`, `OPENAI_API_KEY`) reach the agent CLIs as their own
variables ([REFERENCE.md](REFERENCE.md#set-by-xe-forge-for-the-programs-it-starts)).

A workspace holds:

```
CLAUDE.md, config.yaml          the workflow and its settings (AGENTS.md for codex)
.claude/commands/optimize-kernel.md
.claude/agents/                 tool-runner; kernel-locator and port-back with --kernel-repo;
                                workload-inspector with --dataset-record
test_kernels/<name><ext>        the baseline
test_kernels/<name>_pytorch.py  the reference, when there is one
test_kernels/<name>.yaml        the spec
knowledge_base -> ...           a link to the repository's knowledge base
trials/<name>/                  the trial tree
output/                         the winner, and with --kernel-repo the patch
claude-session.log              the headless session's output (agent-session.log for dspy-agent)
```

## DSLs and their stages

`--stages` (dspy engine) picks stages by name; the plan is filtered to those the DSL
supports (`src/xe_forge/dsl_registry.py`). An unknown name is skipped with a warning.

| Stage | What it does | triton | gluon | sycl | opencl | cuda |
|---|---|---|---|---|---|---|
| `analysis` | Finds the issues; always first | yes | yes | yes | yes | yes |
| `algorithmic` | Mathematical rewrites: common subexpressions, reassociation, tree reductions | yes | yes | yes | yes | yes |
| `dtype_fix` | Accumulator precision, float64 removal, needless casts | yes | yes | yes | yes | yes |
| `fusion` | Fuses launches and elementwise or reduction chains | yes | yes | yes | yes | yes |
| `memory_access` | Coalescing, inner-loop transposes, register pressure | yes | yes | yes | yes | yes |
| `tensor_descriptors` | Pointer arithmetic and block pointers to `tl.make_tensor_descriptor` (`block_pointers` is an alias) | yes | | | | |
| `persistent_kernel` | Persistent-kernel scheduling | yes | | | | yes |
| `device_specific` | Tile sizes, warps, swizzling for the device | yes | yes | yes | yes | yes |
| `autotuning` | Adds or widens `@triton.autotune` configurations | yes | yes | | | yes |
| `discovery` | Optimizations no other stage covers | yes | yes | yes | yes | yes |

An OpenCL kernel is a Python file holding the OpenCL C source as a string, built and
launched through `xe_forge.opencl`. Adding a DSL: [DSL.md](DSL.md).

## Trials and the benchmark gate

In a workspace, the session saves each attempt as a trial (`xe-forge-skill trial save`)
and measures it with `xe-forge-skill benchmark --trial-id`, which records the result in
the tree. `trial finalize --require-measured` keeps only results the tool recorded.

- `benchmark` measures with the host's command (`--external-benchmark`) or, when asked by
  name, with the built-in executor (`--builtin-benchmark`, which the workspace adds when no
  host command is set). With neither it fails with `VERDICT: NO_BENCHMARK_CONFIGURED`.
- The first passing benchmark pins the tree to its variant; a recorded benchmark on
  another variant is refused.
- Two times closer than `BENCHMARK_MIN_EFFECT` count as equal. Parity is finalized; a
  regression never is.
- **Weighted shapes.** With `weights` in the spec, `benchmark` without `--variant` runs
  every weighted variant and reports one result. Every shape must pass. A shape measured
  slower makes the trial a regression at that shape's ratio (`VERDICT: SLOWER_ON`),
  whatever the others gain. Otherwise the result is the time-weighted total,
  `1 / sum(weight / ratio)`, where a shape too close to call counts as 1. `--baseline-us`
  is refused, since each shape's baseline is timed afresh.
- **Baseline against captured time.** For every shape run, weighted or one `--variant`,
  when `captured_us` names it and the baseline runs there more than `XE_FORGE_BASELINE_SLACK` times
  slower than the captured time, `benchmark` fails with
  `VERDICT: BASELINE_NOT_CAPTURED_KERNEL` and prints both times. The baseline is then not
  the kernel the model ran, and every trial would be scored against the wrong thing; make
  the baseline run that kernel, through the library's own call where part of it cannot be
  built from source.

The dspy engine has no trial session: each stage's optimizer retries on compile, run and
compare feedback, the kernel is re-analyzed after each stage, and the final kernel is
measured against the original.

## What decides correctness

Every run has one oracle, decided by the flags. A trial that disagrees with it is wrong.
The table is for the workspace engines. The dspy engine compares each stage's output with
the kernel the stage started from; it reads the reference only in analysis.

| You have | Oracle | Flags (plus `--engine claude\|codex\|dspy-agent --dsl <dsl> --workspace <ws>`) |
|---|---|---|
| A PyTorch op, no kernel (`sycl`, `cuda`: a `.py` DSL takes `op.py` as the baseline) | the op, written to `test_kernels/<name>_pytorch.py`; the session writes the kernel | `-i op.py -n <name> -s spec.yaml` |
| A kernel and a reference | the reference: baseline and every trial must match it | `-i kernel.cpp --reference ref.py -n <name> -s spec.yaml` |
| A kernel, no reference | the input kernel: trials must match its outputs | `-i kernel.cpp -n <name> -s spec.yaml` |
| A kernel in another repository, no reference | the copy of the repository's kernel | `--kernel-repo <repo> -n <name> -s spec.yaml` |
| A kernel in another repository and a reference | the reference | `--kernel-repo <repo> --reference ref.py -n <name> -s spec.yaml` |
| A host project that measures | the host command's `CORRECTNESS` and `VERDICT` lines | `--external-benchmark "<cmd> {trial} {baseline} {variant}"` |

Rules:

1. The oracle is read-only. A session never edits it to make a trial pass.
2. With a reference, the built-in `benchmark` checks the baseline as well as the trial
   against it, every time. When the reference builds the inputs, the session must show the
   baseline passes before it creates a trial; with a baseline oracle, that the baseline
   builds, runs and repeats deterministically.
3. A host result with no correctness verdict is never a pass, and `DONE` must be the last
   line the host command prints (`src/xe_forge/external.py` has the contract).
4. `--no-correctness` switches checking off for the dspy engine only. A workspace always
   checks.

A baseline oracle proves "the same as before", not "correct": a bug the kernel already has
passes every check. If the op matters, write a reference and pass it with `--reference`.

## A kernel in another repository

`--kernel-repo PATH` optimizes a kernel that lives in a git checkout, with any workspace
engine. The checkout is read-only.

1. The `kernel-locator` agent explores the repository and writes
   `experiments/kernel_profile.md`: the commit, the file and line of every region, the
   build target and flags, the repository's test and benchmark.
2. Without `-i`, the session writes the baseline as a self-contained copy of the
   repository's kernel, every kernel the op launches in launch order. With an `-i` in the
   target DSL, that file is the baseline and must be a copy of the repository's kernel,
   since port-back diffs baseline against winner.
3. Without a reference, the spec decides the inputs. A spec without `inputs:` makes the
   baseline build them the way the repository's test or call site does, citing file and
   line, and the session first checks that the copy builds, runs and repeats
   deterministically. A spec with `inputs:` uses random tensors, and for a standard op the
   locator writes `<name>_pytorch.py` from the kernel's source.
4. Trials run; `trial finalize` writes `output/<name>_optimized<ext>`.
5. The `port-back` agent turns the winner into a patch (below).

`--target library` optimizes how the repository calls a library (a GEMM, a convolution)
instead of a kernel: the baseline reproduces the call through the library's API, a trial
changes only how the call is made, and the patch goes to the call site, never the library.

`--integration-repo PATH` (repeatable) names repositories the kernel is wired into; a
winner may then change the op's interface and its call sites there.

## Port-back output

The port-back agent clones each repository privately (`xe-forge-skill upstream clone`),
commits the change there, writes `output/PR.md`, and runs `xe-forge-skill upstream patch`,
which writes:

- `output/upstream.patch` (and `output/upstream-<name>.patch` for each further clone)
  with a `.mbox` of the commits beside each;
- `output/PORT.txt`: the kind of port, base commit and touched files per repository, the
  best trial, and what was verified;
- the measured numbers, appended to `output/PR.md`.

`--kind upstream` keeps the op's schema, signature and call sites. `--kind serving-only`
is for a winner that needs a dispatch by shape or a changed call site; it wires the winner
into the stack and is not for upstream. Touched Python files are compiled; other files are
reported unbuilt.

With `--stack-python`, the agent also builds the port beside that interpreter's stack (in a
new venv under `stack/`, the stack itself unchanged), runs the repository's tests for the op
unpatched and patched, runs the model briefly, and records the verdict with
`upstream patch --stack "PASS|FAIL <evidence>"`. `upstream patch` refuses to finish
without a verdict when a stack is configured, and a port is done only on PASS.

## Knowledge base

The YAML knowledge base is off by default (`KNOWLEDGE_BASE_ENABLED`) and is read by the
dspy engine; workspaces get a link to it. It is loaded from `KNOWLEDGE_DIR` in this order:

```
knowledge_base/
  common/                 every DSL
  <dsl>/common/           every device of the DSL
  <dsl>/<device>/         this DSL on this device, e.g. triton/xpu/
    *.yaml                patterns and constraints, each tagged with a stage
    examples/index.yaml   reference kernels and the stages they show
```

Each stage sees only its own patterns, constraints and examples. The file format is in
[DSL.md](DSL.md#step-7-knowledge-base).

## Troubleshooting

| Symptom | Cause |
|---|---|
| `Model.__init__() missing required positional argument` | The spec has no `inits`, or an `inits` value names no dim of the variant |
| `spec dim X is not a parameter of the reference's get_inputs/get_init_inputs` | The variant's dims and the reference's functions disagree |
| `VERDICT: NO_BENCHMARK_CONFIGURED` | `benchmark` was run with neither a host command nor `--builtin-benchmark` |
| `VERDICT: BASELINE_NOT_CAPTURED_KERNEL` | The baseline is not the kernel the model ran (see the benchmark gate) |
| `'claude' CLI not found in PATH; auto-launch requested` | The agent CLI is not installed; the workspace was only generated |
| `session completed but recorded no correct trial at or above baseline` | No trial was correct and at least as fast as the baseline; the session log says why |
| Connection errors or retries (dspy) | The prompt may be too large: run fewer `--stages`, or use a model with a larger context |
