# Framework frontend: from a running model to a kernel manifest

`xe-forge capture | analyze | optimize | run` take a model running under its inference
framework and turn what it executed into Xe-Forge runs:

```
xe-forge run --framework vllm --model Qwen/Qwen2.5-7B-Instruct -o runs/qwen \
    --framework-python <vllm-venv>/bin/python --top 3
```

produces the kernel manifest, the specs, one Xe-Forge workspace per chosen kernel, and
`runs/qwen/results/` with every measured win (optimized kernel and port-back patch) and a
`SUMMARY.md` accounting for every kernel the model ran. The step-by-step procedure, with
the stack check and the prove step, is [FRAMEWORK_E2E.md](FRAMEWORK_E2E.md).

The frontend has one job: decide *what* to optimize and at *which* shapes, from
measurement. It never writes kernel or reference code. It hands Xe-Forge the same thing a
person writes by hand for a model (`scripts/nemotron30b/kernels.yaml` and its specs), and
Xe-Forge's own workspace does everything after that, exactly as for a hand-written
manifest:

| Frontend (`src/xe_forge/frontend/`) | Xe-Forge (unchanged) |
|---|---|
| run the framework, record ops, kernels, shapes, dtypes, device time, calls | locate the kernel in its repository (`kernel-locator`) |
| canonicalize, deduplicate, aggregate into workloads | build inputs; the baseline copy is the oracle |
| rank by end-to-end share; decide optimize or skip, with the reason | plan and optimize (engines, stages) |
| **name** each kernel: the symbol to grep for, its repository, its DSL | benchmark and validate (`TrialManager`) |
| emit `kernels.yaml`, no-inputs specs, a `DatasetRecord` per kernel | port back (`upstream.patch`) |

## Pipeline

```
 capture   (framework interpreter, e.g. the vLLM venv; this package on PYTHONPATH)
   adapter.capture()  -> raw/trace.json.gz, raw/run.json, raw/schemas.json
   torch_trace        -> CapturedOp rows (owner-charged device time)
   canonical          -> Workloads (var axes measured)        -> capture.json
 analyze   (Xe-Forge venv)
   gates + rank       -> analysis.json, gates.log, report.md
   manifest.emit      -> kernels.yaml, specs/<kernel>.yaml, records/<kernel>.json
 ══════════ boundary: YAML only ══════════
 optimize  one `python -m xe_forge.cli --name --kernel-repo --spec --variant` per entry
           (the command scripts/gen_kernel_sbatch.py writes; same manifest works there)
 collect   TrialManager.best_measured + keepable -> results/<kernel>__<variant>/, SUMMARY.md
```

Each stage is a subcommand and can be re-run alone; `run` reuses an existing
`capture.json`, and `xe-forge capture --reparse <run>/raw -o <run>/capture.json` rebuilds
the capture from the raw trace without running the model again.

## Capture

### One parser for every PyTorch framework

vLLM and SGLang both run on PyTorch, so both are captured as a `torch.profiler` chrome
trace (`record_shapes=True`, `with_stack=True`) and read by one parser,
`torch_trace.py`:

- **Attribution.** A device kernel names the host op that launched it through
  `External id`. That op is charged to its *outermost* enclosing dispatched op
  (`namespace::op`), so an op whose work happens in nested calls collects its children's
  time and charges never overlap -- the rule FlashInfer-Bench's `device/attribution.py`
  applies to in-memory profiler events, here on the exported trace so a capture can be
  re-read anywhere.
- **Python-registered ops.** `vllm::unified_attention_with_output` reaches its kernel
  through `_vllm_fa2_C::varlen_fwd`. When the launcher chain under an owner holds a
  non-ATen op, the row is keyed on *that* op with *its* arguments (the paged KV cache, the
  block table), and the owner is kept in `framework_meta.owner`. ATen helpers inside the
  Python op stay with the owner.
- **Kernels launched outside any op** (vLLM's Triton kernels called from Python) are their
  own `kernel::<name>` rows, with no argument shapes.
- **Call site.** The `nn.Module` frames enclosing the owner, instance numbers dropped:
  `Qwen2ForCausalLM/Qwen2DecoderLayer/Qwen2MLP/MergedColumnParallelLinear`.
- **Argument names** come from the op's schema (`torch.ops.<ns>.<op>`), looked up in the
  framework's interpreter where provider ops are registered, and kept in
  `raw/schemas.json` so a re-parse elsewhere has them.

### Adapters

#### Adding a framework

One file in `adapters/` and one line in `adapters/__init__.py:ADAPTERS`. The module
provides:

| | required | does |
|---|---|---|
| `add_arguments(parser)` | yes | the framework's own flags |
| `capture(args, raw_dir) -> dict` | yes | runs the framework (in its own venv; capture runs under `--framework-python` with this package on `PYTHONPATH`), leaves artefacts in `raw/`, returns run metadata |
| `parse(raw_dir) -> TraceSummary` | if `raw/` is not torch.profiler traces | `CapturedOp` rows: op, kernel symbols with device time, calls, argument shapes, call site |
| `name_op(op) -> Naming` | if ops are not torch ops | route (`kernel`/`library`/`aten`/`unnamed`), and for a kernel its name, repository and DSL |

Everything after that -- canonical workloads, ranking, the manifest, optimize, collect --
is shared. Where a framework reports no per-call shapes, `parse` puts the call's
`tokens` and `context` in `framework_meta`; they become the `TOKENS`/`CONTEXT` dims, the
token axis and its histogram, and the agent maps them onto the kernel's arguments.
A kernel whose DSL Xe-Forge does not write is named and ranked but stopped by the
`buildable` gate, so the manifest still says where the time goes.

OpenVINO's GPU kernels are OpenCL C, and `opencl` is an Xe-Forge DSL (`models.DSL`):
the trial module keeps the `.cl` source as a string and runs it with
`xe_forge.opencl.load`, which compiles it with oneAPI's `clang` and `llvm-spirv` and
loads the SPIR-V through SYCL on torch's XPU queue. Benchmark, correctness, trials and
port-back are the same as for SYCL. A kernel built by OpenVINO's kernel_selector gets the
generator's JIT constants as `defines`, read by the kernel-locator from its
`GetJitConstants`. A patch to an OpenVINO `.cl` takes effect once the GPU plugin is
rebuilt.

| framework | status | mechanism | framework-specific code |
|---|---|---|---|
| `torch` | done | profile any `module:factory` callable | ~60 lines; also the shared profiler region |
| `vllm` | done | in-process `LLM` (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, `enforce_eager`, TP 1); one warm-up request, then one profiled `generate` of `--batch` synthetic prompts of `--input-len` tokens and `--output-len` decode steps | ~100 lines |
| `sglang` | done | runs SGLang's own `python -m sglang.benchmark.one_batch --profile --profile-record-shapes --profile-activities CPU XPU` (in-process `ModelRunner`; `--attention-backend intel_xpu --page-size 64 --disable-cuda-graph`, TP 1), one process per phase -- on XPU Kineto cannot open a second profiling session in a process -- and every decode step profiled; the two exported traces are parsed and merged. Depends on SGLang's command line, not its internals | ~90 lines |
| `openvino` | done | optimum-intel `OVModelForCausalLM` on the exported IR, compiled with `PERF_COUNT`; a proxy around its infer request reads `get_profiling_info()` after every prefill/decode step; the runtime model gives port shapes and, where profiling says `undef`, `primitiveType`. OpenCL primitives are named as their source names them (`ocl::sdpa::opt__f16` -> `sdpa_opt`, repo `openvino`, dsl `opencl`); `undef` FullyConnected is oneDNN (ONEDNN_VERBOSE: matmul `jit:gemm:any`) -> library | ~150 lines |
| `llamacpp` | planned | `llama-eval-callback` for the graph (op, tensor, ggml type, shape), each distinct op timed by `test-backend-ops perf`; GGUF metadata via `gguf` | `parse()` + perf runner + tables |
| `fib` | planned | import FlashInfer-Bench `discovered.json` | `parse()` |

Each framework runs in its own venv (vLLM and SGLang pin different torch builds), and the
capture process imports only `xe_forge.frontend`: the package's top-level names resolve
lazily, so nothing of Xe-Forge's own dependency stack is needed in a framework's venv.
Several traces from one run (one per profiled phase) are parsed separately -- `External id`
restarts in each profiler session -- and merged row by row.

Limits of the vLLM adapter as built: eager mode only (graph replay would hide the ops, so
shares are the eager ones); tensor parallel 1; the profiled window is one request batch,
so prefill is one step and decode `--output-len` steps.

## `capture.json` (`xe_forge.capture/v1`)

Two levels. `ops` is what the trace showed, one row per (op, argument signature, call
site), raw strides included. `workloads` is the canonical form several ops share; analysis
reads only workloads, and each lists its `ops` so a decision traces back to observation.
Anything only one framework can say lives in `framework_meta` and is never read by
analysis. Abridged, from a Qwen2.5-0.5B capture:

```json
{
 "schema": "xe_forge.capture/v1",
 "run": {"id": "Qwen2.5-0.5B-Instruct-vllm-20261009T202523", "framework": "vllm",
         "framework_version": "0.28.1rc1.dev628+g2a02f6efe.xpu",
         "model": {"name": "Qwen/Qwen2.5-0.5B-Instruct", "hidden_size": 896, "...": "..."},
         "runtime": {"tp": 1, "enforce_eager": true, "batch": 1, "input_len": 512, "output_len": 16},
         "device": {"type": "xpu", "name": "..."},
         "total_device_us": 42553.9, "attributed_pct": 99.37, "kernel_invocations": 4307,
         "raw": ["raw/trace.json.gz"], "command": "...", "env": {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"}},
 "ops": [{"id": "op-0007", "framework_op": "_C::fused_add_rms_norm",
          "args": [{"shape": [1, 896], "dtype": "bfloat16", "stride": [896, 1]}, "...", null],
          "scalars": [null, null, null, "9.9999999999999995e-07"],
          "kernel_symbols": {"vllm::fused_add_rms_norm_kernel<...>": 1403.901},
          "calls": 705, "device_us": 1403.901,
          "call_site": "Qwen2ForCausalLM/Qwen2DecoderLayer/RMSNorm",
          "impl_class": "provider",
          "framework_meta": {"arg_names": ["input", "residual", "weight", "epsilon"]}}],
 "workloads": [{"workload_id": "norm/64226c0fa8", "family": "norm",
                "dims": {"INPUT_D0": "var", "INPUT_D1": 896, "RESIDUAL_D0": "var",
                         "RESIDUAL_D1": 896, "WEIGHT_D0": 896},
                "dtypes": ["bfloat16", "bfloat16", "bfloat16"], "layout": ["c", "c", "c"],
                "attrs": {"epsilon": "9.9999999999999995e-07"},
                "var_axes": {"INPUT_D0": {"1": {"calls": 720, "device_us": 1434.418},
                                          "448": {"calls": 48, "device_us": 170.809}}},
                "calls": 768, "device_us": 1605.227, "share": 0.037722,
                "ops": ["op-0007", "op-0030", "op-0042", "op-0060"],
                "call_sites": ["Qwen2ForCausalLM/Qwen2DecoderLayer/RMSNorm", "Qwen2ForCausalLM/RMSNorm"],
                "naming": {"route": "kernel", "name": "fused_add_rms_norm",
                           "repo": "vllm-xpu-kernels", "dsl": "sycl", "via": "torch.ops._C"}}]
}
```

### Canonicalization rules (`canonical.py`)

- **Dims** are named after the op's schema: `INPUT_D1` is dim 1 of the argument called
  `input`; an integer scalar argument is a dim too (`MAX_SEQLEN_K`, `HEAD_SIZE`). GEMM-shaped
  ATen ops (`linear`, `mm`, `addmm`, `bmm`, `matmul`) get `M`/`N`/`K`[`/B`] instead.
- **Layout class** per tensor is read from the innermost dimension only (`c` contiguous,
  `t` last two transposed, `s` strided). Outer pitches depend on the token count -- a
  one-row view of a fused QKV output reads as contiguous -- so they stay in the op's raw
  strides instead of splitting a workload.
- **Var axes are measured, not declared.** The token count of a call is the leading dim of
  its first tensor (`M` for a GEMM). A dim is var when it equals the token count in every
  call of its group and the token count varied. Integer arguments that still differ once
  the tensor shapes agree are runtime sizes -- a context length growing every decode step --
  and are var too; the value each token count was captured with is kept beside it
  (`var_axes[axis][value].with`), so a decode variant carries a real context length.
  The same holds for a tensor dim that takes one value per token count and changes with
  it -- a page table one page wider once decode runs past the prompt -- while two values
  at one token count are two workloads (two norms of different width).
- **Identity.** `workload_id = family/sha1(op, dtypes, layout, attrs, dims with var
  masked)[:10]`. Two calls are one workload exactly when one kernel at one configuration
  could serve both. Family (gemm, attention, norm, rope, activation, moe, cache, sampling,
  copy, elementwise, other) is a regex on the op name, falling back to the Python owner's.
- **Aggregation.** Ops sum per (op, signature, call site); workloads sum over ops, layers
  collapsed by the call site and phases folded into the var histogram.

### Naming: an agent (`frontend-name.md.j2` -> `names.json`)

Where an op is implemented is read from the framework and the repositories, not matched
against tables in code. After capture, `xe-forge run` gives one agent session the ops
covering `--name-coverage` of device time. For each one it gets the op, its owner, the
kernels it launched, its argument names, shapes and dtypes, and its call site, plus the
model config and the kernel repositories on the machine. The agent answers per op key:

| field | meaning |
|---|---|
| `route` | `kernel` (a kernel in a repository Xe-Forge rewrites), `library` (a library primitive: tune the call), `aten` (inside PyTorch), `unnamed` (not established, with the reason) |
| `name`, `repo`, `dsl`, `via` | what the kernel-locator greps for, where, in which language, and the evidence |
| `family`, `paged` | the op family; whether it reads a paged buffer (no streaming bound) |
| `gemm` | for a matrix multiply: which argument is the activation, the weight, what is read with the weight, the output, where N sits, packing along K, grouping and the expert-count argument |

It also answers once per model with `model.topk` and `model.experts`. The capture is then
rebuilt with `--names`, and so are the prove arms' captures, so both arms key their
workloads alike. The key is the implementing op (OpenVINO: node type plus implementation),
which a patched build keeps. Without `names.json`, only `aten::` ops are named, by
PyTorch's own rule.

## Analyze: the decision

Per workload over the profiled window: `S` = share of device time; `G` = recoverable time
/ total, the end-to-end ceiling. Until the part is calibrated nothing bounds how far a
workload could go, so `G = S` and the roofline gate reports `UNKNOWN` rather than passing.
Ranking by share is the right order when all that is known is where the time goes: a 5 us
kernel called 100,000 times in a 30 s run (1.7%) outranks a 300 us kernel called twice
(0.002%), which fails `worth` whatever its latency.

Gates, in order, one line each in `gates.log`
(`GATE <workload_id> <gate> PASS|REJECT|UNKNOWN <evidence>`):

| gate | passes when |
|---|---|
| `routable` | naming found a kernel in an editable repository |
| `buildable` | its DSL is one Xe-Forge writes (`models.DSL`); otherwise "Xe-Forge has no <dsl> path" |
| `shaped` | the trace recorded argument shapes (a spec needs dims) |
| `bounded` | below its roofline bound -- `UNKNOWN` until calibrated |
| `worth` | `G >= --min-gain` (default 0.005) |
| `top` | its kernel is among the `--top` kernels by summed `G` |

Workloads are then grouped by kernel: one manifest entry per (repository, name), its spec
carrying the variants of every workload the kernel served. Rejected kernels stay in the
manifest, `enabled: false`, with the reason.

**Where an LLM belongs** (not built; the hooks are the gate list and the `reason` field):
naming an op no rule matched (its proposed dims checked against the recorded shapes), and
demoting -- never promoting -- a shortlisted kernel with a stated reason. Both via DSPy
signatures whose instructions live in `prompts/templates/` like the existing ones, the
LM configured once in `llm_setup.py`. Planning *how* to optimize is already Xe-Forge's.

## Manifest and specs

`kernels.yaml` is the format `scripts/gen_kernel_sbatch.py` consumes, plus provenance
fields it ignores:

```yaml
kernels:
- name: varlen_fwd                 # the symbol the kernel-locator greps for
  repo: vllm-xpu-kernels           # a path, or a directory beside this checkout
  dsl: sycl
  enabled: true
  workload_ids: [attention/b74679a4d2]
  priority: 0.109445               # summed G
  reason: 10.94% of device time; 384 calls; bound unknown
  spec: specs/varlen_fwd.yaml      # relative to the manifest
  record: records/varlen_fwd.json
  variants: [bench-t1, bench-t448] # costliest first; optimize runs the first by default
- name: linear
  repo: null
  enabled: false
  reason: '78.73% of device time; 1552 calls; library primitive: change the call, not the kernel'
```

`specs/varlen_fwd.yaml` is a no-inputs spec (as `scripts/nemotron30b/*.yaml`): the dims
go as keyword arguments to the baseline's `get_init_inputs`/`get_inputs`. Variants are
`bench-t<token count>`: per workload, the smallest set of captured token counts covering
90% of its time (at most 4); `ci` is the smallest. Every spec is validated with
`spec_loader.load_spec` when written.

`records/<kernel>.json` is a `DatasetRecord` (`--dataset-record`): `dataset` is the
capture, `variants` maps each variant to `<workload_id>@<axis>=<value>`, and
`input_fidelity` is `shapes-only` -- the shapes are the model's, the values are not.

## Optimize and collect

`xe-forge optimize kernels.yaml` runs, per enabled entry (`--only` to pick, first variant
unless `--all-variants`), sequentially:

```
python -m xe_forge.cli --engine claude --auto-launch --dsl <dsl> --device xpu \
    --workspace <run>/workspaces/<kernel>__<variant> --max-turns N \
    --name <kernel> --kernel-repo <repo> --spec <spec> --variant <variant> \
    --dataset-record <record> [--compiler-flags <XE_FORGE_SYCL_FLAGS>] [-- passthrough]
```

with the log beside the workspace (`<kernel>__<variant>.log`, `.status.json`). A repository
not found under `--kernel-repos` (default: the directory holding this checkout) is skipped
and recorded; nothing is cloned. `{kernel_repo}` in `--compiler-flags` expands per entry.
The environment -- GPU driver, oneAPI toolchain, the Claude Code CLI and the endpoint
configuration Xe-Forge reads -- is the caller's, as for a hand-launched run.

`collect` asks each workspace's trial tree for its best *measured* non-baseline trial and
keeps it only when `TrialManager.keepable` -- the test `finalize` and `upstream patch`
apply, so a measured regression is never copied. A kept result gets
`results/<kernel>__<variant>/output/` (the workspace's `output/`: finalized kernel, and
`<repo>.patch` when the session ported back) and `result.json`; `SUMMARY.md` has one row
per manifest entry: win / parity / no win / skipped / not run, with the reason.

## Library calls (oneDNN): tuned only below the card's roofline

A GEMM served by oneDNN is not a kernel Xe-Forge rewrites, but how the framework calls it
can still leave time on the table. Whether it does is measured, not assumed:

- `xe-forge calibrate` (GPU node) records the card's read bandwidth (a reduction over
  random data: constant data compresses and would read faster than any real tensor) and
  large-square matmul throughput per dtype it can multiply (a dtype it cannot is `None`,
  with the reason), under `$XE_FORGE_CACHE/calibration/`.
- `xe-forge analyze --calibration <json>` bounds every M/N/K GEMM workload, at each captured
  token count: `max((M*K + K*N + M*N) * size / bandwidth, 2*M*N*K / peak)`. A workload at
  `--at-roofline` (default 0.8) or more of its bound is rejected "at roofline" with its
  efficiency; below it, its recoverable time (not its whole share) ranks it, and a library
  workload becomes an entry with `target: library`, at the call site the naming agent named.
- `xe-forge optimize` passes `--target library`: the workspace reproduces the call as the
  repository makes it (oneDNN API, `ONEDNN_VERBOSE` for the implementation picked) and trials
  change only how it is called -- prepacked weights, attributes, primitive reuse, splitting.
  The port changes the call site, never oneDNN; the stack check and `compare` apply as for
  kernels.

Without `--calibration` nothing is tuned: library entries are listed per call site and stay
skipped.

## Bounds for any op, any architecture

- **Model metadata** is stored whole (`run.model.config`). What a family calls its top-k and
  expert count, and where a multimodal model nests its language model, is the naming
  agent's to read (`names.json` `model` -> `run.model_facts`).
- **Bytes are measured from the captured tensors** the naming agent identified: weight,
  scales, zero points and bias over E·N·K, the activation over M·K, the output over M·N.
  No quantization-format table. The compute peak is the calibration's for the activation
  dtype; for a dtype it has no measured peak for, the fastest measured one.
- **A grouped (MoE) call never gets E=1.** E is the local count the kernel was given (the
  argument the agent named, or the weight's leading dim); or the model's experts divided by
  the expert-parallel size; or the bound is unknown. A
  measured `active_experts` overrides the uniform-routing estimate. Each point's bound is
  capped at its measured time, so a violated assumption shows no headroom rather than
  invented headroom.
- **A bound registry:** `gemm` gets the roofline. Every other op gets the streaming bound
  (all its tensors read or written once, over bandwidth), except ops the agent marked
  `paged` (KV cache, block or page tables), which stay unknown until a family bounder
  (attention, MLA, linear attention, SSM, collective) is added as one entry. Ranking uses
  recoverable time, so an op already streaming at its bandwidth is no longer offered.

## Mergeable patches: the stack check

A per-kernel win is measured on a standalone copy. Whether the port works in its stack --
built the way the stack builds it, dispatched the way the stack dispatches it, for every
shape the op accepts -- is a separate question, answered before a patch is reported.

- **By the agent** (`xe-forge run --stack-check`, i.e. Xe-Forge `--stack-python <python>`):
  - The port-back agent builds the port into a new environment beside the stack's, never
    into it.
  - It runs the repository's own tests for the op, unpatched and patched, and runs the
    model briefly.
  - It records its verdict with `upstream patch --stack "PASS|FAIL <evidence>"`.
- **By a command of the user's** (`--external-stack-check CMD`, placeholders
  `{clone} {clones} {kernel} {kind} {workspace}`): `upstream patch` runs it once per port,
  over all its changed clones (`{clones}`, `os.pathsep`-joined), without LLM credentials in
  its environment. It builds from a copy, never inside a clone, prints
  `STACK: PASS|FAIL <reason>` and `DONE` last, and passes only with exit code 0.
- Either way `PORT.txt` records one `stack check <clones>: PASS|FAIL` line per port. A FAIL makes the port
  fail, and the port-back agent fixes the port and checks it again.
- `kind: upstream` must break no shape the op accepts. Either the new code is general, or
  the repository's own selection (a kernel selector's validation and priority, an
  implementation's applicability check) admits it only where it is valid, with the
  original kernel unchanged elsewhere.
- `kind: serving-only` is still produced for a winner valid only at the served shapes. The
  check then proves only that the stack runs with it.
- `results/SUMMARY.md` gains `kind`, `stack` and `mergeable` (upstream and PASS). A port
  that failed is kept as `*.patch.rejected`, not handed out.

With neither set, nothing changes: ports are reported `unbuilt` as before.

## Roadmap

| stage | adds | testable by |
|---|---|---|
| 1 (this) | IR, trace parser, canonical, naming, analyze, manifest, `torch` + `vllm` adapters, optimize runner, collect, `run` | `tests/test_frontend.py` (synthetic traces, no device); a GPU `run` on a vLLM model |
| 2 | calibration (bandwidth, matmul throughput, launch floor per part; `None` when unmeasured, never 0), roofline bounds and the `bounded` gate; a results index over `trials/*/state.json` keyed by `workload_id` (`done`/`failed`); `fib` adapter | bound arithmetic units; index over existing run trees |
| 3 | DSPy naming of unnamed ops and demote-only triage; `llm_setup` provider table keyed by model prefix | `DummyLM` validator tests |
| 4 | SGLang adapter; tensor-parallel capture (rank merge: max for ops, min for collectives); eager-vs-compiled dual pass | trace fixtures |
| 5 | OpenVINO adapter (report-only entries) | profiling-info fixture |
| 6 | llama.cpp adapter; per-workload before/after comparison of two captures | log fixtures |
