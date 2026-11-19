# Reference

Every flag of `xe-forge` and `xe-forge-skill`, and every environment variable Xe-Forge reads.
Each command prints the same list with `--help`.

Where a flag and a variable set the same thing, the flag wins: `xe-forge` writes the flag into
the environment before it loads the configuration (`src/xe_forge/cli.py`). A variable already
set in the shell wins over the same variable in `.env`, which is loaded without overriding
(`src/xe_forge/config.py`).

`xe-forge` and `xe-forge-skill` load `.env` before anything reads the environment: the one
in the working directory or above it, then the one at the repository root.

## `xe-forge`: optimize one kernel

`xe-forge [flags]`. With `run`, `optimize` or `calibrate` as its first word it runs that
command instead (below).

| Flag | Default | What it does |
|---|---|---|
| `-i`, `--input` | none | Kernel file. A file in the `--dsl` language is the baseline; a file in another language is the reference, and a workspace engine writes the kernel. Not needed with `--kernel-repo` and a workspace engine |
| `--reference` | `<stem>_pytorch.py` beside a baseline `--input`, when it exists | Semantic reference the kernel must match |
| `-n`, `--name` | `kernel` | Kernel function name |
| `-o`, `--output` | none | Where the optimized kernel (or a directory winner) is written |
| `-s`, `--spec` | none | YAML spec: shapes, dtypes, tolerances, FLOP per variant |
| `--variant` | spec's `default_variant`, else `bench-gpu` (`bench-xpu` for `--dsl sycl`) | Spec variant to run |
| `--stages` | chosen from what analysis finds | Comma-separated stages (dspy engine): `analysis`, `algorithmic`, `dtype_fix`, `fusion`, `memory_access`, `tensor_descriptors`, `persistent_kernel`, `device_specific`, `autotuning`, `discovery`. An unknown name is skipped with a warning |
| `--model` | `LLM_MODEL` | LLM model |
| `--api-base` | `OPENAI_API_BASE` | LLM endpoint URL |
| `--api-key` | `OPENAI_API_KEY` | LLM API key |
| `--device` | `DEVICE_TYPE` (`xpu`) | `xpu`, `cuda` or `cpu` |
| `--dsl` | `DSL` (`triton`) | `triton`, `gluon`, `sycl`, `cuda` or `opencl` |
| `--num-warps` | `DEFAULT_NUM_WARPS` | Default number of warps |
| `--tile-size` | `PREFERRED_TILE_M` / `_N` | Preferred tile size, M = N |
| `--target-dtype` | `TARGET_DTYPE` (none) | `float16`, `bfloat16` or `float32` |
| `--no-correctness` | off | Skip the output comparison (performance only) |
| `--rtol` | spec, else `CORRECTNESS_RTOL` | Relative tolerance of the correctness check |
| `--atol` | spec, else `CORRECTNESS_ATOL` | Absolute tolerance of the correctness check |
| `--engine` | `ENGINE` (`dspy`) | `dspy`, `claude`, `codex` or `dspy-agent`. `claude` and `codex` drive an agent workspace with that CLI; `dspy-agent` runs a DSPy tool-calling agent in it |
| `--target` | `TARGET` (`kernel`) | Workspace engines: optimize the kernel, or `library`: tune how the repository calls a library |
| `--max-trials` | `MAX_TRIALS` (10) | Trial limit |
| `--trials-dir` | `TRIALS_DIR` (`./trials`) | Trial state directory |
| `--no-trials` | off | Disable trial tracking |
| `--unitrace` / `--no-unitrace` | `UNITRACE_ENABLED` (off) | Profile trials with unitrace |
| `--unitrace-bin` | `UNITRACE_BIN` (`unitrace`) | unitrace binary |
| `--vtune` / `--no-vtune` | `VTUNE_ENABLED` (off) | Profile trials with VTune |
| `--vtune-bin` | `VTUNE_BIN` (`vtune`) | VTune binary |
| `--workspace` | `WORKSPACE` (`./`) | Workspace directory of the workspace engines; also sets `TORCH_EXTENSIONS_DIR` to `<workspace>/.torch_extensions` when unset |
| `--auto-launch` | `AUTO_LAUNCH` (off) | `claude` / `codex`: run the agent session headless; without it the workspace is only generated. `dspy-agent` always runs |
| `--max-turns` | `MAX_TURNS` (80) | Turn limit of one headless session (claude; codex is bounded by time only) |
| `--compiler-flags` | `COMPILER_FLAGS` (none) | Extra flags for the target compiler (e.g. `icpx`), written into the workspace for the session's own build |
| `--integration-repo PATH` | none | With `--kernel-repo`: a repository the kernel is wired into; a winner may change the op's interface and its call sites, ported as a serving-only patch. Repeatable |
| `--debug` | off | Debug logging |


### Host integration

Delegate building, correctness and timing to a project that owns them.

| Flag | Default | What it does |
|---|---|---|
| `--build-backend` | `BUILD_BACKEND` (built-in) | Build backend: a registered name, an entry point, or `module:attr` |
| `--external-benchmark` | `EXTERNAL_BENCHMARK` (built-in) | Command run instead of the built-in benchmark. Placeholders: `{kernel}` `{baseline}` `{trial}` `{spec}` `{variant}` `{device}` `{dsl}` `{workspace}` |
| `--external-validate` | `EXTERNAL_VALIDATE` (built-in) | Command run instead of the built-in validator. Placeholders: `{kernel}` `{trial}` `{dsl}` `{stage}` `{workspace}` |
| `--stack-python` | `STACK_PYTHON` (none) | Interpreter of the stack the kernel serves; the port-back agent builds its port beside that stack, tests it there and records the verdict |
| `--dataset-record` | `DATASET_RECORD` (none) | JSON record of the data behind the spec's variants, rendered into the workspace (`xe_forge.core.dataset_record`) |
| `--lessons DIR` | `LESSONS_DIR` (none) | Directory outside the workspace where sessions record what they measured; each session reads all of it (`xe_forge.core.lessons`) |
| `--kernel-repo PATH` | `KERNEL_REPO` (none) | External repository holding the kernel; the kernel-locator agent explores it before the first trial |

## `xe-forge run`: optimize a model in its framework

`xe-forge run -o DIR --framework NAME --model ID [flags] [-- FLAGS]`. Everything after `--` is
passed to every per-kernel `xe-forge` run (for example `-- --no-unitrace`).

| Flag | Default | What it does |
|---|---|---|
| `-o`, `--output` | required | Run directory |
| `--framework` | required | What runs the model, by its project's name; free text handed to the agents |
| `--model` | required | Hugging Face id or local path |
| `--framework-python` | none | The framework's interpreter, already installed. Without it the setup agent builds a venv, which later runs reuse |
| `--framework-source URL[@REF]` | the framework's upstream, latest release | Repository and tag, branch or commit the setup agent builds from. Not with `--framework-python` |
| `--framework-venv DIR` | `$XE_FORGE_CACHE/frameworks/<framework>-<hash of the source>` | Where the setup agent builds. Not with `--framework-python` |
| `--framework-note TEXT` | none | A constraint the capture agent cannot discover, handed to it verbatim. Repeatable |
| `--reference-python` | the framework's interpreter | GPU torch + transformers, for a patch's reference tokens |
| `--workload` | `text` | `text`, `vision` or `action` |
| `--batch` | 1 | Batch size |
| `--quantization` | `auto` | `auto` (the checkpoint's own), a scheme name (`fp8`, `awq`), a stored dtype (`int8`), or `<scheme>:<dtype>` (`awq:int4`). The agent never picks it |
| `--input-len` | 512 | Prompt tokens [text] |
| `--output-len` | 32 | Generated tokens [text] |
| `--rows` | none | `rows.yaml` of several `(batch, input_len, output_len, weight)`, replacing the three flags above [text] |
| `--prompt-file` | synthetic token ids | Text the prompts are sliced from, at least batch x input-len tokens [text] |
| `--precision` | `auto` | Compute dtype: `auto` (the framework's default) or a torch dtype name |
| `--kv-cache` | `auto` | KV cache dtype: `auto` or a torch dtype name such as `float8_e4m3fn` [text] |
| `--steps` | 28 [vision]; the policy's own [action] | Denoising steps |
| `--height` | 1024 | Image height [vision] |
| `--width` | 1024 | Image width [vision] |
| `--frames` | 1 | Frames to generate; 1 is a still image [vision] |
| `--input-image PATH` | none | Input image: the one edited, a video's first frame, or a policy's camera view. Repeatable. Required for action [vision, action] |
| `--prompt` | a fixed prompt / instruction | Text prompt [vision] or instruction [action] |
| `--negative-prompt` | none | Negative prompt [vision] |
| `--calls` | 8 | Policy calls in one measured run [action] |
| `--state PATH` | zeros of the policy's state size | Robot state, `.npy` or `.json` [action] |
| `--top` | 3 | Kernels to optimize |
| `--at-roofline` | 0.8 | Efficiency of its bound at which a kernel is left alone |
| `--min-gain` | 0.005 | Smallest share of device time worth optimizing |
| `--calibration` | none (counter bounds only) | `xe-forge calibrate` output. Before naming, a matmul peak the capture's GEMMs and costs need and the file lacks is measured into it; a dtype that cannot be measured has no compute bound, is bounded by bandwidth alone, and is never stopped at roofline |
| `--name-coverage` | 0.99 | Share of device time whose ops the naming agent must place |
| `--analyze-only` | off | Stop after the manifest (`kernels.yaml`) |
| `--agent-max-turns` | `--max-turns` | Turn limit of every frontend agent |
| `--prove` | off | Build the stack with the winning patches and measure them in the model |
| `--prove-repeat` | 3 | Captures per arm in prove |
| `--fuse` | off | After optimizing, fuse chains of ops into one kernel |
| `--fuse-top` | 2 | Fusions to optimize |
| `--stack-check` | off | Each port-back agent builds and tests its port in the framework's stack (`--stack-python`) |

`run` also takes every `xe-forge optimize` flag except the manifest.

## `xe-forge optimize`: optimize the kernels of a manifest

`xe-forge optimize MANIFEST [flags] [-- FLAGS]`. Runs one `xe-forge` per enabled kernel of a
`kernels.yaml` from `run --analyze-only`, in that run's directory; `--` as for `run`.

| Flag | Default | What it does |
|---|---|---|
| `MANIFEST` | required | `kernels.yaml` |
| `--engine` | `XE_FORGE_AGENT`, else `claude` | `claude` or `codex`: the agent CLI of every agent; `dspy-agent`: for the kernels only |
| `--device` | `XE_FORGE_DEVICE`, else the first device torch sees | Device; exported as `XE_FORGE_DEVICE` to every step |
| `--max-turns` | 80 | Turn limit of one kernel session |
| `--only NAME ...` | every enabled kernel | Kernels to run |
| `--kernel-repos` | the directory holding this checkout | Directory of the kernel repositories |
| `--compiler-flags` | `XE_FORGE_SYCL_FLAGS` | Compile flags for `sycl` entries; `{kernel_repo}` expands to the entry's repository |

## `xe-forge calibrate`: measure the device's ceilings

| Flag | Default | What it does |
|---|---|---|
| `-o`, `--output` | `$XE_FORGE_CACHE/calibration/<device name>.json` | Output file |
| `--read-bytes` | 1073741824 (1 GiB) | Bytes read by the bandwidth test; must exceed the device's caches |
| `--matmul-n` | 8192 | Square matmul size |
| `--iters` | 20 | Iterations per measurement |
| `--dtypes DTYPE ...` | all | Matmul dtypes to measure |

## `xe-forge-skill`: the tools agents call

`xe-forge-skill <tool> ...`. A tool's verdict is its exit code. `--trials-dir` defaults to
`./trials` wherever it appears.

### `analyze PYTORCH_FILE`

AST analysis of a PyTorch reference. No flags.

### `validate KERNEL`

Static validation of a source file or a directory of them.

| Flag | Default | What it does |
|---|---|---|
| `--dsl` | `triton` | `triton`, `sycl`, `gluon`, `cuda` or `opencl` |
| `--stage` | none | Current optimization stage |
| `--external-validate` | `EXTERNAL_VALIDATE` | Host command to validate with instead |

### `benchmark BASELINE OPTIMIZED`

Correctness and timing of a trial against the baseline.

| Flag | Default | What it does |
|---|---|---|
| `-s`, `--spec` | none | Spec for shape-based inputs |
| `--reference` | none | Semantic reference; without `--spec` its `get_inputs()` / `get_init_inputs()` make the inputs |
| `--variant` | every weighted variant when the spec has `weights`, else its `default_variant` | Spec variant |
| `--baseline-us` | none | Cached baseline time, in microseconds |
| `--device` | `XE_FORGE_DEVICE`, else the first device torch sees | Device; exported as `XE_FORGE_DEVICE` to every step |
| `--dsl` | `triton` | As for `validate` |
| `--triton-baseline` | off | The baseline is a Triton kernel |
| `--external-benchmark` | `EXTERNAL_BENCHMARK` | Host command to benchmark with instead |
| `--kernel-name` | none | Trial tree to record the result in |
| `--trial-id` | none | Saved trial the optimized file is; records the result |
| `--builtin-benchmark` | `XE_FORGE_BUILTIN_BENCHMARK` (off) | Measure with the built-in executor; required when no host benchmark is configured |

### `trial`

| Subcommand | Arguments and flags | What it does |
|---|---|---|
| `init` | `KERNEL_NAME BASELINE_FILE [--triton-baseline] [--require-profiles {ComputeBasic,EuStallSampling,VTune} ...]` | Start a trial tree; `--require-profiles` names profiles every correct trial must attempt before the next save or finalize |
| `save` | `KERNEL_NAME TRIAL_FILE [--parent ID] [--strategy TEXT]` | Save a trial (a file or a directory of sources) |
| `result` | `KERNEL_NAME TRIAL_ID [--validation pass\|fail] [--correctness pass\|fail] [--speedup X] [--baseline-us US] [--kernel-us US] [--verdict TEXT]` | Record a trial's results; `--triton-us` is an alias of `--kernel-us`; `--verdict` records why a host withheld `--speedup` |
| `status` | `KERNEL_NAME` | Show the tree |
| `best` | `KERNEL_NAME` | Show the best trial |
| `baseline-us` | `KERNEL_NAME` | Print the recorded baseline time |
| `finalize` | `KERNEL_NAME OUTPUT_FILE [--require-measured]` | Write the best trial out; `--require-measured` considers only trials `benchmark --trial-id` measured |

### `profile KERNEL_FILE`

Hardware counters of one kernel. See [PROFILING.md](PROFILING.md).

| Flag | Default | What it does |
|---|---|---|
| `-s`, `--spec` | none | Spec; with `--reference` its variant supplies dims, dtype and tolerances |
| `--reference` | none | Reference defining the input factories and state validation |
| `--variant` | none | Spec variant |
| `--warmup` | 5 | Warmup iterations |
| `--iters` | 20 | Profiled iterations |
| `--tool` | `auto` | `auto` (unitrace `ComputeBasic` first, then VTune), `unitrace` or `vtune` |
| `--unitrace-bin` | `unitrace` | unitrace binary (does not read `UNITRACE_BIN`) |
| `--metric-group` | `ComputeBasic` | `ComputeBasic`, `EuStallSampling` or `all` (both, in sequential runs) |
| `--sampling-interval-us` | `XE_FORGE_SAMPLING_INTERVAL_US` (25) | unitrace sampling interval |
| `--assembly` | off | Capture IGC shader dumps (disables persistent JIT caches during collection) |
| `--parent-profile` | none | Retained unitrace artifact directory to compare kernel resources against |
| `--vtune-bin` | `vtune` | VTune binary |
| `--kernel-name` | none | Trial tree to record the attempt in |
| `--trial-id` | none | Saved trial this kernel file is |

### `upstream`

| Subcommand | Arguments and flags | What it does |
|---|---|---|
| `clone` | `REPO [--name N] [--rev REV]` | Private shared clone into `upstream/` (`upstream-<name>/` with `--name`), at `--rev` (default the repository's HEAD) |
| `patch` | `KERNEL_NAME [--kind upstream\|serving-only] [--notes TEXT] [--stack "PASS\|FAIL <evidence>"]` | Write and verify `output/upstream.patch`; `--kind` defaults to `upstream`; `--stack` records the stack-check verdict |

### `capture`

| Subcommand | Arguments and flags | What it does |
|---|---|---|
| `run` | `LAUNCHER -o RAW [--unitrace BIN] [--sampling-interval US] [--trace-steps N]` | Run a launcher (it writes into `$RAW_DIR`) timed, then under unitrace, into `raw/`. `--unitrace` defaults to `UNITRACE_BIN`, `--sampling-interval` to `XE_FORGE_SAMPLING_INTERVAL_US` (25), `--trace-steps` to `XE_FORGE_TRACE_STEPS` (2; 0 runs the full workload under unitrace too) |
| `parse` | `RAW -o CAPTURE_JSON [--names NAMES_JSON] [--min-known PCT]` | `raw/` to `capture.json`, checked against unitrace; `--names` applies a `names.json`; at least `--min-known` (90) % of op time must be in kernels unitrace saw |

### `patch check DIR`

Gate a patch that makes a framework run a model: a real diff, a fresh apply, an import, and
tokens scored against transformers. `DIR` holds `patch.json` and `generate.sh`.

| Flag | Default | What it does |
|---|---|---|
| `--repos` | required | The clones' directory (`--kernel-repos`) |
| `--framework-python` | required | The unpatched framework's interpreter |
| `--reference-python` | the framework's | GPU torch + transformers |
| `--model` | required | Model |
| `--batch` | required | Batch size |
| `--input-len` | none | Prompt tokens; needed by a model that returns tokens |
| `--min-match` | 32 | Tokens scored per sequence |
| `--tolerance` | 0.3 | Nats a token may fall below the reference's best |
| `--trust-remote-code` | off | Pass `trust_remote_code` |
| `--prompt-file` | none | The run's prompt text |
| `--precision` | `bfloat16` | Float dtype scored first: the run's declared compute |
| `--no-reference` | off | Skip only the reference comparison (for image or video output) |
| `--margin` | 0.1 | Non-text output: how much worse than the native run's worst the patched run's worst may be against a float32 reference, as a relative squared error (for PSNR, 10·log10(1 + margin) dB) |

### `bound CAPTURE_JSON`

Each workload's efficiency against its bound.

| Flag | Default | What it does |
|---|---|---|
| `--calibration` | none | `xe-forge calibrate` output |
| `--min-share` | 0.005 | Smallest share of device time listed |
| `--json` | off | Rows as JSON |
| `--names` | none | A draft `names.json` to apply first, to a capture parsed without `--names` (exit 2 otherwise) |
| `--ops` | none | The namer's `ops.json`, to validate `--names` |
| `--repos` | none | The kernel repositories `--names` refers to |

### `inputs check INPUTS_JSON`

Each chosen shape was captured and the weights sum to 1. `INPUTS_JSON` is
`{kernel: {shapes, why}}`, each kernel keyed by its spec's stem (its name unless two
entries share one).

| Flag | Default | What it does |
|---|---|---|
| `--manifest` | required | `kernels.yaml` |

### `fuse check FUSIONS_JSON`

The fused ops were captured, each listed once, the reference fits the anchor (its builders
name every dim; `**kwargs` does not), and the fusions are ranked. An empty `fusions` list
with a `why` passes.

| Flag | Default | What it does |
|---|---|---|
| `--capture` | required | `capture.json` |
| `--results` | none | `results/summary.json` of the kernels optimized before |

## Environment variables

Commands: `xe-forge` is the
kernel pipeline (also each kernel run that `run` / `optimize` starts); "agents" means every
command that starts an agent session, which builds its endpoint from the LLM variables.
Read but unused means the code loads the value and nothing uses it.

### LLM endpoint

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `LLM_MODEL` | `openai/gpt-4o` | Model, LiteLLM style; `bedrock/<id>` routes to AWS Bedrock. Passed to claude / codex without the provider prefix | `xe-forge`, agents |
| `OPENAI_API_BASE` | none | Endpoint URL; claude gets it as `ANTHROPIC_BASE_URL`, codex as `openai_base_url` | `xe-forge`, agents |
| `OPENAI_API_KEY` | none | API key; claude gets it as `ANTHROPIC_AUTH_TOKEN`, codex as `CODEX_API_KEY` | `xe-forge`, agents |
| `LLM_TEMPERATURE` | 0.1 | Sampling temperature (dspy, dspy-agent) | `xe-forge` |
| `LLM_MAX_TOKENS` | 8192 | Token limit of one completion (dspy, dspy-agent) | `xe-forge` |
| `AWS_REGION_NAME`, `AWS_REGION` | `us-east-1` | Bedrock region, the first one set | `xe-forge` |

Bedrock credentials (`AWS_BEARER_TOKEN_BEDROCK`, or `AWS_ACCESS_KEY_ID` +
`AWS_SECRET_ACCESS_KEY`) are read by LiteLLM, not by Xe-Forge (`src/xe_forge/llm_setup.py`).

### Engine and agents

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `ENGINE` | `dspy` | `--engine` | `xe-forge` |
| `TARGET` | `kernel` | `--target` | `xe-forge` |
| `WORKSPACE` | `./` | `--workspace` | `xe-forge` |
| `WORKSPACE_GIT_INIT` | false | Initialize the generated workspace as a git repository | `xe-forge` |
| `AUTO_LAUNCH` | false | `--auto-launch` | `xe-forge` |
| `MAX_TURNS` | 80 | `--max-turns` | `xe-forge` |
| `XE_FORGE_AGENT` | `claude` | Agent CLI of the frontend agents, `claude` or `codex`; default of `--engine` for `run` / `optimize` | `run`, `optimize` |
| `XE_FORGE_AGENT_TIMEOUT` | 21600 (6 h) | Seconds one frontend agent session may run | `run` |
| `XE_FORGE_AGENT_RESUMES` | 5 | Times a frontend agent session is resumed before it fails | `run` |
| `AGENT_STRATEGY` | `cover` | dspy optimizer: `cover`, `react` or `coordinator` | `xe-forge` |
| `AGENT_MAX_ITERATIONS` | 5 | dspy optimizer iterations per stage | `xe-forge` |
| `TRITON_OPT_DUMP_DIR` | `./outputs/kernels` | Where the dspy agents dump kernels they generate | `xe-forge` |
| `USE_COVER` | true | Read but unused | `xe-forge` |

### Correctness, trials and timing

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `REQUIRE_CORRECTNESS` | true | false = `--no-correctness` | `xe-forge` |
| `CORRECTNESS_RTOL` | 1e-2 | Relative tolerance when the spec gives none | `xe-forge` |
| `CORRECTNESS_ATOL` | 1e-5 | Absolute tolerance when the spec gives none | `xe-forge` |
| `TARGET_DTYPE` | none | `--target-dtype` | `xe-forge` |
| `TRIALS_ENABLED` | true | false = `--no-trials` | `xe-forge` |
| `TRIALS_DIR` | `./trials` | `--trials-dir` | `xe-forge` |
| `MAX_TRIALS` | 10 | `--max-trials` | `xe-forge` |
| `BENCHMARK_MIN_EFFECT` | 0.02 | Relative difference below which two timings count as equal; a finite number above 0 | `xe-forge`, `benchmark` |
| `XE_FORGE_BUILTIN_BENCHMARK` | off | `1`/`true`/`yes`/`on` = `benchmark --builtin-benchmark` | `benchmark` |
| `XE_FORGE_BASELINE_SLACK` | 1.5 | How much slower than the captured kernel a baseline may run before it is not that kernel | `benchmark` |
| `XE_FORGE_BENCHMARK_TIMEOUT` | `EXTERNAL_TIMEOUT` | Seconds per shape before `benchmark` gives up on a hang | `benchmark` |
| `MAX_ATTEMPTS_PER_STAGE` | 3 | Read but unused | `xe-forge` |
| `VALIDATE_EACH_STAGE` | true | Read but unused | `xe-forge` |
| `TARGET_SPEEDUP` | 2.0 | Read but unused | `xe-forge` |

### Host integration

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `BUILD_BACKEND` | none | `--build-backend` | `xe-forge` |
| `EXTERNAL_BENCHMARK` | none | `--external-benchmark` | `xe-forge`, `benchmark` |
| `EXTERNAL_VALIDATE` | none | `--external-validate` | `xe-forge`, `validate` |
| `EXTERNAL_TIMEOUT` | 1800 | Seconds a host command (and each benchmark shape) may run | `xe-forge`, `benchmark` |
| `STACK_PYTHON` | none | `--stack-python` | `xe-forge` |
| `DATASET_RECORD` | none | `--dataset-record` | `xe-forge` |
| `LESSONS_DIR` | none | `--lessons` | `xe-forge` |
| `KERNEL_REPO` | none | `--kernel-repo` | `xe-forge` |
| `INTEGRATION_REPOS` | none | `--integration-repo`, paths joined with `:` | `xe-forge` |
| `COMPILER_FLAGS` | none | `--compiler-flags` | `xe-forge` |

### Device and DSL

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `DEVICE_TYPE` | `xpu` | `--device` | `xe-forge` |
| `DSL` | `triton` | `--dsl` | `xe-forge` |
| `XPU_DEVICE` | `DEVICE_TYPE` | Device name when the device is not `cuda`; set, it replaces `--device xpu`/`cpu` | `xe-forge` |
| `DEFAULT_NUM_WARPS` | 32 (`cuda`: 4) | Default warps given to the optimizer | `xe-forge` |
| `DEFAULT_NUM_STAGES` | 2 (`cuda`: 3) | Default pipeline stages | `xe-forge` |
| `PREFERRED_TILE_M` | 256 (`cuda`: 128) | Preferred tile M | `xe-forge` |
| `PREFERRED_TILE_N` | 256 (`cuda`: 128) | Preferred tile N | `xe-forge` |
| `PREFERRED_TILE_K` | 32 | Preferred tile K | `xe-forge` |
| `GRF_MODE` | `256` | Triton GRF mode suggested on XPU | `xe-forge` |
| `GROUP_SIZE_M` | 4 | GEMM group size suggested on XPU | `xe-forge` |
| `SYCL_TLA_DIR` | none | sycl-tla checkout whose headers SYCL kernels compile against | `xe-forge` |
| `MKL_INCLUDE` | `$MKLROOT/include` | MKL headers for SYCL kernels | `xe-forge` |
| `MKLROOT` | none | Set by oneAPI's environment script; used when `MKL_INCLUDE` is unset | `xe-forge` |
| `XE_FORGE_SYCL_FLAGS` | none | Default of `optimize --compiler-flags` | `run`, `optimize` |
| `XE_FORGE_XPU_ARCH` | the device id torch reports | AOT target of the generated SYCL torch-extension seed | the generated workspace |
| `XE_FORGE_OPENCL_OPTIONS` | `-cl-std=CL3.0 -O2` | OpenCL C compile options | `xe_forge.opencl` (opencl kernels) |
| `XE_FORGE_SPIRV_OPTIONS` | `--spirv-ext=+all,-SPV_KHR_untyped_pointers` | `llvm-spirv` options | `xe_forge.opencl` (opencl kernels) |
| `XE_FORGE_OPENCL_CACHE` | `$XE_FORGE_CACHE/opencl` | OpenCL build cache | `xe_forge.opencl` (opencl kernels) |
| `XE_FORGE_CLANG_DIR` | none | Directory searched first for `clang` / `llvm-spirv` | `xe_forge.opencl` (opencl kernels) |
| `CMPLR_ROOT` | none | Set by oneAPI's environment script; searched after `XE_FORGE_CLANG_DIR` | `xe_forge.opencl` (opencl kernels) |

The SYCL compile goes through the `ai_bench` dependency, which reads `AIBENCH_SYCL_COMPILER`
(default `icpx`), `AIBENCH_SYCL_FLAGS` (replaces its default flags) and `AIBENCH_SYCL_TARGET`
(used only when torch reports no XPU device). Its `AIBENCH_SYCL_INCLUDE` is ignored: Xe-Forge
passes the include directories itself.

### Profiling

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `UNITRACE_ENABLED` | false | `--unitrace` | `xe-forge` |
| `UNITRACE_BIN` | `unitrace` | unitrace binary | `xe-forge`, `run`, `optimize`; `capture run` |
| `UNITRACE_METRIC_GROUP` | `all` | Metric group the workspace's profile step collects | `xe-forge` |
| `VTUNE_ENABLED` | false | `--vtune` | `xe-forge` |
| `VTUNE_BIN` | `vtune` | `--vtune-bin` | `xe-forge` |
| `XE_FORGE_SAMPLING_INTERVAL_US` | 25 | unitrace sampling interval in microseconds; too coarse misses short kernels, too fine stops unitrace partway with no records | `profile`, `capture run` |
| `XE_FORGE_TRACE_STEPS` | 2 | Default of `capture run --trace-steps` | `capture run` |
| `VTUNE_WARMUP` | 5 | Read but unused | `xe-forge` |
| `VTUNE_ITERS` | 20 | Read but unused | `xe-forge` |

### Knowledge base, logs and cache

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `KNOWLEDGE_BASE_ENABLED` | false | Load the YAML knowledge base (dspy engine) | `xe-forge` |
| `KNOWLEDGE_DIR` | `./knowledge_base` | Knowledge base directory | `xe-forge` |
| `AUTO_LOAD_KNOWLEDGE` | true | Read but unused | `xe-forge` |
| `LOG_LEVEL` | `INFO` | Pipeline log level | `xe-forge` |
| `LOG_DIR` | `./outputs/logs` | Pipeline logs | `xe-forge` |
| `KERNEL_DIR` | `./outputs/kernels` | Saved kernels | `xe-forge` |
| `SAVE_INTERMEDIATE` | true | Save each stage's kernel | `xe-forge` |
| `XE_FORGE_DEVICE` | the first device torch sees | Device of `run`, `optimize`, `calibrate`, `benchmark` and the processes they start; set from `--device` | `run`, `optimize`, `calibrate`, `benchmark` |
| `XE_FORGE_CACHE` | `~/.cache/xe_forge` | Cache root: framework venvs, calibrations, OpenCL builds | `run`, `calibrate`, `xe_forge.opencl` |

### Set by Xe-Forge for the programs it starts

| Variable | Set by | For |
|---|---|---|
| `RAW_DIR` | `capture run` | The launcher: where it writes its output |
| `XE_FORGE_TRACE_STEPS` | `capture run` | The launcher, during the unitrace pass only |
| `PTI_ENABLE_COLLECTION` | `profile` | unitrace: collect only around the profiled iterations |
| `ZE_FLAT_DEVICE_HIERARCHY=FLAT` | `profile`, `capture run` | Level Zero, under unitrace |
| `TORCH_EXTENSIONS_DIR` | `xe-forge --workspace` | torch extension builds, when unset |
| `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL` | agents | claude, from the LLM variables, when unset |
| `CODEX_API_KEY` | agents | codex, from `OPENAI_API_KEY`, when unset |
| `FRAMEWORK_PYTHON`, `XF_BATCH`, `XF_INPUT_LEN`, `XF_OUTPUT_LEN` | `run` | the launcher: the framework's interpreter and the text workload |
| `XE_FORGE_NATIVE` | `patch check` | `generate.sh`: run the framework's own path, the reference arm |

The framework launchers `capture run` and `patch check` start get the environment without
`ANTHROPIC_*` and `OPENAI_*` variables.
