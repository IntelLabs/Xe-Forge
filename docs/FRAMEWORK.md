# Optimizing a model in its framework

```
x-forge run --framework <name> --model <hub id or path> \
    --batch 4 --input-len 512 --output-len 32 \
    --calibration <calibration.json> --kernel-repos <repos> --top 3 --prove -o runs/<name>
```

profiles the model in the framework that serves it, names the kernel behind each op, picks the
kernels with the most recoverable device time, optimizes each with X-Forge, and writes
`runs/<name>/results/SUMMARY.md`: every kernel the model ran, what was done with it and why,
and each measured win with its patch. Flags and environment variables are in
[REFERENCE.md](REFERENCE.md#x-forge-run-optimize-a-model-in-its-framework).

Without `--framework-python`, a setup agent first builds the framework into a venv under
`$X_FORGE_CACHE/frameworks/` (or `--framework-venv`), from `--framework-source URL[@REF]` or
the latest release; later runs asking for the same framework and source reuse it.

Agents do the framework-, model- and hardware-specific work: installing the framework, driving
it, reading its source, choosing shapes. Code measures and checks what they produce, through
`x-forge-skill` tools whose exit code is the verdict. `--framework` is free text handed to the
agents; nothing in the code names a framework.

## Workloads

| `--workload` | One measured run | Inputs |
|---|---|---|
| `text` (default) | a decoder's prefill and decode steps | `--batch`, `--input-len`, `--output-len`, or `--rows`; `--prompt-file` |
| `vision` | a diffusion model's denoising steps at a resolution | `--steps`, `--height`, `--width`, `--frames`, `--prompt`, `--negative-prompt`, `--input-image` |
| `action` | a robot policy called `--calls` times on what it sees | `--input-image` (required), `--prompt`, `--state`, `--steps` |

Text prompts are token ids from `x_forge.frontend.profile.token_ids`, the same for every
framework and run: fixed synthetic ids, or slices of the `--prompt-file` text. The run copies
that file to `inputs/prompts.txt` and the launcher records it in `meta.json` as
`runtime.inputs` (`{name: {"file", "sha256"}}`); `capture parse` fails when a file there is
missing or its hash changed, and a resumed run with another prompt file is refused until `capture.json` is removed.
`--precision`, `--quantization` and `--kv-cache` are passed to the capture agent, which never
picks them itself; `capture parse` checks that the trace ran what was declared.

## Steps

| Step | Who | Writes | Checked by |
|---|---|---|---|
| setup | agent (`frontend-setup.md.j2`): builds the venv from the project's own install docs for this device, lists what it drops, edits no source | `<venv>/x-forge-setup.json`, `setup/framework.json` | the python is inside the venv, torch sees `--device` under it; a capture made on another commit is refused |
| capture | agent (`frontend-capture.md.j2`): decides whether the model fits, converts it if the framework needs that, writes `launcher/run.sh` | `raw/`, `capture.json` | `capture run`, `capture parse`; `run` parses again itself |
| patch (when the framework lacks the architecture or an op) | agent (`frontend-patch.md.j2`): a patch in a clone under `--kernel-repos`, built into a layered venv, with `generate.sh` and `PR.md` | `patch-<n>/`, `stack.json` | `patch check`; capture then runs again on the patched stack, at most three patches |
| name | agent (`frontend-name.md.j2`): finds each op's source, clones or fast-forwards its repository | `name/ops.json`, `names.json` | `naming.validate`: each name a symbol, its repository under `--kernel-repos`, its dsl one X-Forge has; with `--calibration`, a stated cost above any measured call is dropped |
| analyze | code (`frontend/analyze.py`) | `analysis.json`, `kernels.yaml`, `specs/`, `records/` | bound arithmetic: GEMM roofline, counter speed-of-light, the costs the namer stated; the whole model against its roofline (`Model roofline:` in the output and SUMMARY.md: every bounded workload's bound summed, over the time the model took: on a host its wall time with the per-node hook off, else all the device time) |
| inputs | agent (`frontend-inputs.md.j2`): per kernel, one shape per group of captured shapes that take the same path, with weights | `inputs/inputs.json`, the specs' `weights:` | `inputs check`: every captured shape in one group, a group within one configuration, weights summing to 1 |
| optimize | one X-Forge run per kernel | `workspaces/`, `results/` | the benchmark gates below; for a repository the run's stack built with a build script (a generated framework), on a host, each trial is built into it, its kernel checked and timed there on its own against the baseline's (`x-forge-skill inmodel`); only `--prove` times the whole engine |
| fuse (`--fuse`) | agent (`frontend-fuse.md.j2`): chains of ops one kernel could replace, each with a `_pytorch.py` reference; each passing one is optimized as a kernel, on clones carrying the winning patches | `fuse/` | `fuse check`: the ops were captured, the reference takes the anchor's dims; ranked by the chain's time after the kernels' wins |
| prove (`--prove`) | agent (`frontend-build-arms.md.j2`) builds an unpatched and a patched stack; code re-runs the launcher in each | `prove/` | device time over `--prove-repeat` captures per arm, in ABBA order; on a patched framework, the tokens scored again on the patched arm (`patch score`) before any capture, judged against arm A |

**Name.** Each op gets a route: `kernel` (a kernel in a repository X-Forge can rewrite),
`library` (the time is a library primitive; what is tuned is the call, in the repository
making it), `aten` (inside PyTorch, nothing here to edit) or `unnamed` (with the reason). A
kernel that a library launches when the repository calls it also carries that `call`, and the
call is tuned as a library route beside the kernel's rewrite, on the same shapes. For each
kernel or library route the namer states a tolerance (`rtol`, `atol`) from the op's dtypes,
how the kernel accumulates, and the repository's own tests. An op without one is not optimized
(`kernels.yaml` says "the naming agent stated no tolerance"): there is no default.

**Analyze.** A kernel is skipped with one reason: at roofline (`--at-roofline`, 0.8 of its bound), not
routable, no X-Forge DSL, no shapes, below `--min-gain`, or beyond `--top`. Without a bound,
a kernel is ranked by its whole share of device time.

**Optimize.** `benchmark` times every weighted shape of the spec: every shape must be correct,
none slower, and the score is the time-weighted total. The spec records each shape's time per
call in the model (`captured_us`). A baseline more than `X_FORGE_BASELINE_SLACK` times
slower than that is refused: it is not the kernel the model ran, and a gain over it is
not one over that kernel.

**Prove.** When the patches together fail in the model, each is proved alone; one that fails
alone is marked "not proven in the model" and not mergeable in `results/`, and the rest are
proved without it. On a patched framework (`stack.json`), the patched arm's output is scored
again by the patch check's scoring, with the arguments the check had
(`patch-<n>/check-args.json`), before anything is timed. What is scored is `generate.sh`'s run
of the model on the check's prompt, so a kernel the launcher reaches and `generate.sh` does not
is not judged by it. A failure counts against the patches only when arm A, built the same way
without them, passes; output that could not be judged (the reference failed) is said, and the
time is still measured. With `--quantization`, the quantized tokens are reported beside the
check's, not gated, as the check reports them.

## Run directory

| Path | Holds |
|---|---|
| `capture-request.json` | what the capture was asked for: framework, model, workload, declared dtypes |
| `setup/` | the setup agent's session and `framework.json`, the build this run used |
| `launcher/run.sh`, `raw/`, `capture.json` | the launcher, its instrumented output, the parsed ops |
| `patch-<n>/`, `stack.json` | a patch the model needed, its check (`check.json`, `quality.json`), the patched interpreter |
| `name/`, `names.json` | the ops shown to the namer; its answer, stamped with the capture it names |
| `rows/` | with `--rows`: each further row's capture and the merged one |
| `analysis.json`, `kernels.yaml`, `specs/`, `records/` | the ranking and the per-kernel specs |
| `inputs/` | `inputs.json`, and `prompts.txt` with `--prompt-file` |
| `workspaces/<kernel>__<variant>/` | one X-Forge workspace, with `.status.json` and `.log` beside it |
| `results/` | `SUMMARY.md`, `summary.json`, each kernel's winning patch, `PR.md`, `PORT.txt` |
| `fuse/`, `prove/` | the optional steps |
| `outputs/` | images, videos and `actions.npy` the model produced, collected for a person to look at |

Every agent step leaves `prompt.md` and `session.log` in its directory.

## Resuming

Run the same command again. Reused: the framework venv (by its record), `capture.json` when it
is a capture of what this run asks, each finished `patch-<n>/` (checked again, not redone),
`names.json` when it was written for this capture, row captures this launcher made at their
sizes, and each workspace whose `.status.json` has exit 0. Analyze runs again; the inputs,
fuse and prove agents are asked again. To redo a step, remove what it wrote (`capture.json`,
`names.json`, a workspace). A `capture.json` of another workload or model is refused, not
reused. `x-forge optimize runs/<name>/kernels.yaml` re-runs only the optimize step, for
example after `--analyze-only`.

## The framework's environment

The setup build only adds to the environment: its `env` sets variables that are unset, and its
`prepend` puts directories inside the venv before a path variable's own value. A record that
replaces a set value, or prepends a directory outside the venv, goes back to the agent. These
additions reach only the framework's processes (the agents' sessions under it, the launcher,
`patch check`); X-Forge's own optimize sessions run with the environment as it was before
setup (`frontend/cli.py`: `_setup_environ`, `_session_env`).

## Tools

| Tool | Does |
|---|---|
| `x-forge-skill capture run <launcher> -o raw` | runs the launcher plain, then under unitrace (device timing, the launch timeline, `ComputeBasic` counters when the device lists them); records the launcher's sha256 and a `MANIFEST.json` of `raw/` |
| `x-forge-skill capture parse raw -o capture.json [--names names.json]` | builds the op list and checks it: `raw/` unchanged, at least 90% (`--min-known`) of op time in kernels unitrace saw, every pattern of an op map matches a kernel and none is claimed twice, `runtime.inputs` hashes hold, declared dtypes ran |
| `x-forge-skill patch check <dir>` | the patch is git's own diff, applies to a fresh clone, imports from outside the framework's venv; then the output check below |
| `x-forge-skill patch score <dir> --python <py> --out <json>` | the output check alone, with `generate.sh` under another build of the stack; `--prove` runs it on the patched arm |
| `x-forge-skill bound capture.json` | each workload's efficiency against the stricter known bound |
| `x-forge-skill inputs check inputs.json --manifest kernels.yaml` | each chosen shape's captured share beside its weight |
| `x-forge-skill fuse check fusions.json --capture capture.json` | the fusions' ops, references and ranking |
| `x-forge calibrate` | the device's read bandwidth and matmul throughput per dtype, for `--calibration` |

Counters, sampling interval and the libraries unitrace needs: [PROFILING.md](PROFILING.md).

## The launcher contract

The launcher runs one batch of the workload into `$RAW_DIR`, under `$FRAMEWORK_PYTHON` when it
is set (prove sets it per arm). It writes `meta.json` (framework, model, runtime settings,
device, and `repeats`: how many times the process ran the identical workload, since unitrace
times the whole process; first-use work stays before those runs) and either a torch.profiler
trace (`trace*.json.gz`; `x_forge.frontend.profile.profile_region` does it for a callable) or,
for a framework that is not PyTorch, `ops.json`: each node, regexes of the kernel names it ran,
its argument shapes, its calls. A text launcher also reads `$XF_BATCH`, `$XF_INPUT_LEN` and
`$XF_OUTPUT_LEN`, so `--rows` can rerun it with no agent. The full text is
`src/x_forge/claude/templates/frontend-capture.md.j2`.

## Output-quality checks

A **text** patch is scored against Hugging Face transformers, or against the patch agent's
own runner (`patch.json`'s `"reference"`: a file with `load(model, dtype, device)`) when no
transformers loads the model, teacher-forced: each of the first 32 generated tokens
(`--min-match`) must be within 0.3 nat (`--tolerance`) of the reference's best token. A sequence that fails is judged again in float32 over several
slightly perturbed forwards.

An **image, video or action** patch is compared with the same `generate.sh` run under
`X_FORGE_NATIVE=1`, the framework's own path for that op (`frontend/quality.py`). Stages:

| Stage | File | Role | Metric | Warn / hard limit |
|---|---|---|---|---|
| `prompt` | `prompt_embeds.npy` | gate | relative squared error | 1e-3 / 1e-1 |
| `step1` | `latent_step1.npy` | gate | relative squared error | 1e-3 / 1e-1 |
| `step1_image` | `output.png` or `frames/*.npy` | gate | PSNR (dB) | 40 / 25 |
| `latent` | `latent_final.npy` | guard | relative squared error | 1e-2 / 1e-1 |
| `image` | `output.png` or `frames/*.npy` | report | PSNR (dB) | 40 / 25 |
| `actions` | `actions.npy` | report | relative squared error | 1e-2 / 1e-1 |

A is the native run, B the patched one. Byte-identical outputs pass. With a float32 reference
in the patch directory's `reference/<stage>.npy` (or `.png`), B passes when its worst run is
within `--margin` (10% in squared error; for PSNR the same, 10·log10(1 + margin) = 0.41 dB) of A's worst against it, fails when every B run is further from it
than A's worst, and is otherwise inconclusive. Without a reference, A against B passes within A's
own run-to-run spread or the warn limit, warns past it, and fails past the hard limit; these
limits are a convention, and the verdict says so. NaN or a different shape fails. The worst gate
stage decides; a guard or report stage fails the whole only on a catastrophe (NaN, a different
shape, past its hard limit). With no gate artifact the guard, then the report stage, is
promoted. The verdict goes to `quality.json`; fail,
inconclusive, or nothing to compare fails the check.

An optimized kernel is kept on its per-kernel tolerance, and prove compares device time. For
vision and action runs, look at `outputs/` before using a patch.

## Adding a framework

Nothing in the code. Run it with `--framework <its name>`, then record what to pass and
what the agents did on a page in [frameworks/](frameworks/README.md), from
[frameworks/TEMPLATE.md](frameworks/TEMPLATE.md). Knowledge the agents should start from goes
in `knowledge_base/frameworks/<its name>/`, with a `framework.yaml` for the facts the run acts
on (see [frameworks/native.md](frameworks/native.md)).
