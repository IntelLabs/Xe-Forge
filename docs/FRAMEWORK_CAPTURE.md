# Framework frontend: from a running model to optimized kernels

```
xe-forge run --framework <framework> --model <hub id> \
    --framework-python <framework-venv>/bin/python --batch <n> --input-len <n> --output-len <n> \
    --calibration <calibration.json> --kernel-repos <dir> --top 3 -o runs/<name>
```

profiles the model in its serving framework, names the kernel behind each op, picks the ones
with the most recoverable device time, optimizes each with Xe-Forge and writes
`runs/<name>/results/SUMMARY.md`: every kernel the model ran, what was done with it and why,
and each measured win with its patch.

Nothing in the code knows a framework. **Agents do the framework- and model-specific work;
code measures and checks it**, through `xe-forge-skill` tools the agents call themselves.

## Steps

| Step | Who | Writes | Checked by |
|---|---|---|---|
| capture | agent (`frontend-capture.md.j2`): decides whether the model fits, converts it if the framework needs that, writes `launcher/run.sh` | `raw/`, `capture.json` | `capture run`, `capture parse` (below); `run` re-runs `parse` itself |
| name | agent (`frontend-name.md.j2`): finds each op's kernel source, clones or fast-forwards its repository | `names.json` | `naming.validate`: the symbol greps in the repo, the dsl exists, every stated cost holds at every measured shape |
| analyze | code (`analyze.py`) | `kernels.yaml`, `specs/`, `analysis.json` | bound arithmetic: GEMM roofline, counter speed-of-light, agent-stated costs |
| inputs | agent (`frontend-inputs.md.j2`): per kernel, one shape per configuration (each captured workload), merging only the token counts that take the same path, with weights | `inputs/inputs.json`, the specs' `weights:` | `inputs check`: every captured shape stands in one group, a group within one configuration, weights summing to 1 |
| optimize | Xe-Forge, one workspace per kernel | `workspaces/`, `results/` | Xe-Forge's benchmark and trial gates; `benchmark` times every weighted shape: all correct, none slower, scored on the time-weighted total |
| fuse (`--fuse`) | agent (`frontend-fuse.md.j2`): after optimizing, proposes chains of ops one kernel could replace, each with its chain as a `_pytorch.py` reference; then each passing one goes through inputs and optimize as a kernel does, on clones carrying the kernels' winning patches (`fuse/stacked/`), against the optimized chain | `fuse/fusions.json`, `fuse/kernels.yaml`, `fuse/workspaces/`, `fuse/results/` | `fuse check`: the ops captured, the reference's inputs the anchor's dims, the repositories there; ranked by the chain's time after the kernels' wins. A fusion that changes call sites is serving-only |
| prove (`--prove`) | agent builds two stacks and makes the launcher hold on each; code re-runs it in each | `prove/` | median and spread over `--prove-repeat` alternating captures |

`run` reuses `capture.json`, `names.json` and each workspace that finished when they exist
(remove one to redo that step);
`xe-forge optimize runs/<name>/kernels.yaml` re-runs only the optimize step.

## Tools

| Tool | Does |
|---|---|
| `xe-forge-skill capture run <launcher> -o raw` | runs the launcher once as is (its own trace), once under `unitrace --device-timing --metric-sampling --group ComputeBasic` (every kernel by its real name, its time and counters); records the launcher's sha256 and a `MANIFEST.json` of raw/ |
| `xe-forge-skill capture parse raw -o capture.json [--names names.json]` | builds the op list and checks it: raw/ unchanged since `run`; >= 90% of op time in kernels unitrace saw; for an op map, every kernel pattern matches a kernel unitrace saw and none is claimed twice. Exit 1 prints each problem |
| `xe-forge-skill bound capture.json [--calibration c.json]` | each workload's efficiency against the stricter known bound, recoverable us and gain |
| `xe-forge-skill inputs check inputs.json --manifest kernels.yaml` | each chosen shape's captured share beside its weight, and the captured time they cover; exit 1 on a shape the capture did not run or weights that do not sum to 1 |
| `xe-forge calibrate` | the card's read bandwidth and matmul TFLOP/s per dtype |

## The launcher contract

The launcher runs one batch of the workload into `$RAW_DIR`, under `$FRAMEWORK_PYTHON` when
it is set (prove sets it per arm). It writes `meta.json` (framework, model, runtime settings,
device, and `repeats`: how many times the process ran the identical workload, since unitrace
times the whole process; first-use work such as a weight prepack stays before those runs) and either a torch.profiler trace (`trace*.json.gz`, any PyTorch
framework; `xe_forge.frontend.profile.profile_region` does it for a callable) or, for any
other framework, `ops.json`: each node, the regexes of the kernel names it ran, its argument
shapes, its calls. The model's input token ids come from `xe_forge.frontend.profile.token_ids`,
the same for every framework and run: fixed synthetic ids, or with `xe-forge run --prompt-file`
slices of that text. Real text is closer to use and leaves fewer near-ties between the top two
tokens. The run's inputs go into `meta.json` as `runtime.inputs` (`{name: {"file",
"sha256"}}`); `capture parse` checks every file there still has its hash, whatever its kind,
which leaves room for images or observations. Runs whose tokens are compared may differ only
at a near-tie (0.3 nat, the patch check's tolerance). The full text is `frontend-capture.md.j2`.

Adding a framework: nothing to add.

## Counters

`ComputeBasic` gives each kernel's bytes read and written (GB/s against the calibrated
bandwidth) and its ALU0/1/2 utilization (ALU2 is the XMX pipe); its efficiency is the larger
of the two. A counter that was not measured is unknown, never 0, and such a kernel takes no
counter bound. Stall sampling is the optimize agent's `xe-forge-skill profile`.
