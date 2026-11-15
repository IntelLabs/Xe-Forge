# Framework frontend: from a running model to optimized kernels

```
xe-forge run --framework vllm --model Qwen/Qwen2.5-0.5B-Instruct \
    --framework-python <framework-venv>/bin/python --batch 4 --input-len 512 --output-len 16 \
    --calibration <calibration.json> --kernel-repos <dir> --top 3 -o runs/qwen
```

profiles the model in its serving framework, names the kernel behind each op, picks the ones
with the most recoverable device time, optimizes each with Xe-Forge and writes
`runs/qwen/results/SUMMARY.md`: every kernel the model ran, what was done with it and why,
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
| prove (`--prove`) | agent builds two stacks; code re-runs the cached launcher in each | `prove/` | median and spread over `--prove-repeat` alternating captures, and `quality.json`: is the output still right |

`run` reuses `capture.json` and `names.json` when they exist (remove one to redo that step);
`xe-forge optimize runs/qwen/kernels.yaml` re-runs only the optimize step.

## Tools

| Tool | Does |
|---|---|
| `xe-forge-skill capture run <launcher> -o raw` | runs the launcher once as is (its own trace), once under `unitrace --device-timing --chrome-kernel-logging` (every kernel by its real name, its time, and the launch timeline), adding `--metric-sampling --group ComputeBasic --sampling-interval 10` only when the device lists that group); records the launcher's sha256 and a `MANIFEST.json` of raw/ |
| `xe-forge-skill capture parse raw -o capture.json [--names names.json]` | builds the op list and checks it: raw/ unchanged since `run`; >= 90% of op time in kernels unitrace saw; for an op map, every kernel pattern matches a kernel unitrace saw and none is claimed twice. Exit 1 prints each problem |
| `xe-forge-skill bound capture.json [--calibration c.json]` | each workload's efficiency against the stricter known bound, recoverable us and gain |
| `xe-forge-skill inputs check inputs.json --manifest kernels.yaml` | each chosen shape's captured share beside its weight, and the captured time they cover; exit 1 on a shape the capture did not run or weights that do not sum to 1 |
| `xe-forge calibrate` | the card's read bandwidth and matmul TFLOP/s per dtype |

## The launcher contract

The launcher runs one batch of the workload into `$RAW_DIR`, under `$FRAMEWORK_PYTHON` when
it is set (prove sets it per arm). It writes `meta.json` (framework, model, runtime settings,
device, and `repeats`: how many times the process ran the identical workload, since unitrace
times the whole process) and either a torch.profiler trace (`trace*.json.gz`, any PyTorch
framework; `xe_forge.frontend.profile.profile_region` does it for a callable) or, for any
other framework, `ops.json`: each node, the regexes of the kernel names it ran, its argument
shapes, its calls. The model's input token ids come from `xe_forge.frontend.profile.token_ids`,
the same for every framework and run. The full text is `frontend-capture.md.j2`.

Adding a framework: nothing to add. The B70 experiments that replaced the per-framework
adapters (Qwen2.5-0.5B, batch 4, 512 + 16 tokens): the agent's capture matched the old vLLM
and SGLang adapters within 0.4% of device time at 99.9% attribution, and for OpenVINO gave
concrete shapes for 100% of device time (the adapter: 0%) and counters for 97.6% (8.3%);
code re-running each cached launcher twice spread by under 0.3%.

## Counters

`ComputeBasic` gives each kernel's bytes read and written (GB/s against the calibrated
bandwidth) and its ALU0/1/2 utilization (ALU2 is the XMX pipe); its efficiency is the larger
of the two. A counter that was not measured is unknown, never 0, and such a kernel takes no
counter bound. Stall sampling is the optimize agent's `xe-forge-skill profile`.

Metric sampling is time-based, so the interval decides whether the counters are usable at
all -- and it fails in **both** directions. Too coarse and a kernel shorter than one tick is
never sampled, so its counters cover under half its measured time and it is dropped as
unmeasured. Too fine and unitrace stops partway, writing a truncated metrics file that
parses to zero records, with exit 0 and no error. Share of device time keeping a counter
bound, measured on two real launchers (a 0.5B and a 9B text model on an Arc Pro B70):

| interval | 0.5B, 44 ms | 9B, 1013 ms | |
|---|---|---|---|
| 50 us (unitrace's default) | 0.2% | 0.1% | works, useless |
| **25 us** | **86.5%** | **94.6%** | works |
| 10 us | 0.0% | 0.0% | no records |

So `capture run` samples at 25 us (`--sampling-interval`, env
`XE_FORGE_SAMPLING_INTERVAL_US`). 10 us is not reliably broken, which is what makes it the
wrong default: it returned 96% on the 0.5B once and nothing on the same launcher and node
afterwards. Device time moved under 2% across intervals, so sampling shares the pass with
`--device-timing`. When sampling is asked for and no records come back, `capture run` says
so and keeps the metrics files in `metrics-no-records/` beside raw/ -- otherwise the
capture succeeds, every bound quietly falls back to arithmetic, and the evidence goes with
the scratch directory. `--metric-query`, which queries every kernel instance instead of
sampling, wrote nothing these parsers could read and is not used.

## Installing what unitrace needs for counters

Reading the counters needs **two** libraries, and a GPU stack may ship only one:

| | |
|---|---|
| metrics **discovery** (`libigdmd`) | lists the metric groups -- every stack here ships it |
| metrics **library** (`libigdml`) | collects them -- **no stack here ships it** |

Without the library a card lists `EuStallSampling` and nothing else, because EU stall
sampling comes through a different path in the driver. That looks exactly like a card with
no counters, and it is not: on an Arc Pro B70 the library turns 1 group into 13,
ComputeBasic among them, on both the b70 and bmtxb70 nodes. It is plain user-space code and
needs no root -- the kernel side is already open (`/proc/sys/dev/xe/observation_paranoid`
is 0):

```sh
git clone --depth 1 --branch metrics-library-1.0.200 \
    https://github.com/intel/metrics-library.git ~/opt/metrics-library-src
cmake -S ~/opt/metrics-library-src -B ~/opt/metrics-library-build \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/opt/metrics-library
cmake --build ~/opt/metrics-library-build -j && \
cmake --install ~/opt/metrics-library-build          # -> ~/opt/metrics-library/lib64
```

Put that directory on `LD_LIBRARY_PATH` before the capture runs -- nothing in the package
does it for you, since the launcher inherits its environment from whoever starts it. Check it
took effect with `unitrace --metric-list`, which should name ComputeBasic; without the library
a card lists `EuStallSampling` alone, and `capture run` says so rather than reporting a device
with no counters.

A version note: the installed discovery here is 1.16.194, newer than any public release
(1.14.182), so this stack is an internal build and the officially paired library may be
internal too. 1.0.200 is the newest public one and works. If a future stack stops listing
ComputeBasic, ask for the `intel-metrics-library` matching that discovery version.

Asking for a group a device really lacks is not a missing counter -- unitrace exits 255
before it runs the launcher, so the device-timing pass is lost with it. So `capture run`
asks the device first (`unitrace --metric-list`) and simply goes without counters when the
group is absent. If that ever happens here, the metrics library is the first thing to check.

Without counters `analyze` falls back to the GEMM roofline and the costs the naming agent
states and proves. That is weaker than a counter bound, not absent: a kernel with neither
gets no bound at all, so it is ranked by its share of device time and can never be skipped
as already at roofline.
