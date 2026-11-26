# Profiling

X-Forge reads hardware counters with two tools: **unitrace** (device timing,
`ComputeBasic` counters, EU stall sampling) and **VTune** (`gpu-offload` hotspots). The
flags are in [REFERENCE.md](REFERENCE.md#profile-kernel_file).

## Where profiles are used

- `x-forge-skill profile KERNEL_FILE` profiles one kernel. `--tool auto` (the default)
  tries unitrace `ComputeBasic` first, then VTune; `--tool unitrace` or `--tool vtune` picks
  one. A tool that is not installed is skipped.
- In a workspace (`--engine claude|codex|dspy-agent`), `--unitrace` and `--vtune` add the
  profile step to the session's workflow. With `--unitrace` and `UNITRACE_METRIC_GROUP=all`
  (the default) every correct trial must attempt both `ComputeBasic` and `EuStallSampling`
  before the next save or finalize.
- The dspy engine uses VTune only: with `--vtune` and a spec, each stage after the first is profiled
  and the report goes into the next stage's prompt.
- `x-forge run` captures a model under unitrace (`capture run`), with device timing and
  `ComputeBasic` sampling in the same pass ([FRAMEWORK.md](FRAMEWORK.md)).

`profile` runs the kernel's `Model` with `--warmup` untimed iterations, then collects
around the `--iters` profiled ones only. With `--reference`, the reference builds the
inputs, as in `benchmark`.

## unitrace

- **Device timing**: each kernel's device time and launch count.
- **`ComputeBasic`**: each kernel's bytes read and written and its ALU0/1/2 utilization
  (ALU2 is the XMX pipe), plus busy, active, stall and occupancy percentages. A counter that
  was not measured is unknown, never 0.
- **`EuStallSampling`** (`--metric-group EuStallSampling`): stall causes sampled at
  instruction addresses. `--assembly` also captures IGC shader dumps for mapping them.
- `--metric-group all` collects both groups in sequential runs.
- `--parent-profile DIR` compares kernel resources against an earlier profile's kept
  artifacts.

### The sampling interval

Metric sampling is time-based, and the interval fails in both directions:

- **Too coarse** (unitrace's own default, 50 us, is coarse for short kernels): a kernel
  shorter than one tick is never sampled, its counters cover too little of its time, and
  it is treated as unmeasured.
- **Too fine**: unitrace stops partway and writes a truncated metrics file that parses to
  zero records, with exit 0 and no error.

The interval is set with `--sampling-interval-us` for `profile`, `--sampling-interval` for
`capture run`, or `X_FORGE_SAMPLING_INTERVAL_US` for both (defaults in REFERENCE.md).
Another device, or a much shorter or longer run, may need another value. When sampling returns no usable records,
`capture run` says so and keeps the metrics files in `metrics-no-records/` beside `raw/`;
`profile` retries one time in a fresh artifacts directory.

### Installing what the counters need

Reading counters needs two libraries, and a GPU stack may ship only the first:

| Library | Role |
|---|---|
| metrics discovery (`libigdmd`) | Lists the metric groups |
| metrics library (`libigdml`) | Collects them |

Without the metrics library a device lists `EuStallSampling` and nothing else, because EU
stall sampling comes through another path in the driver. That looks like a device with no
counters and usually is not. The library is user-space code and needs no root:

```sh
git clone --depth 1 --branch <release tag> \
    https://github.com/intel/metrics-library.git <src>
cmake -S <src> -B <build> -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<prefix>
cmake --build <build> -j && cmake --install <build>
```

Put `<prefix>/lib64` on `LD_LIBRARY_PATH` before profiling or capturing, or install the
library into a `lib/` beside the unitrace binary, which X-Forge adds to `LD_LIBRARY_PATH`
itself. Check with `unitrace --metric-list`: it should name `ComputeBasic`. Pick the
library release that matches the installed metrics discovery version.

Asking unitrace for a group the device lacks makes it exit before it runs the program, so
`capture run` asks `unitrace --metric-list` first and goes without counters when
`ComputeBasic` is absent.

## VTune

Install VTune Profiler (part of the oneAPI Base Toolkit) and source oneAPI's environment so
`vtune` is on `PATH`, or pass `--vtune-bin`. `vtune --version` checks it.

`profile --tool vtune` collects `gpu-offload` data for the profiled iterations, reports the
hottest kernel's XVE active, stalled and idle shares, peak occupancy, L3 and load/store
cache miss ratios and memory bandwidth, and maps them to suggestions:

| Condition | Suggestion |
|---|---|
| XVE stalled above active | Memory-bound: tensor descriptors, narrower inputs, tile swizzling |
| Peak occupancy under 50% | Larger tiles, fewer registers, a persistent kernel |
| XVE idle over 30% | Check the grid and tile swizzling |
| L3 miss ratio over 50% | Smaller tiles, more reuse |
| Load/store cache miss ratio over 30% | Better access patterns |

A collection that runs past 300 seconds is stopped: lower `--iters` or check the kernel
does not hang. Without VTune the profiler returns an error and the optimization goes on
without a profile.
