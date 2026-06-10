# VTune GPU Profiling

Hardware-counter profiling for Triton kernels on Intel XPU using Intel VTune Profiler.

---

## Prerequisites

- Intel VTune Profiler (part of [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html))
- Intel GPU drivers and runtime
- Intel XPU device

## Environment Setup

Source the Intel oneAPI environment before running:

```bash
source /path/to/intel/compiler/latest/env/vars.sh
source /path/to/intel-gpu/latest/intel_gpu_vars.sh

export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
```

Verify VTune is accessible:

```bash
vtune --version
```

---

## Enabling VTune

### CLI

```bash
# Enable profiling
xe-forge -i kernel.py -s spec.yaml --vtune

# With custom VTune binary path
xe-forge -i kernel.py -s spec.yaml --vtune --vtune-bin /opt/intel/vtune/bin/vtune
```

### Environment Variables

```bash
VTUNE_ENABLED=true
VTUNE_BIN=vtune       # default, assumes vtune on $PATH
VTUNE_WARMUP=5        # warmup iterations (not profiled)
VTUNE_ITERS=20        # iterations to profile
```

### Standalone

```bash
# Profile a kernel directly
xe-forge-skill profile kernel.py --spec spec.yaml

# With custom warmup and iterations
xe-forge-skill profile kernel.py --spec spec.yaml --warmup 10 --iters 50

# With custom VTune path
xe-forge-skill profile kernel.py --spec spec.yaml --vtune-bin /opt/intel/vtune/bin/vtune
```

---

## How It Works

1. Generates a runner script that loads the kernel's `Model` class
2. Runs warmup iterations (not profiled)
3. Collects `gpu-offload` data via VTune for the profiled iterations
4. Extracts hotspot report, filters overhead kernels (Fill, Copy, Cast)
5. Returns metrics and optimization recommendations

---

## Metrics Collected

| Metric | Description |
|--------|-------------|
| XVE Active % | Percentage of time XVE (Xe Vector Engine) cores are executing |
| XVE Stalled % | Percentage of time XVE cores are stalled (waiting for data) |
| XVE Idle % | Percentage of time XVE cores are idle (no work scheduled) |
| Peak Occupancy % | Peak thread occupancy across XVE cores |
| L3 Miss Ratio % | L3 cache miss ratio |
| GPU Memory BW Read/Write | GPU memory bandwidth in GB/s |
| LSC Miss Ratio % | Load/Store Cache miss ratio |
| LSC BW Read/Write | Load/Store Cache bandwidth in GB/s |

---

## Recommendations

The profiler maps metric thresholds to optimization guidance:

| Condition | Diagnosis | Suggested Action | KB Reference |
|-----------|-----------|-----------------|--------------|
| XVE Stalled > Active | Memory-bound | Tensor descriptors, bf16 inputs, tile swizzling | `xpu_optimizations.yaml` |
| Peak Occupancy < 50% | Low occupancy | Larger tiles, fewer registers, persistent kernel | `xpu_optimizations.yaml` |
| XVE Idle > 30% | Work distribution | Check grid dimensions and tile swizzling | `xpu_optimizations.yaml` |
| L3 Miss > 50% | Cache thrashing | Reduce tile sizes, improve data reuse | `memory_patterns.yaml` |
| LSC Miss > 30% | Poor cache locality | Improve access patterns | `memory_patterns.yaml` |

---

## Integration with Engines

### DSPy Engine

The profiler runs automatically after each optimization stage (starting from stage 2). Its output is passed as `vtune_report` to the next stage's LLM prompt, giving the optimizer hardware-level feedback.

```bash
xe-forge -i kernel.py -s spec.yaml --vtune --engine dspy
```

### Claude Code Engine

When `vtune_enabled`, the generated `CLAUDE.md` workflow adds a "Profile" step after the first benchmarked trial. Claude calls `xe-forge-skill profile` and uses the recommendations to guide subsequent trials.

```bash
xe-forge -i kernel.py -s spec.yaml --vtune --engine claude --workspace ./workspace
```

---

## Troubleshooting

**"VTune not found"** -- Ensure `vtune` is on `$PATH` after sourcing the oneAPI environment, or specify the path with `--vtune-bin`.

**No GPU kernels in report** -- Check that `ONEAPI_DEVICE_SELECTOR="level_zero:gpu"` and `IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"` are set.

**Collection timeout (>300s)** -- Reduce `--iters` or verify the kernel isn't hanging. The default timeout is 300 seconds.

**Graceful degradation** -- If VTune is not available, the profiler returns an empty result with a warning. The optimization pipeline continues without profiling data.
