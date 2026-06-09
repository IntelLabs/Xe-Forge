# VTune GPU Profiling

Hardware-counter profiling on Intel XPU using Intel VTune Profiler. Two paths:
**Triton/PyTorch** kernels (`gpu-offload` around a generated Python runner) and
**SYCL/CUTLASS** `.cpp` kernels (`gpu-hotspots` characterization on the compiled
binary). The path is selected by `--dsl`; see [SYCL kernels](#sycl-kernels) below.

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

## SYCL kernels

SYCL/CUTLASS kernels are compiled `.cpp` binaries, so they are profiled
differently from Triton: there is no Python runner. `SyclProfiler`
(`core/sycl_profiler.py`) compiles the kernel via `SyclExecutor`, generates the
same deterministic file-IO inputs the benchmark uses, and runs the binary
directly under VTune `gpu-hotspots` in **characterization** mode — which exposes
richer Intel Xe metrics than `gpu-offload`.

```bash
# Profile a SYCL kernel (point --vtune-bin at a 2026.x build if needed)
xe-forge-skill profile examples/sycl/gemm.cpp \
    --spec examples/sycl/gemm.yaml --dsl sycl --variant bench-xpu \
    --iters 200 --vtune-bin /data/swtools/intel/vtune/2026.0/bin64/vtune
```

Under the hood:

```bash
vtune -collect gpu-hotspots \
    -knob gpu-profiling-mode=characterization \
    -knob characterization-mode=overview \
    -result-dir <dir> \
    -- <binary> --m=M --n=N --k=K --input_dir=<in> --output_dir=<out> \
       --iterations=200 --verify=0
```

### Metrics collected (SYCL)

| Metric | Meaning |
|--------|---------|
| XVE Active / Stalled / Idle | Xe Vector Engine execution / stall / idle time |
| Peak XVE Threads Occupancy | Thread occupancy (with Work-Size / SLM / Barrier sub-limiters) |
| XMX (DPAS) Active | Fraction of time the matrix engine is busy — the key GEMM-efficiency signal |
| GPU L3 Miss Ratio | L3 cache miss ratio |
| GPU Memory Bandwidth Read/Write | GB/s to/from GPU memory |

### Metric → CUTLASS knob (SYCL)

| Condition | Diagnosis | Action | KB |
|-----------|-----------|--------|----|
| XVE Stalled > Active | Memory-bound mainloop | ↑ PipelineStages; 2D-block/VNNI copy atoms; ↓ TileK | `sycl_vtune.yaml` |
| Peak occupancy < 50% | Grid too small / register pressure | Smaller TileShape (256→128); check 256-GRF | `sycl_vtune.yaml` |
| XVE Idle > 30% | Work-distribution / tail | TileShape vs M/N; stream-K / persistent scheduler | `sycl_vtune.yaml` |
| XMX active < 20% | Matrix engine underutilized | Larger N-per-subgroup; SubgroupLayout vs DPAS atom | `sycl_vtune.yaml` |
| L3 miss > 50% | Cache thrashing | Reduce tiles; improve K-blocking/reuse | `sycl_vtune.yaml` |
| Mem BW ≈ peak, low TFLOPS | Bandwidth-bound | Accept, or change algorithm | `sycl_vtune.yaml` |

### SYCL-specific notes

- **Self-checker false negative**: `vtune-self-checker.sh` may report GPU
  profiling as unsupported (its bundled DPC++ app fails to launch), yet a real
  AOT-compiled `bmg-g31` kernel profiles fine. Don't gate on the self-checker.
- **`xe` kernel driver** (newer than `i915`) is supported by VTune 2026 for
  `gpu-hotspots`; `perf_event_paranoid=0` helps.
- **VTune version**: the config default `vtune_bin` may point at an older build;
  pass `--vtune-bin /data/swtools/intel/vtune/2026.0/bin64/vtune` (or set
  `VTUNE_BIN`) to use 2026.x.

---

## Troubleshooting

**"VTune not found"** -- Ensure `vtune` is on `$PATH` after sourcing the oneAPI environment, or specify the path with `--vtune-bin`.

**No GPU kernels in report** -- Check that `ONEAPI_DEVICE_SELECTOR="level_zero:gpu"` and `IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"` are set.

**Collection timeout (>300s)** -- Reduce `--iters` or verify the kernel isn't hanging. The default timeout is 300 seconds.

**Graceful degradation** -- If VTune is not available, the profiler returns an empty result with a warning. The optimization pipeline continues without profiling data.
