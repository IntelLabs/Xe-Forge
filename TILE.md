# Tile Search

LLM-driven tile configuration tuning for CUTLASS SYCL kernels on Intel Xe GPUs.

## Overview

CUTLASS SYCL kernels (GEMM, Flash Attention) use compile-time tile shapes that
determine performance. Tile search uses a propose-validate-benchmark loop:

1. **Propose** — An LLM (via DSPy) proposes tile configurations based on problem
   shape, hardware constraints, and history of previous results.
2. **Validate** — Each proposed config is checked against Intel Xe DPAS hardware
   constraints (atom shapes, subgroup limits, SLM capacity).
3. **Benchmark** — Valid configs are compiled into C++ templates and executed on
   the GPU. Performance (TFLOPS, time) is measured.
4. **Feedback** — Results are fed back to the LLM for the next round.

The loop supports GEMM and Flash Attention V2 kernels through a strategy pattern
(`KernelStrategy` protocol).

## Prerequisites

- **icpx** — Intel DPC++ compiler (oneAPI 2025.x)
- **sycl-tla** — CUTLASS SYCL port (set `SYCL_TLA_DIR`)
- **Intel Arc B580** (or compatible Xe2 GPU)
- **Python deps** — `dspy`, `pyyaml`, `torch` (with XPU support)

## Environment Setup

### sycl-tla headers

`SYCL_TLA_DIR` must point to the sycl-tla checkout. The executor derives all
include paths from it automatically based on `KernelType`:

| Include path | Kernel types | Contents |
|---|---|---|
| `$SYCL_TLA_DIR/include` | all | Core CUTLASS headers (`cutlass/`, `cute/`) |
| `$SYCL_TLA_DIR/tools/util/include` | all | Utility headers |
| `$SYCL_TLA_DIR/examples/common` | all | Shared example helpers (`helper.h`, `sycl_common.hpp`) |
| `$SYCL_TLA_DIR/applications` | FA, DUAL_GEMM | Application kernels (`flash_attention_v2/`, `dual_gemm/`) |
| `$SYCL_TLA_DIR/examples/06_bmg_flash_attention` | FA only | FA runner (`xe_fmha_fwd_runner.hpp`, dispatch) |

```bash
export SYCL_TLA_DIR=/path/to/sycl-tla
```

### Intel compiler toolchain

Source the Intel compiler and GPU runtime before running:

```bash
source /opt/intel/compiler/latest/env/vars.sh
source /opt/intel-gpu/latest/intel_gpu_vars.sh
```

The `ai_bench` SYCL compiler uses `icpx` by default. Override with:

| Variable | Default | Description |
|---|---|---|
| `AIBENCH_SYCL_COMPILER` | `icpx` | Path to the SYCL compiler binary |
| `AIBENCH_SYCL_TARGET` | `""` | AOT target device (e.g. `bmg-g31` for B580) |
| `AIBENCH_SYCL_FLAGS` | `-O2 -std=c++17 -fno-sycl-instrument-device-code` | Extra compiler flags |

### GPU runtime flags

These are critical for performance on Xe2 (B580):

```bash
# 256-GRF mode — halves occupancy but eliminates register spills in
# compute-heavy kernels. Required for GEMM/FA tile shapes that use
# large tiles (256x256, etc).
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"

# Force Level Zero GPU backend (required for Xe2 features)
export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"

# Optional: disable conservative vector alias analysis that can
# split loads/stores and reduce throughput
export IGC_VectorAliasBBThreshold=100000000000
```

### MKL headers (optional)

Used by some CUTLASS utilities. The executor adds this include path
automatically if the directory exists:

```bash
export MKL_INCLUDE=/opt/intel/mkl/latest/include   # default
```

### oneDNN headers

If your kernels link against oneDNN (e.g. for reference implementations or
post-ops fusion), set:

```bash
export ONEDNN_DIR=/path/to/onednn
# Then add to AIBENCH_SYCL_FLAGS or compiler include paths:
#   -I$ONEDNN_DIR/include -L$ONEDNN_DIR/lib -ldnnl
```

Currently tile search kernels (GEMM, FA) are self-contained CUTLASS templates
and do not require oneDNN.

### Xe-Forge mode

```bash
export DSL=sycl
export DEVICE_TYPE=xpu
```

### LLM configuration

Tile search uses the same `.env` / config as the optimization pipeline:

```bash
# .env file or environment
export LLM_MODEL=openai/gpt-4o
export OPENAI_API_BASE=https://your-endpoint/v1
export OPENAI_API_KEY=sk-...
```

## Quick Start

### Single GEMM shape

```bash
python -m xe_forge.cli --dsl sycl --tile-tune \
    --m 8192 --gemm-n 4096 --k 4096 \
    --max-rounds 5 --gemm-dtype bf16 \
    --tune-output gemm_results.json
```

### YAML config (multiple workloads)

```bash
python -m xe_forge.cli --dsl sycl --tune-config tune_gemm_shapes.yaml
```

## YAML Config Format

```yaml
mode: gemm          # "gemm" or "fa"
dtype: bf16         # bf16, f16, tf32, f32, int8
max_rounds: 5       # LLM proposal rounds per workload
output: results.json

tune-xpu:
  - name: "Large square"
    dims:
      M: 8192
      N: 8192
      K: 8192

  - name: "LLM FFN"
    dims:
      M: 8192
      N: 4096
      K: 12288
```

For Flash Attention:

```yaml
mode: fa
dtype: bf16
max_rounds: 5
output: fa_results.json

tune-xpu:
  - name: "Llama 3 8B"
    dims:
      head_dim: 128
      batch: 1
      num_heads_q: 32
      num_heads_kv: 8
      seq_qo: 4096
      seq_kv: 4096
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--tile-tune` | Single GEMM shape tuning mode |
| `--tune-config FILE` | Multi-workload tuning from YAML |
| `--m`, `--gemm-n`, `--k` | GEMM dimensions (with `--tile-tune`) |
| `--max-rounds N` | Max LLM proposal rounds (default: 5) |
| `--gemm-dtype TYPE` | Data type (default: bf16) |
| `--tune-output FILE` | Output JSON path |
| `--debug` | Verbose logging |

## Architecture

```
src/xe_forge/core/tile_search/
    __init__.py           # Public API
    agent.py              # TileTuningAgent + KernelStrategy + GEMMStrategy + FAStrategy
    config.py             # YAML config parser (TuneConfig, workload dataclasses)
    templates/
        gemm.py           # GEMM C++ template generator
        fa.py             # FA V2 C++ template generator
    validators/
        gemm.py           # GEMM tile constraint validator
        fa.py             # FA tile constraint validator + KNOWN_FA_CONFIGS
```

Key classes:

- **`TileTuningAgent`** — Orchestrates the propose-validate-benchmark loop.
  Takes an `SyclExecutor` and a `KernelStrategy`.
- **`KernelStrategy`** (protocol) — Defines kernel-specific behavior: how to
  build problem strings, validate tiles, generate C++ source, build CLI args.
- **`GEMMStrategy`** / **`FAStrategy`** — Concrete implementations.
- **`KernelType`** (enum in `sycl_executor.py`) — Controls include paths for
  compilation: `GEMM`, `FA`, `DUAL_GEMM`.

The executor's `execute_raw()` method compiles and runs kernels with arbitrary
CLI args (not just `--m/--n/--k`), enabling FA and other kernel types.

## Adding New Kernel Types

1. Implement `KernelStrategy` protocol (see `GEMMStrategy` as reference)
2. Add a C++ template generator in `templates/`
3. Add a tile validator in `validators/`
4. Add `KernelType` enum value if new include paths are needed
5. Optionally add a DSPy signature with kernel-specific prompting

## Output Format

Results are saved as JSON:

```json
{
  "workloads": [
    {
      "problem_shape": {"M": 8192, "N": 4096, "K": 4096},
      "best_config": {"wg": [256, 128, 32], "sg": [4, 8, 1]},
      "best_tflops": 95.2,
      "best_time_ms": 0.1234,
      "configs_tested": [...]
    }
  ],
  "tile_shapes": [
    {"wg": [256, 128, 32], "sg": [4, 8, 1]}
  ]
}
```

The `tile_shapes` array is compatible with `SYCL_TLA_ADDITIONAL_TILE_SHAPES`.
