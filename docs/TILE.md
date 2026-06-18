# Tile Search

LLM-driven tile configuration tuning for CUTLASS SYCL kernels on Intel Xe GPUs.

## Overview

CUTLASS SYCL kernels (GEMM, Flash Attention V2, MoE GEMM, Grouped GEMM) use
compile-time tile shapes that determine performance. Tile search uses a
propose-validate-benchmark loop:

1. **Propose** — An LLM (via DSPy) proposes tile configurations based on problem
   shape, hardware constraints, and history of previous results.
2. **Validate** — Each proposed config is checked against Intel Xe DPAS hardware
   constraints (atom shapes, subgroup limits, SLM capacity).
3. **Benchmark** — Valid configs are compiled from Jinja2 C++ templates and
   executed on the GPU. Performance (TFLOPS, time) is measured.
4. **Feedback** — Results are fed back to the LLM for the next round.

The loop supports four kernel types through a strategy pattern
(`KernelStrategy` protocol): GEMM, Flash Attention V2, MoE GEMM, and
Grouped GEMM.

## Prerequisites

- **icpx** — Intel DPC++ compiler (oneAPI 2025.x)
- **sycl-tla** — CUTLASS SYCL port (set `SYCL_TLA_DIR`)
- **Intel Xe2 GPU** — Arc B580, B570, or Battlemage (BMG) family
- **Python deps** — `dspy`, `pyyaml`, `torch` (with XPU support via `uv sync --extra intel`)

## Step-by-Step Guide

### Step 1 — Environment setup

```bash
# Intel compiler + GPU runtime
source /opt/intel/compiler/latest/env/vars.sh
source /opt/intel-gpu/latest/intel_gpu_vars.sh

# Point to CUTLASS SYCL headers
export SYCL_TLA_DIR=/path/to/sycl-tla

# AOT target device (required for SPIRV extensions)
export AIBENCH_SYCL_TARGET=bmg-g31

# GPU performance flags (256-GRF mode for large tiles)
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"

# Xe-Forge mode
export DSL=sycl
export DEVICE_TYPE=xpu

# LLM endpoint
export LLM_MODEL=openai/gpt-4o
export OPENAI_API_BASE=https://your-endpoint/v1
export OPENAI_API_KEY=sk-...
```

### Step 2 — Single GEMM tile search (quick demo)

```bash
python -m xe_forge.cli --dsl sycl --tile-tune \
    --m 4096 --gemm-n 4096 --k 4096 \
    --max-rounds 3 --gemm-dtype bf16 \
    --tune-output gemm_4k_results.json
```

LLM proposes tile configs, validator filters illegal ones, compile + benchmark
on GPU, best config found in 3 rounds.

### Step 3 — Multi-shape GEMM tuning (real workloads)

Create `tune_gemm.yaml`:

```yaml
mode: gemm
dtype: bf16
max_rounds: 5
output: gemm_results.json

tune-xpu:
  - name: "Large square"
    dims: { M: 8192, N: 8192, K: 8192 }

  - name: "LLM FFN"
    dims: { M: 8192, N: 4096, K: 12288 }

  - name: "Skinny M"
    dims: { M: 32, N: 4096, K: 4096 }
```

```bash
python -m xe_forge.cli --dsl sycl --tune-config tune_gemm.yaml
```

### Step 4 — Flash Attention V2 tuning (model-specific)

Uses `FMHAConfigGenWithTileShape` from sycl-tla's benchmark infrastructure.
The FA V2 template takes high-level tile parameters (WgTileQ, WgTileK,
WgTileV, SgTileQ, SgTileK) and derives ShapeQK, ShapePV, and SubgroupLayout
automatically — no manual shape construction needed.

Create `tune_fa2.yaml`:

```yaml
mode: fa
dtype: bf16
max_rounds: 5
output: fa2_results.json

tune-xpu:
  - name: "Llama 3 8B seq=4096"
    dims:
      head_dim: 128
      batch: 1
      num_heads_q: 32
      num_heads_kv: 8
      seq_qo: 4096
      seq_kv: 4096

  - name: "Gemma 3 27B seq=4096"
    dims:
      head_dim: 256
      batch: 1
      num_heads_q: 16
      num_heads_kv: 16
      seq_qo: 4096
      seq_kv: 4096
```

```bash
python -m xe_forge.cli --dsl sycl --tune-config tune_fa2.yaml
```

#### FA2 mode flags

The FA V2 config supports three optional top-level flags that map to
`FMHAConfigGenWithTileShape` template parameters:

| YAML field | C++ template param | Default | Description |
|---|---|---|---|
| `causal` | `Causal` | `false` | Enable causal (triangular) attention mask |
| `fa_mode` | `FMHAMode` | `prefill` | `prefill` (seq_qo > 1) or `decode` (seq_qo = 1, autoregressive) |
| `persistent` | `Persistent` | `false` | Persistent kernel scheduling |

**Constraint:** `causal` and `persistent` are mutually exclusive (C++ static_assert).
Decode mode automatically uses `pipeline_stages=1`.

Example — causal prefill:

```yaml
mode: fa
dtype: bf16
causal: true
fa_mode: prefill
max_rounds: 5

tune-xpu:
  - name: "Llama 3 8B causal prefill"
    dims:
      head_dim: 128
      batch: 1
      num_heads_q: 32
      num_heads_kv: 8
      seq_qo: 4096
      seq_kv: 4096
```

Example — decode (autoregressive token generation):

```yaml
mode: fa
dtype: bf16
fa_mode: decode
max_rounds: 5

tune-xpu:
  - name: "Llama 3 8B decode KV=2048"
    dims:
      head_dim: 128
      batch: 1
      num_heads_q: 32
      num_heads_kv: 8
      seq_qo: 1
      seq_kv: 2048
```

### Step 5 — MoE GEMM tuning

Create `tune_moe.yaml`:

```yaml
mode: moe
dtype: bf16
max_rounds: 5
output: moe_results.json

tune-xpu:
  - name: "DeepSeek V3 gate_up"
    dims:
      N: 5760
      K: 2880
      num_experts: 32
      total_tokens: 8192
```

```bash
python -m xe_forge.cli --dsl sycl --tune-config tune_moe.yaml
```

### Step 6 — Results

Output JSON contains best tile config per workload:

```json
{
  "workloads": [{
    "problem_shape": {"M": 8192, "N": 8192, "K": 8192},
    "best_config": {"wg": [256, 128, 32]},
    "best_tflops": 95.2,
    "best_time_ms": 0.1234,
    "configs_tested": [...]
  }],
  "tile_shapes": [{"wg": [256, 128, 32]}]
}
```

The `tile_shapes` array is compatible with `SYCL_TLA_ADDITIONAL_TILE_SHAPES`.

### What happens under the hood (per round)

1. LLM sees problem shape + hardware specs + previous results
2. Proposes 3-5 tile configs (WG_M, WG_N, WG_K, SG layout)
3. Validator checks DPAS atom constraints, SLM capacity, subgroup limits
4. Valid configs rendered from Jinja2 C++ template and compiled with icpx
5. Benchmark on Intel Xe2 GPU — TFLOPS measured
6. Results fed back to LLM for next round

## Environment Variables

### sycl-tla headers

`SYCL_TLA_DIR` must point to the sycl-tla checkout. The executor derives all
include paths from it automatically based on `KernelType`:

| Include path | Kernel types | Contents |
|---|---|---|
| `$SYCL_TLA_DIR/include` | all | Core CUTLASS headers (`cutlass/`, `cute/`) |
| `$SYCL_TLA_DIR/tools/util/include` | all | Utility headers |
| `$SYCL_TLA_DIR/examples/common` | all | Shared example helpers (`helper.h`, `sycl_common.hpp`) |
| `$SYCL_TLA_DIR/applications` | FA, DUAL_GEMM | Application kernels (`flash_attention_v2/`, `dual_gemm/`) |
| `$SYCL_TLA_DIR/examples/06_bmg_flash_attention` | FA | FA dispatch and runner |
| `$SYCL_TLA_DIR/benchmarks/flash_attention` | FA | `FMHAConfigGenWithTileShape`, benchmark configs |

### Compiler configuration

| Variable | Default | Description |
|---|---|---|
| `AIBENCH_SYCL_COMPILER` | `icpx` | Path to the SYCL compiler binary |
| `AIBENCH_SYCL_TARGET` | `""` | AOT target device (e.g. `bmg-g31`). **Required** — without this, SPIRV extensions are not enabled and compilation fails |
| `AIBENCH_SYCL_FLAGS` | `-O2 -std=c++17 -fno-sycl-instrument-device-code` | Compiler flags (replaces defaults if set) |

### GPU runtime flags

```bash
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
```

### MKL headers (optional)

```bash
export MKL_INCLUDE=/opt/intel/mkl/latest/include   # default
```

## Supported Kernel Types

### GEMM

Standard CUTLASS SYCL GEMM. Tile parameters: WG_M, WG_N, WG_K with
subgroup layout derived from DPAS atom shapes (M=8, N=16).

### Flash Attention V2

Uses `FMHAConfigGenWithTileShape` from sycl-tla benchmarks. This struct
accepts 7 tile parameters and 5 boolean mode flags, then derives all internal
shapes (ShapeQK, ShapePV, ShapeOutput, SubgroupLayout) automatically.

Tile parameters proposed by the LLM:
- `qk_m`, `qk_n`, `qk_k` — ShapeQK tile dimensions
- `pv_n`, `pv_k` — ShapePV tile dimensions (pv_m = qk_m, shared Q dim)
- `sg_q` — subgroup count partitioning the Q dimension

These are converted to `FMHAConfigGenWithTileShape` parameters:
- `WgTileQ = qk_m`, `WgTileK = qk_n`, `WgTileV = pv_n`
- `SgTileQ = qk_m / sg_q`, `SgTileK = pv_k`
- `HeadDimQK = qk_k` (DPAS K-dim tile, typically 32)
- `HeadDimV = head_dim` (full output head dimension)

Known-good configs per head dimension:
- HD=64: qk=(128,64,32), pv_n=32, pv_k=64, sg_q=8
- HD=128: qk=(256,32,32), pv_n=32, pv_k=32, sg_q=16
- HD=192: qk=(256,64,32), pv_n=32, pv_k=64, sg_q=32
- HD=256: qk=(256,64,32), pv_n=32, pv_k=64, sg_q=32

### MoE GEMM

Mixture-of-Experts GEMM with varying per-expert M dimensions and a tile
scheduler that distributes work across experts dynamically. Tile parameters:
WG_M, WG_N, WG_K.

### Grouped GEMM

Fused multi-GEMM with persistent group tile scheduler. Uses the same tile
parameters as standard GEMM but targets batched problems with different sizes.

## YAML Config Format

```yaml
mode: gemm          # "gemm", "fa", "moe", or "grouped_gemm"
dtype: bf16         # bf16, f16, tf32, f32, int8
max_rounds: 5       # LLM proposal rounds per workload
output: results.json

# FA-specific (optional, ignored for non-FA modes)
causal: false       # causal attention mask
fa_mode: prefill    # "prefill" or "decode"
persistent: false   # persistent kernel scheduling

tune-xpu:
  - name: "workload name"
    dims:
      M: 8192
      N: 8192
      K: 8192
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
| `--no-correctness` | Skip output verification |

## Architecture

```
src/xe_forge/core/tile_search/
    __init__.py           # Public API
    agent.py              # TileTuningAgent + strategies (GEMM, FA, MoE, GroupedGEMM)
    config.py             # YAML config parser (TuneConfig, workload dataclasses)
    templates/
        gemm.py / gemm.cpp.j2               # GEMM C++ template
        fa_v2.py / fa_v2.cpp.j2              # FA V2 template (FMHAConfigGenWithTileShape)
        moe_gemm.py / moe_gemm.cpp.j2       # MoE GEMM template
        grouped_gemm.py / grouped_gemm.cpp.j2  # Grouped GEMM template
    validators/
        gemm.py           # GEMM tile constraint validator
        fa.py             # FA tile constraint validator + KNOWN_FA_CONFIGS
```

Key classes:

- **`TileTuningAgent`** — Orchestrates the propose-validate-benchmark loop.
  Takes an `SyclExecutor` and a `KernelStrategy`.
- **`KernelStrategy`** (protocol) — Defines kernel-specific behavior: how to
  build problem strings, validate tiles, generate C++ source, build CLI args.
- **`GEMMStrategy`** / **`FAStrategy`** / **`MoEGEMMStrategy`** /
  **`GroupedGEMMStrategy`** — Concrete implementations for each kernel type.
- **`SyclExecutor`** — Wraps `SYCLCompiler` for compile/run/parse. Handles
  source string to temp file conversion, AOT device targeting, and SPIRV
  extension flags.
- **`KernelType`** (enum in `sycl_executor.py`) — Controls include paths:
  `GEMM`, `FA`, `DUAL_GEMM`, `GROUPED_GEMM`, `MOE_GEMM`.

### FA V2 Template

The Flash Attention V2 template uses `FMHAConfigGenWithTileShape` from
`sycl-tla/benchmarks/flash_attention/fmha_configuration.hpp`:
- Takes 7 integer tile params + FMHAMode + boolean flags
- Automatically derives ShapeQK, ShapePV, ShapeOutput, and SubgroupLayout
- Configurable: `causal`, `fa_mode` (prefill/decode), `persistent` via YAML
- VarLen, CachedKV, PagedKV reserved for future use (hardcoded false)
- Includes standalone runner with correctness verification and timing

## Adding New Kernel Types

1. Implement `KernelStrategy` protocol (see `GEMMStrategy` as reference)
2. Add a Jinja2 C++ template in `templates/` with a Python generator
3. Add a tile validator in `validators/`
4. Add `KernelType` enum value if new include paths are needed
5. Add a DSPy signature with kernel-specific prompting
6. Register the strategy in `cli.py` (`mode_to_strategy` / `mode_to_kernel_type`)
