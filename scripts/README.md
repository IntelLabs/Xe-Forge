# Roofline plotting scripts

Standalone tools for turning kernel measurements (or just problem shapes) into
[roofline plots](https://en.wikipedia.org/wiki/Roofline_model). None of them
import the `xe_forge` package, so you can copy a results CSV off an XPU box and
plot it anywhere.

Each script is a [PEP 723](https://peps.python.org/pep-0723/) single-file script
with its dependencies declared inline, so `uv run` pulls them ephemerally — there
is nothing to install into the project venv.

| Script | Input | Output | Use it to… |
|--------|-------|--------|------------|
| [`roofline.py`](roofline.py) | a roofline CSV | a `.png` plot | draw the actual plot |
| [`benchmark_to_roofline_csv.py`](benchmark_to_roofline_csv.py) | a benchmark **results** CSV | a roofline CSV | plot measured baseline → optimized pairs |
| [`spec_to_roofline_csv.py`](spec_to_roofline_csv.py) | a kernel **spec** YAML | a roofline CSV | plot the theoretical ceiling for each shape |

The two `*_to_roofline_csv.py` scripts are converters that produce the CSV
`roofline.py` consumes; `roofline.py` can also read a hand-written CSV directly.

---

## `roofline.py` — draw the plot

A roofline plot shows achieved performance (TFLOPS, *y*) against arithmetic
intensity (FLOP/byte, *x*) on log-log axes, overlaid with the hardware "roof":

```
performance_ceiling(AI) = min(peak_compute_tflops,             # flat compute roof
                              AI * peak_bandwidth_gbps / 1000)  # sloped memory roof
```

Points to the left of the ridge are memory-bound (sit under the sloped roof);
points to the right are compute-bound (sit under the flat roof). The closer a
point is to the roof, the better the kernel uses the hardware.

### CSV format

One row per measurement, header row required. Column names are matched
case-insensitively and several aliases are accepted:

| Canonical column | Aliases | Required? | Meaning |
|------------------|---------|-----------|---------|
| `arithmetic_intensity` | `ai`, `intensity`, `flop_per_byte`, `flops_per_byte` | **yes** | FLOP / byte |
| `tflops` | `perf`, `performance`, `throughput` | optional | achieved throughput in TFLOPS |
| `series` | `backend`, `kind`, `category`, `engine` | optional | grouping / legend label → **marker shape** |
| `family` | `group_color`, `op`, `kernel_family` | optional | independent grouping → **marker color** |
| `label` | `name`, `shape`, `config` | optional | per-point annotation |
| `pair` | `group`, `kernel`, `link` | optional | id linking two points with a connector line |

If you omit `tflops`, the point is placed **on** the roof — showing the
theoretical ceiling for that shape. You don't have to precompute `tflops` /
`arithmetic_intensity` either; provide raw measurements and they're derived:

| Provide… | …and you get | Formula |
|----------|--------------|---------|
| `flop` + `time_us` | `tflops` | `flop / time_us / 1e6` |
| `flop` + `bytes` | `arithmetic_intensity` | `flop / bytes` |
| `gflops` | `tflops` | `gflops / 1000` |

Example (`sample_roofline.csv`) — paired baseline/optimized points linked by `pair`:

| series | label | pair | arithmetic_intensity | tflops |
|--------|-------|------|----------------------|--------|
| Original | A=72 S=2k | fa-72-2k | 900 | 35 |
| Optimized | A=72 S=2k | fa-72-2k | 900 | 74 |
| Original | A=32 S=4k | fa-32-4k | 1500 | 18 |
| Optimized | A=32 S=4k | fa-32-4k | 1500 | 71 |

### Hardware presets

The roof is set by a preset (or overridden with `--peak-tflops` /
`--peak-bandwidth`). Run `--list-hardware` to print the table:

| Preset | Device | Peak TFLOPS (FP16/BF16) | Peak BW (GB/s) |
|--------|--------|-------------------------|----------------|
| `arc-pro-b70` | Intel Arc Pro B70 | 160 | 608 |
| `arc-b580` | Intel Arc B580 | 117 | 456 |
| `max-1550` | Intel Data Center GPU Max 1550 (PVC) | 839 | 3276 |
| `max-1100` | Intel Data Center GPU Max 1100 (PVC) | 362 | 1228 |
| `flex-170` | Intel Data Center GPU Flex 170 | 137 | 576 |

### Options

| Flag | Effect |
|------|--------|
| `--hardware <preset>` | use a built-in roof (see table above) |
| `--peak-tflops` / `--peak-bandwidth` | override the roof explicitly (or partially override a preset) |
| `--connect` | draw a baseline → optimized arrow between points sharing a `pair` id |
| `--annotate` | label each point inline (best for a handful of points) |
| `--key` | number each point and list labels in a side key (best for many points) |
| `--series-order a,b,c` | fix the legend/color order |
| `--dpi N` | output resolution (default 200) |
| `--list-hardware` | print the preset table and exit |

### Usage

```bash
# plot with the Arc Pro B70 preset roof
uv run scripts/roofline.py scripts/sample_roofline.csv \
    --hardware arc-pro-b70 --connect --annotate \
    --title "Roofline — FlashAttention Fwd" -o plots/fa.png

# list the built-in hardware presets
uv run scripts/roofline.py --list-hardware

# override the roof explicitly (skips the preset table)
uv run scripts/roofline.py results.csv \
    --peak-tflops 160 --peak-bandwidth 608 -o plots/gemm.png
```

---

## `benchmark_to_roofline_csv.py` — measured baseline vs optimized

Converts a benchmark **results** CSV (one row per kernel variant) into the
roofline CSV above. Each input row carries *measured* times for both the
reference (`baseline_us`) and the optimized Triton kernel (`triton_us`), plus the
optimized throughput (`tflops`). Each row is expanded into **two** output rows
sharing a `pair`:

- **Optimized** — `tflops` as measured.
- **Original** — `tflops × triton_us / baseline_us` (same FLOP, scaled by the
  inverse time ratio, i.e. `tflops / speedup`).

Input CSV (excerpt):

| family | shape_key | config | baseline_us | triton_us | tflops | … |
|--------|-----------|--------|-------------|-----------|--------|---|
| 2_BatchedMoE | bench-gpu | `E=128,M=64,K=768,N=2048` | 826.56 | 784.2 | 32.861 | … |
| 3_FusedMoE | bench-gpu-1 | `E=32,M=64,K=2880,N=5760` | 2141.1 | 2009.19 | 33.818 | … |

Output CSV (two rows per input row):

| series | family | label | pair | arithmetic_intensity | tflops |
|--------|--------|-------|------|----------------------|--------|
| Original | BatchedMoE | qwen3-30b-A3B: … | 2_BatchedMoE:bench-gpu | 57.42 | 31.177 |
| Optimized | BatchedMoE | qwen3-30b-A3B: … | 2_BatchedMoE:bench-gpu | 57.42 | 32.861 |

### Arithmetic-intensity model

AI is `FLOP / compulsory-DRAM-bytes` — the once-through traffic estimate
(each tensor streamed from DRAM exactly once, perfect reuse). This is an *upper
bound*, so points sit at their most compute-bound position; real kernels with
imperfect reuse sit further left. Per family:

| Family | FLOP | DRAM bytes (each tensor once) |
|--------|------|-------------------------------|
| BatchedMoE | `E·M·N·K·2` (all `E` experts active) | `E·M·K·act + E·N·K·w + E·M·N·act + E·4` |
| FusedMoE | `M·TOPK·N·K·2` (token-routed) | `M·TOPK·K·act + Eact·N·K·w + M·TOPK·N·act + M·TOPK·4`, `Eact = min(E, M·TOPK)` |
| UnifiedAttention | `4·TQ·QH·D·MKV` (QK^T + softmax·V) | `TQ·QH·D·2 + 2·MKV·KH·D·kv + TQ·QH·D·2` |

Element widths (`act`, `w`, `kv`) are selected per family by the spec's quant
code. Dims missing from the CSV `config` string (quant widths, MoE `TOPK`) are
read from the example spec YAMLs.

### Options

| Flag | Effect |
|------|--------|
| `-o, --output` | output CSV (default: stdout) |
| `--min-tflops T` | drop a whole pair whose **Optimized** throughput is below `T` (declutters dense memory-bound shapes) |
| `--min-ai A` | drop a pair whose arithmetic intensity is below `A` FLOP/byte (drops the left-most shapes) |

### Usage

```bash
uv run scripts/benchmark_to_roofline_csv.py benchmark.csv -o roofline_input.csv
uv run scripts/roofline.py roofline_input.csv --hardware arc-pro-b70 \
    --connect --key --title "Roofline (baseline vs Triton)" \
    -o plots/roofline.png
```

---

## `spec_to_roofline_csv.py` — theoretical ceilings from a spec

A kernel spec YAML carries only *problem shapes* and a FLOP formula — no measured
times. This script emits each bench variant's **arithmetic intensity**, so when
fed to `roofline.py` *without* a `tflops` column each variant lands on the roof,
showing where the shape sits (memory- vs compute-bound) and the best the hardware
could theoretically deliver.

Bytes use the same compulsory-traffic Fused-MoE model as above. `QUANT` selects
element widths (matching the spec's w13/w2 quant variants):

| `QUANT` | meaning | act bytes | weight bytes |
|---------|---------|-----------|--------------|
| 0 | bf16 | 2 | 2 |
| 1 | fp8 w8a8 | 1 | 1 |
| 2 | int8 w8a8 | 1 | 1 |
| 3 | int8 w8a16 | 2 | 1 |

Output CSV:

| series | label | arithmetic_intensity | flop | bytes |
|--------|-------|----------------------|------|-------|
| bf16 | Qwen3-30B-A3B w13 [M…] | … | … | … |

### Options

| Flag | Effect |
|------|--------|
| `-o, --output` | output CSV (default: stdout) |
| `--variant-prefix P` | only convert variant keys starting with `P` (default: `bench`) |

### Usage

```bash
uv run scripts/spec_to_roofline_csv.py examples/vllm/2_FusedMoE.yaml -o moe.csv
uv run scripts/roofline.py moe.csv --hardware arc-pro-b70 --annotate \
    --title "Roofline — Fused MoE (theoretical ceilings)" -o plots/moe.png
```
