# SYCL Claude Engine Example (Intel Xe / Battlemage)

A worked, hardware-verified example of the **Claude Code engine for SYCL XPU
kernels**: the agent rewrites a whole CUTLASS SYCL `.cpp` GEMM each trial, then
compiles (`icpx`) + benchmarks it via `SyclExecutor`, with correctness checked
against a **golden PyTorch reference** (`numpy.allclose` on the kernel's dumped
`D2.bin`).

| File | Role |
|------|------|
| `gemm.cpp` | Baseline kernel (t0). CUTLASS BMG GEMM `D = A·B0`, tile `256×256×32`. Honours the file-IO contract. |
| `gemm_t1.cpp` | One optimization trial (t1). Same kernel, tile `128×128×32` — ~1.7× faster at 1024³ on an Arc Pro B70. |
| `gemm.yaml` | KernelBench-style spec: GEMM dims `M=N=K=1024`, bf16, `rtol=atol=0.02`. |
| `gemm_pytorch.py` | Golden PyTorch reference: `Model.forward(A, B0) -> A.float() @ B0.float()`. |

## The file-IO contract

Every SYCL kernel optimized by this engine is a standalone executable invoked as:

```
./kernel --m=<M> --n=<N> --k=<K> --input_dir=<dir> --output_dir=<dir> --iterations=<int> --verify=<int>
```

It reads `A.bin` `[M,K]` and `B0.bin` `[K,N]` (raw row-major, bf16 stored as
int16 bits) from `--input_dir`, computes `D = A·B0`, writes `D2.bin` `[M,N]`
(float32, row-major) to `--output_dir`, and prints a `… TFlop/s … ms` line.
Full spec: [`knowledge_base/sycl/xpu/sycl_io_contract.yaml`](../../knowledge_base/sycl/xpu/sycl_io_contract.yaml).

## Environment (Intel XPU box)

```bash
export SYCL_TLA_DIR=/path/to/sycl-tla          # CUTLASS SYCL checkout
export AIBENCH_SYCL_TARGET=bmg-g31             # AOT target (Battlemage: B580/B570/B70)
export MKL_INCLUDE=/path/to/oneapi/include
export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
```

## Reproduce the benchmark

t0 — compiles both kernels, times the baseline, checks the trial vs the golden ref:

```bash
xe-forge-skill benchmark examples/sycl/gemm.cpp examples/sycl/gemm_t1.cpp \
    --spec examples/sycl/gemm.yaml --dsl sycl --variant bench-xpu
```

```
Correctness: PASSED
Performance: baseline_us=193.80, triton_us=109.90, speedup=1.76x
```

t1+ — reuse the cached baseline (no baseline recompile/rerun):

```bash
xe-forge-skill benchmark examples/sycl/gemm.cpp examples/sycl/gemm_t1.cpp \
    --spec examples/sycl/gemm.yaml --dsl sycl --variant bench-xpu --baseline-us 193.80
```

The `triton_us=` token is kept verbatim across DSLs so the trial tooling parses
uniformly; for SYCL it carries the optimized kernel's time in microseconds.

## Generate an agentic workspace

```bash
python -m xe_forge.cli --input examples/sycl/gemm.cpp --name gemm \
    --dsl sycl --engine claude --spec examples/sycl/gemm.yaml \
    --variant bench-xpu --workspace /tmp/ws_sycl
```

This scaffolds a SYCL `CLAUDE.md`, an `/optimize-kernel` command wired with
`--dsl sycl`, the kernel + `gemm_pytorch.py` golden reference under
`test_kernels/`, and a `knowledge_base/` symlink. Passing a PyTorch-only `.py`
input instead substitutes a compilable starter `.cpp` (a copy of `gemm.cpp`)
and uses the `.py` as the golden reference.
