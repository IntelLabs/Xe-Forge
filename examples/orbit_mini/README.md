# `orbit_mini` — Xe-Orbit reference micro-workload

*(plan §15; read that section before changing anything here)*

A two-layer, Qwen-shaped decoder block at toy dimensions — attention with
grouped-query heads and QKV bias, RMSNorm, SwiGLU MLP, RoPE — that runs the
whole Xe-Orbit pipeline end to end in seconds instead of twenty minutes.

```
hidden = 128   heads = 4   kv_heads = 2 (GQA)   head_dim = 32
ffn = 256      seq = 64    batch = 2            layers = 2
```

**It is deliberately adversarial for kernel extraction.** A test workload that
extracts cleanly tests nothing: every trap below corresponds to a real failure
mode the plan describes, and each one is here so the extractor has to get it
right in CI rather than the first time someone points it at vLLM.

## Run it

```bash
# from the repository root
python -m examples.orbit_mini
python -m examples.orbit_mini --steps 5
python -m examples.orbit_mini --json          # structured summary for a test to assert on
```

or from Python:

```python
from examples.orbit_mini import build_model, get_example_inputs, run, main

model = build_model()  # eval mode by default
inputs = get_example_inputs()  # non-contiguous on purpose — see trap 5
summary = run(steps=3)  # dict, not console output
```

**No GPU, no Triton, no numpy required.** Plain PyTorch on CPU is enough, and
that is a hard constraint: the CPU-only CI tier T0 (§16.6) is where most of the
pipeline is tested. The Triton and SYCL paths are present in the source —
that is the point, the file structure has to exercise the §11 language
taxonomy — but every one of them sits behind an availability check with a
pure-torch fallback.

## The traps, and why each one is here

| # | Trap | Where | What it catches |
|---|------|-------|-----------------|
| 1 | Device helpers split across **three modules**, one reached only through an alias re-export | `kernels/helpers_{a,b,c}.py`, `kernels/device_ops.py` | Closure resolution that stops at the kernel's own file or scrapes imports textually (§12.6 step 2). The resulting bundle imports fine on the dev machine — because the source package is on `sys.path` — and fails the isolated-import check (§12.12 step 1). |
| 2 | An **autotune config list** of four points, plus a deterministic selection function | `kernels/rmsnorm.py: RMSNORM_CONFIGS`, `select_config` | Optimizing a specialization the workload never runs. §12.7 requires the winning config to be captured and pinned; §12.12 step 2 requires the pin to match the intercepted launch. `kernels/swiglu.py` carries a *different* config list, so pinning one global config is not good enough. |
| 3 | A **heuristics callable closing over a module-level constant** | `kernels/rmsnorm.py: _num_stages_hint`, `SPLIT_THRESHOLD_ELEMS` | §12.6 step 4. Heuristics lambdas read module state constantly and the constant is the easy thing to drop; the extracted kernel then compiles at a different pipeline depth than the one that was measured. |
| 4 | A **tuned-config JSON keyed by device name**, read by the launch wrapper, with no in-code default | `kernels/tuned_configs.json`, `kernels/tuned.py` | §12.8 and §12.12 step 5. Delete the file and the workload raises `TunedConfigError` — which is what makes "remove each declared data file in turn; each removal must produce a failure" a real test. The block size, epsilon and clamp all come from here, so the same source behaves differently on two machines. |
| 5 | One **deliberately non-contiguous input** | `__init__.py: get_example_inputs` | Synthetic-input reconstruction. The tensor is allocated `(batch, hidden, seq)` and handed back transposed: shape `(2, 64, 128)`, strides `(8192, 1, 64)`. Anything that rebuilds it from a shape and a dtype gets a contiguous tensor and a different launch record (§12.4, §16.4 row 6). The launch wrappers never call `.contiguous()` on the way in, so the strides survive to the record. Holds on CPU, XPU and CUDA — it is a property of the view. |
| 6 | One **hand-written SYCL kernel registered as a dispatcher op**, with its own CMake build | `sycl/orbit_mini_rmsnorm.cpp`, `sycl/CMakeLists.txt`, `kernels/sycl_op.py` | §11 is explicit that SYCL is not the exception. This covers build-graph closure (for a compiled kernel the closure is `compile_commands.json`, not an AST walk — §11.3), the compiler-option sweep and the `icpx` harness (§11.4), and the **P1 rung** of the patch-back ladder (§13) on a SYCL op. **It is not built by default and must never be required.** |
| 7 | One **opaque library call** | `kernels/opaque_gemm.py` | The E4 and `NO_ACTION` paths (§12.5). `torch.matmul` dispatches into a vendor BLAS; there is no source to lift. An extractor reporting E1/E2 here is lying, and an action planner proposing `kernel_rewrite` against it is proposing something it cannot deliver. The op is on the hot path — QKV, output and down projections of both layers — so it holds real time in the catalog. |
| 8 | One **region of three fusable kernels** | `kernels/region.py: MLP_REGION`, `run_region` | The Xe-Fuse path (§12.11). `o_proj GEMM -> post-attention RMSNorm -> SwiGLU` is a genuine producer-consumer chain in `DecoderLayer.forward`, with two intermediates a fused replacement would eliminate. The three members sit at three different extraction levels (E4, E1, E1), and the GEMM is a boundary fusion cannot cross — a candidate claiming to absorb it is wrong, testably. |

Two further details that are deliberate rather than incidental:

- **`kernels/helpers_c.py` is in the closure of two different kernels** (RMSNorm,
  transitively, and RoPE, directly). Per-kernel bundles have to share it
  correctly in both directions; dropping it from either is a bug.
- **`kernels/triton_compat.py`** keeps the source Triton-shaped when Triton is
  not installed. The decorators and the `tl` namespace always exist: with Triton
  they are the real thing, without it they are structural stand-ins that
  preserve everything the closure walk reads (`fn`, `configs`, `key`, `values`,
  `kwargs`, `num_warps`, `num_stages`) and refuse to *execute*. Refusing is safe
  because every launch wrapper checks `HAS_TRITON` and the device type first.

## What still needs a GPU

Everything below is present in the source and unreachable on CPU-only CI. It
belongs to tiers T1/T2 (§16.6), not T0.

- **The Triton launches** — `_rmsnorm_kernel`, `_swiglu_kernel`, `_rope_kernel`.
  Guarded by `HAS_TRITON and device.type in ("cuda", "xpu")`. Their *structure*
  is asserted on CPU; their numerics are not.
- **Real autotuning.** On CPU, `select_config` stands in for Triton's autotuner
  so the "which config actually ran?" question has an answer. Only on silicon
  does a config genuinely win, and only there can §12.10's specialization match
  (kernel name, binary hash, register/spill/SLM counts) be checked.
- **The SYCL extension.** Needs `icpx` and an XPU runtime:

  ```bash
  cd examples/orbit_mini/sycl
  cmake -S . -B build -DCMAKE_CXX_COMPILER=icpx \
        -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
  cmake --build build -j
  export ORBIT_MINI_SYCL_LIB=$PWD/build/liborbit_mini_sycl.so
  python -m examples.orbit_mini --device xpu
  ```

  Without `icpx` the translation unit still compiles and still registers the op,
  but the implementation refuses to run — silently computing something else
  would defeat the §13 dispatch assertion.
- **P1 override verification** (§13): re-profile after registering an override
  and confirm the new kernel appears in the trace *and the old one does not*.
- **Anything measured.** §17's repetition, confidence-interval and
  positive-control requirements are about a device with clocks and power states;
  CPU timings printed by `run()` are a smoke signal, not a measurement.

## Rules for changing this workload

The traps are load-bearing. In particular:

- Do not add a `.contiguous()` to any launch wrapper's input.
- Do not add an in-code default for `tuned_configs.json`.
- Do not flatten the helper modules together, and do not remove the
  `device_ops` re-export hop.
- Do not import anything from `xe_forge.orbit` here. The workload has to stand
  alone, or it is testing the pipeline against itself.
- Keep it running on CPU with plain PyTorch, no Triton, no numpy.
