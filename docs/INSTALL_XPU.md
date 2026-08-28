# Xe-Orbit on Intel XPU — installation and configuration, step by step

Everything below was verified on a real box (Wildcat Lake iGPU, 16 EU, 13.84 GB
shared memory, Arch Linux, 15.6 GB RAM) during the 2026-08 bring-up. Where a step
exists because something broke without it, the trap is named — this document is
the distillation of every dead end so you do not have to rediscover them. Paths
use `~/.cache/orbit-dev` as the source-and-build root; any root works, but set
`ORBIT_XE_FUSE_DIR` / `SYCL_TLA_DIR` if you deviate (they are authoritative in
both directions).

## 0. What you need before starting

| Requirement | Why | Check |
| --- | --- | --- |
| Intel GPU + driver + Level Zero | everything | `ls /usr/include/level_zero/ze_api.h`, `sycl-ls` |
| oneAPI Base Toolkit (icpx 2025.x+) | SYCL kernel compiles, Xe-Fuse, overrides | `source /opt/intel/oneapi/setvars.sh --force && icpx --version` |
| Python 3.12/3.13 + `uv` | the serving venv | `uv --version` |
| ~60 GB free disk | checkouts + wheels + build trees | `df -h ~` |
| **≥30 GB swap** | single cutlass TUs peak >30 GB virtual during kernel builds | `swapon --show` |
| HF account + token | model downloads; Llama/Gemma need license acceptance per model | `huggingface-cli whoami` |

**Swap is not optional** on a ≤16 GB machine. One translation unit of
vllm-xpu-kernels' paged-attention peaked at 10.8 GB RSS + 20.4 GB swap with
`MAX_JOBS=1`. We run 15 GB zram (priority 100) + a 15 GB swapfile.

## 1. oneAPI environment — source it, always, everywhere

```bash
source /opt/intel/oneapi/setvars.sh --force
```

**Trap (measured):** a compile that works in your shell and fails in a subprocess
usually means the subprocess didn't inherit this. `icpx` found by absolute path
still misses MKL headers (`oneapi/mkl/rng/device.hpp: file not found`) at compile
time and `libsycl` at run time. Orbit's Xe-Fuse executor now sources the
`setvars.sh` that owns the resolved compiler by itself; your own scripts must do
the same — put the `source` line *inside* every build/run script, not in your
profile.

## 2. The serving venv: torch XPU + vLLM

```bash
uv venv ~/.cache/orbit-dev/vllmxpu --python 3.13
P=~/.cache/orbit-dev/vllmxpu/bin/python

# torch XPU wheels live on a dedicated index — the resolver cannot see them otherwise
uv pip install --python $P torch==2.13.0+xpu torchvision --index-url https://download.pytorch.org/whl/xpu

# vLLM from source, editable (needed: the in-place patch rungs operate on this tree)
git clone https://github.com/vllm-project/vllm ~/.cache/orbit-dev/vllm-src
cd ~/.cache/orbit-dev/vllm-src
VLLM_TARGET_DEVICE=xpu uv pip install --python $P -e . --extra-index-url https://download.pytorch.org/whl/xpu

# the Intel kernel library (SYCL, compiled) — this is where vLLM's XPU kernels live
uv pip install --python $P vllm-xpu-kernels
```

Facts worth knowing (2026-08):
- As of vLLM v0.16 the IPEX dependency is gone; **all** Intel kernels live in
  `vllm-project/vllm-xpu-kernels` (SYCL, ships compiled) plus in-wheel **Triton**
  `.py` kernels (ship as source — patchable in place).
- **Trap (measured):** `VLLM_ATTENTION_BACKEND` env var no longer exists on
  current vLLM. The knob is the engine argument: `LLM(attention_backend=...)` or
  `vllm serve --attention-backend`. On XPU, `TRITON_ATTN` is the generic fallback
  (any head_size ≥ 32) when the compiled SYCL set lacks your model's shape.
- Build deps for a source build of vllm-xpu-kernels: `uv pip install --python $P
  cmake ninja setuptools-scm build` and use `--no-build-isolation` — **trap
  (measured):** uv's isolated build env cannot resolve `torch==2.13.0+xpu` (it
  doesn't know the XPU index) and fails with "No solution found".

## 3. Kernel-source checkouts, pinned

Compiled wheels ship without sources; Orbit clones the sources and pins them
(§12.10: a tree is identified by URL *and* revision).

```bash
cd ~/.cache/orbit-dev
git clone https://github.com/vllm-project/vllm-xpu-kernels   # pin: see kernel_sources
git clone https://github.com/sgl-project/sgl-kernel-xpu
git clone https://github.com/intel/torch-xpu-ops             # pin: torch's third_party/xpu.txt
git clone https://github.com/codeplaysoftware/sycl-tla       # cutlass SYCL port (Xe-Fuse dep)
git clone https://github.com/IntelLabs/Xe-Fuse
```

Pins live in `knowledge_base/common/framework_*.yaml` under `kernel_sources:`.
**Trap (measured):** vllm-xpu-kernels wheels with a 4th version component
(e.g. 0.1.13.2) are PyPI-only respins with **no git tag** — same code base as the
3-component tag (`v0.1.13` = commit `07d44bcb`), different build config. Pin the
3-component tag's commit and note the inference.

## 4. kineto (GPU-capable) — required for sgl-kernel, useful everywhere

The torch XPU wheel ships **without** `libkineto.a`; sgl-kernel's cmake requires
it. Build the GPU-capable variant (Intel PTI = XPU profiling interface):

```bash
cd ~/.cache/orbit-dev
git clone https://github.com/intel/pti-gpu     # if not present
cd pti-gpu/sdk
source /opt/intel/oneapi/setvars.sh --force
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/.cache/orbit-dev/pti-install \
      -DPTI_BUILD_SAMPLES=OFF -DPTI_BUILD_TESTING=OFF
cmake --build build -j2 && cmake --install build

cd ~/.cache/orbit-dev
git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/pytorch/kineto
cd kineto/libkineto
cmake -B build -DCMAKE_BUILD_TYPE=Release -DKINETO_BUILD_TESTS=OFF \
      -DLIBKINETO_NOCUPTI=ON -DLIBKINETO_NOROCTRACER=ON -DLIBKINETO_NOXPUPTI=OFF \
      -DPti_DIR=$HOME/.cache/orbit-dev/pti-install/lib/cmake/pti \
      -DCMAKE_PREFIX_PATH=$HOME/.cache/orbit-dev/pti-install
cmake --build build -j2
cp build/libkineto.a ~/.cache/orbit-dev/vllmxpu/lib/python3.13/site-packages/torch/lib/
```

Fallback if the XPUPTI variant fails: swap `-DLIBKINETO_NOXPUPTI=OFF` for `ON`
(CPU-only — satisfies the linker, loses GPU-side kineto activity).

## 5. SGLang (source-only on XPU)

No prebuilt XPU wheels exist. Official path (docs/platforms/xpu.md):

```bash
git clone https://github.com/sgl-project/sglang ~/.cache/orbit-dev/sglang
cd ~/.cache/orbit-dev/sglang/python
cp pyproject_xpu.toml pyproject.toml
uv pip install --python $P -v . --extra-index-url https://download.pytorch.org/whl/xpu

cd ~/.cache/orbit-dev/sgl-kernel-xpu
source /opt/intel/oneapi/setvars.sh --force
CMAKE_BUILD_PARALLEL_LEVEL=2 \
  SKBUILD_CMAKE_DEFINE="USE_SYCL_JIT=ON;SGL_BUILD_MEM_FILE_LIMIT_GIB=12" \
  uv pip install --python $P --no-build-isolation -v .
```

- `USE_SYCL_JIT=ON` (env or cmake define): FMHA/MLA/MoE/GDN kernels compile **on
  demand** at first use instead of ahead-of-time — this is what makes the build
  survivable on ≤16 GB machines. First call into a JIT kernel blocks for seconds
  to minutes while icpx runs.
- **LoRA kernels stay AOT** even with JIT on. The repo's own build guards: a
  per-TU RSS limit (`SGL_BUILD_MEM_FILE_LIMIT_GIB`, default 4 — raise via the
  define above on a big-swap box) and a system floor (stops the build when
  MemAvailable < 0.5 GiB; `FORCE`-cached, not overridable — control parallelism
  instead).
- **Trap (measured):** `MAX_JOBS` does nothing here (torch convention);
  scikit-build-core's ninja runs `nproc`-wide by default and three ~4 GiB TUs in
  flight exhaust a 16 GB box. The knob is `CMAKE_BUILD_PARALLEL_LEVEL=2`.
- Serving flag on XPU: `--attention-backend intel_xpu` (SYCL) — Triton is used
  for MoE per Intel's roadmap.

## 6. Xe-Orbit itself

```bash
cd ~/Projects/Xe-Forge
uv pip install --python $P -e .          # or: PYTHONPATH=src for a no-install run
$P -m xe_forge.orbit.cli selftest --quick
```

Environment variables Orbit honors:

| Variable | Meaning |
| --- | --- |
| `ORBIT_XE_FUSE_DIR`, `SYCL_TLA_DIR` | authoritative checkout locations |
| `ORBIT_FUSED_MLP=1` + `ORBIT_FUSED_LIB=<.so>` | opt-in guard for the fused-MLP patch |
| `ORBIT_GPU_UTIL` | vLLM startup-gate utilization (see trap below) |
| `ORBIT_SYCL_SOURCES` | extra source trees for the SYCL backend |

## 7. System configuration for honest measurement

- **Quiet machine, enforced:** every GPU-touching Orbit command takes the
  per-device lease (`~/.cache/orbit-dev/leases/`); a second claimant is refused
  by name. Do not run builds, test suites, or anything CPU-heavy during
  measurement — **measured:** a test suite running during e2e arms throttled the
  shared-TDP iGPU ~35% and poisoned the session.
- **Long builds go through the build lane, under systemd:**
  `xe-orbit build-lane submit --component X --cwd DIR -- bash -c '<cmd>'` then
  `systemd-run --user --collect --unit=orbit-build-lane -p ManagedOOMSwap=auto
  -p ManagedOOMMemoryPressure=auto -p MemoryHigh=11G -p CPUWeight=40
  --setenv=PYTHONPATH=... python -m xe_forge.orbit.cli build-lane run --all`.
  **Traps (measured):** a runner tied to your terminal dies with it (taking the
  compiler with it), and systemd-oomd kills un-exempted units at high swap
  pressure — both settings above exist because each killed a real build once.
- **vLLM memory gate on unified-memory iGPUs:** the free-memory probe does not
  count reclaimable page cache, so back-to-back engine starts fail the 
  `gpu_memory_utilization` check spuriously. Pin the KV cache explicitly
  (`kv_cache_memory_bytes=256MB`) and treat utilization as gate-only
  (`ORBIT_GPU_UTIL=0.12`); wait ~60 s between engine starts.

## 8. Smoke test — the full loop in six commands

```bash
# 1. point-and-start profile of any HF model on vLLM
$P quant_profile.py Qwen/Qwen2.5-0.5B-Instruct /tmp/trace_dir   # or: xe-orbit trace --wrap -- <cmd>
# 2. ingest + rank
xe-orbit trace --from-trace /tmp/trace_dir/rank0.*.pt.trace.json
xe-orbit kernels
xe-orbit regions
# 3. fused-kernel sweep at the model's shapes (deterministic, no agent)
xe-orbit fuse r0 --shapes 16x9728x896 --tiles auto,16x256x32
# 4. guarded patch + ABBA e2e arms + §17 verdict; keeps on ACCEPT, reverts otherwise
xe-orbit fuse-apply --model Qwen/Qwen2.5-0.5B-Instruct --lib <liborbit_fused.so> --e2e
```

A model the stack cannot serve is not a dead end: the failure is classified
(`xe-orbit` prints the enablement diagnosis — rung + concrete next move), and the
ladder climbs: serve flag → source patch → scoped runtime → build lane.
**Measured example:** phi-2's head_dim=80 is missing from the compiled SYCL
paged-decode set; `attention_backend=TRITON_ATTN` served it the same hour at
97 tok/s, no rebuild.
