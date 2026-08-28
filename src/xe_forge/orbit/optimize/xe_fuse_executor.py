"""
Driving Xe-Fuse as a fusion executor (plan §13.4, §9.6).

Xe-Fuse is a checkout, not a pip package: a kernel generator
(`autotune/generate_kernel.py`), a header-only builder library, and sycl-tla headers,
compiled per-shape with icpx and benchmarked by the binary it emits. This module is
the seam that lets Orbit — and, through `FusionTask.executor`, an agent proposing a
region action — run that flow programmatically: locate the checkouts, map the
region's pattern to a preset, generate, compile, run, and parse the measurement.

Measured on Wildcat Lake at the traced Qwen2.5-0.5B decode shapes (M=16, N=9728,
K=896), the k2 preset (GEMM+RMSNorm+SwiGLU in one kernel) beat vLLM's unfused chain
by +3.1% (95% CI [0.24%, 5.94%]) with numerics verified to 0.14% median relative
error — and lost by ~25% at M=128, where oneDNN's GEMM efficiency dominates. A
fusion result is therefore per-shape, never global (§14.3): the caller decides per
regime, and §14.4's specialization set is the deployment shape.

Three upstream findings from that session, encoded here as workarounds until fixed:
the generated benchmark parses `--verify` and never uses it (a skipped check wearing
a flag); `initialize_block` zero-fills float scale buffers, so any future
output-vs-reference check would pass trivially on D == 0; and the tile
auto-selector picks well at small M but poorly at M=128. Verification here is
therefore the caller's job (the differential harness), not the binary's claim.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# The generator's presets, keyed by the region patterns the detector emits (§13.4).
# Knowledge-shaped on purpose: where a pattern maps is a fact about Xe-Fuse, and a
# new preset should be one entry here, not a code change elsewhere.
PRESET_FOR_PATTERN = {
    "gemm+activation": "k2",  # RMSNorm row-scale commutes through the GEMM (§13.4)
    "gemm+rmsnorm+swiglu": "k2",
    "gemm+rmsnorm": "k1",
    "gemm+geglu": "k2_geglu",
}

XE_FUSE_DIR_ENV = "ORBIT_XE_FUSE_DIR"
SYCL_TLA_DIR_ENV = "SYCL_TLA_DIR"

# The SPIR-V translator refuses sycl-tla's split barriers unless the extension is
# named explicitly; found the hard way on the first compile (JIT spir64 path).
_SPIRV_EXT_FLAGS = ("-Xspirv-translator", "--spirv-ext=+SPV_INTEL_split_barrier")


def _setvars_for(compiler: str | Path) -> Path | None:
    """The setvars.sh that owns this compiler, or None if it is not a oneAPI icpx.

    Walk up from the compiler path looking for the oneAPI root's setvars.sh —
    e.g. /opt/intel/oneapi/compiler/<ver>/bin/icpx -> /opt/intel/oneapi/setvars.sh.
    """
    path = Path(compiler).resolve()
    for parent in path.parents:
        candidate = parent / "setvars.sh"
        if candidate.is_file():
            return candidate
    return None

_MS_RE = re.compile(r"\(\s*([0-9.]+)\)ms")
_TFLOPS_RE = re.compile(r"\[([0-9.]+)\]TFlop/s")


def _search_roots() -> list[Path]:
    """The same roots the SYCL source registry searches (§11.2)."""
    from xe_forge.orbit.languages.sources import candidate_roots

    return candidate_roots()


def find_xe_fuse() -> Path | None:
    """Locate an Xe-Fuse checkout: env override, else the source roots.

    An explicit override is authoritative in both directions: when it names a valid
    checkout it wins, and when it names something else the answer is None — falling
    through to a search would turn "use exactly this" into "use whatever is lying
    around", which is how a test or a pinned experiment silently picks up a
    different tree than the one it declared.
    """
    override = os.environ.get(XE_FUSE_DIR_ENV)
    if override:
        candidate = Path(override).expanduser()
        return candidate if (candidate / "autotune" / "generate_kernel.py").is_file() else None
    for root in _search_roots():
        for candidate in (root / "Xe-Fuse", root / "xe-fuse"):
            if (candidate / "autotune" / "generate_kernel.py").is_file():
                return candidate
    return None


def find_sycl_tla() -> Path | None:
    """Locate a sycl-tla checkout, with the same authoritative-override rule."""
    override = os.environ.get(SYCL_TLA_DIR_ENV)
    if override:
        candidate = Path(override).expanduser()
        return candidate if (candidate / "include").is_dir() else None
    for root in _search_roots():
        candidate = root / "sycl-tla"
        if (candidate / "include").is_dir():
            return candidate
    return None


def checkout_available() -> bool:
    """Whether the generate-compile-run flow can work at all on this machine."""
    return find_xe_fuse() is not None and find_sycl_tla() is not None


@dataclass
class XeFuseResult:
    """One generate → compile → run cycle, honestly reported."""

    preset: str
    tile: str
    m: int
    n: int
    k: int
    generated_cpp: Path | None = None
    binary: Path | None = None
    ms: float | None = None
    tflops: float | None = None
    error: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.ms is not None and not self.error

    @property
    def per_iteration_us(self) -> float | None:
        return self.ms * 1000.0 if self.ms is not None else None


def run_preset(
    preset: str,
    m: int,
    n: int,
    k: int,
    output_dir: Path,
    *,
    tile: str = "auto",
    iterations: int = 300,
    xe_fuse_dir: Path | None = None,
    sycl_tla_dir: Path | None = None,
    compiler: str | None = None,
    timeout_s: float = 600.0,
) -> XeFuseResult:
    """Generate, compile and benchmark one Xe-Fuse preset at these shapes.

    Every stage failure is a named error, never an exception: which stage failed
    (generate, compile, run, parse) decides what the operator does next, and
    collapsing them loses that (§12.12's rule about saying which failure it is).

    The returned timing is the binary's own whole-kernel wall clock. Numerical
    correctness is NOT established here — the binary's `--verify` flag is inert
    upstream — so the caller must gate with its own check before believing any
    comparison built on this number.
    """
    result = XeFuseResult(preset=preset, tile=tile, m=m, n=n, k=k)

    fuse = xe_fuse_dir or find_xe_fuse()
    tla = sycl_tla_dir or find_sycl_tla()
    if fuse is None or tla is None:
        missing = [name for name, found in (("Xe-Fuse", fuse), ("sycl-tla", tla)) if found is None]
        result.error = (
            f"checkout(s) not found: {', '.join(missing)}. Set "
            f"{XE_FUSE_DIR_ENV}/{SYCL_TLA_DIR_ENV} or clone under a source root."
        )
        return result

    if compiler is None:
        from xe_forge.orbit.patch.sycl_override import available_compiler

        compiler = available_compiler()
    if compiler is None:
        result.error = "no SYCL compiler (icpx) found; only generation is possible"
        return result

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    stem = f"{preset}_m{m}n{n}k{k}_{tile.replace('x', '_')}"
    cpp = target / f"{stem}.cpp"
    binary = target / stem

    generate = subprocess.run(
        [
            "python3",
            str(fuse / "autotune" / "generate_kernel.py"),
            "--preset",
            preset,
            "--m",
            str(m),
            "--n",
            str(n),
            "--k",
            str(k),
            "--iterations",
            str(iterations),
            "--tile",
            tile,
            "-o",
            str(cpp),
        ],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if generate.returncode != 0 or not cpp.is_file():
        result.error = f"generate failed: {(generate.stderr or generate.stdout).strip()[-300:]}"
        return result
    result.generated_cpp = cpp

    compile_cmd = [
        compiler,
        "-fsycl",
        "-DCUTLASS_ENABLE_SYCL",
        "-DSYCL_INTEL_TARGET",
        f"-I{tla}/include",
        f"-I{tla}/tools/util/include",
        f"-I{tla}/examples/common",
        f"-I{fuse}/include",
        "-O2",
        "-std=c++17",
        "-fsycl-targets=spir64",
        *_SPIRV_EXT_FLAGS,
        "-o",
        str(binary),
        str(cpp),
    ]
    # A compiler resolved from a oneAPI root needs that root's environment too:
    # icpx-by-absolute-path finds the binary but not MKL's headers (CPATH), so the
    # compile would succeed or fail depending on the *caller's* shell — measured
    # live: the same sweep passed from a setvars'd shell and failed from a clean
    # one with `oneapi/mkl/rng/device.hpp not found`. Source setvars ourselves.
    setvars = _setvars_for(compiler)
    if setvars is not None:
        quoted = " ".join(shlex.quote(part) for part in compile_cmd)
        compile_cmd = ["bash", "-c", f"source {shlex.quote(str(setvars))} --force >/dev/null 2>&1 && {quoted}"]
    try:
        compiled = subprocess.run(
            compile_cmd, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired:
        result.error = f"compile timed out after {timeout_s:.0f}s"
        return result
    if compiled.returncode != 0 or not binary.is_file():
        result.error = f"compile failed: {compiled.stderr.strip()[-300:]}"
        return result
    result.binary = binary

    run_cmd = [
        str(binary),
        f"--m={m}",
        f"--n={n}",
        f"--k={k}",
        f"--iterations={iterations}",
    ]
    # The binary needs the same oneAPI runtime the compiler's environment names
    # (libsycl et al. via LD_LIBRARY_PATH) — a clean shell aborts at load/init.
    if setvars is not None:
        quoted = " ".join(shlex.quote(part) for part in run_cmd)
        run_cmd = ["bash", "-c", f"source {shlex.quote(str(setvars))} --force >/dev/null 2>&1 && {quoted}"]
    try:
        ran = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        result.error = f"benchmark timed out after {timeout_s:.0f}s"
        return result
    if ran.returncode != 0:
        result.error = f"benchmark failed (exit {ran.returncode}): {ran.stderr.strip()[-300:]}"
        return result

    ms = _MS_RE.search(ran.stdout)
    tflops = _TFLOPS_RE.search(ran.stdout)
    if ms is None:
        result.error = f"could not parse timing from output: {ran.stdout.strip()[-200:]}"
        return result
    result.ms = float(ms.group(1))
    if tflops is not None:
        result.tflops = float(tflops.group(1))
    result.notes.append(
        "timing only: the binary's --verify is inert upstream, so correctness must "
        "be gated by the caller's differential check before this number is compared"
    )
    return result


def run_region(task, shapes: tuple[int, int, int], output_dir: Path, **kwargs) -> XeFuseResult:
    """Run the preset matching a `FusionTask`'s pattern at (m, n, k)."""
    preset = PRESET_FOR_PATTERN.get(task.pattern)
    if preset is None:
        result = XeFuseResult(preset="?", tile=kwargs.get("tile", "auto"), m=0, n=0, k=0)
        result.error = (
            f"no Xe-Fuse preset maps to pattern {task.pattern!r}; known: "
            f"{sorted(PRESET_FOR_PATTERN)}"
        )
        return result
    m, n, k = shapes
    return run_preset(preset, m, n, k, output_dir, **kwargs)


# ---------------------------------------------------------------------------
# Automated tile autotuning (§11.7): the deterministic sweep before any agent
# ---------------------------------------------------------------------------

# Measured on Wildcat Lake (k2, N=9728, K=896): the generator's auto pick lost at
# every M >= 32, and the winning tile's M dimension tracked the problem M
# (16→16x256x32, 32→32x256x32, 64→64x256x32, 128→128x256x32). Candidate tiles are
# built from that pattern plus near neighbours, with the generator's own "auto" kept
# in the field so an upstream fix shows up as auto winning again.
_TILE_NK_VARIANTS = ((256, 32), (512, 32), (256, 64), (128, 32))
_MAX_TILE_M = 256


def candidate_tiles(m: int) -> list[str]:
    """Tiles worth trying at this problem M, most promising first."""
    pow2 = 1 << max(3, (max(1, m) - 1).bit_length())  # smallest power of two >= m, floor 8
    tile_ms = []
    for tm in (pow2, pow2 // 2, pow2 * 2):
        tm = min(max(8, tm), _MAX_TILE_M)
        if tm not in tile_ms:
            tile_ms.append(tm)
    tiles = ["auto"]
    for tm in tile_ms:
        for tn, tk in _TILE_NK_VARIANTS:
            tiles.append(f"{tm}x{tn}x{tk}")
    return tiles


@dataclass
class XeFuseSweep:
    """A full tile sweep: every result kept, the winner named, failures visible."""

    preset: str
    m: int
    n: int
    k: int
    results: list[XeFuseResult] = field(default_factory=list)

    @property
    def best(self) -> XeFuseResult | None:
        ran = [r for r in self.results if r.ok]
        return min(ran, key=lambda r: r.ms) if ran else None

    def format(self) -> str:
        lines = [
            f"tile sweep: {self.preset} at {self.m}x{self.n}x{self.k} "
            f"({len(self.results)} candidate(s))",
            f"{'TILE':<14} {'us/iter':>10}  NOTE",
            "-" * 48,
        ]
        best = self.best
        for r in sorted(self.results, key=lambda r: (not r.ok, r.ms or 0.0)):
            if r.ok:
                note = "BEST" if r is best else ""
                lines.append(f"{r.tile:<14} {r.per_iteration_us:>10.1f}  {note}")
            else:
                lines.append(f"{r.tile:<14} {'-':>10}  {r.error[:60]}")
        if best is None:
            lines.append("no tile produced a measurement; nothing to choose")
        elif best.tile != "auto":
            auto = next((r for r in self.results if r.tile == "auto" and r.ok), None)
            if auto is not None and auto.ms and best.ms:
                gain = (auto.ms - best.ms) / auto.ms * 100
                lines.append(
                    f"sweep beat the generator's auto pick by {gain:.1f}% — "
                    f"worth reporting upstream (the selector is knowledge, not code)"
                )
        return "\n".join(lines)


def autotune_preset(
    preset: str,
    m: int,
    n: int,
    k: int,
    output_dir: Path,
    *,
    tiles: list[str] | None = None,
    **kwargs,
) -> XeFuseSweep:
    """Sweep tile shapes for one preset and keep everything (§11.7).

    Deterministic and exhaustive over its candidate list — no model in the loop,
    because a tile choice has a measurable answer. A failed tile stays in the table
    with its named stage failure rather than vanishing: silent truncation reads as
    "covered everything" when it did not. The same caveat as `run_preset` applies to
    every number here: timing only, correctness gated by the caller.
    """
    sweep = XeFuseSweep(preset=preset, m=m, n=n, k=k)
    for tile in tiles if tiles is not None else candidate_tiles(m):
        sweep.results.append(run_preset(preset, m, n, k, output_dir, tile=tile, **kwargs))
    return sweep


def autotune_region(task, shapes: tuple[int, int, int], output_dir: Path, **kwargs) -> XeFuseSweep:
    """Autotune the preset matching a region's pattern at (m, n, k)."""
    preset = PRESET_FOR_PATTERN.get(task.pattern)
    if preset is None:
        sweep = XeFuseSweep(preset="?", m=0, n=0, k=0)
        failed = XeFuseResult(preset="?", tile="-", m=0, n=0, k=0)
        failed.error = (
            f"no Xe-Fuse preset maps to pattern {task.pattern!r}; known: "
            f"{sorted(PRESET_FOR_PATTERN)}"
        )
        sweep.results.append(failed)
        return sweep
    m, n, k = shapes
    return autotune_preset(preset, m, n, k, output_dir, **kwargs)
