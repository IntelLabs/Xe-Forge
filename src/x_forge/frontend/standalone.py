"""``python -m x_forge.frontend.standalone``: an engine's kernel on its own, at given shapes.

A generated engine's check module (patch.json ``kernel_check``) offers ``check`` and
``prepare`` (the engine contract). Each build's package is imported here under an alias of its
own, so two builds' libraries load side by side in one process, and X-Forge's host timer
times them in alternating rounds on inputs each build's module made the same way.

    check    one build's kernel against its reference at every shape
    time     two builds' kernels at every shape, in alternating rounds (the caller swaps
             which build is first, so the slot an arm is timed in cancels)
    profile  one build's kernel at every shape, each run for its weight's share of the time,
             each shape's calls one span on CLOCK_MONOTONIC (``perf record -k``)
    probe    one build loads under an alias and offers ``check`` and ``prepare``

Every other mode reads ``--shapes``, a JSON list of ``{"variant", "dims", "dtype", "rtol", "atol",
"weight"}``, and writes its result as JSON to ``-o``. Kernels are timed on the run's device
(``X_FORGE_DEVICE``). Timing counts are the executor's defaults unless
``X_FORGE_KERNEL_WARMUP``, ``X_FORGE_KERNEL_ITERS`` or ``X_FORGE_KERNEL_ROUNDS`` set them;
on a GPU, warm-up is never fewer than ``DEVICE_MIN_WARMUP`` calls.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import struct
import sys
import time
from pathlib import Path

# Where a build's package is, written into its venv by ``inmodel._build``.
PACKAGE_FILE = "x-forge-package"
# ELF symbol binding the dynamic loader makes one per process, whichever library defined it.
STB_GNU_UNIQUE = 10
# Warm-up calls a kernel on a GPU gets at the least: its first calls JIT-compile, allocate and
# ramp clocks, and timing begins only after them.
DEVICE_MIN_WARMUP = 100


def unique_symbols(lib: Path) -> list[str]:
    """The dynamic symbols of the ELF shared object ``lib`` bound ``STB_GNU_UNIQUE``. The loader
    binds each to the first library that defined it, ``RTLD_LOCAL`` or not, so a second build
    of the same library would read the first build's copy."""
    data = lib.read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 2:  # ELF64 only: the class a 64-bit host builds
        return []
    end = "<" if data[5] == 1 else ">"
    (shoff,) = struct.unpack_from(end + "Q", data, 0x28)
    shentsize, shnum = struct.unpack_from(end + "HH", data, 0x3A)
    sections = [struct.unpack_from(end + "IIQQQQIIQQ", data, shoff + i * shentsize)
                for i in range(shnum)]  # fmt: skip
    out = []
    for _, kind, _, _, off, size, link, _, _, entsize in sections:
        if kind != 11 or not entsize:  # SHT_DYNSYM
            continue
        strtab = sections[link][4]
        for i in range(size // entsize):
            name, info = struct.unpack_from(end + "IB", data, off + i * entsize)
            if info >> 4 == STB_GNU_UNIQUE:
                out.append(data[strtab + name : data.index(b"\0", strtab + name)].decode())
    return out


def _mapped() -> set[str]:
    """The shared objects this process has mapped."""
    try:
        text = Path("/proc/self/maps").read_text()
    except OSError:
        return set()
    return {f for f in text.split() if f.startswith("/") and ".so" in Path(f).name}


def load(alias: str, pkgdir: Path, package: str, module: str, root: Path | None = None):
    """The check module of the build whose package is ``pkgdir``, imported as ``alias``. The
    package must import itself relatively: one that imports its own name loads whichever build
    that name finds, and two builds would then time one library. No library the import maps
    from the build (under ``root``, its venv, or ``pkgdir``) may bind a symbol once per
    process (:func:`unique_symbols`)."""
    if not module.startswith(package + "."):
        raise RuntimeError(f"the check module {module!r} is not inside the package {package!r}")
    if package in sys.modules:
        raise RuntimeError(f"{package} is already imported: a build cannot load under an alias")
    spec = importlib.util.spec_from_file_location(
        alias, pkgdir / "__init__.py", submodule_search_locations=[str(pkgdir)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"{pkgdir} is not a package")
    before = _mapped()
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[alias] = pkg
    try:
        spec.loader.exec_module(pkg)
        mod = importlib.import_module(alias + module[len(package) :])
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == package or exc.name.startswith(package + ".")):
            raise RuntimeError(f"{package} imports itself by name: two builds cannot load side "
                               "by side; its modules must import one another relatively") from exc  # fmt: skip
        raise
    roots = [Path(r).resolve() for r in (root, pkgdir) if r]
    for lib in sorted(_mapped() - before):
        if not any(Path(lib).resolve().is_relative_to(r) for r in roots):
            continue
        if names := unique_symbols(Path(lib)):
            raise RuntimeError(f"{Path(lib).name} binds {len(names)} symbols once per process "
                               f"(STB_GNU_UNIQUE: {', '.join(names[:3])}): two builds of it in "
                               "one process would share them; build it without them (GCC: "
                               "-fno-gnu-unique)")  # fmt: skip
    if package in sys.modules:
        raise RuntimeError(f"{package} imports itself by name: two builds cannot load side by "
                           "side; its modules must import one another relatively")  # fmt: skip
    missing = [n for n in ("check", "prepare") if not callable(getattr(mod, n, None))]
    if missing:
        raise RuntimeError(f"{module} offers no {' or '.join(missing)}(): the engine predates the "
                           "contract's standalone kernel calls; bring it up to the contract")  # fmt: skip
    return mod


class _Call:
    """A prepared kernel call as the host timer drives it: ``reset`` untimed, ``forward``
    timed, the call itself both."""

    def __init__(self, call):
        self.call = call

    def reset(self) -> None:
        reset = getattr(self.call, "reset", None)
        if reset is not None:
            reset()

    def forward(self):
        return self.call.forward()

    def __call__(self):
        self.reset()
        return self.forward()


def _executor():
    """The host timer on the run's device (``X_FORGE_DEVICE``), at the counts the environment
    sets. Off the host, at least :data:`DEVICE_MIN_WARMUP` warm-up calls."""
    from x_forge.core.executor import KernelBenchExecutor
    from x_forge.frontend.profile import device

    counts = {}
    for env, name in (("X_FORGE_KERNEL_WARMUP", "warmup_iters"),
                      ("X_FORGE_KERNEL_ITERS", "benchmark_iters"),
                      ("X_FORGE_KERNEL_ROUNDS", "rounds")):  # fmt: skip
        if os.environ.get(env):
            counts[name] = int(os.environ[env])
    dev = device()
    warmup = counts.get("warmup_iters", DEVICE_MIN_WARMUP)
    if dev.split(":")[0] != "cpu" and warmup < DEVICE_MIN_WARMUP:
        print(f"standalone: {counts['warmup_iters']} warm-up calls on {dev} raised to "
              f"{DEVICE_MIN_WARMUP}", file=sys.stderr)  # fmt: skip
        counts["warmup_iters"] = DEVICE_MIN_WARMUP
    return KernelBenchExecutor(device=dev, **counts)


def _check(mod, kernel: str, shapes: list[dict], threads: int) -> dict:
    out = []
    for s in shapes:
        try:
            got = mod.check(kernel, s["dims"], s["dtype"], s["rtol"], s["atol"], threads=threads)
            out.append({"variant": s["variant"], "ok": bool(got["ok"]),
                        "max_abs": got.get("max_abs"), "max_rel": got.get("max_rel")})  # fmt: skip
        except (Exception, SystemExit) as exc:  # the kernel's or its inputs': the caller judges
            out.append(
                {"variant": s["variant"], "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            )
    return {"shapes": out}


def _time(mods: list, kernel: str, shapes: list[dict], threads: int) -> dict:
    ex = _executor()
    out = []
    for s in shapes:
        calls = [_Call(m.prepare(kernel, s["dims"], s["dtype"], threads=threads)) for m in mods]
        first, second, ratios = ex.time_interleaved(calls[0], (), calls[1], ())
        out.append({"variant": s["variant"], "first_us": first, "second_us": second,
                    "ratios": ratios})  # fmt: skip
        del calls
    return {"shapes": out, "flush_note": ex.flush_note, "rounds": ex.rounds,
            "warmup_iters": ex.warmup_iters, "benchmark_iters": ex.benchmark_iters}  # fmt: skip


def _profile(mod, kernel: str, shapes: list[dict], threads: int, seconds: float) -> dict:
    """Each shape's calls, restore and forward, run for its weight's share of ``seconds`` (the
    time of one, after a warm one): one span per shape on CLOCK_MONOTONIC, and its calls."""
    total = sum(s.get("weight") or 0.0 for s in shapes) or 1.0
    spans = []
    for s in shapes:
        call = _Call(mod.prepare(kernel, s["dims"], s["dtype"], threads=threads))
        call()
        t = time.monotonic_ns()
        call()
        per = max(time.monotonic_ns() - t, 1)
        n = max(1, round((s.get("weight") or 0.0) / total * seconds * 1e9 / per))
        t = time.monotonic_ns()
        for _ in range(n):
            call()
        spans.append([s["variant"], t, time.monotonic_ns(), n])
        del call
    return {"spans": spans}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m x_forge.frontend.standalone", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("mode", choices=["check", "time", "profile", "probe"])
    p.add_argument("--build", action="append", required=True,
                   help="a build's venv, holding the package's directory in x-forge-package; "
                   "time takes two, first slot first")  # fmt: skip
    p.add_argument("--package", required=True, help="the package's name (patch.json)")
    p.add_argument("--module", required=True, help="the check module (patch.json kernel_check)")
    p.add_argument("--kernel")
    p.add_argument("--shapes", help="JSON file of the shapes")
    p.add_argument("--threads", type=int, default=0, help="the kernel's team (0: OpenMP's)")
    p.add_argument("--seconds", type=float, default=0.0, help="profile: the calls' total time")
    p.add_argument("-o", "--out")
    a = p.parse_args(argv)
    want = 2 if a.mode == "time" else 1
    if len(a.build) != want:
        p.error(f"{a.mode} takes {want} --build")
    if a.mode != "probe" and not (a.kernel and a.shapes and a.out):
        p.error(f"{a.mode} takes --kernel, --shapes and -o")
    try:
        mods = [load(f"_xf_build{i}", Path((Path(v) / PACKAGE_FILE).read_text().strip()),
                     a.package, a.module, Path(v)) for i, v in enumerate(a.build)]  # fmt: skip
    except (RuntimeError, OSError) as exc:
        print(f"standalone: {exc}", file=sys.stderr)
        return 2
    if a.mode == "probe":
        return 0
    shapes = json.loads(Path(a.shapes).read_text())
    if a.mode == "check":
        result = _check(mods[0], a.kernel, shapes, a.threads)
    elif a.mode == "time":
        result = _time(mods, a.kernel, shapes, a.threads)
    else:
        result = _profile(mods[0], a.kernel, shapes, a.threads, a.seconds)
    Path(a.out).write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
