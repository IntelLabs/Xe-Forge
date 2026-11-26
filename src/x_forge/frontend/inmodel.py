"""``x-forge-skill inmodel``: a kernel trial built into the run's engine and timed on its own.

For a kernel whose repository the run itself built into its stack (a patch whose patch.json
names a ``build`` script and a ``kernel_check`` module), a trial is a directory of that
repository's files at their repository paths. This builds the repository with them in place,
checks the trial's kernel against its PyTorch reference at every shape, times it against the
baseline's kernel at every weighted shape -- both builds' kernels in one process, through the
check module's ``prepare``, in alternating rounds by X-Forge's host timer, twice with the
slots swapped (:mod:`x_forge.frontend.standalone`) -- and scores the build's output with the
patch's own check. The model is not run for timing: only ``--prove`` times the whole engine.
It answers in :mod:`x_forge.external`'s contract, so ``benchmark``'s gates (every shape
correct, the baseline-is-the-captured-kernel check, regressions per shape, the noise band)
apply unchanged.

Correctness of a trial: it builds and the repository's own tests pass (``build.sh``); its
kernel is within tolerance of its reference at every shape the spec holds; the build's output
passes the patch check's scoring again (``patch score``).

One measurement serves every variant of a (baseline, trial) pair: ``benchmark`` asks once per
weighted shape, and the answer is read from ``<workspace>/.inmodel/<key>/result.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from x_forge.frontend.standalone import PACKAGE_FILE  # where a build's package is


def provider(run_dir: Path, repo: Path) -> dict | None:
    """The run's facts that let a kernel of ``repo`` be built and checked: the patch that built
    ``repo`` into the stack, with a ``build`` script, a ``kernel_check`` module, and its check's
    arguments and record. None otherwise."""
    stack = run_dir / "stack.json"
    if not stack.is_file():
        return None
    for d in reversed(json.loads(stack.read_text()).get("patches") or []):
        work = Path(d)
        meta = (
            json.loads((work / "patch.json").read_text()) if (work / "patch.json").is_file() else {}
        )
        if not meta.get("build") or Path(str(meta.get("repo"))).name != repo.name:
            continue
        need = [work / "check-args.json", work / "check.json", work / meta["build"]]
        if all(p.is_file() for p in need) and meta.get("kernel_check") and meta.get("package"):
            return {"patch": work, "meta": meta}
    return None


def _digest(path: Path) -> str:
    """Content digest of a file or a directory's files (paths and bytes)."""
    h = hashlib.sha256()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for f in files:
        h.update(str(f.relative_to(path) if path.is_dir() else f.name).encode() + b"\0")
        h.update(f.read_bytes())
    return h.hexdigest()


def _overlay_files(trial: Path) -> list[tuple[str, Path]]:
    """(repository path, file) of every file a trial directory holds; refuses what could land
    outside the tree."""
    if not trial.is_dir():
        raise ValueError(f"{trial} is not a directory of repository files")
    out = []
    for f in sorted(trial.rglob("*")):
        rel = f.relative_to(trial)
        if f.is_symlink():
            raise ValueError(f"{rel}: a symbolic link, not a file")
        if f.is_file():
            if ".." in rel.parts:
                raise ValueError(f"{rel}: outside the repository")
            out.append((rel.as_posix(), f))
    return out


def _git(repo: Path, *args: str, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=text)


def _changed(repo: Path, trial: Path) -> list[str]:
    """The repository paths whose content the trial changes from HEAD (new files included)."""
    out = []
    for rel, f in _overlay_files(trial):
        got = _git(repo, "show", f"HEAD:{rel}", text=False)
        if got.returncode != 0 or got.stdout != f.read_bytes():
            out.append(rel)
    return out


def _baseline_problem(repo: Path, baseline: Path, head: str) -> str | None:
    """Why the baseline directory is not the repository at the checked commit, or None."""
    now = _git(repo, "rev-parse", "HEAD").stdout.strip()
    if now != head:
        return (f"REPOSITORY_MOVED: {repo} is at {now[:12]}, not {head[:12]}, the commit the "
                "patch check judged")  # fmt: skip
    files = _overlay_files(baseline)
    if not files:
        return f"BASELINE_NOT_REPOSITORY: {baseline} holds no file"
    for rel, f in files:
        got = _git(repo, "show", f"HEAD:{rel}", text=False)
        if got.returncode != 0:
            return f"BASELINE_NOT_REPOSITORY: {rel} is not a file of {repo} at HEAD"
        if got.stdout != f.read_bytes():
            return f"BASELINE_NOT_REPOSITORY: {rel} differs from {repo} at HEAD"
    return None


def _tree(repo: Path, dest: Path, overlay: Path | None) -> None:
    """``repo`` at HEAD in ``dest``, with ``overlay``'s files at their paths over it."""
    dest.mkdir(parents=True)
    with tempfile.TemporaryFile() as tar:
        subprocess.run(["git", "-C", str(repo), "archive", "HEAD"], stdout=tar, check=True)
        tar.seek(0)
        with tarfile.open(fileobj=tar) as t:
            t.extractall(dest, filter="data")
    for rel, f in _overlay_files(overlay) if overlay else []:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f.read_bytes())


def _build(facts: dict, src: Path, venv: Path, log: Path, clone: Path | None = None) -> str | None:
    """Build ``src`` into ``venv`` with the patch's build script; why it failed, or None. The
    package must then import from this build, and no library it loads may come from
    ``clone`` (the repository the trees are made from): a build that links the clone's
    library runs the same code for every trial. The package's directory is written to
    ``venv/PACKAGE_FILE``."""
    work, meta = facts["patch"], facts["meta"]
    env = dict(os.environ, SRC=str(src), VENV=str(venv))
    with open(log, "w") as f:
        code = subprocess.call(["bash", str(work / meta["build"])], cwd=work, env=env,
                               stdout=f, stderr=subprocess.STDOUT)  # fmt: skip
    if code != 0:
        return f"the build or the repository's tests failed (exit {code})"
    python = venv / "bin" / "python"
    pkg = meta.get("package", "")
    from x_forge.claude.render import render

    where = subprocess.run([str(python), "-c", render("inmodel-where.py.j2", pkg=pkg)], capture_output=True,
                           text=True)  # fmt: skip
    if where.returncode != 0:
        return f"{python} cannot import {pkg!r}: {where.stderr.strip()[-300:]}"
    lines = where.stdout.strip().splitlines()
    got = Path(lines[-1]).resolve() if lines else Path("/")
    if not (got.is_relative_to(venv.resolve()) or got.is_relative_to(src.resolve())):
        return f"{pkg} imports from {got}, not from this build"
    if clone is not None and len(lines) >= 2:
        import ast

        theirs = [
            p
            for p in ast.literal_eval(lines[-2])
            if Path(p).resolve().is_relative_to(clone.resolve())
        ]
        if theirs:
            return f"{pkg} loads {theirs[0]} from {clone}, not from this build"
    (venv / PACKAGE_FILE).write_text(str(got.parent))
    return None


def _shapes(spec: dict, record: dict) -> list[dict]:
    """Every recorded variant's shapes in the spec, each with its tolerance and its weight
    (0 for a variant the spec does not weigh)."""
    weights = spec.get("weights") or {}
    return [{"variant": v, "dims": s["dims"], "dtype": str(s.get("dtype")), "rtol": s["rtol"],
             "atol": s["atol"], "weight": weights.get(v, 0.0)}
            for v in record["variants"] for s in spec.get(v) or []]  # fmt: skip


def _standalone(facts: dict, mode: str, builds: list[Path], kernel: str, shapes: list[dict],
                where: Path, *wrap: str, **extra: str) -> tuple[dict | None, str]:  # fmt: skip
    """(result, what it said) of :mod:`x_forge.frontend.standalone` in its own process under
    this interpreter, with ``builds``' venvs, its kernel on the team the run's model ran on
    (``facts["threads"]``); the result is None when it failed or outlived the external
    command's timeout."""
    from x_forge.config import get_config

    meta = facts["meta"]
    extra = {"threads": str(facts.get("threads") or 0), **extra}
    where.mkdir(parents=True, exist_ok=True)
    (where / "shapes.json").write_text(json.dumps(shapes))
    argv = [*wrap, sys.executable, "-m", "x_forge.frontend.standalone", mode,
            *[x for b in builds for x in ("--build", str(b))], "--package", str(meta["package"]),
            "--module", str(meta["kernel_check"]), "--kernel", kernel,
            "--shapes", str(where / "shapes.json"), "-o", str(where / "result.json"),
            *[x for k, v in extra.items() for x in (f"--{k}", v)]]  # fmt: skip
    try:
        got = subprocess.run(argv, capture_output=True, text=True,
                             timeout=get_config().external.timeout)  # fmt: skip
    except subprocess.TimeoutExpired as exc:
        said = f"{mode} did not finish in {exc.timeout} s (external.timeout)"
        (where / "standalone.log").write_text(said + "\n")
        return None, said
    said = (got.stdout + got.stderr).strip()[-3000:]
    (where / "standalone.log").write_text(said + "\n")
    if got.returncode != 0 or not (where / "result.json").is_file():
        return None, said
    return json.loads((where / "result.json").read_text()), said


def _numeric(facts: dict, pkgs: dict, kernel: str, shapes: list[dict], where: Path) -> str | None:
    """The trial's kernel against its PyTorch reference at every shape and tolerance, in one
    process: why it fails, or None. A shape the baseline's kernel fails too says nothing of
    the trial: that raises (the check's fault)."""
    arm = "B" if "B" in pkgs else "A"
    got, said = _standalone(facts, "check", [pkgs[arm]], kernel, shapes, where / f"check-{arm}")
    if got is None:  # the process died: whose kernel killed it, the baseline's run says
        base, base_said = _standalone(
            facts, "check", [pkgs["A"]], kernel, shapes, where / "check-A"
        )
        if arm == "A" or base is None:
            raise RuntimeError(f"the baseline's kernel check did not finish:\n{base_said}")
        if not all(s["ok"] for s in base["shapes"]):
            raise RuntimeError("the baseline's kernel fails its own check, so the check is not "
                               "judging it")  # fmt: skip
        return f"the kernel check died on the trial's build, where the baseline's passes:\n{said}"
    failed = [(s, r) for s, r in zip(shapes, got["shapes"], strict=True) if not r["ok"]]
    if not failed:
        return None
    if arm == "B":
        base, base_said = _standalone(facts, "check", [pkgs["A"]], kernel,
                                      [s for s, _ in failed], where / "check-A")  # fmt: skip
        if base is None:
            raise RuntimeError(f"the baseline's kernel check did not finish:\n{base_said}")
        bad = [s for (s, _), b in zip(failed, base["shapes"], strict=True) if not b["ok"]]
    else:
        bad = [s for s, _ in failed]
    if bad:
        raise RuntimeError(f"the baseline's kernel fails its own check at {bad[0]['variant']} "
                           f"{bad[0]['dims']}, so the check is not judging it")  # fmt: skip
    s, r = failed[0]
    return (f"{s['variant']} {s['dims']}: not within rtol {s['rtol']}, atol {s['atol']} of its "
            f"reference, where the baseline's is: {json.dumps(r)}"
            + (f" ({len(failed)} shapes fail)" if len(failed) > 1 else ""))  # fmt: skip


def _timed(facts: dict, pkgs: dict, kernel: str, shapes: list[dict], where: Path) -> dict | str:
    """Per variant ``(baseline_us, trial_us, ratios)``, both arms timed in one process at each
    weighted shape, twice with the slots swapped; or what the timer said when it failed on the
    trial. Raises RuntimeError when it fails on the baseline alone too (the check module's or
    the baseline's fault, not the trial's)."""
    from x_forge.core.executor import combine_swapped

    b = pkgs.get("B", pkgs["A"])
    ab, said = _standalone(facts, "time", [pkgs["A"], b], kernel, shapes, where / "time-AB")
    ba = None
    if ab is not None:
        ba, said = _standalone(facts, "time", [b, pkgs["A"]], kernel, shapes, where / "time-BA")
    if ab is None or ba is None:
        if "B" in pkgs:  # whose failure: the baseline against itself says
            aa, base_said = _standalone(facts, "time", [pkgs["A"], pkgs["A"]], kernel, shapes,
                                        where / "time-AA")  # fmt: skip
            if aa is not None:
                return said
            said = base_said
        raise RuntimeError(f"timing the baseline's kernel on its own failed:\n{said}")
    out = {}
    for x, y in zip(ab["shapes"], ba["shapes"], strict=True):
        a_us, b_us, ratios = combine_swapped((x["first_us"], x["second_us"], x["ratios"]),
                                             (y["first_us"], y["second_us"], y["ratios"]))  # fmt: skip
        out[x["variant"]] = {"baseline_us": round(a_us, 3), "trial_us": round(b_us, 3),
                             "ratios": [round(r, 4) for r in ratios]}  # fmt: skip
    return {"variants": out, "flush_note": ab.get("flush_note", ""), "rounds": ab.get("rounds"),
            "iters": ab.get("benchmark_iters"), "warmup": ab.get("warmup_iters")}  # fmt: skip


def measure(run_dir: Path, facts: dict, repo: Path, record: dict, baseline: Path,
            trial: Path, where: Path, spec: dict) -> dict:  # fmt: skip
    """The result of one (baseline, trial) pair at every variant of ``record``. Raises
    RuntimeError on what is not the trial's (the baseline arm does not build or check)."""
    from x_forge.frontend.prove import _score_arm

    same = not _changed(repo, trial)  # the tree it builds, not the files it holds
    (where / "trial.digest").write_text(_digest(trial))  # a profile of it reuses its build
    if same:
        (where / "same").touch()
    arms = {"A": None} if same else {"A": None, "B": trial}
    venvs, pkgs = {}, {}
    for arm, overlay in arms.items():
        src, venv, log = where / arm / "src", where / arm / "venv", where / f"build-{arm}.log"
        _tree(repo, src, overlay)
        why = _build(facts, src, venv, log, clone=repo)
        if why and arm == "A":
            raise RuntimeError(f"the baseline arm: {why}; see {log}")
        if why:
            return {"correctness": False, "verdict": "BUILD_FAILED",
                    "why": why, "log": log.read_text(errors="replace")[-3000:]}  # fmt: skip
        venvs[arm], pkgs[arm] = venv / "bin" / "python", venv
    kernel = record.get("definition", "")
    shapes = _shapes(spec, record)
    # The kernel against its reference first: a wrong kernel is never timed.
    why = _numeric(facts, pkgs, kernel, shapes, where)
    if why:
        return {"correctness": False, "verdict": "NUMERIC_FAILED", "why": why}
    # The tokens only: the step report belongs to the patch check and prove, not to a trial.
    argv = [
        a for a in json.loads((facts["patch"] / "check-args.json").read_text()) if a != "--steps"
    ]
    scored = _score_arm(facts["patch"], argv, str(venvs["B" if not same else "A"]),
                        where / "output.json")  # fmt: skip
    if scored is None or scored.get("unscored") or scored.get("exit") not in (0, 1):
        raise RuntimeError(f"the output could not be scored ({where / 'output.json'})")
    if not scored["pass"]:
        if same:
            raise RuntimeError(f"the baseline's own output fails its check: {scored['problems']}")
        return {
            "correctness": False,
            "verdict": "OUTPUT_FAILED",
            "why": "; ".join(scored["problems"]) or "the model's output fails its check",
        }
    # Timed at the shapes the benchmark asks for: the weighted ones, one shape per variant.
    weighted = spec.get("weights") or {}
    first, timed = set(), []
    for s in shapes:
        if (not weighted or s["variant"] in weighted) and s["variant"] not in first:
            first.add(s["variant"])
            timed.append(s)
    got = _timed(facts, pkgs, kernel, timed, where)
    if isinstance(got, str):
        return {"correctness": False, "verdict": "TIMING_FAILED",
                "why": f"timing the kernels on their own failed:\n{got}"}  # fmt: skip
    return {"correctness": True, "same": same, **got}


# The profile's events: where a kernel's cycles go, how many instructions retire in them, and
# what misses. Sampled per symbol inside the kernel's own calls.
EVENTS = os.environ.get("X_FORGE_MODEL_EVENTS", "cycles,instructions,cache-misses,branch-misses")
# How long the profiled calls run in all, each shape for its weight's share.
PROFILE_SECONDS = os.environ.get("X_FORGE_KERNEL_PROFILE_SECONDS", "10")
_SAMPLE = re.compile(
    r"^\s*(\d+)\s+(\d+)\.(\d{9}):\s+(\d+)\s+(\S+?):\s+[0-9a-f]+\s+(.*)\s+\(([^()]*)\)\s*$"
)


def _event(name: str) -> str:
    """perf's printed event name as asked for: ``cycles`` of ``cycles:u`` or ``cpu_core/cycles/``."""
    if "/" in name:
        name = name.split("/")[1] or name
    return name.split(":")[0]


def _built(store: Path, trial: Path) -> Path | None:
    """A venv a measurement already built for this trial's content, or None."""
    want = _digest(trial)
    for mark in store.glob("*/trial.digest"):
        if mark.read_text() == want:
            arm = "A" if (mark.parent / "same").exists() else "B"
            venv = mark.parent / arm / "venv"
            if (venv / PACKAGE_FILE).is_file():
                return venv
    return None


def _sample(facts: dict, pkg: Path, kernel: str, shapes: list[dict],
            where: Path) -> tuple[dict, dict, str]:  # fmt: skip
    """The kernel's calls at ``shapes`` under ``perf record``: per shape its time and calls
    (restore and forward), per shape, symbol and event the samples' count inside its calls,
    and what perf said."""
    import bisect

    perf = os.environ.get("PERF_BIN", "perf")
    data = where / "perf.data"
    cmd = [perf, "record", "-q", "-e", EVENTS, "-k", "CLOCK_MONOTONIC", "-o", str(data), "--"]
    got, said = _standalone(facts, "profile", [pkg], kernel, shapes, where, *cmd,
                            seconds=PROFILE_SECONDS)  # fmt: skip
    if got is None or not data.is_file():
        raise RuntimeError(f"perf record or the kernel failed: {said[-600:]}")
    spans = sorted((s, e, name) for name, s, e, _ in got["spans"])
    starts = [s for s, _, _ in spans]
    steps = {name: {"us": (e - s) / 1e3, "calls": n} for name, s, e, n in got["spans"]}
    counts: dict[str, dict[str, dict[str, float]]] = {}  # shape -> symbol -> event -> count
    proc = subprocess.Popen([perf, "script", "-i", str(data), "-F",
                             "tid,time,event,period,ip,sym,dso", "--ns"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  # fmt: skip
    assert proc.stdout is not None
    for line in proc.stdout:  # streamed: every thread's samples can exceed memory
        m = _SAMPLE.match(line)
        if not m:
            continue
        t = int(m[2]) * 10**9 + int(m[3])
        i = bisect.bisect_right(starts, t) - 1
        if i < 0 or t > spans[i][1]:
            continue  # outside every call: imports, inputs, warm-up
        dso = m[7] if m[7].startswith("[") else f"[{Path(m[7]).name}]"
        sym = counts.setdefault(spans[i][2], {}).setdefault(f"{m[6]} {dso}", {})
        ev = _event(m[5])
        sym[ev] = sym.get(ev, 0.0) + int(m[4])
    err = proc.stderr.read() if proc.stderr else ""
    if proc.wait() != 0:
        raise RuntimeError(f"perf script failed: {err.strip()[-300:]}")
    data.unlink()  # large, and the counts are kept
    return steps, counts, err


def profile(facts: dict, repo: Path, trial: Path, where: Path, kernel: str, spec: dict,
            record: dict) -> str:  # fmt: skip
    """This trial's kernel on its own under ``perf record`` (:data:`EVENTS`), at every weighted
    shape for its weight's share of the time, every sample inside a call kept: per symbol its
    events and instructions per cycle, the kernel's own symbols first; per shape its time per
    call (``profile.json`` holds each shape's symbols too). Raises RuntimeError when the build
    or the kernel does not run."""
    venv = _built(where.parent, trial)
    if venv is None:
        src, venv, log = where / "src", where / "venv", where / "build.log"
        _tree(repo, src, trial)
        if why := _build(facts, src, venv, log, clone=repo):
            raise RuntimeError(f"{why}; see {log}")
    shapes = [s for s in _shapes(spec, record) if s["weight"] > 0] or _shapes(spec, record)
    steps, counts, err = _sample(facts, venv, kernel, shapes, where)
    per_sym: dict[str, dict[str, float]] = {}
    for syms in counts.values():
        for sym, evs in syms.items():
            for ev, n in evs.items():
                per_sym.setdefault(sym, {}).setdefault(ev, 0.0)
                per_sym[sym][ev] += n
    events = [e for e in EVENTS.split(",") if any(e in v for v in per_sym.values())]
    if not events:
        raise RuntimeError(
            f"perf recorded no sample of {EVENTS} inside a call: {err.strip()[-300:]}"
        )
    first = events[0]
    order = sorted(per_sym, key=lambda s: (kernel not in s, -per_sym[s].get(first, 0.0)))[:15]
    total = sum(v.get(first, 0.0) for v in per_sym.values()) or 1.0
    lines = [f"Profile of {trial.name}'s {kernel} on its own ({', '.join(events)}; samples inside "
             f"its calls at the spec's {len(shapes)} weighted shapes, each run for its weight's "
             "share of the time):",
             f"{'symbol':60s} {first + ' %':>10s} " +
             " ".join(f"{e:>14s}" for e in events) + "        IPC"]  # fmt: skip
    for sym in order:
        v = per_sym[sym]
        ipc = (v.get("instructions", 0) / v["cycles"]) if v.get("cycles") else None
        lines.append(f"{sym[:60]:60s} {100 * v.get(first, 0.0) / total:9.1f}% "
                     + " ".join(f"{v.get(e, 0.0):14.3g}" for e in events)
                     + (f" {ipc:10.2f}" if ipc is not None else ""))  # fmt: skip
    lines.append("Shapes (time per call under perf; profile.json holds each shape's symbols):")
    for name, st in sorted(steps.items(), key=lambda kv: -kv[1]["us"])[:10]:
        lines.append(f"  {name}: {st['us'] / st['calls']:.2f} us x {st['calls']}")
    text = "\n".join(lines) + "\n"
    (where / "profile.txt").write_text(text)
    (where / "profile.json").write_text(json.dumps(
        {"events": events, "symbols": per_sym, "shapes": steps, "by_shape": counts}, indent=1))  # fmt: skip
    return text


def answer(result: dict, variant: str) -> str:
    """One variant's answer in the external benchmark's contract."""
    if not result["correctness"]:
        # Free text indented: a log line reading "CORRECTNESS: ..." is never a field here.
        text = "\n".join(f"| {ln}" for ln in f"{result.get('why', '')}\n{result.get('log', '')}"
                         .strip().splitlines())  # fmt: skip
        return f"{text}\nCORRECTNESS: fail\nVERDICT: {result['verdict']}\nDONE\n"
    v = result["variants"].get(variant)
    if v is None:
        return (f"CORRECTNESS: fail\nVERDICT: NO_SUCH_VARIANT\n{variant} is not a timed variant "
                "of the record\nDONE\n")  # fmt: skip
    from x_forge.config import noise_band

    ratios, band = v["ratios"], noise_band()
    speedup = sorted(ratios)[len(ratios) // 2]  # as the executor reads its rounds
    timer = (f"TIMER: the kernel on its own, the baseline's and the trial's builds in one process, "
             f"{result.get('rounds')} alternating rounds of {result.get('iters')} calls after "
             f"{result.get('warmup')} warm-up calls each, twice with the slots swapped; per-round "
             f"speedup {min(ratios):.3f}-{max(ratios):.3f}x.{result.get('flush_note', '')}")  # fmt: skip
    lines = ["CORRECTNESS: pass", f"BASELINE_US: {v['baseline_us']}", f"TRIAL_US: {v['trial_us']}"]
    agree = min(ratios) > 1 or max(ratios) < 1
    if result.get("same"):
        lines.append("VERDICT: SAME_SOURCE")
    elif not agree or abs(speedup - 1.0) < band:
        lines.append("VERDICT: INDISTINGUISHABLE")  # the rounds straddle 1, or inside the band
    else:
        lines += [f"SPEEDUP: {speedup:.4f}", "VERDICT: MEASURED"]
    return "\n".join([*lines, timer, "DONE"]) + "\n"


def _say(verdict: str, text: str, ok: bool = False) -> str:
    """A verdict in the contract, its free text indented so no line of it reads as a field."""
    body = "\n".join(f"| {ln}" for ln in text.strip().splitlines())
    return f"{body}\nCORRECTNESS: {'pass' if ok else 'fail'}\nVERDICT: {verdict}\nDONE\n".lstrip(
        "\n"
    )


def _threads(run_dir: Path) -> int:
    """The thread count the run's model ran on (its capture's ``runtime.threads``), or 0 for
    OpenMP's default: a kernel is timed on the team the model gives it."""
    try:
        rt = json.loads((run_dir / "capture.json").read_text())["run"].get("runtime") or {}
    except (OSError, ValueError, KeyError):
        return 0
    return int(rt.get("threads") or 0)


# A failure that may not recur on the same code (the timer's process died) is not kept.
TRANSIENT = {"TIMING_FAILED"}
# What the timer's counts are: a measurement under other counts is another measurement.
COUNTS = ("X_FORGE_KERNEL_WARMUP", "X_FORGE_KERNEL_ITERS", "X_FORGE_KERNEL_ROUNDS")


def run(args) -> int:
    import fcntl
    import shutil

    import yaml

    run_dir, repo = Path(args.run_dir).resolve(), Path(args.repo).resolve()
    facts = provider(run_dir, repo)
    if facts is None:
        print(f"inmodel: {run_dir} built no {repo.name} into its stack with a build script and "
              "a kernel_check module")  # fmt: skip
        return 1
    trial = Path(args.trial).resolve()
    try:
        _overlay_files(trial)
    except ValueError as exc:
        print(_say("BAD_TRIAL", str(exc)), end="")
        return 1
    # A trial is judged by the repository's check and references, never by its own edits to them.
    module = str(facts["meta"].get("kernel_check") or "")
    own = [p for p in _changed(repo, trial) if p.startswith("reference/")
           or (module and p.endswith(module.split(".")[-1] + ".py"))]  # fmt: skip
    if own:
        print(_say("BAD_TRIAL", f"the trial changes what checks it ({', '.join(own)}): a trial "
                   "changes kernels, never their references or their check"), end="")  # fmt: skip
        return 1
    head = json.loads((facts["patch"] / "check.json").read_text()).get("head", "")
    store = Path(args.workspace).resolve() / ".inmodel"
    store.mkdir(parents=True, exist_ok=True)
    # One build or measurement at a time per workspace: a second caller (a retry, a parallel
    # validate) waits for the first, rather than perturbing its timing or removing its tree.
    with open(store / "lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        facts["threads"] = _threads(run_dir)
        if args.profile:  # the trial's kernel on its own, under perf
            if not (args.spec and args.record):
                print("inmodel: a profile runs the kernel at the spec's shapes: pass --spec and "
                      "--record")  # fmt: skip
                return 1
            where = store / f"profile-{_digest(trial)[:16]}"
            shutil.rmtree(where, ignore_errors=True)
            where.mkdir(parents=True)
            try:
                spec = yaml.safe_load(Path(args.spec).read_text())
                record = json.loads(Path(args.record).read_text())
                print(profile(facts, repo, trial, where, args.kernel or "", spec, record), end="")
            except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as exc:
                print(f"inmodel: profile failed: {exc}")
                return 1
            print(f"Artifacts: {where}")
            return 0
        if args.build_only:  # validate: the trial builds and the repository's tests pass
            with tempfile.TemporaryDirectory() as tmp:
                _tree(repo, Path(tmp) / "src", trial)
                log = Path(tmp) / "build.log"
                why = _build(facts, Path(tmp) / "src", Path(tmp) / "venv", log, clone=repo)
                if why:
                    print(_say("BUILD_FAILED", f"{why}\n{log.read_text(errors='replace')[-3000:]}"),
                          end="")  # fmt: skip
                    return 1
            print(_say("BUILT", "", ok=True), end="")
            return 0
        baseline = Path(args.baseline).resolve()
        if why := _baseline_problem(repo, baseline, head):
            verdict, _, text = why.partition(": ")
            print(_say(verdict, text), end="")
            return 1
        if not args.spec:
            print(_say("NO_SPEC", "a trial is checked and timed at the spec's shapes: pass --spec"),
                  end="")  # fmt: skip
            return 1
        record = json.loads(Path(args.record).read_text())
        spec = yaml.safe_load(Path(args.spec).read_text())
        sources = [_digest(baseline), _digest(trial), head, _digest(Path(args.record)),
                   _digest(Path(args.spec)), _digest(facts["patch"] / facts["meta"]["build"]),
                   _digest(facts["patch"] / "check-args.json"), str(facts["meta"]["package"]),
                   str(facts["meta"]["kernel_check"]), str(facts["threads"]),
                   os.environ.get("OMP_NUM_THREADS", ""), os.environ.get("X_FORGE_DEVICE", ""),
                   *(os.environ.get(k, "") for k in COUNTS), "standalone"]  # fmt: skip
        where = store / hashlib.sha256("\0".join(sources).encode()).hexdigest()[:16]
        done = where / "result.json"
        if not done.is_file():
            shutil.rmtree(where, ignore_errors=True)  # a measurement that did not finish
            where.mkdir(parents=True)
            try:
                result = measure(run_dir, facts, repo, record, baseline, trial, where, spec)
            except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as exc:
                print(f"inmodel: {exc}")  # no DONE: not the trial's, and nothing is recorded
                return 1
            if result.get("verdict") not in TRANSIENT:
                done.write_text(json.dumps(result, indent=1) + "\n")
        else:
            result = json.loads(done.read_text())
    print(answer(result, args.variant), end="")
    return 0 if result["correctness"] and args.variant in result["variants"] else 1
