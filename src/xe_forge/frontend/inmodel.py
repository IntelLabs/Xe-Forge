"""``xe-forge-skill inmodel``: a kernel trial timed inside the model it was captured from.

For a kernel whose repository the run itself built into its stack (a patch whose patch.json
names a ``build`` script), a trial is a directory of that repository's files at their
repository paths. This builds the repository with them in place, runs the run's own launcher
on the build, reads the kernel's time per call from the capture -- the same launcher, the
same ``capture parse --names`` and the same division as the spec's ``captured_us``, so the
baseline is the captured kernel by construction -- and scores the build's output with the
patch's own check. It answers in :mod:`xe_forge.external`'s contract, so ``benchmark``'s gates
(every shape correct, the baseline-is-the-captured-kernel check, regressions per shape, the
noise band) apply unchanged.

Correctness of a trial: it builds and the repository's own tests pass (``build.sh``); every
weighted shape runs as often as in the baseline (the model did the same work); the build's
output passes the patch check's scoring again (``patch score``).

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
from statistics import median

# Captures per arm, in ABBA order. A ratio counts only beyond both arms' ranges summed: with
# five captures an identical pair passes that by chance about once in five thousand shapes,
# and a trial is judged at every weighted shape.
REPEATS = int(os.environ.get("XE_FORGE_MODEL_REPEATS", "5"))


def provider(run_dir: Path, repo: Path) -> dict | None:
    """The run's facts that let a kernel of ``repo`` be timed in the model, or None: the patch
    that built ``repo`` into the stack, with a ``build`` script, its check's arguments and
    record, the launcher and names.json. A run with several rows is not covered."""
    stack = run_dir / "stack.json"
    if not stack.is_file() or (run_dir / "rows" / "capture.json").is_file():
        return None
    for d in reversed(json.loads(stack.read_text()).get("patches") or []):
        work = Path(d)
        meta = (
            json.loads((work / "patch.json").read_text()) if (work / "patch.json").is_file() else {}
        )
        if not meta.get("build") or Path(str(meta.get("repo"))).name != repo.name:
            continue
        need = [run_dir / "launcher" / "run.sh", run_dir / "names.json", work / "check-args.json",
                work / "check.json", work / meta["build"]]  # fmt: skip
        # A launcher that names its interpreter itself runs that build, never a trial's.
        if all(p.is_file() for p in need) and "FRAMEWORK_PYTHON" in need[0].read_text():
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
    library runs the same code for every trial."""
    work, meta = facts["patch"], facts["meta"]
    env = dict(os.environ, SRC=str(src), VENV=str(venv))
    with open(log, "w") as f:
        code = subprocess.call(["bash", str(work / meta["build"])], cwd=work, env=env,
                               stdout=f, stderr=subprocess.STDOUT)  # fmt: skip
    if code != 0:
        return f"the build or the repository's tests failed (exit {code})"
    python = venv / "bin" / "python"
    pkg = meta.get("package", "")
    from xe_forge.claude.generator import render

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
    return None


def _capture(run_dir: Path, out: Path, python: Path) -> Path | None:
    """The run's launcher on this build, timeline only, read with the run's names.json."""
    from xe_forge.frontend.cli import _tool

    raw, cap = out / "raw", out / "capture.json"
    if _tool(sys.executable, "capture", "run", str(run_dir / "launcher" / "run.sh"), "-o",
             str(raw), "--timeline-only", FRAMEWORK_PYTHON=str(python)):  # fmt: skip
        return None
    if _tool(sys.executable, "capture", "parse", str(raw), "-o", str(cap), "--names",
             str(run_dir / "names.json")):  # fmt: skip
        return None
    return cap


def _check_one(module: str, python: Path, kernel: str, shape: dict) -> tuple[int, str]:
    """(exit code, what it said) of the engine's ``kernel_check`` on one shape."""
    argv = [str(python), "-m", module, "--kernel", kernel, "--dims", json.dumps(shape["dims"]),
            "--dtype", str(shape.get("dtype")), "--rtol", str(shape["rtol"]), "--atol",
            str(shape["atol"])]  # fmt: skip
    got = subprocess.run(argv, capture_output=True, text=True)
    return got.returncode, (got.stdout + got.stderr).strip()[-1500:]


def _numeric(facts: dict, pythons: dict, kernel: str, spec: dict, record: dict) -> str | None:
    """The trial's kernel against its PyTorch reference at every recorded variant's shape and
    tolerance (the engine's ``kernel_check``, patch.json): why it fails, or None. A shape the
    baseline's kernel fails too says nothing of the trial: that raises (the check's fault)."""
    module = facts["meta"].get("kernel_check")
    if not module:
        raise RuntimeError(
            'patch.json names no "kernel_check": no kernel can be checked on its own'
        )
    arm = "B" if "B" in pythons else "A"
    for variant in record["variants"]:
        for shape in spec.get(variant) or []:
            code, said = _check_one(module, pythons[arm], kernel, shape)
            if code == 0:
                continue
            if arm == "A" or _check_one(module, pythons["A"], kernel, shape)[0] != 0:
                raise RuntimeError(f"the baseline's kernel fails its own check at {variant} "
                                   f"{shape['dims']}, so the check is not judging it:\n{said}")  # fmt: skip
            return (f"{variant} {shape['dims']}: not within rtol {shape['rtol']}, atol "
                    f"{shape['atol']} of its reference, where the baseline's is:\n{said}")  # fmt: skip
    return None


def measure(run_dir: Path, facts: dict, repo: Path, record: dict, baseline: Path,
            trial: Path, where: Path, repeats: int = REPEATS, spec: dict | None = None) -> dict:  # fmt: skip
    """The result of one (baseline, trial) pair at every variant of ``record``. Raises
    RuntimeError on what is not the trial's (the baseline arm does not build or run)."""
    from xe_forge.frontend import ir, manifest
    from xe_forge.frontend.prove import _score_arm

    same = not _changed(repo, trial)  # the tree it builds, not the files it holds
    (where / "trial.digest").write_text(_digest(trial))  # a profile of it reuses its build
    if same:
        (where / "same").touch()
    arms = {"A": None} if same else {"A": None, "B": trial}
    venvs = {}
    for arm, overlay in arms.items():
        src, venv, log = where / arm / "src", where / arm / "venv", where / f"build-{arm}.log"
        _tree(repo, src, overlay)
        why = _build(facts, src, venv, log, clone=repo)
        if why and arm == "A":
            raise RuntimeError(f"the baseline arm: {why}; see {log}")
        if why:
            return {"correctness": False, "verdict": "BUILD_FAILED",
                    "why": why, "log": log.read_text(errors="replace")[-3000:]}  # fmt: skip
        venvs[arm] = venv / "bin" / "python"
    # The kernel on its own first, against its reference at every shape: cheap, and a wrong
    # kernel is refused before the model is run on it.
    if spec is not None:
        why = _numeric(facts, venvs, record.get("definition", ""), spec, record)
        if why:
            return {"correctness": False, "verdict": "NUMERIC_FAILED", "why": why}
    times: dict[str, dict[str, list[float]]] = {a: {} for a in arms}
    calls: dict[str, dict[str, int]] = {a: {} for a in arms}
    caps: dict[str, list[Path]] = {a: [] for a in arms}
    for r in range(1, repeats + 1):
        for arm in (("A", "B") if r % 2 else ("B", "A")) if not same else ("A",):
            cap_path = _capture(run_dir, where / f"{arm}-r{r}", venvs[arm])
            if cap_path is None:
                if arm == "A":
                    raise RuntimeError(f"the baseline arm did not capture ({where / f'A-r{r}'})")
                return {"correctness": False, "verdict": "MODEL_FAILED",
                        "why": f"the model did not run or capture on the trial's build "
                               f"({where / f'{arm}-r{r}'})"}  # fmt: skip
            caps[arm].append(cap_path)
            cap = ir.load(cap_path)
            for variant, ref in record["variants"].items():
                got = manifest.per_call(cap, ref)
                if got is None:
                    if arm == "A":
                        raise RuntimeError(f"the baseline arm has no workload {ref}")
                    return {"correctness": False, "verdict": "WORK_DIFFERS",
                            "why": f"{variant} ({ref}) did not run on the trial's build"}  # fmt: skip
                times[arm].setdefault(variant, []).append(got[0])
                calls[arm].setdefault(variant, got[1])
    for variant, n in calls.get("B", {}).items():
        if n != calls["A"][variant]:
            return {"correctness": False, "verdict": "WORK_DIFFERS",
                    "why": f"{variant} ran {n} times on the trial's build, {calls['A'][variant]} "
                           "on the baseline's: the model did not do the same work"}  # fmt: skip
    model = None if same else _model(caps, record)
    if model and model["slower"]:
        return {"correctness": False, "verdict": "MODEL_SLOWER", "why": _model_text(model)}
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

    def stats(xs):
        m = median(xs)
        return round(m, 3), round((max(xs) - min(xs)) / m, 4) if m else 0.0

    variants = {}
    for variant in record["variants"]:
        a, sa = stats(times["A"][variant])
        b, sb = stats(times["B" if not same else "A"][variant])
        variants[variant] = {"baseline_us": a, "trial_us": b, "spread": round(sa + sb, 4)}
    return {"correctness": True, "same": same, "variants": variants, "repeats": repeats,
            "model": model}  # fmt: skip


def _beyond(row: dict) -> bool:
    """A compare row whose two arms differ by more than their captures' spreads together."""
    sp = (row.get("spread_a") or 0.0) + (row.get("spread_b") or 0.0)
    return bool(row.get("speedup")) and abs(row["speedup"] - 1) > sp


def _model(caps: dict[str, list[Path]], record: dict) -> dict:
    """The whole model on both builds, from the trial's own captures, as prove compares it:
    its total, and every workload that moved beyond its spread. A kernel can make the model
    slower outside its own time -- what it leaves in the caches, the threads, the clocks --
    which its own time never shows. Slower: the total is, or the time the moved workloads
    gained and lost sums to a loss (the kernel saves less than it costs elsewhere)."""
    from xe_forge.frontend.compare import compare

    try:
        rows = compare(caps["A"], caps["B"])
    except SystemExit as exc:  # the arms' captures ran different work
        raise RuntimeError(str(exc)) from exc
    total, own = rows[0], {ref.partition("@")[0] for ref in record["variants"].values()}
    kernel = record.get("definition") or ""
    moved = [r for r in rows[1:] if r["status"] == "matched" and _beyond(r)]
    net = round(sum(r["delta_us"] for r in moved), 3)
    others = sorted((r for r in moved if r["workload_id"] not in own and r["kernel"] != kernel),
                    key=lambda r: r["delta_us"])  # fmt: skip
    keep = ("workload_id", "kernel", "us_a", "us_b", "speedup")
    return {
        "baseline_us": total["us_a"], "trial_us": total["us_b"], "speedup": total["speedup"],
        "spread": round((total["spread_a"] or 0.0) + (total["spread_b"] or 0.0), 4),
        "net_us": net, "slower": (_beyond(total) and total["speedup"] < 1) or net < 0,
        "others": [{k: r[k] for k in keep} for r in others],
    }  # fmt: skip


def _model_text(model: dict) -> str:
    """The model's total and the other workloads that moved, most time lost first."""
    lines = [f"the whole model: {model['baseline_us']} -> {model['trial_us']} us "
             f"(x{model['speedup']}, spread {model['spread']}); the workloads that moved beyond "
             f"their spread save {model['net_us']} us in all"]  # fmt: skip
    for r in model["others"]:
        lines.append(f"  {r['kernel'] or '-'} {r['workload_id']}: {r['us_a']} -> {r['us_b']} us "
                     f"(x{r['speedup']}), not this kernel's own workload")  # fmt: skip
    return "\n".join(lines)


# The profile's events: where a kernel's cycles go, how many instructions retire in them, and
# what misses. Sampled per symbol inside the model's own steps.
EVENTS = os.environ.get("XE_FORGE_MODEL_EVENTS", "cycles,instructions,cache-misses,branch-misses")
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
            if (venv / "bin" / "python").exists():
                return venv
    return None


def profile(run_dir: Path, facts: dict, repo: Path, trial: Path, where: Path, kernel: str) -> str:
    """The model on this trial's build under ``perf record`` (:data:`EVENTS`), every sample in a
    step of a timed window kept: per symbol, its events and instructions per cycle, the
    kernel's own symbols first; per step type, its time per call (``profile.json`` holds each
    step's symbols too). Raises RuntimeError when the build or the model does not run."""
    import bisect

    from xe_forge.skills.capture import _env, _launch

    venv = _built(where.parent, trial)
    if venv is None:
        src, venv, log = where / "src", where / "venv", where / "build.log"
        _tree(repo, src, trial)
        if why := _build(facts, src, venv, log, clone=repo):
            raise RuntimeError(f"{why}; see {log}")
    raw, data = where / "raw", where / "perf.data"
    raw.mkdir()
    perf = os.environ.get("PERF_BIN", "perf")
    cmd = [perf, "record", "-q", "-e", EVENTS, "-k", "CLOCK_MONOTONIC", "-o", str(data), "--"]
    env = dict(_env(), FRAMEWORK_PYTHON=str(venv / "bin" / "python"))
    code = _launch(run_dir / "launcher" / "run.sh", raw, cmd, where / "launcher.log", env, where)
    if code or not data.is_file() or not (raw / "host_timeline.json").is_file():
        raise RuntimeError(
            f"perf record or the model failed (exit {code}); see {where / 'launcher.log'}"
        )
    timeline = json.loads((raw / "host_timeline.json").read_text())
    timed = [w for w in timeline.get("windows") or [] if w.get("timed")]
    spans = sorted((s, e, name) for w in timed for name, s, e in w.get("nodes") or [])
    starts = [s for s, _, _ in spans]
    steps: dict[str, dict] = {}
    for s, e, name in spans:
        st = steps.setdefault(name, {"us": 0.0, "calls": 0})
        st["us"] += (e - s) / 1e3
        st["calls"] += 1
    counts: dict[str, dict[str, dict[str, float]]] = {}  # step -> symbol -> event -> count
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
            continue  # outside every step: load, warm-up, between steps
        dso = m[7] if m[7].startswith("[") else f"[{Path(m[7]).name}]"
        sym = counts.setdefault(spans[i][2], {}).setdefault(f"{m[6]} {dso}", {})
        ev = _event(m[5])
        sym[ev] = sym.get(ev, 0.0) + int(m[4])
    err = proc.stderr.read() if proc.stderr else ""
    if proc.wait() != 0:
        raise RuntimeError(f"perf script failed: {err.strip()[-300:]}")
    data.unlink()  # large, and the counts are kept
    per_sym: dict[str, dict[str, float]] = {}
    for syms in counts.values():
        for sym, evs in syms.items():
            for ev, n in evs.items():
                per_sym.setdefault(sym, {}).setdefault(ev, 0.0)
                per_sym[sym][ev] += n
    events = [e for e in EVENTS.split(",") if any(e in v for v in per_sym.values())]
    if not events:
        raise RuntimeError(
            f"perf recorded no sample of {EVENTS} inside a step: {err.strip()[-300:]}"
        )
    first = events[0]
    order = sorted(per_sym, key=lambda s: (kernel not in s, -per_sym[s].get(first, 0.0)))[:15]
    total = sum(v.get(first, 0.0) for v in per_sym.values()) or 1.0
    lines = [f"In-model profile of {trial.name} ({', '.join(events)}; samples inside the model's "
             "steps of its timed windows):", f"{'symbol':60s} {first + ' %':>10s} " +
             " ".join(f"{e:>14s}" for e in events) + "        IPC"]  # fmt: skip
    for sym in order:
        v = per_sym[sym]
        ipc = (v.get("instructions", 0) / v["cycles"]) if v.get("cycles") else None
        lines.append(f"{sym[:60]:60s} {100 * v.get(first, 0.0) / total:9.1f}% "
                     + " ".join(f"{v.get(e, 0.0):14.3g}" for e in events)
                     + (f" {ipc:10.2f}" if ipc is not None else ""))  # fmt: skip
    lines.append("Steps (time per call under perf; profile.json holds each step's symbols):")
    for name, st in sorted(steps.items(), key=lambda kv: -kv[1]["us"])[:10]:
        lines.append(f"  {name}: {st['us'] / max(st['calls'], 1):.2f} us x {st['calls']}")
    text = "\n".join(lines) + "\n"
    (where / "profile.txt").write_text(text)
    (where / "profile.json").write_text(json.dumps(
        {"events": events, "symbols": per_sym, "steps": steps, "by_step": counts}, indent=1))  # fmt: skip
    return text


def answer(result: dict, variant: str) -> str:
    """One variant's answer in the external benchmark's contract."""
    timer = (f"TIMER: in-model, the run's launcher on each build, {result.get('repeats')} "
             "captures per arm in ABBA order, the kernel's time per call from the engine's "
             "timeline (median)")  # fmt: skip
    if not result["correctness"]:
        # Free text indented: a log line reading "CORRECTNESS: ..." is never a field here.
        text = "\n".join(f"| {ln}" for ln in f"{result.get('why', '')}\n{result.get('log', '')}"
                         .strip().splitlines())  # fmt: skip
        return f"{text}\nCORRECTNESS: fail\nVERDICT: {result['verdict']}\nDONE\n"
    v = result["variants"].get(variant)
    if v is None:
        return (
            f"CORRECTNESS: fail\nVERDICT: NO_SUCH_VARIANT\n{variant} is not in the record\nDONE\n"
        )
    model = result.get("model")
    told = [f"| {ln}" for ln in _model_text(model).splitlines()] if model else []
    lines = [
        *told,
        "CORRECTNESS: pass",
        f"BASELINE_US: {v['baseline_us']}",
        f"TRIAL_US: {v['trial_us']}",
    ]
    ratio = v["baseline_us"] / v["trial_us"] if v["trial_us"] else None
    if result.get("same"):
        lines.append("VERDICT: SAME_SOURCE")
    elif ratio is None or abs(ratio - 1) <= v["spread"]:
        lines.append("VERDICT: INDISTINGUISHABLE")  # inside the captures' own spread
    else:
        lines += [f"SPEEDUP: {ratio:.4f}", "VERDICT: MEASURED"]
    return "\n".join([*lines, timer, "DONE"]) + "\n"


def _say(verdict: str, text: str, ok: bool = False) -> str:
    """A verdict in the contract, its free text indented so no line of it reads as a field."""
    body = "\n".join(f"| {ln}" for ln in text.strip().splitlines())
    return f"{body}\nCORRECTNESS: {'pass' if ok else 'fail'}\nVERDICT: {verdict}\nDONE\n".lstrip(
        "\n"
    )


# A failure that may not recur on the same code (the model did not run or capture) is not kept.
TRANSIENT = {"MODEL_FAILED"}


def run(args) -> int:
    import fcntl
    import shutil

    run_dir, repo = Path(args.run_dir).resolve(), Path(args.repo).resolve()
    facts = provider(run_dir, repo)
    if facts is None:
        print(f"inmodel: {run_dir} built no {repo.name} into its stack with a build script, or "
              "its launcher does not run under $FRAMEWORK_PYTHON")  # fmt: skip
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
    # validate) waits for the first, rather than perturbing its captures or removing its tree.
    with open(store / "lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.profile:  # the trial's build in the model, under perf
            where = store / f"profile-{_digest(trial)[:16]}"
            shutil.rmtree(where, ignore_errors=True)
            where.mkdir(parents=True)
            try:
                print(profile(run_dir, facts, repo, trial, where, args.kernel or ""), end="")
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
        record = json.loads(Path(args.record).read_text())
        import yaml

        spec = yaml.safe_load(Path(args.spec).read_text()) if args.spec else None
        sources = [_digest(baseline), _digest(trial), head, _digest(Path(args.record)),
                   _digest(Path(args.spec)) if args.spec else "",
                   _digest(run_dir / "names.json"), _digest(run_dir / "launcher" / "run.sh"),
                   _digest(facts["patch"] / facts["meta"]["build"]), str(REPEATS), "model"]  # fmt: skip
        where = store / hashlib.sha256("\0".join(sources).encode()).hexdigest()[:16]
        done = where / "result.json"
        if not done.is_file():
            shutil.rmtree(where, ignore_errors=True)  # a measurement that did not finish
            where.mkdir(parents=True)
            try:
                result = measure(run_dir, facts, repo, record, baseline, trial, where, REPEATS,
                                 spec)  # fmt: skip
            except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as exc:
                print(f"inmodel: {exc}")  # no DONE: not the trial's, and nothing is recorded
                return 1
            if result.get("verdict") not in TRANSIENT:
                done.write_text(json.dumps(result, indent=1) + "\n")
        else:
            result = json.loads(done.read_text())
    print(answer(result, args.variant), end="")
    return 0 if result["correctness"] else 1
