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


_WHERE = """import sys
import {pkg}
maps = open("/proc/self/maps").read().split()
print(sorted({{p for p in maps if p.startswith("/") and ".so" in p}}))
print({pkg}.__file__)"""


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
    where = subprocess.run([str(python), "-c", _WHERE.format(pkg=pkg)], capture_output=True,
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


def measure(run_dir: Path, facts: dict, repo: Path, record: dict, baseline: Path,
            trial: Path, where: Path, repeats: int = REPEATS) -> dict:  # fmt: skip
    """The result of one (baseline, trial) pair at every variant of ``record``. Raises
    RuntimeError on what is not the trial's (the baseline arm does not build or run)."""
    from xe_forge.frontend import ir, manifest
    from xe_forge.frontend.prove import _score_arm

    same = not _changed(repo, trial)  # the tree it builds, not the files it holds
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
    times: dict[str, dict[str, list[float]]] = {a: {} for a in arms}
    calls: dict[str, dict[str, int]] = {a: {} for a in arms}
    for r in range(1, repeats + 1):
        for arm in (("A", "B") if r % 2 else ("B", "A")) if not same else ("A",):
            cap_path = _capture(run_dir, where / f"{arm}-r{r}", venvs[arm])
            if cap_path is None:
                if arm == "A":
                    raise RuntimeError(f"the baseline arm did not capture ({where / f'A-r{r}'})")
                return {"correctness": False, "verdict": "MODEL_FAILED",
                        "why": f"the model did not run or capture on the trial's build "
                               f"({where / f'{arm}-r{r}'})"}  # fmt: skip
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
    scored = _score_arm(facts["patch"], json.loads((facts["patch"] / "check-args.json").read_text()),
                        str(venvs["B" if not same else "A"]), where / "output.json")  # fmt: skip
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
    return {"correctness": True, "same": same, "variants": variants, "repeats": repeats}


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
    lines = ["CORRECTNESS: pass", f"BASELINE_US: {v['baseline_us']}", f"TRIAL_US: {v['trial_us']}"]
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
    head = json.loads((facts["patch"] / "check.json").read_text()).get("head", "")
    store = Path(args.workspace).resolve() / ".inmodel"
    store.mkdir(parents=True, exist_ok=True)
    # One build or measurement at a time per workspace: a second caller (a retry, a parallel
    # validate) waits for the first, rather than perturbing its captures or removing its tree.
    with open(store / "lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
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
        sources = [_digest(baseline), _digest(trial), head, _digest(Path(args.record)),
                   _digest(run_dir / "names.json"), _digest(run_dir / "launcher" / "run.sh"),
                   _digest(facts["patch"] / facts["meta"]["build"]), str(REPEATS)]  # fmt: skip
        where = store / hashlib.sha256("\0".join(sources).encode()).hexdigest()[:16]
        done = where / "result.json"
        if not done.is_file():
            shutil.rmtree(where, ignore_errors=True)  # a measurement that did not finish
            where.mkdir(parents=True)
            try:
                result = measure(run_dir, facts, repo, record, baseline, trial, where, REPEATS)
            except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as exc:
                print(f"inmodel: {exc}")  # no DONE: not the trial's, and nothing is recorded
                return 1
            if result.get("verdict") not in TRANSIENT:
                done.write_text(json.dumps(result, indent=1) + "\n")
        else:
            result = json.loads(done.read_text())
    print(answer(result, args.variant), end="")
    return 0 if result["correctness"] else 1
