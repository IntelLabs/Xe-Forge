"""x-forge-skill upstream: carry a finalized winner back to the repository it came from.

A kernel imported from a repository is optimized as a self-contained copy; the result
is only useful once it is a patch against the original files. The edits are the
session's to make. Everything around them is mechanical, and done here so that every
engine does it the same way and none needs a shell:

    upstream clone <repo> [--name N] [--rev REV]
        A private ``git clone --shared`` into ``upstream/`` (``upstream-N/`` with a name,
        for a further repository such as the stack the kernel is wired into) at REV
        (default: the repository's HEAD). The repository itself is never written.
    upstream patch <kernel_name> [--kind upstream|serving-only] [--notes TEXT]
        ``output/<clone>.patch`` (the diff) and ``output/<clone>.mbox`` (the commits, as
        ``git format-patch``) of what each clone committed over its base commit,
        every touched Python file compiled, and ``output/PORT.txt`` saying which kind of port it is and
        what was and was not verified, and the measured numbers appended to the session's
        ``output/PR.md``. C++/SYCL changes are reported unbuilt: rebuilding a provider is its
        owner's action. ``serving-only`` wires a winner into a stack without being fit for
        upstream (a dispatch by shape, a changed call site).
"""

from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path

UPSTREAM = Path("upstream")
OUTPUT = Path("output")
BASE_REF = "refs/x-forge/base"  # the commit a clone was made at; the agent commits on top
MEASURED = "<!-- Measured by x-forge; everything below this line is generated. -->"


def commit_problems(repo: Path, base: str) -> list[str]:
    """What keeps the commits base..HEAD in ``repo`` from being the whole, owner's change."""
    out = []
    git = ["git", "-C", str(repo)]
    dirty = subprocess.run([*git, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()  # fmt: skip
    if dirty:
        out.append(f"{repo.name} has uncommitted changes, which no patch carries: commit them")
    log = subprocess.run([*git, "log", "--format=%B", f"{base}..HEAD"],
                         capture_output=True, text=True).stdout  # fmt: skip
    if "Signed-off-by:" in log:
        out.append(f"a commit in {repo.name} carries a Signed-off-by: that is its owner's to add")
    return out


def write_mbox(repo: Path, base: str, path: Path) -> None:
    """The commits as ``git format-patch`` writes them: messages and authorship kept."""
    path.write_text(subprocess.run(["git", "-C", str(repo), "format-patch", "--stdout",
                                    f"{base}..HEAD"], capture_output=True, text=True).stdout)  # fmt: skip


def measured_pr(path: Path, facts: dict) -> bool:
    """Append ``facts`` to the session's PR description at ``path``; False when it wrote none.

    The prose is the session's; every number in it comes from here, so none is typed by hand.
    Re-running replaces the block rather than adding another.
    """
    text = path.read_text().split(MEASURED)[0].strip() if path.is_file() else ""
    if not text:
        return False
    rows = [f"| {k} | {v} |" for k, v in facts.items() if v not in (None, "", [])]
    path.write_text(f"{text}\n\n{MEASURED}\n\n## Measurements\n\n| | |\n|---|---|\n"
                    + "\n".join(rows) + "\n")  # fmt: skip
    return True


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", *args], capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc


def _clone(args) -> None:
    repo = Path(args.repo).expanduser().resolve()
    named = args.name and args.name != str(UPSTREAM)
    if named and (Path(args.name).name != args.name or args.name in (".", "..")):
        raise SystemExit("--name must be a single path component")
    target = Path(f"{UPSTREAM}-{args.name}") if named else UPSTREAM
    if (target / ".git").exists():
        # Reused, never re-cloned: it holds the session's edits. A clone of another
        # repository or revision is refused rather than edited in its place.
        base = _base(target)
        origin = _git("-C", str(target), "remote", "get-url", "origin", check=False)
        if Path(origin.stdout.strip()).resolve() != repo:
            raise SystemExit(
                f"{target}/ is a clone of {origin.stdout.strip() or 'unknown'}, not {repo}; "
                "use another --name, or remove it to clone afresh"
            )
        if args.rev:
            want = _git("-C", str(repo), "rev-parse", "--verify", f"{args.rev}^{{commit}}")
            if want.stdout.strip() != base:
                raise SystemExit(
                    f"{target}/ is at {base}, not --rev {args.rev} ({want.stdout.strip()}); "
                    "use another --name, or remove it to clone afresh"
                )
        print(f"{target}/ already exists at {base}; edit it, then run `upstream patch`.")
        return
    rev = args.rev or _git("-C", str(repo), "rev-parse", "HEAD").stdout.strip()
    _git("clone", "--quiet", "--shared", "--no-checkout", str(repo), str(target))
    _git("-C", str(target), "checkout", "--quiet", "--detach", rev)
    _git("-C", str(target), "update-ref", BASE_REF, "HEAD")
    print(f"Cloned {repo} at {rev} into {target}/. Edit files there, then run `upstream patch`.")


def _base(clone: Path) -> str:
    """The commit ``clone`` was made at (its HEAD, for a clone made before the ref existed)."""
    ref = _git("-C", str(clone), "rev-parse", "--verify", "-q", BASE_REF, check=False)
    return ref.stdout.strip() or _git("-C", str(clone), "rev-parse", "HEAD").stdout.strip()


def _best_trial(kernel_name: str, trials_dir: str) -> dict | None:
    """The measured winner a port carries, or ``None`` when there is none to carry."""
    from x_forge.core.trial_manager import TrialManager

    mgr = TrialManager(trials_dir)
    best = mgr.best_measured(kernel_name) if mgr.exists(kernel_name) else None
    # The same test `finalize` applies: a measured regression is not a winner.
    if best is None or not mgr.keepable(best):
        return None
    variant = (mgr._load_state(kernel_name).get("measured_against") or {}).get("variant")
    return {**best, "measured_variant": variant}


def _port_one(clone: Path) -> tuple[list[str], bool] | None:
    """Write ``output/<clone>.patch`` and ``.mbox`` and compile what it touches; ``None`` when
    the clone has no commits over its base."""
    base = _base(clone)
    diff = _git("-C", str(clone), "diff", "--binary", f"{base}..HEAD").stdout
    if not diff.strip():
        return None
    origin = _git("-C", str(clone), "remote", "get-url", "origin", check=False).stdout.strip()
    touched = [
        name
        for name in _git(
            "-C", str(clone), "diff", "--name-only", "-z", f"{base}..HEAD"
        ).stdout.split("\0")
        if name
    ]
    patch = (OUTPUT / f"{clone.name}.patch").resolve()
    patch.write_text(diff)
    write_mbox(clone, base, patch.with_suffix(".mbox"))

    ok = True
    checks = []
    for name in touched:
        path = clone / name
        if path.suffix == ".py" and path.exists():
            # Bytecode goes outside the clone: a __pycache__ there would join the next patch.
            check_file = OUTPUT / ".py_compile-check.pyc"
            try:
                py_compile.compile(str(path), cfile=str(check_file), doraise=True)
                checks.append(f"py_compile {name}: ok")
            except py_compile.PyCompileError as exc:
                ok = False
                checks.append(f"py_compile {name}: FAILED: {exc.msg.strip()}")
            finally:
                check_file.unlink(missing_ok=True)
    unbuilt = [n for n in touched if Path(n).suffix != ".py"]
    if unbuilt:
        checks.append(f"unbuilt (rebuilding is the owner's action): {', '.join(unbuilt)}")
    lines = [
        f"{OUTPUT / patch.name}:",
        f"  repository: {origin or 'unknown'}",
        f"  base commit: {base}",
        f"  touched files: {', '.join(touched)}",
        "  verified:",
        *(f"    - {c}" for c in checks),
    ]
    return lines, ok


def _patch(args) -> int:
    clones = sorted(p for p in Path(".").glob(f"{UPSTREAM}*") if (p / ".git").exists())
    if not clones:
        print("No upstream/ clone; run `upstream clone <repo>` first.")
        return 1
    best = _best_trial(args.kernel_name, args.trials_dir)
    problems = [p for c in clones for p in commit_problems(c, _base(c))]
    if problems:
        print("\n".join(problems))
        return 1
    if best is None:
        print(
            f"No measured winner for {args.kernel_name} (none, or a regression); nothing to port."
        )
        return 1
    OUTPUT.mkdir(exist_ok=True)
    ported = [(c, _port_one(c)) for c in clones]
    ported = [(c, r) for c, r in ported if r is not None]
    if not ported:
        print("No clone has commits over its base; commit the port, then run this again.")
        return 1

    lines = [
        f"kind: {args.kind}"
        + (
            " -- not for upstream; wires the winner into the stack"
            if args.kind == "serving-only"
            else ""
        ),
        f"best trial: {best['id']}: speedup "
        f"{best.get('speedup') or best.get('verdict') or 'parity'}",
    ]
    for _, (block, _) in ported:
        lines += block
    stack_ok = True
    names = "+".join(c.name for c, _ in ported)
    verdict, _, evidence = getattr(args, "stack", "").strip().partition(" ")
    # Strip the punctuation an agent naturally writes after the verdict ("PASS:"). The
    # verdict is one word from an agent, so normalise it rather than demanding exact
    # spelling.
    verdict = verdict.strip(":;,.-").upper()
    if verdict:
        stack_ok &= verdict == "PASS"
        lines.append(f"stack check {names}: {verdict} {evidence}".rstrip())
    built = all(ok for _, (_, ok) in ported)
    from x_forge.config import get_config

    if built and not verdict and get_config().external.stack_python:
        print("A stack check is configured: check the port in its stack, then pass "
              '--stack "PASS|FAIL <evidence>".')  # fmt: skip
        return 1
    if args.notes:
        lines += ["notes:", args.notes]
    facts = {
        "kernel": args.kernel_name, "port": args.kind, "trial": best["id"],
        "strategy": best.get("strategy"), "measured at": best.get("measured_variant"),
        "baseline (us)": best.get("baseline_us"), "patched (us)": best.get("triton_us"),
        "speedup": best.get("speedup"), "correctness": best.get("correctness"),
        **{f"base commit ({c.name})": _base(c) for c, _ in ported},
        "stack check": f"{verdict} {evidence}".strip() if verdict else None,
    }  # fmt: skip
    if not measured_pr(OUTPUT / "PR.md", facts):
        print("No output/PR.md: write the pull-request description, then run this again.")
        return 1
    (OUTPUT / "PORT.txt").write_text("\n".join(lines) + "\n")
    print(f"Wrote {', '.join(f'output/{c.name}.patch' for c, _ in ported)}, the .mbox of each, "
          "output/PORT.txt and the measurements in output/PR.md")  # fmt: skip
    print("\n".join(lines))
    return 0 if stack_ok and all(ok for _, (_, ok) in ported) else 1


def run(args):
    if args.upstream_command == "clone":
        _clone(args)
        return
    sys.exit(_patch(args))
