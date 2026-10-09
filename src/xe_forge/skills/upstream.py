"""xe-forge-skill upstream: carry a finalized winner back to the repository it came from.

A kernel imported from a repository is optimized as a self-contained copy; the result
is only useful once it is a patch against the original files. The edits are the
session's to make. Everything around them is mechanical, and done here so that every
engine does it the same way and none needs a shell:

    upstream clone <repo> [--name N] [--rev REV]
        A private ``git clone --shared`` into ``upstream/`` (``upstream-N/`` with a name,
        for a further repository such as the stack the kernel is wired into) at REV
        (default: the repository's HEAD). The repository itself is never written.
    upstream patch <kernel_name> [--kind upstream|serving-only] [--notes TEXT]
        ``output/<clone>.patch`` from the edits in every clone, with its base commit,
        every touched Python file compiled, and ``output/PORT.txt`` saying which kind of port it is and
        what was and was not verified. C++/SYCL changes are reported unbuilt: rebuilding a
        provider is its owner's action. ``serving-only`` wires a winner into a stack
        without being fit for upstream (a dispatch by shape, a changed call site).
"""

from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path

UPSTREAM = Path("upstream")
OUTPUT = Path("output")


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
        base = _git("-C", str(target), "rev-parse", "HEAD").stdout.strip()
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
    print(f"Cloned {repo} at {rev} into {target}/. Edit files there, then run `upstream patch`.")


def _best_trial(kernel_name: str, trials_dir: str) -> str | None:
    """The measured winner a port carries, or ``None`` when there is none to carry."""
    from xe_forge.core.trial_manager import TrialManager

    mgr = TrialManager(trials_dir)
    best = mgr.best_measured(kernel_name) if mgr.exists(kernel_name) else None
    # The same test `finalize` applies: a measured regression is not a winner.
    if best is None or not mgr.keepable(best):
        return None
    return f"{best['id']}: speedup {best.get('speedup') or best.get('verdict') or 'parity'}"


def _port_one(clone: Path) -> tuple[list[str], bool] | None:
    """Write ``output/<clone>.patch`` and compile what it touches; ``None`` when unchanged."""
    # New files are part of the change; intent-to-add puts them in the diff.
    _git("-C", str(clone), "add", "--intent-to-add", "--all")
    diff = _git("-C", str(clone), "diff", "--binary").stdout
    if not diff.strip():
        return None
    base = _git("-C", str(clone), "rev-parse", "HEAD").stdout.strip()
    origin = _git("-C", str(clone), "remote", "get-url", "origin", check=False).stdout.strip()
    touched = [
        name
        for name in _git("-C", str(clone), "diff", "--name-only", "-z").stdout.split("\0")
        if name
    ]
    patch = (OUTPUT / f"{clone.name}.patch").resolve()
    patch.write_text(diff)

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
    if best is None:
        print(
            f"No measured winner for {args.kernel_name} (none, or a regression); nothing to port."
        )
        return 1
    OUTPUT.mkdir(exist_ok=True)
    ported = [(c, _port_one(c)) for c in clones]
    ported = [(c, r) for c, r in ported if r is not None]
    if not ported:
        print("No clone has changes; nothing to port.")
        return 1

    lines = [
        f"kind: {args.kind}"
        + (
            " -- not for upstream; wires the winner into the stack"
            if args.kind == "serving-only"
            else ""
        ),
        f"best trial: {best}",
    ]
    for _, (block, _) in ported:
        lines += block
    if args.notes:
        lines += ["notes:", args.notes]
    (OUTPUT / "PORT.txt").write_text("\n".join(lines) + "\n")
    print(f"Wrote {', '.join(f'output/{c.name}.patch' for c, _ in ported)} and output/PORT.txt")
    print("\n".join(lines))
    return 0 if all(ok for _, (_, ok) in ported) else 1


def run(args):
    if args.upstream_command == "clone":
        _clone(args)
        return
    sys.exit(_patch(args))
