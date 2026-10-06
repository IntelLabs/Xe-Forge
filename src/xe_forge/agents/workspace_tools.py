"""The workspace a Claude session gets, as tools a DSPy agent can call.

A generated workspace (``claude/generator.py``) is already everything an optimizing
session needs: the policy in ``CLAUDE.md``, the seed, reference and spec under
``test_kernels/``, the knowledge base, and ``xe-forge-skill`` as the only way to
validate, measure, profile and record. These tools hand the same workspace to a DSPy
agent, so the two engines differ in the model driving them and in nothing else.

What the tools enforce rather than ask for:

* reads stay inside the workspace and the roots it links to (the knowledge base, a
  kernel repository the host named);
* writes stay inside the workspace and never touch what a skill writes -- the trial
  tree and the finalized output. What a measurement was taken against is pinned by the
  skills themselves, by digest, for every engine alike;
* ``skill`` runs ``xe-forge-skill`` with an argv split from the call, never through a
  shell, in a child process: a kernel that hangs or crashes the device takes the child
  with it, not the agent.

Every number the agent can act on comes back from a skill. Nothing here measures.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import dspy

__all__ = ["WorkspacePolicy", "make_workspace_tools"]

SKILLS = ("analyze", "validate", "benchmark", "trial", "profile", "upstream")
# Written only by a skill (`trial save`, `finalize`, `upstream patch`).
SKILL_OWNED = ("trials", "output")


@dataclass(frozen=True)
class WorkspacePolicy:
    """What the agent may read, write and run in one workspace."""

    root: Path
    read_roots: tuple[Path, ...] = ()
    # Files outside the workspace the agent may also write: the lessons ledger entry
    # the session owns, which outlives the workspace by design.
    writable_files: tuple[Path, ...] = ()
    # Files whose text the agent already has in its instructions; re-reading them would
    # put a second copy into a history that is re-sent every step.
    given: tuple[str, ...] = ()
    allowed_skills: tuple[str, ...] = SKILLS
    skill_timeout_s: int = 3600
    max_output_chars: int = 12_000
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def for_workspace(
        cls,
        root: str | Path,
        *,
        kernel_repo: str | None = None,
        integration_repos: list[str] = (),
        lessons_file: str | None = None,
        **kw,
    ):
        """The policy for a generated workspace: it may also read what it links to."""
        root = Path(root).resolve()
        reads = []
        if lessons_file:
            lessons = Path(lessons_file).resolve()
            reads.append(lessons.parent)
            kw["writable_files"] = (lessons,)
        kb = root / "knowledge_base"
        if kb.exists():
            reads.append(kb.resolve())
        for repo in (kernel_repo, *integration_repos):
            if repo:
                reads.append(Path(repo).expanduser().resolve())
        return cls(root=root, read_roots=tuple(reads), **kw)

    def _inside(self, path: Path, roots) -> bool:
        return any(path == r or r in path.parents for r in roots)

    def readable(self, path: str) -> Path:
        resolved = (self.root / path).resolve()
        if not self._inside(resolved, (self.root, *self.read_roots)):
            raise PermissionError(f"{path} is outside the workspace and its linked roots")
        return resolved

    def writable(self, path: str) -> Path:
        lexical = Path(os.path.normpath(self.root / path))
        resolved = lexical.resolve()
        if lexical in self.writable_files:
            return lexical
        # resolved != lexical: the path goes through a symlink, which can lead anywhere.
        if lexical != resolved or not self._inside(resolved, (self.root,)) or resolved == self.root:
            raise PermissionError(f"{path} is not a writable path in the workspace")
        parts = resolved.relative_to(self.root).parts
        if ".git" in parts:
            # A hook or a config key there is a command git would run for us later.
            raise PermissionError(f"{path} is inside a git directory; it is not writable")
        if parts[0] in SKILL_OWNED:
            raise PermissionError(
                f"{path} is under {parts[0]}, which only a skill writes; change it through "
                "the skill (e.g. `trial save`), not directly"
            )
        return resolved

    def clip(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        half = self.max_output_chars // 2
        dropped = len(text) - 2 * half
        return f"{text[:half]}\n... [{dropped} characters omitted] ...\n{text[-half:]}"


def make_workspace_tools(policy: WorkspacePolicy) -> list[dspy.Tool]:
    """Tools over *policy*'s workspace. Paths are relative to the workspace root."""

    def list_files(path: str = ".", pattern: str = "*") -> str:
        """List files under `path` whose name matches the glob `pattern` (at most 300)."""
        base = policy.readable(path)
        if base.is_file():
            return path
        found = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith((".git", "__pycache__")))
            for name in sorted(filenames):
                if fnmatch.fnmatch(name, pattern):
                    full = Path(dirpath) / name
                    try:
                        found.append(str(full.relative_to(policy.root)))
                    except ValueError:
                        found.append(str(full))
                    if len(found) >= 300:
                        return "\n".join(found) + "\n... (truncated at 300)"
        return "\n".join(found) or "(no files)"

    def read_file(path: str, start: int = 1, lines: int = 400) -> str:
        """Read `lines` lines of a text file from line `start` (1-based), numbered."""
        resolved = policy.readable(path)
        if any(resolved == (policy.root / g).resolve() for g in policy.given):
            return f"{path} is already in your instructions; it is not repeated here."
        text = resolved.read_text(errors="replace").splitlines()
        start = max(1, int(start))
        window = text[start - 1 : start - 1 + int(lines)]
        body = "\n".join(f"{start + i:6d}\t{line}" for i, line in enumerate(window))
        tail = f"\n... ({len(text)} lines total)" if start - 1 + len(window) < len(text) else ""
        return policy.clip(body + tail)

    def search(regex: str, path: str = ".", glob: str = "*") -> str:
        """Search files under `path` matching `glob` for the Python regex; file:line: text."""
        pattern = re.compile(regex)
        base = policy.readable(path)

        def walk():
            # Pruned in place, so a linked repository's object database is never entered;
            # sorted in place, so the order -- and what a truncation keeps -- is stable.
            for d, dirs, names in os.walk(base):
                dirs[:] = sorted(x for x in dirs if x not in (".git", "__pycache__"))
                yield from (Path(d) / f for f in sorted(names) if fnmatch.fnmatch(f, glob))

        files = [base] if base.is_file() else walk()
        hits = []
        for file in files:
            try:
                if file.stat().st_size > 2_000_000:
                    continue
                for number, line in enumerate(file.read_text(errors="strict").splitlines(), 1):
                    if pattern.search(line):
                        hits.append(
                            f"{file.relative_to(policy.root) if policy.root in file.parents else file}:{number}: {line.strip()[:200]}"
                        )
                        if len(hits) >= 200:
                            return policy.clip("\n".join(hits) + "\n... (truncated at 200 matches)")
            except (UnicodeDecodeError, OSError):
                continue
        return policy.clip("\n".join(hits) or "(no matches)")

    def write_file(path: str, content: str) -> str:
        """Create or overwrite a file in the workspace (kernels, harnesses, notes).

        The lessons ledger file is the exception: content is appended to it, never
        replaces it, because earlier sessions' entries are in the same file.
        """
        target = policy.writable(path)
        if target in policy.writable_files:
            with open(target, "a") as ledger:
                ledger.write(content if content.startswith("\n") else "\n" + content)
            return f"appended {len(content)} chars to {path}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"wrote {path} ({len(content)} chars)"

    def copy_file(src: str, dst: str) -> str:
        """Copy a file (e.g. the baseline, or a trial) to start a new file from it."""
        source = policy.readable(src)
        target = policy.writable(dst)
        if target in policy.writable_files:
            raise PermissionError(f"{dst} is appended to, not replaced; use write_file")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        return f"copied {src} -> {dst}"

    def edit_file(path: str, old: str, new: str, replace_all: bool = False) -> str:
        """Replace the exact text `old` with `new` in a file; `old` must occur exactly once
        unless replace_all. Change a large file this way instead of rewriting it."""
        target = policy.writable(path)
        if target in policy.writable_files:
            raise PermissionError(f"{path} is appended to, not edited; use write_file")
        text = target.read_text()
        count = text.count(old)
        if not old or count == 0:
            return f"not changed: the text to replace was not found in {path}"
        if count > 1 and not replace_all:
            return (
                f"not changed: the text occurs {count} times in {path}; include more "
                "surrounding lines, or pass replace_all"
            )
        target.write_text(text.replace(old, new))
        return f"edited {path}: {count if replace_all else 1} replacement(s)"

    def skill(args: str) -> str:
        """Run `xe-forge-skill <args>`; e.g. "trial save k cand.py --parent t0 --strategy '...'".

        Subcommands: analyze, validate, benchmark, trial, profile, upstream. Returns the output and
        exit code. Benchmark and profile results are the only measurements there are.
        """
        argv = shlex.split(args)
        if argv[:1] == ["xe-forge-skill"]:
            argv = argv[1:]
        if not argv or argv[0] not in policy.allowed_skills:
            return f"refused: the first argument must be one of {', '.join(policy.allowed_skills)}"
        command = [sys.executable, "-c", "from xe_forge.skills import main; main()", *argv]
        try:
            proc = subprocess.run(
                command,
                cwd=policy.root,
                capture_output=True,
                text=True,
                timeout=policy.skill_timeout_s,
                env={**os.environ, **policy.env},
            )
        except subprocess.TimeoutExpired:
            return f"exit=timeout after {policy.skill_timeout_s}s"
        return policy.clip(f"exit={proc.returncode}\n{proc.stdout}{proc.stderr}".rstrip())

    return [
        dspy.Tool(f)
        for f in (list_files, read_file, search, write_file, copy_file, edit_file, skill)
    ]
