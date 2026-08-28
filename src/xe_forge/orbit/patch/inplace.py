"""Crash-durable in-place edits to an installed tree: the original and journal entry
land on disk before the target is touched, writes are atomic, and reverts are
digest-verified and idempotent. Design rationale: docs/DESIGN.md."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

JOURNAL_NAME = "inplace_journal.json"


class PatchSafetyError(RuntimeError):
    """Raised when an edit cannot be made safely, before anything is modified."""


class RecoveryOutcome(StrEnum):
    RESTORED = "restored"
    # Already matches the recorded original; nothing to undo.
    ALREADY_CLEAN = "already_clean"
    # Matches neither the original nor the patched digest: a third party edited it.
    CONFLICT = "conflict"
    MISSING = "missing"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class PatchRecord:
    """One in-place edit, with everything needed to undo it without the original file."""

    target: str
    original_digest: str
    patched_digest: str
    original_path: str
    kernel_id: str = ""
    reason: str = ""

    def to_json(self) -> dict[str, str]:
        return {
            "target": self.target,
            "original_digest": self.original_digest,
            "patched_digest": self.patched_digest,
            "original_path": self.original_path,
            "kernel_id": self.kernel_id,
            "reason": self.reason,
        }

    @classmethod
    def from_json(cls, raw: dict[str, str]) -> PatchRecord:
        return cls(
            target=raw.get("target", ""),
            original_digest=raw.get("original_digest", ""),
            patched_digest=raw.get("patched_digest", ""),
            original_path=raw.get("original_path", ""),
            kernel_id=raw.get("kernel_id", ""),
            reason=raw.get("reason", ""),
        )


def _fsync_dir(path: Path) -> None:
    """Fsync the directory so a rename survives power loss; best-effort where the
    platform does not permit it."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write(target: Path, data: bytes) -> None:
    """Replace `target` with `data`, or leave it exactly as it was.

    The temp file lives in the target's own directory so `os.replace` is an atomic
    same-filesystem rename, never an interruptible copy.
    """
    target = Path(target)
    directory = target.parent
    handle = tempfile.NamedTemporaryFile(
        dir=str(directory), prefix=f".{target.name}.orbit-", delete=False
    )
    tmp = Path(handle.name)
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp), str(target))
        _fsync_dir(directory)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class InPlacePatcher:
    """Applies and reverts edits to an installed tree, with a crash-durable journal."""

    def __init__(self, journal_dir: Path, sandbox_roots: list[Path] | None = None) -> None:
        self.journal_dir = Path(journal_dir)
        # Paths outside these roots are refused; agent-proposed paths are untrusted.
        self.sandbox_roots = [Path(r).resolve() for r in (sandbox_roots or [])]

    # -- safety checks -----------------------------------------------------

    def check(self, target: Path) -> None:
        """Refuse anything that cannot be safely edited. Raises, never returns False."""
        target = Path(target)

        if not target.exists():
            raise PatchSafetyError(f"{target} does not exist")
        if target.is_symlink():
            # Writing through a symlink could modify a file outside the sandbox.
            raise PatchSafetyError(
                f"{target} is a symlink; writing through it would modify its target, "
                f"possibly outside the sandbox"
            )
        if not target.is_file():
            raise PatchSafetyError(f"{target} is not a regular file")
        if not os.access(target, os.W_OK):
            raise PatchSafetyError(
                f"{target} is not writable; an edit that cannot be reverted must not be attempted"
            )
        if not os.access(target.parent, os.W_OK):
            raise PatchSafetyError(
                f"{target.parent} is not writable, so the atomic replace cannot be "
                f"staged there and the write could not be made crash-safe"
            )

        if self.sandbox_roots:
            resolved = target.resolve()
            if not any(_is_within(resolved, root) for root in self.sandbox_roots):
                raise PatchSafetyError(
                    f"{resolved} is outside every sandbox root "
                    f"({', '.join(str(r) for r in self.sandbox_roots)})"
                )

    # -- journal -----------------------------------------------------------

    @property
    def journal_path(self) -> Path:
        return self.journal_dir / JOURNAL_NAME

    def _load_journal(self) -> list[PatchRecord]:
        if not self.journal_path.is_file():
            return []
        try:
            raw = json.loads(self.journal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return [PatchRecord.from_json(r) for r in raw if isinstance(r, dict)]

    def _save_journal(self, records: list[PatchRecord]) -> None:
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        atomic_write(
            self.journal_path,
            json.dumps([r.to_json() for r in records], indent=2).encode("utf-8"),
        )

    # -- apply / revert ----------------------------------------------------

    def apply(
        self,
        target: Path,
        new_content: bytes,
        kernel_id: str = "",
        reason: str = "",
    ) -> PatchRecord:
        """Edit `target` in place, leaving enough on disk to undo it after a crash.

        Order matters and is the whole safety argument: the original is copied and the
        journal fsynced *before* the target is touched. A crash between those steps
        leaves a journal entry whose recorded digest still matches the file, which
        `recover()` reads as "nothing to do".
        """
        target = Path(target)
        self.check(target)

        original = target.read_bytes()
        original_hash = digest(original)

        if digest(new_content) == original_hash:
            raise PatchSafetyError(
                f"the patch would leave {target} byte-identical; recording a no-op edit "
                f"would make the journal claim a change that never happened"
            )

        self.journal_dir.mkdir(parents=True, exist_ok=True)
        journal = self._load_journal()

        # A second patch to the same target inherits the first record's true original
        # (never the current, already-patched content); one record per target.
        existing = next((r for r in journal if r.target == str(target)), None)
        if existing is not None:
            original_hash = existing.original_digest
            backup = Path(existing.original_path)
            journal = [r for r in journal if r.target != str(target)]
        else:
            backup = self.journal_dir / f"{original_hash[:16]}-{target.name}.orig"
            atomic_write(backup, original)

        record = PatchRecord(
            target=str(target),
            original_digest=original_hash,
            patched_digest=digest(new_content),
            original_path=str(backup),
            kernel_id=kernel_id,
            reason=reason,
        )
        # Journal first. From here on, a crash is recoverable.
        self._save_journal([*journal, record])

        atomic_write(target, new_content)
        return record

    def revert(self, record: PatchRecord, force: bool = False) -> RecoveryOutcome:
        """Put the original back, refusing to discard a third party's edit.

        `force` overrides that refusal.
        """
        target = Path(record.target)
        if not target.exists():
            self._forget(record)
            return RecoveryOutcome.MISSING

        current = digest(target.read_bytes())
        if current == record.original_digest:
            # Never written, or already restored. Reverting twice must be harmless.
            self._forget(record)
            return RecoveryOutcome.ALREADY_CLEAN

        if current != record.patched_digest and not force:
            return RecoveryOutcome.CONFLICT

        backup = Path(record.original_path)
        if not backup.is_file():
            raise PatchSafetyError(
                f"the recorded original for {target} is missing at {backup}; the file "
                f"cannot be restored, which is why the backup is written before the edit"
            )

        restored = backup.read_bytes()
        if digest(restored) != record.original_digest:
            raise PatchSafetyError(
                f"the backup at {backup} does not match its recorded digest; restoring "
                f"it would write content we cannot vouch for"
            )

        atomic_write(target, restored)
        self._forget(record)
        return RecoveryOutcome.RESTORED

    def revert_all(self, force: bool = False) -> dict[str, RecoveryOutcome]:
        return {r.target: self.revert(r, force=force) for r in self._load_journal()}

    def recover(self, force: bool = False) -> dict[str, RecoveryOutcome]:
        """Undo edits left behind by a run that died. Safe to call at startup."""
        return self.revert_all(force=force)

    def _forget(self, record: PatchRecord) -> None:
        remaining = [r for r in self._load_journal() if r.target != record.target]
        if remaining:
            self._save_journal(remaining)
        else:
            self.journal_path.unlink(missing_ok=True)
        Path(record.original_path).unlink(missing_ok=True)

    # -- reporting ---------------------------------------------------------

    @property
    def outstanding(self) -> list[PatchRecord]:
        return self._load_journal()

    def format(self) -> str:
        records = self._load_journal()
        if not records:
            return "in-place patches: none outstanding; the installed tree is unmodified"
        lines = [f"in-place patches outstanding: {len(records)}", "-" * 72]
        for record in records:
            state = "?"
            path = Path(record.target)
            if path.is_file():
                current = digest(path.read_bytes())
                state = (
                    "applied"
                    if current == record.patched_digest
                    else ("clean" if current == record.original_digest else "CONFLICT")
                )
            lines.append(f"  [{state:>8}] {record.target}")
            if record.reason:
                lines.append(f"             {record.reason}")
        lines.append("-" * 72)
        lines.append("Run `xe-orbit patch recover` to restore the recorded originals.")
        return "\n".join(lines)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
