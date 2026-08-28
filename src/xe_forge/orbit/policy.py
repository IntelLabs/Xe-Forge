"""
The minimal policy gate: an action allowlist, the sandbox invariants, and a
single-writer lock, enforced before an intent mutates state. Every refusal names
the invariant that refused. Design rationale: docs/DESIGN.md
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from xe_forge.orbit.patch.inplace import InPlacePatcher, PatchSafetyError

logger = logging.getLogger(__name__)

# The one action the optimization loop performs against the tree.
DEFAULT_ALLOWED_ACTIONS = frozenset({"apply_patch"})

# Retries after breaking a stale lock; a slot re-taken immediately means a live holder.
_ACQUIRE_ATTEMPTS = 2


class PolicyViolation(Exception):
    """Raised when the gate refuses an intent, before any state has changed.

    The message names the invariant that refused — allowlist, sandbox, or
    single-writer — and why.
    """


class PolicyGate:
    """Allowlist + sandbox + single-writer, checked before any state changes.

    The sandbox invariants are delegated to the patcher given at construction;
    this class adds only which actions the context permits and who else may be
    writing.
    """

    def __init__(
        self,
        patcher: InPlacePatcher,
        allowed_actions: Iterable[str] = DEFAULT_ALLOWED_ACTIONS,
        lock_dir: Path | None = None,
    ) -> None:
        self.patcher = patcher
        self.allowed_actions = frozenset(allowed_actions)
        # Locks default to the journal directory — the run's designated writable disk.
        self.lock_dir = Path(lock_dir) if lock_dir is not None else patcher.journal_dir

    # -- action allowlist --------------------------------------------------

    def check_action(self, action: str) -> None:
        """Refuse any action the current context has not explicitly granted."""
        if action not in self.allowed_actions:
            granted = ", ".join(sorted(self.allowed_actions)) or "nothing"
            raise PolicyViolation(
                f"action allowlist: {action!r} is not permitted in this context "
                f"(granted: {granted}); refused before any state changed"
            )

    # -- sandbox invariants ------------------------------------------------

    def check_write(self, target: Path) -> None:
        """Refuse a write the patcher could not make safely, as a policy decision.

        The invariants are `InPlacePatcher.check`'s, not duplicated here; its
        refusal is re-raised as `PolicyViolation` with the same reason.
        """
        try:
            self.patcher.check(Path(target))
        except PatchSafetyError as exc:
            raise PolicyViolation(f"sandbox: {exc}") from exc

    # -- single-writer -----------------------------------------------------

    def _lock_path(self, target: Path) -> Path:
        # Keyed by the resolved path so two spellings of one file contend for one lock.
        target = Path(target)
        key = hashlib.sha256(str(target.resolve()).encode("utf-8")).hexdigest()[:16]
        return self.lock_dir / f"{key}-{target.name}.lock"

    @contextmanager
    def single_writer(self, target: Path) -> Iterator[None]:
        """Hold the advisory per-target write lock for the duration of the block.

        Acquisition refuses — naming the holder — if a live process already
        holds the lock; a stale lock, whose recorded holder is dead, is broken
        with a logged note rather than silently. Release removes the lock file.
        """
        lock = self._acquire(Path(target))
        try:
            yield
        finally:
            lock.unlink(missing_ok=True)

    def _acquire(self, target: Path) -> Path:
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        lock = self._lock_path(target)
        holder: int | None = None
        for _ in range(_ACQUIRE_ATTEMPTS):
            # O_CREAT|O_EXCL is the atomicity: no window between "check" and "take".
            try:
                fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                holder = self._read_holder(lock)
                if holder is not None and _pid_alive(holder):
                    raise PolicyViolation(
                        f"single-writer: {target} is already locked by live pid "
                        f"{holder} ({lock}); a second concurrent writer on one "
                        f"target would interleave journals and reverts"
                    ) from None
                # Stale lock: breaking it is correct, but never silently.
                if holder is None:
                    logger.warning(
                        "single-writer: breaking stale lock %s on %s; the lock file "
                        "names no readable holder, so its writer died before stamping it",
                        lock,
                        target,
                    )
                else:
                    logger.warning(
                        "single-writer: breaking stale lock %s on %s; holder pid %d is dead",
                        lock,
                        target,
                        holder,
                    )
                lock.unlink(missing_ok=True)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"pid": os.getpid(), "target": str(target)}, handle)
            return lock
        # Only reachable if a broken stale lock was re-taken before the retry.
        raise PolicyViolation(
            f"single-writer: {target} could not be locked; a stale lock was broken "
            f"but another writer re-took {lock} immediately, so a live contender "
            f"is racing for this target"
        )

    @staticmethod
    def _read_holder(lock: Path) -> int | None:
        """The pid recorded in the lock file, or None if it cannot be read."""
        try:
            raw = json.loads(lock.read_text(encoding="utf-8"))
            return int(raw["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            return None


def _pid_alive(pid: int) -> bool:
    """Whether the process holding a lock still exists.

    `/proc/<pid>` is the source of truth where procfs exists; elsewhere,
    signal 0 probes for existence without touching the process — a
    `PermissionError` means the pid exists but belongs to someone else, which
    still counts as alive.
    """
    proc = Path("/proc")
    if proc.is_dir():
        return (proc / str(pid)).exists()
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class ResourceLease:
    """Exclusive per-device lease. A second claimant is refused with the holder's
    name, reason and start time; a stale lease is broken with a logged note.

    An optional ``probe`` runs after acquisition and before the caller proceeds
    (the intended probe is the quiet-machine check); a probe failure releases
    the lease and refuses.
    """

    def __init__(
        self,
        resource: str = "xpu0",
        lease_dir: Path | None = None,
        probe: Callable[[], None] | None = None,
    ) -> None:
        self.resource = resource
        # ORBIT_LEASE_DIR isolates tests and CI from the machine-wide lease;
        # operators on one device share the default deliberately.
        default = os.environ.get("ORBIT_LEASE_DIR") or Path.home() / ".cache/orbit-dev/leases"
        self.lease_dir = Path(lease_dir) if lease_dir else Path(default)
        self.probe = probe

    @property
    def _lease_path(self) -> Path:
        return self.lease_dir / f"{self.resource}.lease"

    @contextmanager
    def hold(self, reason: str) -> Iterator[None]:
        """Hold the device exclusively for the duration of the block."""
        lease = self._acquire(reason)
        try:
            if self.probe is not None:
                try:
                    self.probe()
                except Exception as exc:
                    raise PolicyViolation(
                        f"lease: acquired {self.resource} but the measurement "
                        f"precondition probe refused: {exc}"
                    ) from exc
            yield
        finally:
            lease.unlink(missing_ok=True)

    def _acquire(self, reason: str) -> Path:
        self.lease_dir.mkdir(parents=True, exist_ok=True)
        lease = self._lease_path
        for _ in range(_ACQUIRE_ATTEMPTS):
            try:
                fd = os.open(str(lease), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                stamp = self._read_stamp(lease)
                pid = stamp.get("pid") if stamp else None
                if isinstance(pid, int) and _pid_alive(pid):
                    raise PolicyViolation(
                        f"lease: {self.resource} is held by live pid {pid} "
                        f"({stamp.get('reason', '?')}, since {stamp.get('since', '?')}); "
                        f"two claimants on one device would corrupt both measurements "
                        f"— wait, or investigate the holder"
                    ) from None
                logger.warning(
                    "lease: breaking stale lease %s on %s; %s",
                    lease,
                    self.resource,
                    f"holder pid {pid} is dead" if isinstance(pid, int) else "no readable holder stamped",
                )
                lease.unlink(missing_ok=True)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "pid": os.getpid(),
                        "resource": self.resource,
                        "reason": reason,
                        "since": _now_iso(),
                    },
                    handle,
                )
            return lease
        raise PolicyViolation(
            f"lease: {self.resource} could not be acquired; a stale lease was broken "
            f"but another claimant re-took it immediately, so a live contender is "
            f"racing for this device"
        )

    @staticmethod
    def _read_stamp(lease: Path) -> dict | None:
        try:
            raw = json.loads(lease.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else None
        except (OSError, ValueError):
            return None


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
