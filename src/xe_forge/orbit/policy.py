"""
The minimal policy gate (plan §24 Tier C).

This is deliberately the *minimal* gate — an action allowlist, the sandbox
invariants, and a single-writer lock — enforced before an intent mutates state.
It is the affordable half of Hyperloom's PolicyGate (§5.6 item 2), which also
enforces phase ordering, resource leases and data dependencies inside a
PRELUDE→CLOSE phase machine; that half stays deferred, because a phase machine
without phases would be a name for a component that does not exist — the exact
failure §24 tells this file's caller to stop committing.

What the minimal gate buys, concretely:

* **An allowlist, not a denylist.** An agent proposes actions, and an
  agent-proposed action is untrusted input. The gate admits what a context has
  explicitly granted and refuses everything else by default, so a new action
  is powerless until someone decides it should not be.
* **One exception type at the boundary.** The path invariants are
  `InPlacePatcher.check`'s — symlinks, writability, sandbox roots — and are
  *reused*, not re-implemented: a second copy of a safety check is a second
  place for the two to disagree. The patcher's refusal is wrapped in
  `PolicyViolation` so a caller sees one gate and one exception type.
* **Single-writer.** Two concurrent loops patching the same file interleave
  their journals and each other's reverts. An advisory per-target lock file
  refuses the second writer by name, and a lock whose holder is dead is broken
  with a logged note — never silently, because a silently broken lock is
  indistinguishable from a lock that never worked.

Every refusal says which invariant refused and why: a gate whose decisions the
caller cannot explain is indistinguishable from a bug (§20.4).
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

# The one action the optimization loop performs against the tree. A context that
# grants nothing else has granted exactly what the loop needs, and an agent
# proposal naming any other action is refused before anything happens.
DEFAULT_ALLOWED_ACTIONS = frozenset({"apply_patch"})

# How many times acquisition retries after breaking a stale lock. One retry is
# enough: if the slot is taken again immediately, the new holder is live and the
# refusal below names it.
_ACQUIRE_ATTEMPTS = 2


class PolicyViolation(Exception):
    """Raised when the gate refuses an intent, before any state has changed.

    The message always names the invariant that refused — allowlist, sandbox,
    or single-writer — and why, so the caller can relay a decision rather than
    report a mystery.
    """


class PolicyGate:
    """The §24 Tier C gate: allowlist + sandbox + single-writer, checked first.

    The sandbox invariants are delegated to the patcher given at construction;
    this class adds only what the patcher does not know about — which actions
    the current context permits, and who else may be writing.
    """

    def __init__(
        self,
        patcher: InPlacePatcher,
        allowed_actions: Iterable[str] = DEFAULT_ALLOWED_ACTIONS,
        lock_dir: Path | None = None,
    ) -> None:
        self.patcher = patcher
        self.allowed_actions = frozenset(allowed_actions)
        # Locks live alongside the patch journal by default: the journal
        # directory is already the run's designated writable scrap of disk, and
        # a lock next to the journal it protects is easy to find in a post-mortem.
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

        The invariants themselves — existence, symlinks, writability, sandbox
        roots — are `InPlacePatcher.check`'s and are not duplicated here; the
        patcher's refusal is re-raised as a `PolicyViolation` carrying the same
        reason, so the caller catches one exception type for every gate.
        """
        try:
            self.patcher.check(Path(target))
        except PatchSafetyError as exc:
            raise PolicyViolation(f"sandbox: {exc}") from exc

    # -- single-writer -----------------------------------------------------

    def _lock_path(self, target: Path) -> Path:
        # Keyed by the resolved path so two spellings of one file contend for
        # one lock; the digest keeps the name filesystem-safe and the suffix
        # keeps it human-readable, mirroring the journal's backup naming.
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
            # O_CREAT|O_EXCL is the atomicity: exactly one contender can create
            # the file, so there is no window between "check" and "take".
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
                # The holder is dead (or the lock file is unreadable, meaning
                # its writer died between creating and stamping it). Breaking it
                # is correct, and doing so silently is not: a broken lock is an
                # event the operator should be able to find afterwards.
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
        # Only reachable if a stale lock was broken and the slot was taken again
        # before the retry — meaning a live contender is racing for this target.
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
    """Exclusive per-device lease (plan §24 Tier E, item E2).

    §17.5 requires the GPU to be quiet during a measurement, and until now that
    was discipline — a rule the operator follows — rather than a mechanism. The
    measured motivation: a served arm was once launched while an eval still held
    the GPU, and only a fast manual kill kept the numbers honest. This class
    makes the collision impossible instead of disciplined-against: every
    GPU-touching command holds the lease for the duration, and a second claimant
    is refused with the holder's name, reason and start time.

    The mechanism is the single-writer lock's, deliberately — one lock idiom in
    this module, not two: ``O_CREAT|O_EXCL`` for atomicity, a JSON stamp naming
    the holder, stale-holder breaking with a logged note, and every refusal
    naming its invariant. An optional ``probe`` runs after acquisition and
    before the caller proceeds (the §17.5 quiet-machine check is the intended
    probe), so taking the lease and validating the measurement precondition are
    one gesture; a probe failure releases the lease and refuses.
    """

    def __init__(
        self,
        resource: str = "xpu0",
        lease_dir: Path | None = None,
        probe: Callable[[], None] | None = None,
    ) -> None:
        self.resource = resource
        self.lease_dir = Path(lease_dir) if lease_dir else Path.home() / ".cache/orbit-dev/leases"
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
                        f"(§17.5) — wait, or investigate the holder"
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
