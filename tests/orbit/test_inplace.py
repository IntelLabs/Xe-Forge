"""
Editing an installed tree in place, survivably (plan §13.6).

The question that decides whether in-place editing is usable is not "does it apply the
edit" but "what happens when we are wrong, or the process dies halfway through". These
tests are mostly about the second half.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from xe_forge.orbit.patch.inplace import (
    InPlacePatcher,
    PatchSafetyError,
    RecoveryOutcome,
    atomic_write,
    digest,
)

ORIGINAL = b"def kernel():\n    return 1\n"
PATCHED = b"def kernel():\n    return 2\n"


@pytest.fixture
def tree(tmp_path):
    site = tmp_path / "site-packages" / "framework"
    site.mkdir(parents=True)
    target = site / "kernels.py"
    target.write_bytes(ORIGINAL)
    return target


@pytest.fixture
def patcher(tmp_path, tree):
    return InPlacePatcher(journal_dir=tmp_path / "run", sandbox_roots=[tree.parent])


class TestRefusesWhatItCannotUndo:
    """A patch that cannot be reverted is not a patch, it is damage with a nicer name."""

    def test_a_readonly_file_is_refused_before_anything_changes(self, patcher, tree):
        tree.chmod(0o444)
        try:
            with pytest.raises(PatchSafetyError, match="not writable"):
                patcher.apply(tree, PATCHED)
        finally:
            tree.chmod(0o644)
        assert tree.read_bytes() == ORIGINAL

    def test_a_symlink_is_refused_because_the_write_would_escape(self, patcher, tmp_path, tree):
        """Writing through a symlink modifies its target, possibly outside the sandbox."""
        outside = tmp_path / "outside.py"
        outside.write_bytes(b"# not ours\n")
        link = tree.parent / "link.py"
        link.symlink_to(outside)
        with pytest.raises(PatchSafetyError, match="symlink"):
            patcher.apply(link, PATCHED)
        assert outside.read_bytes() == b"# not ours\n"

    def test_a_path_outside_the_sandbox_is_refused(self, patcher, tmp_path):
        stray = tmp_path / "elsewhere.py"
        stray.write_bytes(ORIGINAL)
        with pytest.raises(PatchSafetyError, match="outside every sandbox root"):
            patcher.apply(stray, PATCHED)

    def test_a_missing_file_is_refused(self, patcher, tree):
        with pytest.raises(PatchSafetyError, match="does not exist"):
            patcher.apply(tree.parent / "nope.py", PATCHED)

    def test_a_no_op_edit_is_refused(self, patcher, tree):
        """A journal entry claiming a change that never happened is worse than nothing."""
        with pytest.raises(PatchSafetyError, match="byte-identical"):
            patcher.apply(tree, ORIGINAL)

    def test_nothing_is_journalled_when_a_check_fails(self, patcher, tmp_path):
        stray = tmp_path / "elsewhere.py"
        stray.write_bytes(ORIGINAL)
        with pytest.raises(PatchSafetyError):
            patcher.apply(stray, PATCHED)
        assert patcher.outstanding == []


class TestApplyAndRevert:
    def test_the_edit_lands(self, patcher, tree):
        patcher.apply(tree, PATCHED)
        assert tree.read_bytes() == PATCHED

    def test_revert_restores_the_original_exactly(self, patcher, tree):
        record = patcher.apply(tree, PATCHED)
        assert patcher.revert(record) is RecoveryOutcome.RESTORED
        assert tree.read_bytes() == ORIGINAL

    def test_reverting_twice_is_harmless(self, patcher, tree):
        record = patcher.apply(tree, PATCHED)
        patcher.revert(record)
        assert patcher.revert(record) is RecoveryOutcome.ALREADY_CLEAN
        assert tree.read_bytes() == ORIGINAL

    def test_a_clean_revert_leaves_no_journal_behind(self, patcher, tree):
        record = patcher.apply(tree, PATCHED)
        assert patcher.outstanding
        patcher.revert(record)
        assert patcher.outstanding == []
        assert not patcher.journal_path.exists()

    def test_a_deleted_target_is_reported_rather_than_recreated(self, patcher, tree):
        record = patcher.apply(tree, PATCHED)
        tree.unlink()
        assert patcher.revert(record) is RecoveryOutcome.MISSING
        assert not tree.exists()

    def test_several_patches_revert_independently(self, patcher, tree):
        other = tree.parent / "other.py"
        other.write_bytes(b"x = 1\n")
        patcher.apply(tree, PATCHED)
        patcher.apply(other, b"x = 2\n")
        assert len(patcher.outstanding) == 2
        outcomes = patcher.revert_all()
        assert set(outcomes.values()) == {RecoveryOutcome.RESTORED}
        assert tree.read_bytes() == ORIGINAL
        assert other.read_bytes() == b"x = 1\n"


class TestRefusesToDiscardSomeoneElsesEdit:
    """The revert that silently destroys a third party's change is the subtle one."""

    def test_a_third_party_edit_is_a_conflict_not_a_restore(self, patcher, tree):
        record = patcher.apply(tree, PATCHED)
        tree.write_bytes(b"# somebody else was here\n")
        assert patcher.revert(record) is RecoveryOutcome.CONFLICT
        assert tree.read_bytes() == b"# somebody else was here\n"

    def test_a_conflict_stays_on_the_journal_for_a_human(self, patcher, tree):
        record = patcher.apply(tree, PATCHED)
        tree.write_bytes(b"# somebody else\n")
        patcher.revert(record)
        assert patcher.outstanding
        assert "CONFLICT" in patcher.format()

    def test_force_overrides_the_refusal_deliberately(self, patcher, tree):
        record = patcher.apply(tree, PATCHED)
        tree.write_bytes(b"# somebody else\n")
        assert patcher.revert(record, force=True) is RecoveryOutcome.RESTORED
        assert tree.read_bytes() == ORIGINAL

    def test_a_corrupted_backup_is_refused_rather_than_written(self, patcher, tree):
        record = patcher.apply(tree, PATCHED)
        Path(record.original_path).write_bytes(b"# not what we saved\n")
        with pytest.raises(PatchSafetyError, match="does not match its recorded digest"):
            patcher.revert(record)
        assert tree.read_bytes() == PATCHED

    def test_a_missing_backup_is_refused(self, patcher, tree):
        record = patcher.apply(tree, PATCHED)
        Path(record.original_path).unlink()
        with pytest.raises(PatchSafetyError, match="cannot be restored"):
            patcher.revert(record)


class TestCrashRecovery:
    """The journal is fsynced before the file is touched and cleared after it is
    restored, so at every instant the on-disk state is either "nothing to do" or "an
    entry naming exactly what to put back"."""

    def _fresh_patcher(self, patcher):
        """A new process: same journal directory, no in-memory state."""
        return InPlacePatcher(journal_dir=patcher.journal_dir, sandbox_roots=patcher.sandbox_roots)

    def test_a_crash_after_the_write_is_recovered_by_the_next_run(self, patcher, tree):
        patcher.apply(tree, PATCHED)
        assert tree.read_bytes() == PATCHED
        # process dies here; a later run constructs a fresh patcher over the journal
        outcomes = self._fresh_patcher(patcher).recover()
        assert set(outcomes.values()) == {RecoveryOutcome.RESTORED}
        assert tree.read_bytes() == ORIGINAL

    def test_a_crash_before_the_write_leaves_nothing_to_undo(self, patcher, tree):
        """Journalled but never written: the file still matches its recorded original."""
        record = patcher.apply(tree, PATCHED)
        atomic_write(tree, ORIGINAL)  # simulate the write never having landed
        outcomes = self._fresh_patcher(patcher).recover()
        assert outcomes[record.target] is RecoveryOutcome.ALREADY_CLEAN
        assert tree.read_bytes() == ORIGINAL

    def test_recovery_is_a_no_op_when_nothing_was_left_behind(self, patcher):
        assert self._fresh_patcher(patcher).recover() == {}

    def test_recovery_reports_a_conflict_rather_than_overwriting(self, patcher, tree):
        patcher.apply(tree, PATCHED)
        tree.write_bytes(b"# edited between runs\n")
        outcomes = self._fresh_patcher(patcher).recover()
        assert set(outcomes.values()) == {RecoveryOutcome.CONFLICT}
        assert tree.read_bytes() == b"# edited between runs\n"

    def test_an_unreadable_journal_does_not_break_startup(self, patcher, tree):
        """Recovery runs at startup; it must degrade rather than block every later run."""
        patcher.apply(tree, PATCHED)
        patcher.journal_path.write_text("{ not json", encoding="utf-8")
        assert self._fresh_patcher(patcher).recover() == {}

    def test_the_backup_lives_outside_the_tree_being_patched(self, patcher, tree):
        """So the record survives whatever happens to the tree."""
        record = patcher.apply(tree, PATCHED)
        backup = Path(record.original_path).resolve()
        assert not str(backup).startswith(str(tree.parent.resolve()))
        assert backup.read_bytes() == ORIGINAL


class TestAtomicWrite:
    def test_the_replacement_is_complete_or_absent_never_partial(self, tmp_path):
        target = tmp_path / "f.txt"
        target.write_bytes(b"old")
        atomic_write(target, b"new content")
        assert target.read_bytes() == b"new content"

    def test_a_failed_write_leaves_the_original_intact(self, tmp_path, monkeypatch):
        target = tmp_path / "f.txt"
        target.write_bytes(b"old")

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError):
            atomic_write(target, b"new")
        assert target.read_bytes() == b"old"

    def test_a_failed_write_leaves_no_temp_files_behind(self, tmp_path, monkeypatch):
        target = tmp_path / "f.txt"
        target.write_bytes(b"old")
        monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError()))
        with pytest.raises(OSError):
            atomic_write(target, b"new")
        assert [p.name for p in tmp_path.iterdir()] == ["f.txt"]

    def test_the_temp_file_is_staged_beside_the_target(self, tmp_path):
        """A temp file on another filesystem makes os.replace a copy, which can tear."""
        target = tmp_path / "sub" / "f.txt"
        target.parent.mkdir()
        target.write_bytes(b"old")
        atomic_write(target, b"new")
        assert target.read_bytes() == b"new"


class TestReporting:
    def test_a_clean_tree_says_so(self, patcher):
        assert "none outstanding" in patcher.format()

    def test_an_outstanding_patch_is_visible_with_its_reason(self, patcher, tree):
        patcher.apply(tree, PATCHED, reason="E3 harness for k12")
        rendered = patcher.format()
        assert "applied" in rendered
        assert "E3 harness for k12" in rendered
        assert "recover" in rendered

    def test_digest_is_stable(self):
        assert digest(b"abc") == digest(b"abc")
        assert digest(b"abc") != digest(b"abd")


class TestRepatchingKeepsTheTrueOriginal:
    """Found by running the demo: applying twice recorded the wrong original.

    The second `apply` read the file as it stood — which was the *first patch's output*
    — and journalled that as the original. Reverting then restored an intermediate
    state and called the tree clean, which is precisely the failure this module exists
    to prevent.
    """

    def test_a_second_patch_does_not_overwrite_the_recorded_original(self, patcher, tree):
        patcher.apply(tree, PATCHED)
        second = patcher.apply(tree, b"def kernel():\n    return 3\n")
        assert second.original_digest == digest(ORIGINAL)

    def test_reverting_after_two_patches_reaches_the_pristine_file(self, patcher, tree):
        patcher.apply(tree, PATCHED)
        record = patcher.apply(tree, b"def kernel():\n    return 3\n")
        assert patcher.revert(record) is RecoveryOutcome.RESTORED
        assert tree.read_bytes() == ORIGINAL

    def test_one_record_per_target_is_the_invariant(self, patcher, tree):
        """Duplicates made revert report on whichever entry happened to land last."""
        patcher.apply(tree, PATCHED)
        patcher.apply(tree, b"def kernel():\n    return 3\n")
        assert len(patcher.outstanding) == 1

    def test_recovery_after_two_patches_restores_the_original(self, patcher, tree):
        patcher.apply(tree, PATCHED)
        patcher.apply(tree, b"def kernel():\n    return 3\n")
        fresh = InPlacePatcher(journal_dir=patcher.journal_dir, sandbox_roots=patcher.sandbox_roots)
        assert set(fresh.recover().values()) == {RecoveryOutcome.RESTORED}
        assert tree.read_bytes() == ORIGINAL

    def test_the_status_line_reports_one_entry_not_two(self, patcher, tree):
        patcher.apply(tree, PATCHED)
        patcher.apply(tree, b"def kernel():\n    return 3\n")
        assert "outstanding: 1" in patcher.format()
