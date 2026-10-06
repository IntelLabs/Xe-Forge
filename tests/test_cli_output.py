"""``--output`` replaces its target, so it must never overlap the winner it copies."""

from __future__ import annotations

import pytest

from xe_forge.cli import _copy_winner_dir


@pytest.fixture
def winner(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "trials" / "k" / "t1"
    src.mkdir(parents=True)
    (src / "kernel.cpp").write_text("winner")
    return src


@pytest.mark.parametrize("target", [".", "..", "sub"])
def test_an_output_overlapping_the_winner_is_refused(winner, target):
    error = _copy_winner_dir(winner, winner / target)
    assert error and "winner" in error
    assert (winner / "kernel.cpp").read_text() == "winner"


def test_the_output_is_replaced_by_the_winner(winner, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale.cpp").write_text("old")
    assert _copy_winner_dir(winner, out) is None
    assert sorted(p.name for p in out.iterdir()) == ["kernel.cpp"]
