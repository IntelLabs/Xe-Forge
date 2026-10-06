"""The host contract is complete only when ``DONE`` is the last line of stdout."""

from __future__ import annotations

import sys

import pytest

from xe_forge.external import ExternalCommandError, run_external


def _host(tmp_path, body: str) -> str:
    script = tmp_path / "host.py"
    script.write_text("import sys\n" + body)
    return f"{sys.executable} {script}"


def test_done_followed_by_more_stdout_is_not_complete(tmp_path):
    cmd = _host(tmp_path, "print('CORRECTNESS: pass')\nprint('DONE')\nprint('Traceback ...')\n")
    with pytest.raises(ExternalCommandError, match="not the last line"):
        run_external(cmd)


def test_done_last_on_stdout_parses_despite_stderr_after_it(tmp_path):
    cmd = _host(
        tmp_path,
        "print('CORRECTNESS: pass')\nprint('SPEEDUP: 1.5')\nprint('DONE')\n"
        "sys.stdout.flush()\nprint('shutdown warning', file=sys.stderr)\n",
    )
    result = run_external(cmd)
    assert result.correctness is True
    assert result.speedup == 1.5
