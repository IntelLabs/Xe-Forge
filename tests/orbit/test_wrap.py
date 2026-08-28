"""The point-and-start profiler wrapper (`xe-orbit trace --wrap`).

The wrapper's contract: any single-process torch workload profiles with zero
profiler code of its own, the trace lands where the trace stage looks, and a
workload whose GPU work the wrapper cannot reach is told so rather than handed an
empty trace as though something was measured.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[2] / "src"


def _run_wrap(out: Path, workload: list[str]):
    return subprocess.run(
        [sys.executable, "-m", "xe_forge.orbit.profiling.wrap", "--out", str(out), "--", *workload],
        capture_output=True,
        text=True,
        timeout=300,
        env={"PYTHONPATH": str(REPO_SRC), "PATH": "/usr/bin:/bin"},
    )


class TestWrapModule:
    def test_script_path_form_produces_a_chrome_trace(self, tmp_path):
        script = tmp_path / "workload.py"
        script.write_text(
            "import torch\n"
            "x = torch.randn(64, 64)\n"
            "for _ in range(3):\n"
            "    x = x @ x.T\n"
            "print('workload done', float(x.sum()))\n"
        )
        out = tmp_path / "trace.json"
        result = _run_wrap(out, ["python", str(script)])
        assert result.returncode == 0, result.stderr[-500:]
        assert "workload done" in result.stdout
        payload = json.loads(out.read_text())
        assert payload.get("traceEvents"), "trace has no events"

    def test_cpu_only_workload_gets_the_honest_zero_device_note(self, tmp_path):
        script = tmp_path / "cpu_only.py"
        script.write_text("import torch; torch.ones(4).sum()\n")
        out = tmp_path / "trace.json"
        result = _run_wrap(out, [str(script)])
        assert result.returncode == 0
        # On a machine with a live GPU the note may legitimately not fire; what must
        # hold everywhere is that the event count is printed rather than implied.
        assert "device-side events" in result.stdout

    def test_module_form_and_workload_argv(self, tmp_path):
        pkg = tmp_path / "wl"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text(
            "import sys, torch\n"
            "torch.zeros(2)\n"
            "print('args:', sys.argv[1:])\n"
        )
        out = tmp_path / "trace.json"
        result = subprocess.run(
            [
                sys.executable, "-m", "xe_forge.orbit.profiling.wrap",
                "--out", str(out), "--", "python", "-m", "wl", "--steps", "2",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            env={"PYTHONPATH": f"{REPO_SRC}:{tmp_path}", "PATH": "/usr/bin:/bin"},
        )
        assert result.returncode == 0, result.stderr[-500:]
        assert "args: ['--steps', '2']" in result.stdout
        assert out.is_file()

    def test_workload_sysexit_zero_is_success(self, tmp_path):
        script = tmp_path / "exits.py"
        script.write_text("import sys, torch; torch.ones(1); sys.exit(0)\n")
        result = _run_wrap(tmp_path / "t.json", [str(script)])
        assert result.returncode == 0

    def test_failing_workload_fails_the_wrapper(self, tmp_path):
        script = tmp_path / "boom.py"
        script.write_text("raise RuntimeError('boom')\n")
        result = _run_wrap(tmp_path / "t.json", [str(script)])
        assert result.returncode != 0
        assert "boom" in result.stderr
