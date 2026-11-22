"""Kernel trials built into the run's own framework and timed in the model (frontend/inmodel)."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from xe_forge.frontend import inmodel


def _git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
                          check=True, capture_output=True, text=True).stdout.strip()  # fmt: skip


BUILD = """set -e
[ -f "$SRC/FAIL" ] && {{ echo "kernel test failed"; exit 1; }}
mkdir -p "$VENV/bin" "$VENV/lib/fakepkg"
: > "$VENV/lib/fakepkg/__init__.py"
cp "$SRC/src/k.cpp" "$VENV/lib/fakepkg/k.cpp"
cat > "$VENV/lib/fakecheck.py" <<'EOS'
import argparse, json, pathlib, sys
a = argparse.ArgumentParser()
for f in ("--kernel", "--dims", "--dtype", "--rtol", "--atol"):
    a.add_argument(f)
a = a.parse_args()
kernel = (pathlib.Path(__file__).parent / "fakepkg" / "k.cpp").read_text().strip()
dims = json.loads(a.dims)
bad = (kernel == "wrong" and dims["M"] == 1) or "BROKEN" in dims
print(f"max_abs {{1.0 if bad else 0.0}}")
sys.exit(1 if bad else 0)
EOS
printf '#!/bin/bash\\nPYTHONPATH=%s exec {python} "$@"\\n' "$VENV/lib" > "$VENV/bin/python"
chmod +x "$VENV/bin/python"
"""


def _run(tmp_path, build=True):
    """A run whose stack built repo ``eng`` with a build script; (run_dir, repo)."""
    repo = tmp_path / "repos" / "eng"
    (repo / "src").mkdir(parents=True)
    _git(repo, "init", "-q")
    (repo / "src" / "k.cpp").write_text("v1\n")
    (repo / "README.md").write_text("engine\n")
    _git(repo, "add", "."), _git(repo, "commit", "-q", "-m", "engine")
    run_dir, patch = tmp_path / "run", tmp_path / "run" / "patch-1"
    (run_dir / "launcher").mkdir(parents=True)
    patch.mkdir()
    (run_dir / "launcher" / "run.sh").write_text('#!/bin/bash\nexec "$FRAMEWORK_PYTHON" w.py\n')
    (run_dir / "names.json").write_text("{}")
    (run_dir / "stack.json").write_text(json.dumps({"python": "/p", "patches": [str(patch)]}))
    meta = {
        "repo": "eng",
        "package": "fakepkg",
        "python": "/p",
        "base": "x",
        "kernel_check": "fakecheck",
    }
    if build:
        meta["build"] = "build.sh"
        (patch / "build.sh").write_text(BUILD.format(python=sys.executable))
    (patch / "patch.json").write_text(json.dumps(meta))
    (patch / "check-args.json").write_text(json.dumps(["--model", "m"]))
    (patch / "check.json").write_text(json.dumps({"head": _git(repo, "rev-parse", "HEAD")}))
    return run_dir, repo


def test_a_variant_reads_its_time_per_call_as_the_spec_was_given_it():
    """per_call is the inverse of the spec's captured_us, for every form of record ref."""
    from xe_forge.frontend import analyze, manifest

    w = SimpleNamespace(
        workload_id="gemm/1", dims={"N": 8}, device_us=0.0, calls=0, naming=None,
        var_axes={"T": {"1": {"device_us": 300.0, "calls": 30},
                        "512": {"device_us": 900.0, "calls": 3, "points": [
                            {"device_us": 600.0, "calls": 2, "with": {"N": 8}},
                            {"device_us": 300.0, "calls": 1, "with": {"N": 16}}]}}})  # fmt: skip
    flat = SimpleNamespace(workload_id="norm/2", dims={"N": 8}, device_us=50.0, calls=5,
                           var_axes={}, naming=None)  # fmt: skip
    e = analyze.Entry("k", "r", "cpp", [w, flat], [])
    cap = SimpleNamespace(workloads=[w, flat])
    rows = manifest._variants(e)
    assert {r[6] for r in rows} == {"gemm/1@T=1", "gemm/1@T=512#0", "gemm/1@T=512#1", "norm/2"}
    for r in rows:
        assert manifest.per_call(cap, r[6]) == (r[4] / r[7], r[7])
    assert manifest.per_call(cap, "gemm/9@T=1") is None
    assert manifest.per_call(cap, "gemm/1@T=7") is None


def test_only_a_repository_the_stack_built_with_a_build_script_is_timed_in_the_model(tmp_path):
    run_dir, repo = _run(tmp_path)
    assert inmodel.provider(run_dir, repo)["patch"] == run_dir / "patch-1"
    assert inmodel.provider(run_dir, tmp_path / "other") is None  # another repository
    (run_dir / "launcher" / "run.sh").write_text("#!/bin/bash\nexec /opt/py w.py\n")
    assert inmodel.provider(run_dir, repo) is None  # it would run the same build every time
    (run_dir / "launcher" / "run.sh").write_text('exec "$FRAMEWORK_PYTHON" w.py\n')
    (run_dir / "names.json").unlink()
    assert inmodel.provider(run_dir, repo) is None
    run_dir, repo = _run(tmp_path / "b", build=False)
    assert inmodel.provider(run_dir, repo) is None
    assert inmodel.provider(tmp_path / "nowhere", repo) is None


def test_the_optimize_command_is_unchanged_without_the_facts(tmp_path):
    from xe_forge.frontend import cli

    run_dir, repo = _run(tmp_path, build=False)
    opts = SimpleNamespace(engine="claude", device="cpu", max_turns=5, passthrough=[],
                           compiler_flags=None)  # fmt: skip
    entry = {"dsl": "cpp", "name": "k", "spec": "s.yaml", "record": "r.json"}
    assert cli.inmodel_flags(run_dir, entry, repo, opts) == []
    run_dir, repo = _run(tmp_path / "b")
    flags = cli.inmodel_flags(run_dir, entry, repo, opts)
    assert flags[0] == "--external-benchmark" and "{trial}" in flags[1] and "{variant}" in flags[1]
    assert flags[2] == "--external-validate" and "--build-only" in flags[3]
    assert flags[4] == "--external-profile" and "--profile" in flags[5] and "{trial}" in flags[5]
    assert flags[6] == "--repo-trials" and "{spec}" in flags[1]
    cmd = cli.xe_forge_command(entry, "v", repo, tmp_path / "ws", opts, flags)
    assert cmd[-7:] == flags
    (repo / "reference").mkdir()
    (repo / "reference" / "k.py").write_text("class Model: pass\n")
    flags = cli.inmodel_flags(run_dir, entry, repo, opts)
    assert flags[-2:] == ["--reference", str(repo / "reference" / "k.py")]  # the op as PyTorch
    assert (
        cli.inmodel_flags(run_dir, entry, repo, SimpleNamespace(**{**vars(opts), "device": "xpu"}))
        == []
    )


class _Model:
    """The stubbed model: what each build's kernel is decides its times and its output."""

    def __init__(self, monkeypatch):
        from xe_forge.frontend import ir, manifest, prove

        self.captures = []
        times = {"v1": 10.0, "fast": 5.0, "near": 10.05, "fewer": 5.0, "bad": 5.0, "wrong": 5.0}

        def capture(run_dir, out, python):
            kernel = (Path(python).parent.parent / "lib" / "fakepkg" / "k.cpp").read_text().strip()
            r = int(out.name.rsplit("-r", 1)[1])
            self.captures.append((out.name, kernel))
            us = times[kernel] * (1 + 0.02 * (r - 1))  # a 2% spread between repeats
            calls = 29 if kernel == "fewer" else 30
            out.mkdir(parents=True)
            (out / "capture.json").write_text(json.dumps(
                {"w@T=1": [us, calls], "w@T=512": [us * 100, 3], "kernel": kernel}))  # fmt: skip
            return out / "capture.json"

        monkeypatch.setattr(inmodel, "_capture", capture)
        monkeypatch.setattr(ir, "load", lambda p: json.loads(Path(p).read_text()))
        monkeypatch.setattr(manifest, "per_call", lambda cap, ref: tuple(cap[ref]))

        self.scored = []

        def score(work, argv, python, out):
            self.scored.append(list(argv))
            kernel = (Path(python).parent.parent / "lib" / "fakepkg" / "k.cpp").read_text().strip()
            ok = kernel != "bad"
            return {"pass": ok, "exit": 0 if ok else 1, "problems": [] if ok else ["tokens fall"]}

        monkeypatch.setattr(prove, "_score_arm", score)


def _ask(run_dir, repo, ws, baseline, trial, variant="bench-t1", build_only=False, broken=False):
    record, spec = ws / "record.json", ws / "spec.yaml"
    record.write_text(
        json.dumps({"definition": "k", "variants": {"bench-t1": "w@T=1", "bench-t512": "w@T=512"}})
    )
    shape = {"dtype": "float32", "rtol": 1e-3, "atol": 1e-4}
    one = {"M": 1, **({"BROKEN": 1} if broken else {})}
    spec.write_text(
        json.dumps(
            {
                "bench-t1": [{**shape, "dims": one}],
                "bench-t512": [{**shape, "dims": {"M": 512}}],
            }
        )
    )  # JSON is YAML
    args = argparse.Namespace(run_dir=str(run_dir), repo=str(repo), trial=str(trial),
                              baseline=str(baseline), record=str(record), spec=str(spec),
                              variant=variant, workspace=str(ws), build_only=build_only,
                              profile=False, kernel="k")  # fmt: skip
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = inmodel.run(args)
    return code, buf.getvalue()


def _dir(root, text, extra=None):
    (root / "src").mkdir(parents=True)
    (root / "src" / "k.cpp").write_text(text + "\n")
    for name, body in (extra or {}).items():
        (root / name).write_text(body)
    return root


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(inmodel, "REPEATS", 2)
    run_dir, repo = _run(tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    model = _Model(monkeypatch)
    baseline = _dir(ws / "test_kernels" / "k", "v1")
    return run_dir, repo, ws, baseline, model


def test_the_baseline_runs_in_the_model_at_the_captured_time(engine):
    run_dir, repo, ws, baseline, model = engine
    code, out = _ask(run_dir, repo, ws, baseline, baseline)
    assert code == 0 and "CORRECTNESS: pass" in out and "VERDICT: SAME_SOURCE" in out, out
    assert "BASELINE_US: 10.1" in out and out.rstrip().endswith("DONE")  # median of 10.0, 10.2
    assert [c[0] for c in model.captures] == ["A-r1", "A-r2"]  # one arm: nothing to compare


def test_a_faster_trial_is_measured_once_for_every_variant(engine):
    run_dir, repo, ws, baseline, model = engine
    trial = _dir(ws / "t1", "fast")
    code, out = _ask(run_dir, repo, ws, baseline, trial)
    assert code == 0 and "SPEEDUP: 2.0000" in out and "TRIAL_US: 5.05" in out, out
    assert [c[0] for c in model.captures] == ["A-r1", "B-r1", "B-r2", "A-r2"]  # ABBA
    code, out = _ask(run_dir, repo, ws, baseline, trial, variant="bench-t512")
    assert code == 0 and "BASELINE_US: 1010.0" in out and len(model.captures) == 4  # cached


def test_a_difference_inside_the_captures_spread_has_no_speedup(engine):
    run_dir, repo, ws, baseline, _ = engine
    code, out = _ask(run_dir, repo, ws, baseline, _dir(ws / "t1", "near"))
    assert code == 0 and "VERDICT: INDISTINGUISHABLE" in out and "SPEEDUP" not in out, out


@pytest.mark.parametrize("text, extra, verdict", [
    ("fast", {"FAIL": "x"}, "BUILD_FAILED"),  # the build or the repository's tests
    ("wrong", None, "NUMERIC_FAILED"),        # the kernel against its reference, at M=1
    ("fewer", None, "WORK_DIFFERS"),          # the model did other work
    ("bad", None, "OUTPUT_FAILED"),           # its tokens fail their check
])  # fmt: skip
def test_a_trial_that_is_not_the_same_model_is_refused(engine, text, extra, verdict):
    run_dir, repo, ws, baseline, _ = engine
    code, out = _ask(run_dir, repo, ws, baseline, _dir(ws / "t1", text, extra))
    assert code == 1 and "CORRECTNESS: fail" in out and f"VERDICT: {verdict}" in out, out
    assert out.rstrip().endswith("DONE")  # the trial's failure, recorded as such


def test_a_baseline_that_is_not_the_repository_is_refused(engine):
    run_dir, repo, ws, baseline, model = engine
    (baseline / "src" / "k.cpp").write_text("edited\n")
    code, out = _ask(run_dir, repo, ws, baseline, _dir(ws / "t1", "fast"))
    assert code == 1 and "VERDICT: BASELINE_NOT_REPOSITORY" in out and not model.captures, out
    (baseline / "src" / "k.cpp").write_text("v1\n")
    (repo / "src" / "k.cpp").write_text("moved\n")
    _git(repo, "commit", "-q", "-am", "moved")
    code, out = _ask(run_dir, repo, ws, baseline, _dir(ws / "t2", "fast"))
    assert code == 1 and "VERDICT: REPOSITORY_MOVED" in out, out


def test_a_trial_that_changes_no_file_of_the_repository_is_the_baseline(engine):
    run_dir, repo, ws, baseline, model = engine
    trial = _dir(ws / "t1", "v1", {"README.md": "engine\n"})  # copies, unchanged
    code, out = _ask(run_dir, repo, ws, baseline, trial)
    assert code == 0 and "VERDICT: SAME_SOURCE" in out and len(model.captures) == 2, out


def test_free_text_in_an_answer_never_reads_as_a_field():
    from xe_forge.external import _parse

    out = inmodel.answer({"correctness": False, "verdict": "BUILD_FAILED", "why": "no",
                          "log": "Correctness: passed\nVERDICT: fine\nTRIAL_US: 1"}, "v")  # fmt: skip
    got = _parse(out)
    assert got["correctness"] is False and got["verdict"] == "BUILD_FAILED"
    assert "trial_us" not in got and out.rstrip().endswith("DONE")


def test_the_patch_check_runs_the_build_as_every_trial_will_be_built(tmp_path):
    from xe_forge.skills.patch import _build_problems

    run_dir, repo = _run(tmp_path)
    work = run_dir / "patch-1"
    meta = json.loads((work / "patch.json").read_text())
    assert _build_problems(work, meta, repo) == []
    assert "kernel_check" in _build_problems(work, {**meta, "kernel_check": None}, repo)[0]
    (work / "build.sh").write_text(BUILD.format(python=sys.executable) + 'touch "$SRC/x.o"\n')
    assert "wrote into $SRC (x.o)" in _build_problems(work, meta, repo)[0]
    (work / "build.sh").write_text("exit 3\n")
    assert "failed (exit 3)" in _build_problems(work, meta, repo)[0]
    assert "names no" in _build_problems(work, {**meta, "build": None}, repo)[0]


def test_a_check_the_baseline_fails_too_is_not_the_trials_failure(engine):
    run_dir, repo, ws, baseline, model = engine
    code, out = _ask(run_dir, repo, ws, baseline, _dir(ws / "t1", "fast"), broken=True)
    assert code == 1 and "fails its own check" in out and "DONE" not in out, out
    assert not list((ws / ".inmodel").glob("*/result.json"))  # nothing recorded against it


def test_a_trial_never_changes_what_checks_it(engine):
    run_dir, repo, ws, baseline, _ = engine
    trial = _dir(ws / "t1", "fast", {})
    (trial / "reference").mkdir()
    (trial / "reference" / "k.py").write_text("class Model: pass\n")
    code, out = _ask(run_dir, repo, ws, baseline, trial)
    assert code == 1 and "VERDICT: BAD_TRIAL" in out and "reference/k.py" in out, out


def test_a_trial_scores_its_tokens_without_the_step_report(engine):
    run_dir, repo, ws, baseline, model = engine
    args = run_dir / "patch-1" / "check-args.json"
    args.write_text(json.dumps(["--model", "m", "--steps"]))
    assert _ask(run_dir, repo, ws, baseline, _dir(ws / "t1", "fast"))[0] == 0
    assert model.scored == [["--model", "m"]]


def test_a_profile_line_of_perf_is_read_with_its_event():
    line = (
        "3142931 1063349.533525987:    4786240 instructions:      153572e01029 "
        "rangeiter_next (/usr/lib64/libpython3.12.so.1.0)"
    )
    m = inmodel._SAMPLE.match(line)
    assert m and m[4] == "4786240" and m[5] == "instructions" and m[6] == "rangeiter_next"
    assert m[7] == "/usr/lib64/libpython3.12.so.1.0"
    # as perf prints events it had to restrict, or a hybrid part's core PMU
    for printed in ("cycles:u", "cpu_core/cycles/"):
        m = inmodel._SAMPLE.match(line.replace("instructions", printed))
        assert m and inmodel._event(m[5]) == "cycles"


def test_a_trial_cannot_write_outside_the_repository(engine, tmp_path):
    run_dir, repo, ws, baseline, _ = engine
    trial = _dir(ws / "t1", "fast")
    (trial / "link").symlink_to(tmp_path)
    code, out = _ask(run_dir, repo, ws, baseline, trial, build_only=True)
    assert code == 1 and "VERDICT: BAD_TRIAL" in out and "symbolic link" in out, out


def test_validate_builds_the_trial_and_runs_the_repositorys_tests(engine):
    run_dir, repo, ws, baseline, _ = engine
    code, out = _ask(run_dir, repo, ws, baseline, _dir(ws / "t1", "fast"), build_only=True)
    assert code == 0 and "VERDICT: BUILT" in out, out
    code, out = _ask(run_dir, repo, ws, baseline, _dir(ws / "t2", "fast", {"FAIL": "x"}),
                     build_only=True)  # fmt: skip
    assert code == 1 and "BUILD_FAILED" in out and "kernel test failed" in out, out


def test_the_workspace_speaks_of_repository_files_when_trials_run_in_the_model(tmp_path):
    from xe_forge.claude.generator import generate_workspace
    from xe_forge.config import ConfigManager

    cfg = ConfigManager().config
    cfg.device_config.device, cfg.device_config.dsl = "cpu", "cpp"
    cfg.external.kernel_repo, cfg.external.repo_trials = str(tmp_path), True
    cfg.external.benchmark = "inmodel {trial}"
    cfg.external.profile = "inmodel --profile --trial {trial} --workspace {workspace}"
    generate_workspace(tmp_path / "ws", cfg, "k", "", spec_path=None)
    text = (tmp_path / "ws" / "CLAUDE.md").read_text()
    assert "`inmodel --profile --trial <trial dir> --workspace .` builds the trial" in text
    assert "Profile** (MANDATORY every trial)" in text
    assert "trial directories `t<N>/`" in text and "BASELINE_NOT_REPOSITORY" in text
    assert "load_inline" not in text and "self-contained" not in text
    assert "trial finalize k output/k_optimized --require-measured" in text
    assert "perf_enabled: false" in (tmp_path / "ws" / "config.yaml").read_text()
    port = (tmp_path / "ws" / ".claude" / "agents" / "port-back.md").read_text()
    assert "copy each into the clone at the" in port
