"""A kernel's trials are scored on the shapes and weights an agent chose: the spec's
``weights``, checked by ``inputs check``, enforced by ``benchmark`` over every shape."""

from __future__ import annotations

import argparse
import re

import pytest
import yaml

from xe_forge.core.spec_loader import load_spec
from xe_forge.skills import benchmark, inputs

SPEC = {"default_variant": "dec", "weights": {"dec": 0.75, "pre": 0.25},
        "ci": [{"dims": {"ROWS": 1}}], "dec": [{"dims": {"ROWS": 4}}],
        "pre": [{"dims": {"ROWS": 2048}}]}  # fmt: skip


def _args(tmp_path, **kw):
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump(SPEC))
    return argparse.Namespace(spec=str(spec), variant=None, baseline_us=None, **kw)


def _shape(results):
    """One shape's benchmark, as the single-shape path prints it."""

    def single(args):
        ok, base, kern, verdict = results[args.variant]
        print(f"Variant: {args.variant}\nCorrectness: {'PASSED' if ok else 'FAILED'}")
        if not ok:
            print("Error: mismatch")
            return 1
        print("TIMER: t")
        ratio = "none" if verdict else f"{base / kern:.2f}x"
        print(f"Performance: baseline_us={base:.2f}, kernel_us={kern:.2f}, speedup={ratio}")
        if verdict:
            print(f"VERDICT: {verdict}")
        return 0

    return single


def _gate(tmp_path, capsys, results):
    args = _args(tmp_path)
    code = benchmark._weighted(args, lambda: _shape(results)(args))
    text = capsys.readouterr().out
    return code, text, benchmark.parse_outcome(text), args


def test_the_spec_carries_weights_and_old_specs_none(tmp_path):
    assert load_spec(_args(tmp_path).spec).weights == {"dec": 0.75, "pre": 0.25}
    (tmp_path / "old.yaml").write_text(yaml.safe_dump({"ci": [{"dims": {"ROWS": 1}}]}))
    assert load_spec(tmp_path / "old.yaml").weights == {}


def test_the_score_is_the_time_weighted_total(tmp_path, capsys):
    code, text, out, args = _gate(tmp_path, capsys, {"dec": (True, 10, 5, None),
                                                     "pre": (True, 100, 80, None)})  # fmt: skip
    assert code == 0 and out["correctness"] == "pass"
    assert out["speedup"] == pytest.approx(1 / (0.75 / 2 + 0.25 / 1.25), abs=0.01)
    assert "verdict" not in out and args.variant is None  # pinned to the set, not one shape
    assert len(re.findall(r"^Performance:", text, re.M)) == 1  # per-shape lines are indented


def test_a_shape_measured_slower_is_a_regression_whatever_the_others_gain(tmp_path, capsys):
    _, text, out, _ = _gate(tmp_path, capsys, {"dec": (True, 10, 2, None),
                                               "pre": (True, 100, 125, None)})  # fmt: skip
    assert out["speedup"] == pytest.approx(0.8) and out["verdict"] == "SLOWER_ON"
    assert "Slower on: pre" in text


def test_a_gated_shape_counts_as_parity(tmp_path, capsys):
    _, _, out, _ = _gate(tmp_path, capsys, {"dec": (True, 10, 10, "INDISTINGUISHABLE"),
                                            "pre": (True, 100, 100, "BELOW_FLOOR")})  # fmt: skip
    assert "speedup" not in out and out["verdict"] == "INDISTINGUISHABLE"
    _, _, out, _ = _gate(tmp_path, capsys, {"dec": (True, 10, 5, None),
                                            "pre": (True, 100, 100, "INDISTINGUISHABLE")})  # fmt: skip
    assert out["speedup"] == pytest.approx(1 / (0.75 / 2 + 0.25), abs=0.01)


def test_a_shape_that_fails_correctness_ends_the_trial(tmp_path, capsys):
    code, text, out, _ = _gate(tmp_path, capsys, {"dec": (True, 10, 5, None),
                                                  "pre": (False, 0, 0, None)})  # fmt: skip
    assert code == 1 and out["correctness"] == "fail" and "Variant: pre" in text


def test_one_variant_asked_for_is_one_shape(tmp_path, capsys):
    args = _args(tmp_path)
    args.variant = "pre"
    benchmark._weighted(args, lambda: _shape({"pre": (True, 100, 50, None)})(args))
    assert benchmark.parse_outcome(capsys.readouterr().out)["speedup"] == 2.0


def _manifest(tmp_path):
    """One kernel serving two configurations (N=896 and N=4864), each at 4 and 2048 rows."""
    from types import SimpleNamespace

    from xe_forge.frontend import ir, manifest
    from xe_forge.frontend.analyze import Entry
    from xe_forge.frontend.ir import CaptureRun, Naming, Workload

    def w(wid, n, dec, pre):
        return Workload(workload_id=wid, family="gemm", dims={"M": "var", "N": n, "K": 896},
                        dtypes=["float16"], layout=[], attrs={}, calls=20, device_us=dec + pre,
                        share=0.5, var_axes={"M": {"4": {"calls": 19, "device_us": dec},
                                                   "2048": {"calls": 1, "device_us": pre}}},
                        ops=[], naming=Naming(route="kernel"))  # fmt: skip

    ws = [w("gemm/a", 896, 30.0, 20.0), w("gemm/b", 4864, 30.0, 20.0)]
    e = Entry(name="mm", repo="r", dsl="sycl", workloads=ws, decisions=[], enabled=True)
    cap = CaptureRun(run={"model": {"name": "m"}}, ops=[], workloads=ws)
    ir.dump(cap, tmp_path / "capture.json")
    return manifest, SimpleNamespace(entries=[e], run_id="r1"), cap


def test_every_captured_shape_is_a_variant_weighted_by_its_time_or_the_agents(tmp_path):
    manifest, analysis, cap = _manifest(tmp_path)
    path = manifest.emit(analysis, cap, tmp_path / "capture.json", tmp_path / "out")
    spec = load_spec(tmp_path / "out" / "specs" / "mm.yaml")
    assert spec.weights == {"bench-t4-w0": 0.3, "bench-t4-w1": 0.3, "bench-t2048-w0": 0.2,
                            "bench-t2048-w1": 0.2}  # fmt: skip
    chosen = {"mm": {"bench-t2048-w0": 0.5, "bench-t2048-w1": 0.5}}
    manifest.emit(analysis, cap, tmp_path / "capture.json", tmp_path / "out", chosen)
    assert load_spec(tmp_path / "out" / "specs" / "mm.yaml").weights == chosen["mm"]
    assert yaml.safe_load(path.read_text())["kernels"][0]["variants"] == list(chosen["mm"])
    manifest.emit(analysis, cap, tmp_path / "capture.json", tmp_path / "out", {})
    entry = yaml.safe_load(path.read_text())["kernels"][0]
    assert not entry["enabled"] and "no checked trial shapes" in entry["reason"]


def _g(*vs, w):
    return {"stands_for": list(vs), "weight": w}


GOOD = {"bench-t4-w0": _g("bench-t4-w0", "bench-t2048-w0", w=0.5),
        "bench-t4-w1": _g("bench-t4-w1", "bench-t2048-w1", w=0.5)}  # fmt: skip


@pytest.mark.parametrize(
    "shapes, problem",
    [(GOOD, None),
     ({"bench-t4-w0": _g("bench-t4-w0", "bench-t2048-w0", "bench-t4-w1", "bench-t2048-w1",
                         w=1.0)}, "different configurations"),
     ({"bench-t4-w0": _g("bench-t4-w0", "bench-t2048-w0", w=1.0)},
      "no shape stands for bench-t4-w1, bench-t2048-w1"),
     ({**GOOD, "bench-t2048-w1": _g("bench-t2048-w1", w=0.1)}, "stands with both"),
     ({"bench-t4-w0": _g("bench-t2048-w0", w=0.5), "bench-t4-w1": GOOD["bench-t4-w1"]},
      "must be in its own stands_for"),
     ({"bench-t9-w0": _g("bench-t9-w0", w=1.0)}, "not a shape the capture ran"),
     ({k: {**g, "weight": 0.4} for k, g in GOOD.items()}, "sum to 0.8000")],
)  # fmt: skip
def test_inputs_check_holds_every_configuration(tmp_path, shapes, problem):
    from xe_forge.frontend import manifest as m

    manifest, analysis, cap = _manifest(tmp_path)
    path = manifest.emit(analysis, cap, tmp_path / "capture.json", tmp_path / "out")
    problems, lines = inputs.check({"mm": {"shapes": shapes, "why": "x"}}, m.load(path))
    if problem is None:
        assert not problems and "2 shapes stand for its 4 captured" in lines[0], problems
    else:
        assert any(problem in p for p in problems.get("mm", [])), problems


@pytest.mark.parametrize("good", [True, False])
def test_run_optimizes_a_kernel_only_on_shapes_that_pass(tmp_path, monkeypatch, good):
    from types import SimpleNamespace

    from xe_forge.frontend import cli

    manifest, analysis, cap = _manifest(tmp_path)
    path = manifest.emit(analysis, cap, tmp_path / "capture.json", tmp_path / "out")
    shapes = GOOD if good else {"bench-t4-w0": GOOD["bench-t4-w0"]}
    answer = {"mm": {"shapes": shapes, "why": "decode and prefill share a path"}}
    monkeypatch.setattr(cli, "_agent", lambda *a, **k: answer)
    got = cli.inputs(path, tmp_path / "out", SimpleNamespace(model="m", framework="fw"), "py",
                     5, tmp_path)  # fmt: skip
    assert got == ({"mm": {"bench-t4-w0": 0.5, "bench-t4-w1": 0.5}} if good else {})


@pytest.mark.parametrize("weighted", [True, False])
def test_a_weighted_spec_benchmarks_every_shape_without_a_variant(tmp_path, weighted):
    from xe_forge.claude.generator import generate_workspace
    from xe_forge.config import Config

    spec = dict(SPEC) if weighted else {k: v for k, v in SPEC.items() if k != "weights"}
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec))
    config = Config()
    config.external.benchmark = "host-bench {trial}"
    ws = tmp_path / "ws"
    generate_workspace(ws, config, "rms", "// kernel\n", reference_code="x = 1\n",
                       spec_path=str(tmp_path / "spec.yaml"), variant_type="dec")  # fmt: skip
    text = (ws / "CLAUDE.md").read_text()
    assert ("--variant dec" not in text.split("| **Benchmark**")[1].split("\n")[0]) is weighted
    assert ("`dec` (weight 0.75), `pre` (weight 0.25)" in text) is weighted
