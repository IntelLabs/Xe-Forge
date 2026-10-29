"""Tests for scripts/spec_to_roofline_csv.py (flop formula evaluation)."""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "spec_to_roofline_csv.py"
_spec = importlib.util.spec_from_file_location("spec_to_roofline_csv", _SCRIPT)
roofline_csv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(roofline_csv)

eval_formula = roofline_csv.eval_formula

DIMS = {"M": 8, "N": 8, "K": 8, "E": 8, "TOPK": 2, "QUANT": 0}

PAYLOAD = (
    "[c for c in ().__class__.__base__.__subclasses__() if c.__name__=='ModuleSpec']"
    "[0].__init__.__globals__['__import__']('os').system('touch {canary}')"
)


@pytest.mark.parametrize(
    "formula, expected",
    [
        ("M*TOPK*N*K*2", 2048.0),
        ("M + N - K", 8.0),
        ("(M + N) * K / 4", 32.0),
        ("K // 3", 2.0),
        ("K % 3", 2.0),
        ("-M + 2 * N", 8.0),
        ("K ** 2", 64.0),
        ("1.5 * M", 12.0),
        ("  M*N  ", 64.0),
    ],
)
def test_valid_formulas(formula, expected):
    assert eval_formula(formula, DIMS) == expected


@pytest.mark.parametrize(
    "formula",
    [
        "__import__('os').system('true')",
        "().__class__",
        "M.__class__",
        "[1, 2][0]",
        "[x for x in (1,)]",
        "lambda: 1",
        "M if N else K",
        "'abc'",
        "None",
        "True",
        "M < N",
        "abs(M)",
    ],
)
def test_unsupported_syntax_rejected(formula):
    with pytest.raises(ValueError):
        eval_formula(formula, DIMS)


def test_unknown_name_rejected():
    with pytest.raises(ValueError, match="unknown name"):
        eval_formula("M * UNKNOWN", DIMS)


def test_non_numeric_dim_rejected():
    with pytest.raises(ValueError, match="not a number"):
        eval_formula("M", {"M": "8"})


def test_non_string_formula_rejected():
    with pytest.raises(ValueError):
        eval_formula(8, DIMS)


def test_syntax_error_propagates():
    with pytest.raises(SyntaxError):
        eval_formula("M *", DIMS)


def test_huge_exponent_rejected():
    with pytest.raises(ValueError, match="exponent"):
        eval_formula("9 ** 9 ** 9", DIMS)


def test_eval_injection_payload_does_not_execute(tmp_path):
    canary = tmp_path / "canary"
    with pytest.raises(ValueError):
        eval_formula(PAYLOAD.format(canary=canary), DIMS)
    assert not canary.exists()


def test_main_rejects_malicious_spec(tmp_path):
    canary = tmp_path / "canary"
    spec = tmp_path / "evil.yaml"
    out = tmp_path / "out.csv"
    flop = PAYLOAD.format(canary=canary).replace('"', '\\"')
    spec.write_text(
        "bench_moe_v1:\n"
        "  - dims: {M: 8, N: 8, K: 8, E: 8, TOPK: 2, QUANT: 0}\n"
        f'    flop: "{flop}"\n'
    )
    with pytest.raises(ValueError):
        roofline_csv.main([str(spec), "-o", str(out)])
    assert not canary.exists()


def test_main_valid_spec(tmp_path):
    spec = tmp_path / "ok.yaml"
    out = tmp_path / "out.csv"
    spec.write_text(
        "bench_moe_v1:\n"
        "  - dims: {M: 8, N: 8, K: 8, E: 8, TOPK: 2, QUANT: 0}\n"
        '    flop: "M*TOPK*N*K*2"\n'
    )
    assert roofline_csv.main([str(spec), "-o", str(out)]) == 0
    assert "2048" in out.read_text()
