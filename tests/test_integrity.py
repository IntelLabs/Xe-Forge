"""Each integrity check catches the cheat it names, and an honest trial passes all of them."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from xe_forge.core.integrity import check_trial, record_oracle, scan_source  # noqa: E402

GPU = "xpu" if torch.xpu.is_available() else None


def _make_inputs(device):
    def make(seed):
        torch.manual_seed(seed)
        return [torch.randn(64, device=device)]

    return make


def _match(expected, actual):
    return torch.allclose(expected, actual)


def _check(trial, oracle=lambda x: x * 2, device="cpu"):
    make = _make_inputs(device)
    return [
        f.split(":")[0]
        for f in check_trial(record_oracle(oracle, make), trial, make, _match, device)
    ]


def _persistent_buffer():
    buffer = {}

    def trial(x):
        out = buffer.setdefault(x.shape, torch.empty_like(x))
        return out.copy_(x * 2)

    return trial


def _cached_by(key):
    cache = {}

    def trial(x):
        return cache.setdefault(key(x), x * 2)

    return trial


def _wrong_after(calls):
    count = [0]

    def trial(x):
        count[0] += 1
        return x * 2 + (count[0] > calls)

    return trial


def test_an_honest_trial_passes():
    assert _check(lambda x: x + x) == []


@pytest.mark.parametrize(
    "trial, oracle, name",
    [
        (lambda x: x, lambda x: x.clone(), "OUTPUT_ALIASES_INPUT"),
        (_persistent_buffer(), None, "OUTPUT_REUSED"),
        (_cached_by(lambda x: x.shape), None, "STALE_RESULT"),
        (_cached_by(lambda x: x.data_ptr()), None, "CACHED_BY_ADDRESS"),
        # Right for the calls before the repeats, wrong on the last repeat only.
        (_wrong_after(5), None, "NONDETERMINISTIC"),
    ],
)
def test_each_cheat_is_named(trial, oracle, name):
    assert name in _check(trial, **({"oracle": oracle} if oracle else {}))


def test_a_defect_the_oracle_shares_is_not_held_against_the_trial():
    assert _check(_persistent_buffer(), oracle=_persistent_buffer()) == []
    assert _check(lambda x: x, oracle=lambda x: x) == []


def test_harness_access_is_refused_by_name():
    found = scan_source("t = start.elapsed_time(end)\nfrom ai_bench import time\n")
    assert found and found[0].startswith("HARNESS_ACCESS")
    assert "ai_bench" in found[0] and "elapsed_time" in found[0]
    assert scan_source("import torch\nev = q.submit(...)  # sycl::event\n") == []


_HONEST = (
    "import torch\nclass Model(torch.nn.Module):\n    def forward(self, x):\n        return x * 2\n"
)
_REUSES = (
    "import torch\n_buf = {}\nclass Model(torch.nn.Module):\n    def forward(self, x):\n"
    "        out = _buf.setdefault(x.shape, torch.empty_like(x))\n        return out.copy_(x * 2)\n"
)


@pytest.mark.parametrize("trial, failures", [(_HONEST, []), (_REUSES, ["OUTPUT_REUSED"])])
def test_spec_path_runs_the_checks_only_when_asked(trial, failures):
    from xe_forge.core.executor import KernelBenchExecutor

    executor = KernelBenchExecutor(device="cpu")
    kwargs = {"kernel_name": "Model", "input_shapes": [(64,)], "dtype": torch.float32}
    assert executor._check_correctness(_HONEST, trial, **kwargs)
    assert executor.integrity_failures == []
    assert executor._check_correctness(_HONEST, trial, integrity=True, **kwargs) == (not failures)
    assert [f.split(":")[0] for f in executor.integrity_failures] == failures


@pytest.mark.parametrize("cheat", [False, True])
def test_reference_path_reports_integrity_before_timing(cheat):
    from unittest.mock import Mock

    from xe_forge.core.executor import KernelBenchExecutor

    buffer = torch.empty(4)

    def compile_module(self, code):
        class Model(torch.nn.Module):
            def forward(self, values):
                if cheat and code == "trial":
                    return buffer.copy_(values * 2)
                return values * 2

        return SimpleNamespace(Model=Model, get_inputs=lambda: [torch.randn(4)])

    executor = KernelBenchExecutor(device="cpu")
    executor._compile_module = compile_module.__get__(executor)
    executor.time = Mock(return_value=10.0)
    result = executor.compare_reference_workload("reference", "baseline", "trial")
    assert result.optimized_correct is not cheat, result.feedback_message
    assert [f.split(":")[0] for f in executor.integrity_failures] == (
        ["OUTPUT_REUSED"] if cheat else []
    )
    assert executor.time.called is not cheat


def test_the_skill_prints_the_integrity_verdict(tmp_path, capsys):
    from xe_forge.skills import benchmark

    (tmp_path / "baseline.py").write_text(_HONEST)
    (tmp_path / "trial.py").write_text(_HONEST + "# torch.xpu.Event(enable_timing=True)\n")
    args = argparse.Namespace(
        baseline=str(tmp_path / "baseline.py"),
        optimized=str(tmp_path / "trial.py"),
        spec=None,
        reference=str(tmp_path / "baseline.py"),
        variant=None,
        baseline_us=None,
        device="cpu",
    )
    assert benchmark._run_builtin(args) == 1
    out = capsys.readouterr().out
    assert "Correctness: FAILED\nVERDICT: INTEGRITY\nError: HARNESS_ACCESS" in out


@pytest.mark.skipif(GPU is None, reason="needs a GPU caching allocator")
def test_unwritten_output_is_caught_on_the_device():
    def trial(x):
        out = torch.empty_like(x)
        out[:-1] = x[:-1] * 2
        return out

    assert "UNWRITTEN_OUTPUT" in _check(trial, device=GPU)
    assert _check(lambda x: x + x, device=GPU) == []
