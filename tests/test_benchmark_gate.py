"""Which of the two measurements answered, and what happens when neither was asked for.

`xe-forge-skill benchmark` has two paths. One delegates to a command the host
supplied, which times the real workload on whatever timer the host calibrated. The
other generates tensors from the spec's shapes and times them with an unfloored
wall clock. Both print the same two lines, because the trial loop branches on them
-- which is exactly why falling from the first to the second must not be silent: a
session that asked for the host's answer and got the generator's cannot tell from
the output, and every trial it ranks afterwards inherits that.

So the built-in is opted into by name. With neither a host command nor the opt-in,
the skill halts and names both, and prints nothing a reader could take as a timing.
"""

from __future__ import annotations

import argparse
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from xe_forge.config import get_config
from xe_forge.skills import benchmark


def _args(**overrides):
    fields = {
        "baseline": "baseline.py",
        "optimized": "trial.py",
        "spec": "spec.yaml",
        "variant": "bench-gpu",
        "baseline_us": None,
        "device": "xpu",
        "dsl": "triton",
        "triton_baseline": False,
        "external_benchmark": None,
        "builtin_benchmark": False,
    }
    fields.update(overrides)
    return argparse.Namespace(**fields)


@pytest.mark.parametrize("prompt_length,decode_length", [(4, 1), (2, 2)])
def test_decode_reference_attention_layout(monkeypatch, prompt_length, decode_length):
    import runpy
    from pathlib import Path

    import torch

    namespace = runpy.run_path(str(Path(__file__).parents[1] / "scripts/deepseek_decode.py"))
    args = namespace["ModelArgs"](
        max_batch_size=2,
        max_seq_len=6,
        dim=12,
        n_heads=3,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
    )
    frequencies = namespace["precompute_freqs_cis"](args)
    shapes = []

    def explicit_attention(query, key, value, *, scale, is_causal):
        shapes.append((tuple(query.shape), tuple(key.shape), tuple(value.shape)))
        assert not is_causal
        return torch.softmax((query @ key.transpose(-2, -1)) * scale, dim=-1) @ value

    with torch.no_grad():
        model = namespace["Model"](
            args,
            torch.randn(2, prompt_length, 12),
            0,
            frequencies[:prompt_length],
        ).eval()
        inputs = (
            torch.randn(2, decode_length, 12),
            prompt_length,
            frequencies[prompt_length : prompt_length + decode_length],
        )
        actual = model(*inputs)
        monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", explicit_attention)
        expected = model(*inputs)
    assert actual.shape == (2, decode_length, 12)
    assert shapes == [
        (
            (2, 3, decode_length, 8),
            (2, 3, prompt_length + decode_length, 8),
            (2, 3, prompt_length + decode_length, 4),
        )
    ]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("legacy_setting", [True, False])
def test_comparison_always_validates_before_timing(legacy_setting):
    from xe_forge.core.executor import KernelBenchExecutor

    executor = KernelBenchExecutor(device="cpu", require_correctness=legacy_setting)
    executor._check_correctness = Mock(return_value=False)
    executor.execute = Mock(side_effect=AssertionError("Timing must not run"))
    result = executor.compare_kernels("reference", "trial", input_shapes=[(2,)])
    assert not result.optimized_correct
    executor._check_correctness.assert_called_once()
    executor.execute.assert_not_called()


@pytest.mark.parametrize("wrong", [False, True])
@pytest.mark.parametrize("cached", [None, 20.0])
def test_extracted_reference_workload_isolates_inputs_and_gates_timing(wrong, cached):
    import torch

    from xe_forge.core.executor import KernelBenchExecutor
    from xe_forge.core.reference_workload import ReferenceWorkload

    initial = torch.zeros(3)
    inputs = [torch.ones(2), 1, {"phases": torch.tensor([1 + 2j, 3 + 4j])}]

    class Reference(torch.nn.Module):
        def __init__(self, cache):
            super().__init__()
            self.register_buffer("cache", cache, persistent=False)

        def forward(self, values, position, metadata):
            self.cache[position] = values.sum()
            return values * metadata["phases"].real, self.cache.clone()

    class Candidate(Reference):
        def forward(self, *arguments):
            output, cache = super().forward(*arguments)
            return output, cache + int(wrong)

    def compile_module(code):
        return SimpleNamespace(
            Model=Reference if code == "reference" else Candidate,
            get_init_inputs=lambda: [initial],
            get_inputs=lambda: inputs,
        )

    with torch.no_grad():
        workload = ReferenceWorkload.prepare(
            compile_module,
            "reference",
            "baseline",
            "trial",
            "cpu",
            1e-2,
            1e-5,
        )
        workload.original.cache.add_(3)
        assert torch.equal(workload.reference.cache, initial)
        assert torch.equal(workload.optimized.cache, initial)
        assert torch.equal(initial, torch.zeros(3))
        copied = workload.copy_inputs()
        copied[0].zero_()
        copied[2]["phases"].zero_()
        assert torch.equal(workload.inputs[0], torch.ones(2))
        assert torch.equal(workload.inputs[2]["phases"], inputs[2]["phases"])
        assert copied[1] == 1

    executor = KernelBenchExecutor(device="cpu")
    executor._compile_module = compile_module
    executor.time = Mock(return_value=10.0)
    result = executor.compare_reference_workload(
        "reference",
        "baseline",
        "trial",
        baseline_us=cached,
    )
    assert result.optimized_correct is not wrong, result.feedback_message
    if wrong:
        executor.time.assert_not_called()
    else:
        assert executor.time.call_count == (1 if cached else 2)
        assert result.speedup == (2.0 if cached else 1.0)


@pytest.mark.parametrize("fail", [False, True])
def test_reference_models_are_offloaded_between_validation_and_timing(fail):
    import torch

    from xe_forge.core.executor import KernelBenchExecutor

    events = []
    active = set()

    def compile_module(self, code):
        class Model(torch.nn.Module):
            def to(self, device):
                assert not active
                active.add(code)
                events.append((code, "load"))
                return self

            def cpu(self):
                active.discard(code)
                events.append((code, "offload"))
                if fail and code == "trial" and (code, "forward") in events:
                    raise RuntimeError("cleanup failed")
                return self

            def forward(self, values):
                assert active == {code}
                events.append((code, "forward"))
                if fail and code == "trial":
                    raise ValueError("candidate failed")
                return values + 1

        return SimpleNamespace(Model=Model, get_inputs=lambda: [torch.ones(2)])

    executor = KernelBenchExecutor(device="cpu")
    executor._compile_module = compile_module.__get__(executor)

    def timer(model, inputs):
        assert len(active) == 1
        assert events.count(("trial", "forward")) == 2
        events.append((next(iter(active)), "time"))
        model(*inputs)
        return 10.0

    executor.time = Mock(side_effect=timer)
    result = executor.compare_reference_workload("reference", "baseline", "trial")
    assert not active
    assert result.optimized_correct is not fail, result.feedback_message
    if fail:
        assert "Validation failed for optimized: candidate failed" in result.feedback_message
        assert "cleanup failed" not in result.feedback_message
    assert [name for name, event in events if event == "load"] == (
        ["reference", "baseline", "trial"]
        if fail
        else ["reference", "baseline", "trial", "baseline", "trial"]
    )
    assert executor.time.call_count == (0 if fail else 2)


@pytest.mark.parametrize("wrong_update", [False, True])
def test_stateful_reference_resets_before_validation_and_each_timed_call(wrong_update):
    import torch

    from xe_forge.core.executor import KernelBenchExecutor

    starts = []

    def compile_module(self, code):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.tensor([5.0]), persistent=False)

            def forward(self, values):
                starts.append((code, self.cache.item()))
                self.cache.add_(values + int(wrong_update and code == "trial"))
                return values * 2

        return SimpleNamespace(Model=Model, get_inputs=lambda: [torch.ones(1)])

    executor = KernelBenchExecutor(device="cpu")
    executor._compile_module = compile_module.__get__(executor)

    def timer(call, inputs):
        call(*inputs)
        call(*inputs)
        call(*inputs)
        return 10.0

    executor.time = Mock(side_effect=timer)
    result = executor.compare_reference_workload("reference", "baseline", "trial")
    assert result.optimized_correct is not wrong_update, result.feedback_message
    assert all(value == 5.0 for code, value in starts)
    if wrong_update:
        executor.time.assert_not_called()
        assert "Updated buffers differ" in result.feedback_message
    else:
        assert executor.time.call_count == 2
        assert "reset plus forward" in result.feedback_message
        assert len(starts) == 12


@pytest.fixture(autouse=True)
def _no_host_command(monkeypatch):
    """No external command from the environment unless a test puts one there."""
    monkeypatch.delenv("EXTERNAL_BENCHMARK", raising=False)
    monkeypatch.delenv(benchmark.BUILTIN_ENV, raising=False)
    get_config(reload=True)
    yield
    get_config(reload=True)


@pytest.mark.parametrize("cached", [None, 20.0])
@pytest.mark.parametrize("correct", [True, False])
def test_builtin_reference_without_spec(tmp_path, monkeypatch, capsys, cached, correct):
    import torch

    from xe_forge.core.executor import KernelBenchExecutor

    class Reference(torch.nn.Module):
        def __init__(self, initial):
            super().__init__()
            self.register_buffer("cache", initial, persistent=False)

        def forward(self, values, position, phases):
            self.cache[position] = values.sum()
            return values * phases.real, self.cache.clone()

    class Candidate(Reference):
        def forward(self, *arguments):
            values, cache = super().forward(*arguments)
            return values + int(not correct), cache

    def compile_module(self, code):
        return SimpleNamespace(
            Model=Reference if code == "reference" else Candidate,
            get_init_inputs=lambda: [torch.zeros(3)],
            get_inputs=lambda: [torch.ones(2), 1, torch.tensor([1 + 2j, 3 + 4j])],
        )

    paths = {}
    for name in ("reference", "baseline", "optimized"):
        path = tmp_path / f"{name}.py"
        path.write_text(name)
        paths[name] = str(path)
    timer = Mock(return_value=10.0)
    monkeypatch.setattr(KernelBenchExecutor, "_compile_module", compile_module)
    monkeypatch.setattr(KernelBenchExecutor, "time", timer)
    code = benchmark._run_builtin(
        _args(
            **paths,
            spec=None,
            variant=None,
            device="cpu",
            baseline_us=cached,
        )
    )
    assert code == (0 if correct else 1)
    output = capsys.readouterr().out
    assert ("Performance:" in output) is correct
    assert ("Correctness: PASSED" in output) is correct
    assert ("TIMER: device_buffer_reset_plus_forward" in output) is correct
    if not correct:
        timer.assert_not_called()
    else:
        assert timer.call_count == (1 if cached else 2)


def _reference_with_spec(tmp_path, monkeypatch, spec_text, scale=1.0):
    """A reference whose workload is built from spec dims; the candidate scales its output."""
    import torch

    from xe_forge.core.executor import KernelBenchExecutor

    class Reference(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.width = width

        def forward(self, values):
            assert values.shape[-1] == self.width
            return values

    class Candidate(Reference):
        def forward(self, values):
            return super().forward(values) * scale

    shapes = []

    def get_inputs(ROWS=1, WIDTH=4, dtype=torch.float32):
        shapes.append((ROWS, WIDTH, dtype))
        return [torch.ones(ROWS, WIDTH, dtype=dtype)]

    def compile_module(self, code):
        return SimpleNamespace(
            Model=Reference if code == "reference" else Candidate,
            get_init_inputs=lambda WIDTH=4: [WIDTH],
            get_inputs=get_inputs,
        )

    paths = {}
    for name in ("reference", "baseline", "optimized"):
        path = tmp_path / f"{name}.py"
        path.write_text(name)
        paths[name] = str(path)
    spec = tmp_path / "decode.yaml"
    spec.write_text(spec_text)
    timer = Mock(return_value=10.0)
    monkeypatch.setattr(KernelBenchExecutor, "_compile_module", compile_module)
    monkeypatch.setattr(KernelBenchExecutor, "time", timer)
    return paths, str(spec), shapes, timer


_SPEC = """
ci:
  - {dims: {ROWS: 2, WIDTH: 8}, dtype: float16, rtol: 0.05, atol: 0.0}
bench-gpu:
  - {dims: {ROWS: 3, WIDTH: 16}, flop: "ROWS*WIDTH*1000", bytes: "ROWS*WIDTH*2000"}
"""


@pytest.mark.parametrize("variant,shape", [("ci", (2, 8, "float16")), (None, (3, 16, "float32"))])
def test_spec_variant_builds_the_reference_workload(tmp_path, monkeypatch, capsys, variant, shape):
    paths, spec, shapes, _ = _reference_with_spec(tmp_path, monkeypatch, _SPEC)
    code = benchmark._run_builtin(_args(**paths, spec=spec, variant=variant, device="cpu"))
    out = capsys.readouterr().out
    assert code == 0, out
    assert {(rows, width, str(dtype).split(".")[-1]) for rows, width, dtype in shapes} == {shape}
    assert f"Variant: {variant or 'bench-gpu'}" in out
    # 48000 FLOP and 96000 bytes in 10 us.
    assert (
        "Throughput: baseline_tflops=0.005, kernel_tflops=0.005, "
        "baseline_gbs=9.6, kernel_gbs=9.6" in out
    ) is (variant is None)


@pytest.mark.parametrize("variant,passes", [("ci", True), ("bench-gpu", False)])
def test_spec_variant_supplies_tolerances(tmp_path, monkeypatch, capsys, variant, passes):
    paths, spec, _, timer = _reference_with_spec(tmp_path, monkeypatch, _SPEC, scale=1.03)
    code = benchmark._run_builtin(_args(**paths, spec=spec, variant=variant, device="cpu"))
    assert code == (0 if passes else 1)
    assert timer.called is passes


@pytest.mark.parametrize(
    "spec_text,error",
    [
        ("bench-gpu:\n  - dims: {ROWS: 2, DEPTH: 3}\n", "spec dim DEPTH is not a parameter"),
        ("bench-gpu:\n  - dims: {ROWS: 2}\n", "Unknown spec variant: ci"),
    ],
)
def test_spec_refuses_a_workload_the_reference_would_not_run(
    tmp_path, monkeypatch, capsys, spec_text, error
):
    paths, spec, _, timer = _reference_with_spec(tmp_path, monkeypatch, spec_text)
    variant = "ci" if "Unknown" in error else None
    code = benchmark._run_builtin(_args(**paths, spec=spec, variant=variant, device="cpu"))
    out = capsys.readouterr().out
    assert code == 1
    assert error in out
    assert "Performance:" not in out
    timer.assert_not_called()


@pytest.mark.parametrize("steps,match", [(0, True), (1, True), (3, False)])
def test_float8_buffers_compare_within_one_step(steps, match):
    import torch

    from xe_forge.core.reference_workload import ReferenceWorkload

    reference = torch.tensor([1.0, -2.0, 0.5]).to(torch.float8_e4m3fn)
    candidate = (torch.tensor([1.0, -2.0, 0.5]) * (1 + 0.125 * steps)).to(torch.float8_e4m3fn)
    workload = ReferenceWorkload(None, None, None, None, rtol=1e-2, atol=1e-5)
    assert workload._values_match(reference, candidate) is match
    if not match:
        with pytest.raises(ValueError, match="differ"):
            workload._assert_values_match(reference, candidate, "Updated buffers differ")


@pytest.mark.parametrize("reference,variant", [(None, None), ("reference.py", "bench-gpu")])
def test_no_spec_requires_reference_and_rejects_variant(capsys, reference, variant):
    code = benchmark._run_builtin(_args(spec=None, reference=reference, variant=variant))
    assert code == 1
    assert "Performance:" not in capsys.readouterr().out


@pytest.fixture
def _paths_not_taken(monkeypatch):
    """Both measurement paths replaced by markers, so a test sees which was chosen."""
    taken = []
    monkeypatch.setenv(benchmark._CHILD_ENV, "1")
    monkeypatch.setattr(benchmark, "_run_builtin", lambda args: taken.append("builtin") or 0)
    monkeypatch.setattr(
        benchmark, "_run_external", lambda args, template: taken.append(template) or 0
    )
    return taken


def test_neither_configured_halts(capsys, _paths_not_taken):
    with pytest.raises(SystemExit) as exit_info:
        benchmark.run(_args())

    assert exit_info.value.code != 0
    assert _paths_not_taken == []
    out = capsys.readouterr().out
    assert "VERDICT: NO_BENCHMARK_CONFIGURED" in out
    assert "EXTERNAL_BENCHMARK" in out
    assert "--builtin-benchmark" in out
    assert benchmark.BUILTIN_ENV in out


def test_the_halt_prints_no_timing(capsys, _paths_not_taken):
    """No line the trial loop can parse as a measurement or a passing kernel."""
    with pytest.raises(SystemExit):
        benchmark.run(_args())

    out = capsys.readouterr().out
    assert "Performance:" not in out
    assert "speedup" not in out
    assert "Correctness: PASSED" not in out


def test_the_flag_runs_the_builtin(_paths_not_taken):
    with pytest.raises(SystemExit) as exit_info:
        benchmark.run(_args(builtin_benchmark=True))

    assert exit_info.value.code == 0
    assert _paths_not_taken == ["builtin"]


def test_a_hung_builtin_is_killed_with_a_verdict(monkeypatch, capsys):
    """A kernel that never completes blocks in the driver; the parent ends it and says so."""
    import subprocess

    popen = subprocess.Popen
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: popen(["sleep", "60"], **kw))
    monkeypatch.setattr(get_config().external, "timeout", 1)

    assert benchmark._run_builtin_watched() == 1
    out = capsys.readouterr().out
    assert "VERDICT: TIMEOUT" in out
    assert "Performance:" not in out


@pytest.mark.parametrize("value", ["1", "true", "yes", "ON"])
def test_the_env_var_runs_the_builtin(monkeypatch, _paths_not_taken, value):
    monkeypatch.setenv(benchmark.BUILTIN_ENV, value)
    with pytest.raises(SystemExit) as exit_info:
        benchmark.run(_args())

    assert exit_info.value.code == 0
    assert _paths_not_taken == ["builtin"]


@pytest.mark.parametrize("value", ["", "0", "no", "later"])
def test_an_unset_looking_env_var_is_not_an_opt_in(monkeypatch, _paths_not_taken, value):
    monkeypatch.setenv(benchmark.BUILTIN_ENV, value)
    with pytest.raises(SystemExit) as exit_info:
        benchmark.run(_args())

    assert exit_info.value.code != 0
    assert _paths_not_taken == []


def test_the_host_command_still_wins(_paths_not_taken):
    """The path that already worked is unchanged, opt-in or not."""
    with pytest.raises(SystemExit) as exit_info:
        benchmark.run(_args(external_benchmark="host-bench {trial}"))

    assert exit_info.value.code == 0
    assert _paths_not_taken == ["host-bench {trial}"]


def test_the_host_command_from_the_environment_still_wins(monkeypatch, _paths_not_taken):
    monkeypatch.setenv("EXTERNAL_BENCHMARK", "host-bench {trial}")
    get_config(reload=True)
    with pytest.raises(SystemExit) as exit_info:
        benchmark.run(_args())

    assert _paths_not_taken == ["host-bench {trial}"]
    assert exit_info.value.code == 0


def test_the_generated_workspace_asks_for_what_it_will_get(tmp_path):
    """A builtin workspace tells the session to pass the flag; a host one does not."""
    from xe_forge.claude.generator import generate_workspace
    from xe_forge.config import Config

    for external, expected in (("host-bench {trial}", False), (None, True)):
        config = Config()
        config.external.benchmark = external
        ws = tmp_path / ("host" if external else "builtin")
        generate_workspace(ws, config, "k", "// kernel\n", reference_code="x = 1\n")
        claude_md = (ws / "CLAUDE.md").read_text()
        runner = (ws / ".claude" / "agents" / "tool-runner.md").read_text()
        assert ("--builtin-benchmark" in claude_md) is expected
        assert ("--builtin-benchmark" in runner) is expected


@pytest.mark.parametrize(
    "outputs_match,execution_success,expected_code",
    [(False, True, 1), (True, True, 0), (True, False, 1)],
)
def test_cached_baseline_requires_reference_correctness(
    tmp_path, monkeypatch, capsys, outputs_match, execution_success, expected_code
):
    from xe_forge.core import spec_loader

    baseline = tmp_path / "baseline.py"
    optimized = tmp_path / "trial.py"
    baseline.write_text("baseline source")
    optimized.write_text("trial source")
    spec = Mock()
    spec.resolve_variant.return_value = "bench-gpu"
    spec.get_input_shapes.return_value = [(8,)]
    spec.get_flop.return_value = 8
    spec.get_dtype.return_value = "float32"
    spec.get_input_dtypes.return_value = ["float32"]
    spec.get_init_args.return_value = [8]
    monkeypatch.setattr(spec_loader, "load_spec", Mock(return_value=spec))

    executor = Mock()
    executor._check_correctness.return_value = outputs_match
    executor.execute.return_value = SimpleNamespace(
        success=execution_success, execution_time_ms=0.005, error_message="execution failed"
    )
    executor_module = ModuleType("xe_forge.core.executor")
    executor_module.KernelBenchExecutor = Mock(return_value=executor)
    monkeypatch.setitem(sys.modules, "xe_forge.core.executor", executor_module)

    result = benchmark._run_builtin(
        _args(baseline=str(baseline), optimized=str(optimized), baseline_us="10,20")
    )

    assert result == expected_code
    executor._check_correctness.assert_called_once_with(
        original_code="baseline source",
        optimized_code="trial source",
        kernel_name="Model",
        input_shapes=[(8,)],
        dtype="float32",
        init_args=[8],
        input_dtypes=["float32"],
    )
    executor.compare_kernels.assert_not_called()
    if outputs_match:
        executor.execute.assert_called_once_with(
            "trial source",
            None,
            [(8,)],
            flop=8,
            dtype="float32",
            init_args=[8],
            input_dtypes=["float32"],
        )
    else:
        executor.execute.assert_not_called()
    output = capsys.readouterr().out
    if expected_code == 0:
        assert "Correctness: PASSED" in output
        assert "baseline_us=15.00, kernel_us=5.00, speedup=3.00x" in output
    else:
        assert "Correctness: FAILED" in output
        assert "Correctness: PASSED" not in output
        assert "Performance:" not in output


@pytest.mark.parametrize("with_spec", [False, True])
def test_generated_commands_share_semantic_reference(tmp_path, with_spec):
    from xe_forge.claude.generator import generate_workspace
    from xe_forge.config import Config

    config = Config()
    config.device_config.dsl = "sycl"
    config.device_config.device = "cpu"
    spec = tmp_path / "source.yaml"
    spec.write_text("name: decode\n")
    workspace = tmp_path / "workspace"
    generate_workspace(
        workspace,
        config,
        "decode",
        "",
        reference_code="reference source",
        spec_path=str(spec) if with_spec else None,
    )
    for path in (
        "CLAUDE.md",
        ".claude/commands/optimize-kernel.md",
        ".claude/agents/tool-runner.md",
    ):
        text = (workspace / path).read_text()
        commands = [line for line in text.splitlines() if "xe-forge-skill benchmark <" in line]
        assert commands
        for command in commands:
            assert "--reference test_kernels/decode_pytorch.py" in command
            assert "--device cpu --dsl sycl" in command
            assert "--builtin-benchmark" in command
            assert ("--spec test_kernels/decode.yaml" in command) is with_spec
    assert (workspace / "test_kernels/decode_pytorch.py").read_text() == "reference source"
    assert (workspace / "test_kernels/decode.yaml").exists() is with_spec
    instructions = (workspace / "CLAUDE.md").read_text()
    assert ("variant supplies the dims" in instructions) is with_spec
    assert "benchmark test_kernels/decode.cpp test_kernels/decode.cpp" in instructions
    assert "Do not create optimized trials before this gate" in instructions
    assert "correctness failure rejects the candidate, not the optimization session" in instructions
    assert "distinct SYCL kernel identities" in instructions
    assert "does not isolate device kernels" in (workspace / "test_kernels/decode.cpp").read_text()
    for path in (workspace / "CLAUDE.md", workspace / ".claude/agents/tool-runner.md"):
        assert (
            "xe-forge-skill profile <kernel_file> --reference test_kernels/decode_pytorch.py"
            in path.read_text()
        )
        assert (
            "--reference test_kernels/decode_pytorch.py --spec test_kernels/decode.yaml"
            in path.read_text()
        ) is with_spec


def test_without_a_reference_the_baseline_copy_is_the_oracle(tmp_path):
    from xe_forge.claude.generator import generate_workspace
    from xe_forge.config import Config

    config = Config()
    config.device_config.dsl = "triton"
    config.device_config.device = "xpu"
    config.external.kernel_repo = str(tmp_path / "repo")
    spec = tmp_path / "source.yaml"
    spec.write_text("name: decode\n")
    workspace = tmp_path / "workspace"
    generate_workspace(workspace, config, "decode", "", spec_path=str(spec), variant_type="bench-x")

    for path in ("CLAUDE.md", ".claude/agents/tool-runner.md"):
        text = (workspace / path).read_text()
        for skill in ("benchmark <", "profile <"):
            commands = [line for line in text.splitlines() if f"xe-forge-skill {skill}" in line]
            assert commands
            for command in commands:
                assert "--reference test_kernels/decode.py" in command
                assert "--variant bench-x" in command
    instructions = (workspace / "CLAUDE.md").read_text()
    assert "There is no PyTorch reference" in instructions
    assert "decode_pytorch.py" not in instructions
    locator = (workspace / ".claude/agents/kernel-locator.md").read_text()
    assert "you write none" in locator
    assert "decode_pytorch.py`. Nothing else" not in locator
    assert "[--variant <name>]" in instructions and "default to variant `bench-x`" in instructions
    assert (workspace / ".claude/agents/port-back.md").exists()
    assert "dispatch the **port-back** agent" in instructions


def test_spec_with_inputs_keeps_its_own_path(tmp_path):
    from xe_forge.claude.generator import generate_workspace
    from xe_forge.config import Config

    config = Config()
    config.device_config.dsl = "sycl"
    config.device_config.device = "cpu"
    spec = tmp_path / "source.yaml"
    spec.write_text(
        "inputs:\n  X: {shape: [N], dtype: float32}\nbench-gpu:\n  - {params: [X], dims: {N: 4}}\n"
    )
    workspace = tmp_path / "workspace"
    generate_workspace(
        workspace, config, "decode", "", reference_code="reference source", spec_path=str(spec)
    )
    instructions = (workspace / "CLAUDE.md").read_text()
    assert "immutable semantic reference" not in instructions
    assert "xe-forge-skill profile <kernel_file> --spec test_kernels/decode.yaml" in instructions


def test_time_forward_off_gpu_times_the_whole_call(monkeypatch):
    from unittest.mock import Mock

    from xe_forge.core.executor import KernelBenchExecutor

    timer = Mock(return_value=10.0)
    monkeypatch.setattr(KernelBenchExecutor, "time", timer)
    call, args = object(), (1,)

    assert KernelBenchExecutor(device="cpu").time_forward(call, args) == 10.0
    timer.assert_called_once_with(call, args)


@pytest.mark.parametrize(
    "times,missing",
    [
        ({"trial_us": 10.0}, "BASELINE_US"),
        ({"baseline_us": 15.0}, "TRIAL_US"),
        ({}, "BASELINE_US, TRIAL_US"),
    ],
)
def test_a_speedup_without_both_times_is_refused(monkeypatch, capsys, times, missing):
    """A ratio the host could not back with both times prints no Performance line."""
    import xe_forge.external as external

    monkeypatch.setattr(
        external,
        "run_external",
        lambda template, **kw: external.ExternalResult(correctness=True, speedup=1.5, **times),
    )
    assert benchmark._run_external(_args(), "host-bench {trial}") == 1
    out = capsys.readouterr().out
    assert "VERDICT: INCOMPLETE_TIMING" in out
    assert f"no {missing}" in out
    assert "Performance:" not in out


def test_the_generated_workspace_profiles_t0_and_ports_back_in_its_clone(tmp_path):
    from xe_forge.claude.generator import generate_workspace
    from xe_forge.config import Config

    config = Config()
    config.external.kernel_repo = str(tmp_path / "repo")
    ws = tmp_path / "ws"
    generate_workspace(ws, config, "k", "// kernel\n", reference_code="x = 1\n")
    claude_md = (ws / "CLAUDE.md").read_text()
    assert "**Profile**" in claude_md
    assert "t1 or later" not in claude_md
    port_back = (ws / ".claude" / "agents" / "port-back.md").read_text()
    assert "are only read" not in port_back
    assert "private clone's\n   working tree" in port_back


def test_compiler_flags_reach_the_seed_and_config_whole(tmp_path):
    """A quoted flag stays one argument, and quotes survive into config.yaml."""
    import ast

    import yaml

    from xe_forge.claude.generator import generate_workspace
    from xe_forge.config import Config

    flags = "-O3 -DNAME='hello world' -DQ=\"x\""
    config = Config()
    config.device_config.dsl = "sycl"
    config.external.benchmark = None
    config.engine.compiler_flags = flags
    ws = tmp_path / "ws"
    generate_workspace(ws, config, "k", "", reference_code="x = 1\n")

    seed = next((ws / "test_kernels").glob("k.*")).read_text()
    line = next(ln for ln in seed.splitlines() if ln.startswith("_EXTRA_SYCL_CFLAGS ="))
    assert ast.literal_eval(line.split("=", 1)[1].strip()) == ["-O3", "-DNAME=hello world", "-DQ=x"]
    assert yaml.safe_load((ws / "config.yaml").read_text())["compiler_flags"] == flags


def test_a_relative_kernel_repo_is_rendered_absolute(tmp_path, monkeypatch):
    from xe_forge.claude.generator import generate_workspace
    from xe_forge.config import Config

    (tmp_path / "repo").mkdir()
    monkeypatch.chdir(tmp_path)
    config = Config()
    config.external.kernel_repo = "repo"
    generate_workspace(tmp_path / "ws", config, "k", "// kernel\n", reference_code="x = 1\n")
    locator = (tmp_path / "ws" / ".claude" / "agents" / "kernel-locator.md").read_text()
    assert f"`{(tmp_path / 'repo').resolve()}`" in locator


def test_a_second_session_does_not_reseed_the_lessons_ledger(tmp_path):
    from xe_forge.claude.generator import generate_workspace
    from xe_forge.config import Config

    config = Config()
    config.external.lessons = str(tmp_path / "lessons")
    generate_workspace(tmp_path / "ws1", config, "k", "// kernel\n", reference_code="x = 1\n")
    ledger = next((tmp_path / "lessons").iterdir())
    ledger.write_text(ledger.read_text() + "\nentry from session one\n")
    generate_workspace(tmp_path / "ws2", config, "k", "// kernel\n", reference_code="x = 1\n")
    assert ledger.read_text().endswith("entry from session one\n")


def test_spec_tolerances_decide_correctness_on_the_spec_path(tmp_path, monkeypatch):
    """A variant's rtol/atol reach the executor whether or not a reference was given."""
    from xe_forge.core import spec_loader

    baseline, optimized = tmp_path / "baseline.py", tmp_path / "trial.py"
    baseline.write_text("baseline source")
    optimized.write_text("trial source")
    spec = Mock()
    spec.inputs = {"X": object()}
    spec.resolve_variant.return_value = "bench-gpu"
    spec.get_rtol.return_value = 0.02
    spec.get_atol.return_value = 0.03
    monkeypatch.setattr(spec_loader, "load_spec", Mock(return_value=spec))

    executor = SimpleNamespace(rtol=1e-2, atol=1e-5)
    seen = {}

    def compare_kernels(**_kwargs):
        seen.update(rtol=executor.rtol, atol=executor.atol)
        return SimpleNamespace(
            original_correct=True,
            optimized_correct=True,
            original_time_us=2.0,
            optimized_time_us=1.0,
            speedup=2.0,
            feedback_message="",
        )

    executor.compare_kernels = compare_kernels
    executor_module = ModuleType("xe_forge.core.executor")
    executor_module.KernelBenchExecutor = Mock(return_value=executor)
    monkeypatch.setitem(sys.modules, "xe_forge.core.executor", executor_module)

    assert benchmark._run_builtin(_args(baseline=str(baseline), optimized=str(optimized))) == 0
    assert seen == {"rtol": 0.02, "atol": 0.03}


@pytest.mark.parametrize("integrated", [False, True])
def test_every_win_ports_and_an_integration_repo_widens_what_may_change(tmp_path, integrated):
    from xe_forge.claude.generator import generate_workspace
    from xe_forge.config import Config

    config = Config()
    config.external.kernel_repo = str(tmp_path / "kernels")
    if integrated:
        config.external.integration_repos = [str(tmp_path / "vllm")]
    ws = tmp_path / "ws"
    generate_workspace(ws, config, "k", "// kernel\n", reference_code="x = 1\n")
    claude_md = (ws / "CLAUDE.md").read_text()
    port_back = (ws / ".claude" / "agents" / "port-back.md").read_text()
    assert "every win must port back" in claude_md and "Removing a launch is in scope" in claude_md
    assert ("serving-only" in claude_md) is integrated
    assert (str(tmp_path / "vllm") in port_back) is integrated
    # upstream is tried first, serving-only second, none only when neither applies.
    assert (
        port_back.index("**upstream**")
        < port_back.index("**serving-only**")
        < port_back.index("**none**")
    )
