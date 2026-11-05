"""The native engine's measuring tools: reading a model's facts from a GGUF header."""

import struct

import pytest

from xe_forge.frontend import surrogate, verify


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def test_gguf_header_becomes_config_and_weights(tmp_path):
    kv = [
        (_s("general.architecture") + struct.pack("<I", 8) + _s("toy")),
        (_s("toy.block_count") + struct.pack("<I", 4) + struct.pack("<I", 3)),
        (
            _s("toy.attention.sliding_window_pattern")
            + struct.pack("<I", 9)
            + struct.pack("<IQ", 7, 3)
            + struct.pack("<???", True, False, True)
        ),
        (
            _s("tokenizer.ggml.tokens")
            + struct.pack("<I", 9)
            + struct.pack("<IQ", 8, 2)
            + _s("a")
            + _s("b")
        ),
    ]
    tensor = _s("blk.0.ffn_up.weight") + struct.pack("<I", 2) + struct.pack("<QQ", 256, 8)
    tensor += struct.pack("<I", 10) + struct.pack("<Q", 0)  # Q2_K
    path = tmp_path / "toy.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, len(kv)) + b"".join(kv) + tensor)

    config = surrogate.merged_config(tmp_path)
    assert config["block_count"] == 3 and config["toy.block_count"] == 3
    assert config["attention.sliding_window_pattern"] == [True, False, True]
    assert config["vocab_size"] == 2 and config["architectures"] == ["toy"]
    assert surrogate.weights(path) == {"blk.0.ffn_up.weight": {"shape": [256, 8], "dtype": "Q2_K"}}


CSV = "# started\n\n{},,exe.amx_busy,1000,100.00,,\n"


def test_perf_count_reads_the_event_or_none():
    assert verify._perf_count(CSV.format(42), "exe.amx_busy") == 42
    assert verify._perf_count(CSV.format("<not supported>"), "exe.amx_busy") is None
    assert verify._perf_count("", "exe.amx_busy") is None


@pytest.mark.parametrize(
    ("on", "off", "passes", "why"),
    [
        (5, 0, True, "exe.amx_busy 5"),
        (0, 0, False, "no tile path ran"),
        (5, 3, False, "switch does not work"),
        (None, 0, False, "could not read"),
        (5, None, False, "could not read"),
    ],
)
def test_amx_gate_verdicts(monkeypatch, on, off, passes, why):
    def run(_engine, _weights, _ids, env):
        busy = off if env.get("XF_NO_AMX") else on
        return {"amx_busy": busy, "tokens": [1, 2], "prefill_ms": 1.0}

    monkeypatch.setattr(verify, "_amx_run", run)
    monkeypatch.setattr(verify.shutil, "which", lambda _: "/usr/bin/perf")
    _, ok, evidence = verify.amx_gate("e", "w", [1, 2, 3], {"XF_NO_AMX": "1"})
    assert ok is passes and why in evidence


def test_amx_gate_rejects_different_tokens_and_a_missing_perf(monkeypatch):
    monkeypatch.setattr(verify.shutil, "which", lambda _: None)
    assert "no perf" in verify.amx_gate("e", "w", [1], {})[2]

    def run(_engine, _weights, _ids, env):
        tokens = [9] if env.get("XF_NO_AMX") else [1]
        return {"amx_busy": 0 if env.get("XF_NO_AMX") else 5, "tokens": tokens}

    monkeypatch.setattr(verify.shutil, "which", lambda _: "/usr/bin/perf")
    monkeypatch.setattr(verify, "_amx_run", run)
    assert "tokens differ" in verify.amx_gate("e", "w", [1], {})[2]


def test_generate_refuses_a_cpu_without_amx(monkeypatch, tmp_path):
    from xe_forge.frontend import generate

    monkeypatch.setattr(generate, "_cpu", lambda: "ISA: avx512f")
    assert generate.main(["--model", "m", "-o", str(tmp_path / "run")]) == 1
    assert not (tmp_path / "run").exists()
