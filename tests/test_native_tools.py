"""The native engine's measuring tools: reading a model's facts from a GGUF header."""

import struct

from xe_forge.frontend import surrogate


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
