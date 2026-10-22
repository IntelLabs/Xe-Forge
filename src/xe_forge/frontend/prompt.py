"""The prompt every adapter runs, so captures from different frameworks see the same work."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Vocab:
    encode: Callable[[str], list[int]]
    ordinary: Sequence[int]


def hf_vocab(tok) -> Vocab:
    """A Hugging Face tokenizer's vocabulary without its special tokens."""
    special = set(getattr(tok, "all_special_ids", ()) or ())
    size = getattr(tok, "vocab_size", None) or len(tok)
    ordinary = [i for i in range(size) if i not in special]
    return Vocab(lambda text: tok.encode(text, add_special_tokens=False), ordinary)


def add_arguments(parser) -> None:
    parser.add_argument("--prompt-file", type=Path, help="text to tokenize (default: synthetic)")
    parser.add_argument("--warmup-tokens", type=int, default=4, help="new tokens in the warm-up")
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="the model's repo ships its own code"
    )


def token_ids(
    vocab: Vocab, batch: int, length: int, prompt_file: Path | None = None
) -> list[list[int]]:
    if prompt_file:
        ids = vocab.encode(Path(prompt_file).read_text())
        if not ids:
            raise SystemExit(f"{prompt_file} tokenizes to nothing")
        stride = max(1, len(ids) // batch)
        return [[ids[(j * stride + i) % len(ids)] for i in range(length)] for j in range(batch)]
    pool = vocab.ordinary
    if not pool:
        raise SystemExit("the tokenizer has no ordinary tokens for a synthetic prompt")
    return [
        [pool[(i * 7919 + j * 104729) % len(pool)] for i in range(length)] for j in range(batch)
    ]


def describe(prompt_file: Path | None) -> str:
    if not prompt_file:
        return "synthetic"
    digest = hashlib.sha1(Path(prompt_file).read_bytes()).hexdigest()[:10]
    return f"{Path(prompt_file).name}@{digest}"
