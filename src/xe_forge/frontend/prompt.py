"""The prompt every adapter runs: one module, so captures from different frameworks see the
same work.

An adapter hands over its model's tokenizer as a ``Vocab`` -- ``encode(text) -> ids`` and
the ids that are ordinary tokens -- and gets one list of token ids per sequence,
``input_len`` long:

- ``--prompt-file``: real text, tokenized by the model's own tokenizer; each sequence starts
  at a different offset into it (wrapping), so a batch is not one sequence repeated. Text
  matters where routing does: an MoE router picks experts from the tokens.
- no file: deterministic ids spread over the ordinary tokens, distinct per sequence; special
  and added tokens (EOS, chat markers, image placeholders) are never used, wherever a
  vocabulary keeps them.

``describe`` is what the capture records, and what ``compare`` matches arms on. A future
framework (llama.cpp) builds its ``Vocab`` from its own tokenizer.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Vocab:
    encode: Callable[[str], list[int]]
    ordinary: Sequence[int]  # ids a synthetic prompt may use


def hf_vocab(tok) -> Vocab:
    """A Hugging Face tokenizer's vocabulary, without its special and added tokens."""
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
