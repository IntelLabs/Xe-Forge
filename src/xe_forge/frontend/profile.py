"""What a capture launcher imports: the model's fixed input token ids (the same work for every
framework and run -- not an LLM prompt; those are claude/templates/*.j2), the device torch
sees, and one torch.profiler region written as ``trace.json.gz`` for ``capture parse``.
"""

from __future__ import annotations

import gzip
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


def device_type() -> str:
    import torch

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def device_info() -> dict:
    import torch

    dev = device_type()
    info = {"type": dev, "torch": torch.__version__}
    if dev == "xpu":
        info["name"] = torch.xpu.get_device_name(0)
    elif dev == "cuda":
        info["name"] = torch.cuda.get_device_name(0)
    return info


def profile_region(fn, raw_dir: Path, with_stack: bool = True) -> Path:
    """Run ``fn`` under torch.profiler and write ``raw_dir/trace.json.gz``."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    dev = device_type()
    activities = [ProfilerActivity.CPU]
    if dev == "xpu":
        activities.append(ProfilerActivity.XPU)
    elif dev == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True, with_stack=with_stack) as prof:
        fn()
        if dev != "cpu":
            getattr(torch, dev).synchronize()

    raw_dir.mkdir(parents=True, exist_ok=True)
    plain = raw_dir / "trace.json"
    prof.export_chrome_trace(str(plain))
    out = raw_dir / "trace.json.gz"
    with open(plain, "rb") as src, gzip.open(out, "wb") as dst:
        shutil.copyfileobj(src, dst)
    plain.unlink()
    return out


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
