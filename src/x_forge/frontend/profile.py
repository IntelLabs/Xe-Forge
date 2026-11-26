"""What a capture launcher imports: the model's input token ids (the same work for every
framework and run: fixed synthetic ids, or slices of the run's prompt file -- not an LLM
prompt for an agent; those are claude/templates/*.j2), the device torch
sees, one torch.profiler region written as ``trace.json.gz`` for ``capture parse``, and, for
a framework on the host that is not torch, :class:`HostTimeline`.
"""

from __future__ import annotations

import gzip
import json
import shutil
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


def device() -> str:
    """The device asked for, index included (``X_FORGE_DEVICE``, which ``run`` sets from
    ``--device``), else the first one torch sees."""
    import os

    if os.environ.get("X_FORGE_DEVICE"):
        return os.environ["X_FORGE_DEVICE"]
    import torch

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def device_type() -> str:
    """:func:`device` without its index."""
    return device().split(":")[0]


def device_info(threads: int | None = None) -> dict:
    """The device record of meta.json. On a host, ``threads`` is the team the framework runs
    with, which the launcher knows; without it, torch's, or None when torch is not there."""
    try:
        import torch
    except ImportError:  # a host framework that is not torch
        torch = None
    dev = device_type()
    info = {"type": dev, "torch": torch.__version__ if torch else None}
    if dev in ("xpu", "cuda"):
        info["name"] = getattr(torch, dev).get_device_name(torch.device(device()).index or 0)
    elif dev == "cpu":
        from x_forge.core.topology import cpu_flags, model_name, placement

        info["name"] = model_name()
        info["flags"] = cpu_flags()
        info["placement"] = placement()  # the launcher's binding, as this process got it
        if threads is None and torch is not None:
            threads = torch.get_num_threads()
        info["threads"] = threads
    return info


def host_placement() -> dict:
    """The launcher's placement of this host process and torch's thread count: what a host
    timing or profile was measured under."""
    import torch

    from x_forge.core.topology import placement

    return {**placement(), "threads": torch.get_num_threads()}


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


class HostTimeline:
    """Wall times of a host framework's graph nodes, for ``capture run`` on a host.

    The launcher brackets each repeat of the workload with :meth:`window` and reports every
    node it ran with :meth:`node`; where it hooks the framework and what it calls a node are
    its choice. A window opened with ``timed=False`` runs the same work with the hook off:
    its wall time is what the hook's own cost is judged against. :meth:`write` saves
    ``$RAW_DIR/host_timeline.json``. The clock is CLOCK_MONOTONIC, the one perf records with
    ``-k CLOCK_MONOTONIC``, so its samples can be joined to the nodes.
    """

    def __init__(self, raw_dir: Path | str, threads: int | None = None) -> None:
        self.path = Path(raw_dir) / "host_timeline.json"
        self.threads = threads
        self.windows: list[dict] = []

    @staticmethod
    def now() -> int:
        return time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    @contextmanager
    def window(self, timed: bool = True) -> Iterator[dict]:
        w = {"timed": timed, "start_ns": self.now(), "end_ns": None, "nodes": []}
        self.windows.append(w)
        try:
            yield w
        finally:
            w["end_ns"] = self.now()

    def node(self, name: str, start_ns: int, end_ns: int) -> None:
        """One node of the open window, between two :meth:`now` readings."""
        self.windows[-1]["nodes"].append([name, int(start_ns), int(end_ns)])

    def write(self) -> Path:
        from x_forge.core.topology import placement

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"placement": placement(), "threads": self.threads,
                                         "windows": self.windows}) + "\n")  # fmt: skip
        return self.path


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
    """``batch`` rows of ``length`` ids: with ``prompt_file``, consecutive slices of its text
    (``x-forge run --prompt-file``), each row its own, the text repeated when it is shorter;
    without, fixed synthetic ids."""
    if prompt_file:
        ids = vocab.encode(Path(prompt_file).read_text())
        if not ids:
            raise SystemExit(f"{prompt_file} tokenizes to no ids")
        ids = ids * -(-batch * length // len(ids))
        return [ids[j * length : (j + 1) * length] for j in range(batch)]
    pool = vocab.ordinary
    if not pool:
        raise SystemExit("the tokenizer has no ordinary tokens for a synthetic prompt")
    return [
        [pool[(i * 7919 + j * 104729) % len(pool)] for i in range(length)] for j in range(batch)
    ]
