"""The model's facts for the architecture agent, and its surrogate: the same architecture at a
reduced config with seeded random weights. Runs under the reference interpreter (transformers).

    python -m xe_forge.frontend.surrogate inspect --model ID -o model.json
    python -m xe_forge.frontend.surrogate build --model ID --overrides '{...}' -o DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def snapshot(model: str) -> Path:
    """The local directory holding the model's config and safetensors (downloaded if missing),
    or a local GGUF file."""
    p = Path(model)
    if p.is_dir() or p.suffix == ".gguf":
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt"]))


_GGML = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
         10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
         17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
         23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
         30: "BF16", 39: "MXFP4"}  # fmt: skip


def gguf_file(path: Path) -> Path | None:
    """The GGUF a model path names: the file itself, or the largest .gguf of a directory
    without a config.json."""
    path = Path(path)
    if path.suffix == ".gguf":
        return path
    if path.is_dir() and not (path / "config.json").is_file():
        files = sorted(path.glob("*.gguf"), key=lambda f: f.stat().st_size)
        return files[-1] if files else None
    return None


def gguf_read(path: Path) -> tuple[dict, dict]:
    """(metadata, tensors) of a GGUF file's header. A string array is kept as its length; a
    tensor's shape is in GGUF order (innermost first), its dtype the ggml type name."""
    import struct

    scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
              10: "<Q", 11: "<q", 12: "<d"}  # fmt: skip
    with open(path, "rb") as f:

        def rd(fmt):
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        def rstr():
            return f.read(rd("<Q")).decode("utf-8", "replace")

        def rval(t):
            if t == 8:
                return rstr()
            if t == 9:
                at, n = rd("<I"), rd("<Q")
                if at == 8:
                    for _ in range(n):
                        rstr()
                    return {"strings": n}
                return [rval(at) for _ in range(n)]
            return rd(scalar[t])

        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        rd("<I")
        n_tensors, n_kv = rd("<Q"), rd("<Q")
        meta = {}
        for _ in range(n_kv):
            key = rstr()
            meta[key] = rval(rd("<I"))
        tensors = {}
        for _ in range(n_tensors):
            name = rstr()
            shape = [rd("<Q") for _ in range(rd("<I"))]
            ggml_type = rd("<I")
            rd("<Q")
            tensors[name] = {"shape": shape, "dtype": _GGML.get(ggml_type, str(ggml_type))}
    return meta, tensors


def weights(directory: Path) -> dict[str, dict]:
    """Tensor name -> {shape, dtype} from the safetensors or GGUF headers (no data read)."""
    if gguf_file(directory):
        return gguf_read(gguf_file(directory))[1]
    out = {}
    for f in sorted(directory.glob("*.safetensors")):
        with open(f, "rb") as fh:
            header = json.loads(fh.read(int.from_bytes(fh.read(8), "little")))
        for name, t in header.items():
            if name != "__metadata__":
                out[name] = {"shape": t["shape"], "dtype": t["dtype"]}
    return out


def merged_config(directory: Path) -> dict:
    """config.json as written, over the config transformers builds from it: fields the config
    class derives (``layer_types``, rope parameters) are there too, and a nested ``text_config``
    is lifted to the top. One arch.json binds against this, for a model and its surrogate."""
    gg = gguf_file(directory)
    if gg:  # GGUF metadata, also without its architecture prefix
        meta = gguf_read(gg)[0]
        arch = meta.get("general.architecture", "")
        short = {
            k[len(arch) + 1 :]: v for k, v in meta.items() if arch and k.startswith(arch + ".")
        }
        tokens = meta.get("tokenizer.ggml.tokens")
        extra = {"vocab_size": tokens["strings"]} if isinstance(tokens, dict) else {}
        return {**short, **meta, **extra, "architectures": [arch]}
    from transformers import AutoConfig

    raw = json.loads((directory / "config.json").read_text())
    try:
        config = AutoConfig.from_pretrained(str(directory)).to_dict()
        resolved = json.loads(json.dumps(config, default=str))
    except (ValueError, OSError, KeyError):
        resolved = {}
    return {**(resolved.get("text_config") or {}), **resolved,
            **(raw.get("text_config") or {}), **raw}  # fmt: skip


def inspect(model: str, out: Path) -> dict:
    import transformers

    d = snapshot(model)
    config = merged_config(d)
    arch = (config.get("architectures") or ["?"])[0]
    modeling = None
    mt = config.get("model_type")
    if mt and not gguf_file(d):
        try:
            import importlib

            mod = importlib.import_module(f"transformers.models.{mt}.modeling_{mt}")
            modeling = mod.__file__
        except ImportError:
            modeling = None
    info = {
        "model": model, "dir": str(d), "architecture": arch, "config": config,
        "modeling": modeling, "transformers": transformers.__version__, "weights": weights(d),
    }  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=1) + "\n")
    return info


def adapter(path: Path) -> dict:
    """A Python file of the architecture agent's, executed fresh (no stale bytecode)."""
    ns: dict = {"__name__": f"xf_{Path(path).stem}", "__file__": str(path)}
    exec(compile(Path(path).read_text(), str(path), "exec"), ns)
    return ns


def build(model: str, overrides: dict, out: Path, seed: int = 1234, std: float = 0.2,
          saver: Path | None = None, reference: Path | None = None) -> Path:  # fmt: skip
    """Write ``out/{config.json, model.safetensors}``: weights N(0, std), norms around 1, stored
    in the original's dtype.

    ``std`` is far above an initializer's: with tiny weights every logit is near zero and the
    greedy token is decided by rounding, which hides real errors and invents fake ones.
    """
    import torch

    # From the raw config.json, so fields the config derives (layer_types, ...) follow the overrides.
    src = snapshot(model)
    raw = merged_config(src) if gguf_file(src) else json.loads((src / "config.json").read_text())
    raw.update(overrides)
    torch.manual_seed(seed)
    if reference is not None:
        # a model transformers does not run: the reference adapter builds it, initialized
        if saver is None:
            raise SystemExit("surrogate: a reference adapter needs surrogate_save.py to write it")
        m = adapter(reference)["random_model"](raw).float()
    else:
        from transformers import AutoConfig, AutoModelForCausalLM

        m = AutoModelForCausalLM.from_config(AutoConfig.for_model(**raw), dtype=torch.float32)
        with torch.no_grad():
            for name, p in m.named_parameters():
                if "norm" in name:
                    p.copy_(1.0 + 0.1 * torch.randn_like(p))
                else:
                    p.normal_(0.0, std)
    out.mkdir(parents=True, exist_ok=True)
    if (raw.get("torch_dtype") or raw.get("dtype")) == "bfloat16":
        m = m.to(torch.bfloat16)  # stored as the original stores it, so the engine reads it alike
    if saver is not None:
        # the architecture agent's: the checkpoint in the original's storage format (a packed
        # format, fused or transposed tensors), which save_pretrained of a dense model is not
        adapter(saver)["save"](m, raw, out)
    else:
        m.save_pretrained(str(out), safe_serialization=True)
    # Keep the original's config layout (save_pretrained may move keys, e.g. rope_theta), so one
    # arch.json binds the model and its surrogate alike.
    if not gguf_file(out):  # a GGUF surrogate carries its config in its own metadata
        (out / "config.json").write_text(json.dumps(raw, indent=1) + "\n")
        (out / "xf_config.json").write_text(json.dumps(merged_config(out), indent=1) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m xe_forge.frontend.surrogate")
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inspect")
    i.add_argument("--model", required=True)
    i.add_argument("-o", "--output", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--model", required=True)
    b.add_argument("--overrides", required=True, help="config overrides: a JSON object or a file")
    b.add_argument("-o", "--output", type=Path, required=True)
    b.add_argument("--seed", type=int, default=1234)
    b.add_argument("--reference", type=Path, help="reference adapter: random_model(config)")
    b.add_argument("--std", type=float, default=0.2)
    b.add_argument(
        "--saver", type=Path, help="save(model, config, out) writing the original's format"
    )
    a = p.parse_args(argv)
    if a.cmd == "inspect":
        info = inspect(a.model, a.output)
        print(f"inspect: {info['architecture']}, {len(info['weights'])} tensors -> {a.output}")
        return 0
    text = a.overrides
    overrides = json.loads(Path(text).read_text() if Path(text).is_file() else text)
    if not isinstance(overrides, dict) or not overrides:
        raise SystemExit("surrogate: --overrides must be a non-empty JSON object")
    print(f"surrogate: {build(a.model, overrides, a.output, a.seed, a.std, a.saver, a.reference)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
