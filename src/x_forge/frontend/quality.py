"""Output quality A/B for `patch check` on a non-text workload: is the patched arm's output
still right? Arm A is the framework's own path, arm B the patched one, from the artifacts the
launcher saved. The stages, each optional (`_STAGES` holds the files and limits):

| stage         | file (first match)                                  | role   |
|---------------|-----------------------------------------------------|--------|
| `prompt`      | `prompt_embeds.npy`                                 | gate   |
| `step1`       | `latent_step1.npy`                                  | gate   |
| `step1_image` | the 1-step picture, when no latent was saved        | gate   |
| `latent`      | `latent_final.npy`                                  | guard  |
| `image`       | `raw_image.png`, `raw/output.png`, `frames/*.npy`   | report |
| `actions`     | `actions.npy` (a policy's returned actions)         | report |

The **gate** decides pass or fail: one denoise step, before rounding is amplified. A **guard**
fails only on a catastrophe (NaN, a different shape, a difference past its hard limit), since a
defect that needs many steps or lives in the VAE cannot show at step 1. A **report** stage only
warns, but for the same catastrophe. If no gate artifact exists, the guard becomes the gate and
the section says so.

Order of judgement, per stage:

1. **Hashes.** One hash per arm and the two equal -> `identical`, pass, nothing measured.
2. **Against a float32 reference** (`<prove>/reference/<stage>.<ext>`, same seed and steps), when
   one exists: the error of every run against it, then `max(e_B) <= max(e_A) * (1 + margin)`
   passes, `min(e_B) > max(e_A)` fails, and a B inside A's own spread passes. Both arms round, so
   the distance to the reference is the only measure that says which arm is *better*; A vs B alone
   mixes two errors.
3. **Absolute limits**, when there is no reference: a documented convention, not a measurement.
   Printed as such.

Linear error is averaged, never dB (a logarithm); dB is derived once, for the report.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

# (stage, role, file patterns, metric, warn, hard) -- warn/hard are the absolute convention of
# step 3, in the stage's metric. "rel" is relative squared error, "psnr" is dB (bigger is better).
_STAGES = (
    # The prompt embedding is one small tensor, and a patch to a shared op (`linear`) reaches the
    # text encoder as well as the transformer. So it is always compared: identical here is what
    # makes "hold the embedding fixed and judge the transformer alone" a safe shortcut.
    ("prompt", "gate", "check", ("prompt_embeds.npy", "raw/prompt_embeds.npy"), "rel", 1e-3, 1e-1),
    ("step1", "gate", "check", ("latent_step1.npy", "raw/latent_step1.npy"), "rel", 1e-3, 1e-1),
    # A framework that cannot hand out a latent still gives the 1-step picture, which is the same
    # idea with the VAE on the end: no amplification, so it gates when no latent was saved.
    ("step1_image", "gate", "check",
     ("output.png", "raw/output.png", "raw_image.png", "frames/*.npy", "raw/frames/*.npy"),
     "psnr", 40.0, 25.0),
    ("latent", "guard", "check", ("latent_final.npy", "raw/latent_final.npy"), "rel", 1e-2, 1e-1),
    ("image", "report", "timed",
     ("raw_image.png", "output.png", "raw/output.png", "frames/*.npy", "raw/frames/*.npy"),
     "psnr", 40.0, 25.0),
    # A policy's actions, as returned: its latents are the action latents, so the stages above
    # gate it, and this reports what the robot would be sent.
    ("actions", "report", "timed", ("actions.npy", "raw/actions.npy"), "rel", 1e-2, 1e-1),
)  # fmt: skip
MARGIN = 0.1  # how much worse than A's worst run B's worst run may be, against the reference


def _find(d: Path, patterns: tuple[str, ...]) -> Path | None:
    for p in patterns:
        hits = sorted(d.glob(p)) if "*" in p else ([d / p] if (d / p).is_file() else [])
        if hits:
            return hits[0]
    return None


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(p: Path):
    """The artifact as float64. An 8-bit PNG rounds away small differences; a .npy does not."""
    import numpy as np

    if p.suffix == ".npy":
        return np.load(p).astype(np.float64), None
    from PIL import Image

    return np.asarray(Image.open(p).convert("RGB")).astype(np.float64), 255.0


def _metrics(x, y, peak: float | None) -> dict:
    """x against y: relative squared error, and PSNR when the data has a known peak."""
    import numpy as np

    if x.shape != y.shape:
        return {"shape": [list(x.shape), list(y.shape)]}
    d = x - y
    mse = float((d**2).mean())
    power = float((y**2).mean())
    out = {
        "rel": mse / power if power else float("inf"),
        "mse": mse,
        "max_abs": float(np.abs(d).max()),
        "differing": float((np.abs(d) > 0).mean()) if d.size else 0.0,
        "nan": bool(not np.isfinite(x).all()),
    }
    p = peak if peak is not None else float(np.abs(y).max())
    out["psnr"] = float("inf") if mse == 0 else 10.0 * np.log10(p * p / mse) if p else None
    return out


def _worse(metric: str, a: float, b: float) -> bool:
    """Is b worse than a, in this metric's direction?"""
    return b > a if metric == "rel" else b < a


def _judge(stage: dict, metric: str, warn: float, hard: float, margin: float = MARGIN) -> dict:
    """Fill in verdict and why, from the hashes, the reference errors or the absolute limits."""
    a, b = stage["a"], stage["b"]
    if not a or not b:
        return {**stage, "verdict": "absent", "why": "the stage's artifact is missing in an arm"}
    ha, hb = {r["sha256"] for r in a}, {r["sha256"] for r in b}
    stage["deterministic"] = {"a": len(ha) == 1, "b": len(hb) == 1}
    if len(ha) == 1 and ha == hb:
        return {**stage, "verdict": "identical", "why": "every run of both arms is byte-identical"}
    bad = [
        r
        for r in a + b
        if r.get("ref", {}).get("nan") or r.get("ab", {}).get("nan") or r.get("ab", {}).get("shape")
    ]
    if bad:
        return {**stage, "verdict": "fail", "catastrophe": True,
                "why": "not a number, or a different shape"}  # fmt: skip
    ea = [r["ref"][metric] for r in a if r.get("ref", {}).get(metric) is not None]
    eb = [r["ref"][metric] for r in b if r.get("ref", {}).get(metric) is not None]
    if len(ea) == len(a) and len(eb) == len(b) and ea and eb:
        # Against the float32 reference. "Worst" is the conservative end of each arm's band.
        wa, wb = (max(ea), max(eb)) if metric == "rel" else (min(ea), min(eb))
        best_b = min(eb) if metric == "rel" else max(eb)
        spread = abs(max(ea) - min(ea))
        # One margin for both: a squared error (1 + margin) times A's is a PSNR
        # 10 log10(1 + margin) dB below A's.
        db = 10 * math.log10(1 + margin)
        allow = wa * (1 + margin) if metric == "rel" else wa - db
        stage["reference"] = {"a": ea, "b": eb, "a_spread": spread}
        if not _worse(metric, allow, wb):
            within = (
                f"{margin:.0%}"
                if metric == "rel"
                else f"{db:.2f} dB ({margin:.0%} in squared error)"
            )
            return {**stage, "verdict": "pass",
                    "why": f"B's worst run ({wb:.4g}) is within {within} of A's worst "
                           f"({wa:.4g}) against the float32 reference"}  # fmt: skip
        if _worse(metric, wa, best_b):
            return {**stage, "verdict": "fail",
                    "why": f"every B run ({best_b:.4g}) is further from the float32 reference "
                           f"than A's worst ({wa:.4g})"}  # fmt: skip
        return {**stage, "verdict": "inconclusive",
                "why": "A's and B's bands against the float32 reference overlap"}  # fmt: skip
    # No reference: the absolute convention, on A against B.
    vals = [r["ab"][metric] for r in b if r.get("ab", {}).get(metric) is not None]
    if not vals:
        return {**stage, "verdict": "absent", "why": f"no {metric} between the arms"}
    worst = max(vals) if metric == "rel" else min(vals)
    stage["ab_worst"] = worst
    inside = stage.get("a_spread_ab")
    if inside is not None and not _worse(metric, inside, worst):
        return {**stage, "verdict": "pass",
                "why": f"A vs B ({worst:.4g}) is inside A's own spread ({inside:.4g})"}  # fmt: skip
    if _worse(metric, hard, worst):
        return {**stage, "verdict": "fail", "catastrophe": True,
                "why": f"A vs B ({worst:.4g}) is past the hard limit {hard:.4g} "
                       f"(a convention, no float32 reference was given)"}  # fmt: skip
    if _worse(metric, warn, worst):
        return {**stage, "verdict": "warn",
                "why": f"A vs B ({worst:.4g}) is past {warn:.4g} but not the hard limit "
                       f"{hard:.4g} (a convention, no float32 reference was given)"}  # fmt: skip
    return {**stage, "verdict": "pass",
            "why": f"A vs B ({worst:.4g}) is within {warn:.4g} "
                   f"(a convention, no float32 reference was given)"}  # fmt: skip


def _stage(name: str, patterns, metric: str, dirs: dict[str, list[Path]], ref: Path | None) -> dict:
    """Read one stage's artifacts in both arms, hash them, and measure what can be measured."""
    out: dict = {"stage": name, "a": [], "b": [], "metric": metric}
    found = {arm: [f for d in ds if (f := _find(d, patterns))] for arm, ds in dirs.items()}
    if not found.get("A") or not found.get("B"):
        return out
    ref_data = _read(ref) if ref and ref.is_file() else None
    base = _read(found["A"][0])
    for arm, key in (("A", "a"), ("B", "b")):
        for f in found[arm]:
            row = {"file": str(f), "sha256": _sha256(f)}
            data, peak = _read(f)
            if ref_data is not None:
                row["ref"] = _metrics(data, ref_data[0], ref_data[1] or peak)
            row["ab"] = _metrics(data, base[0], base[1] or peak)
            out[key].append(row)
    # A's own run-to-run spread in the same metric: the floor, and a lower limit on any tolerance.
    own = [r["ab"][metric] for r in out["a"][1:] if r.get("ab", {}).get(metric) is not None]
    if own:
        out["a_spread_ab"] = max(own) if metric == "rel" else min(own)
    if ref is not None:
        out["reference_file"] = str(ref)
    return out


def quality(
    prove_dir: Path,
    dirs: dict[str, list[Path]],
    check_dirs: dict[str, list[Path]] | None = None,
    margin: float = MARGIN,
) -> dict:
    """Judge every stage found under the arms' directories.

    ``dirs``: the timed captures, {"A": [dirs], "B": [...]} -- the image comes from these, so the
    picture judged is the one the measured run made. ``check_dirs``: the short correctness runs
    (``X_FORGE_CHECK``), which hold the latents. Writes quality.json; returns the record.
    """
    ref_dir = prove_dir / "reference"
    stages = []
    for name, role, source, patterns, metric, warn, hard in _STAGES:
        ref = _find(ref_dir, (f"{name}.npy", f"{name}.png")) if ref_dir.is_dir() else None
        st = _stage(name, patterns, metric, (check_dirs or {}) if source == "check" else dirs, ref)
        st["role"] = role
        stages.append(_judge(st, metric, warn, hard, margin))
    # The step-1 latent gates. Without it the guard decides, and without that the image does;
    # a promoted stage says so, because it is a blunter view of the same question.
    gates: list[dict] = []
    for role in ("gate", "guard", "report"):
        gates = [s for s in stages if s["role"] == role and s["verdict"] != "absent"]
        if gates:
            if role != "gate":
                for s in gates:
                    s["role"] = f"{role}, promoted to gate: no earlier artifact"
            break
    # Several gate stages: the worst of them decides, so a clean prompt embedding cannot cover a
    # bad latent. A guard or report stage fails the whole thing only on a catastrophe.
    rank = ("fail", "inconclusive", "warn", "pass", "identical")
    fails = [s for s in stages if s["verdict"] == "fail" and (s in gates or s.get("catastrophe"))]
    verdict = (
        "fail" if fails
        else min((s["verdict"] for s in gates), key=rank.index, default="absent")
    )  # fmt: skip
    record = {"verdict": verdict, "margin": margin, "stages": stages}
    prove_dir.mkdir(parents=True, exist_ok=True)
    (prove_dir / "quality.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def summary(record: dict) -> str:
    """The `## In model` lines for SUMMARY.md."""
    if record["verdict"] == "absent":
        return "output quality: not judged (the launcher saved no comparable output)"
    lines = [f"output quality: **{record['verdict']}**"]
    for s in record["stages"]:
        if s["verdict"] == "absent":
            continue
        m = s["metric"]
        worst = s.get("ab_worst")
        extra = f", A vs B {m} {worst:.4g}" if worst is not None else ""
        if not all(s.get("deterministic", {}).values()):
            extra += " (an arm is not reproducible run to run)"
        lines.append(f"- `{s['stage']}` ({s['role']}): {s['verdict']} -- {s['why']}{extra}")
    return "\n".join(lines)
