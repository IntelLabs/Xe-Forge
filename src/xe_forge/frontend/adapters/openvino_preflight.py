"""Preflight: is this HF model exportable by optimum-intel (OpenVINO) as-is?

A standalone script (no xe_forge imports) the OpenVINO prepare agent runs under the
framework's interpreter, before exporting.

usage: <framework python> -I openvino_preflight.py <hf_id_or_local_dir>
Prints model_type, whether optimum-intel has an export config and for which tasks,
whether the installed transformers satisfies that config's version range (with the exact
pip pin when it does not), and extra pip packages the model family usually needs.
Exit 0 = ok, 4 = unsupported architecture (needs new export config/patcher),
5 = supported but installed transformers is outside the required range,
6 = installed transformers is too old to even recognize the model_type (needs a newer per-model venv),
7 = Hub repo is gated / private / not found (access problem, not a model problem).
Diffusers pipelines (dirs/repos with model_index.json) are reported too, though the
capture adapter only runs text-generating models.
"""
import difflib, json, sys
from importlib.metadata import PackageNotFoundError, requires, version
from packaging.requirements import Requirement
from packaging.version import Version

mid = sys.argv[1]

# extra pip deps beyond optimum-intel[openvino], keyed by task / family
EXTRA_DEPS = {
    "diffusers": ["diffusers"],
    "automatic-speech-recognition": ["librosa", "soundfile"],
    "image-text-to-text": ["pillow", "num2words (install with --no-deps; uv cannot lock docopt on NFS)"],
    "text-to-image": ["diffusers"],
}
# preferred task when several are registered
TASK_PRIORITY = ["text-generation-with-past", "text-generation", "image-text-to-text",
                 "automatic-speech-recognition-with-past", "automatic-speech-recognition",
                 "text-to-image", "feature-extraction"]


# pipelines wrapped by optimum.intel.openvino.modeling_diffusion.SUPPORTED_OV_PIPELINES (static copy for
# environments without diffusers, where importing that module fails)
OV_DIFFUSERS_PIPELINES = {
    "StableDiffusionPipeline", "StableDiffusionImg2ImgPipeline", "StableDiffusionInpaintPipeline",
    "StableDiffusionXLPipeline", "StableDiffusionXLImg2ImgPipeline", "StableDiffusionXLInpaintPipeline",
    "LatentConsistencyModelPipeline", "LatentConsistencyModelImg2ImgPipeline",
}


def load_model_index(path):
    import os
    p = os.path.join(path, "model_index.json")
    if os.path.isfile(p):
        return json.load(open(p))
    if os.path.isdir(path):
        return None
    from huggingface_hub import hf_hub_download
    try:
        return json.load(open(hf_hub_download(path, "model_index.json")))
    except Exception:  # noqa: BLE001  (no model_index.json -> not a diffusers repo)
        return None


def installed(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def ov_global_bounds():
    """transformers specifier from optimum-intel's own install requirements."""
    for r in requires("optimum-intel") or []:
        req = Requirement(r)
        if req.name == "transformers" and req.marker is None:
            return str(req.specifier)
    return None


def check_transformers(cls):
    tv = installed("transformers")
    lo, hi = getattr(cls, "MIN_TRANSFORMERS_VERSION", None), getattr(cls, "MAX_TRANSFORMERS_VERSION", None)
    spec = ",".join(x for x in (lo and f">={lo}", hi and f"<={hi}") if x)
    res = {"installed": tv, "config_requires": spec or "any", "optimum_intel_requires": ov_global_bounds()}
    ok, pin = True, None
    v = Version(tv)
    if hi and v > Version(hi):
        ok, pin = False, f"transformers<={hi}"
    elif lo and v < Version(lo):
        ok, pin = False, f"transformers>={lo}"
    res["compatible"] = ok
    if not ok:
        res["pin"] = pin
        res["note"] = (f"install `{pin}` (uv pip install '{pin}') before export, and use the SAME transformers "
                       "version for export and for running verify scripts: tokenizer files are version-specific")
    return res


def raw_config(path):
    """config.json as a dict without going through transformers (local dir or Hub id)."""
    import os
    p = path if os.path.isfile(path) else os.path.join(path, "config.json")
    if not os.path.isfile(p):
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(path, "config.json")
    return json.load(open(p))


def access_problem(mid, e):
    """rc 7 if `e` is a Hub access error (gated, private, missing, 401/403); else None. Never prints the token."""
    try:
        from huggingface_hub import get_token
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        return None
    msg = str(e)
    gated = isinstance(e, GatedRepoError) or "gated repo" in msg.lower()
    notfound = isinstance(e, RepositoryNotFoundError) or "is not a valid model identifier" in msg or "is not a local folder" in msg
    if not (gated or notfound or " 401 " in msg or " 403 " in msg or "401 Client Error" in msg or "403 Client Error" in msg):
        return None
    logged_in = get_token() is not None
    out = {"model": mid, "outcome": "hub_access_denied", "kind": "gated" if gated else ("not_found_or_private" if notfound else "http_401_403"),
           "hf_login_configured": logged_in, "error": msg.strip().splitlines()[0][:300]}
    if gated and logged_in:
        out["next_step"] = ("a login is configured but this account has no approval for the repo: request access on the model page "
                            "(manual approval can take time), or supply the config.json yourself")
    elif gated:
        out["next_step"] = ("repo is gated: the user must log in with an approved account themselves (`! huggingface-cli login`, or set "
                            "HF_TOKEN_PATH to the token file; keep HF_HOME elsewhere). Never search for tokens, never print or log the "
                            "token. Meanwhile fall back to the defaults of the installed transformers `<Model>Config` and label results "
                            "as an inference, or ask for the config.json")
    else:
        out["next_step"] = ("check the repo id spelling; private repos and typos both return 'not found' without a login "
                            f"(hf_login_configured={logged_in})")
    print(json.dumps(out, indent=1))
    return 7


def too_old(mid, err):
    """rc 6: installed transformers does not know this model_type. Report a clear next step instead of crashing."""
    raw = raw_config(mid)
    mt = raw.get("model_type")
    out = {"model": mid, "model_type": mt, "architectures": raw.get("architectures"),
           "outcome": "transformers_too_old_or_unknown_model_type",
           "transformers": {"installed": installed("transformers"),
                            "model_config_transformers_version": raw.get("transformers_version")},
           "error": err.strip().splitlines()[0][:300]}
    mins = []
    try:  # optimum-intel's own minimum for this model_type, if it registers one (needs its configs to import)
        import optimum.exporters.openvino.model_configs  # noqa: F401
        from optimum.exporters.tasks import TasksManager
        for _t, p in TasksManager._SUPPORTED_MODEL_TYPE.get(mt, {}).get("openvino", {}).items():
            lo = getattr(getattr(p, "func", p), "MIN_TRANSFORMERS_VERSION", None)
            if lo:
                mins.append(str(lo))
        out["optimum_intel_registers_model_type"] = mt in TasksManager._SUPPORTED_MODEL_TYPE
    except Exception as e2:  # noqa: BLE001
        out["optimum_intel_lookup_error"] = str(e2)[:200]
    cands = [(Version(v), src) for v, src in [(m, "optimum-intel MIN_TRANSFORMERS_VERSION") for m in mins]
             + ([(raw["transformers_version"], "transformers_version in the model's config.json (version it was saved with)")]
                if raw.get("transformers_version") else [])]
    if cands:
        v, src = max(cands, key=lambda c: c[0])
        out["minimum_transformers"] = {"version": str(v), "source": src}
    need = out.get("minimum_transformers", {}).get("version", "a newer release")
    inst = installed("transformers")
    if inst and "minimum_transformers" in out and Version(inst) >= Version(need) and not out.get("optimum_intel_registers_model_type"):
        # the installed transformers is already >= the version the model was saved with: not a version problem; the model_type
        # lives in a custom package (e.g. IBM granite-tsfm for FlowState) or remote code
        out["outcome"] = "no_transformers_or_optimum_intel_integration"
        out["next_step"] = ("transformers does not know this model_type even though it is new enough (installed %s >= %s): the architecture is "
                            "defined by a custom package or remote code. Read the model card, REVIEW the official code before running it, vendor it, "
                            "write an export wrapper (ov.convert_model / torch.export) and validate against the official forward" % (inst, need))
        print(json.dumps(out, indent=1))
        return 4
    out["next_step"] = (f"create a per-model venv with transformers>={need} (`uv venv` + `uv pip install 'transformers>={need}'`; "
                        "do not install optimum-onnx, it downgrades transformers; for a local optimum-intel clone use a .pth file, "
                        "not `uv pip install -e`, on NFS), then re-run preflight there and use the SAME transformers version "
                        "for export and verify")
    print(json.dumps(out, indent=1))
    return 6


out = {"model": mid}
rc = 0
mi = None
try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(mid, trust_remote_code=False)
except (OSError, ValueError, KeyError) as e:
    if "does not recognize this architecture" in str(e):
        sys.exit(too_old(mid, str(e)))
    _rc = access_problem(mid, e)
    if _rc:
        sys.exit(_rc)
    # no config.json (or no model_type) at the root: maybe a diffusers pipeline
    mi = load_model_index(mid)
    if mi is None:
        raise
if mi is not None:
    # diffusers pipeline: no config.json/model_type
    cname = mi.get("_class_name")
    out.update(model_type=f"diffusers:{cname}", library="diffusers")
    try:
        from optimum.intel.openvino.modeling_diffusion import _get_ov_class
        ov_cls = _get_ov_class(cname, throw_error_if_not_exist=False)
        wrapped = ov_cls is not None
        out["ov_class"] = ov_cls.__name__ if wrapped else None
    except ImportError:  # diffusers not installed
        wrapped = cname in OV_DIFFUSERS_PIPELINES
        out["note"] = "diffusers not installed: verdict from static pipeline list; `uv pip install diffusers` before export"
    out["supported"] = wrapped
    if not wrapped:
        out["next_step"] = "pipeline class has no OV wrapper in optimum.intel.openvino.modeling_diffusion"
        rc = 4
    else:
        out["transformers"] = {"installed": installed("transformers"), "optimum_intel_requires": ov_global_bounds(),
                               "compatible": True}
        out["extra_pip_deps"] = EXTRA_DEPS["diffusers"]
else:
    import optimum.exporters.openvino.model_configs  # noqa: F401  (registers configs)
    from optimum.exporters.tasks import TasksManager
    mt = cfg.model_type
    sup = TasksManager._SUPPORTED_MODEL_TYPE
    ov_tasks = sup.get(mt, {}).get("openvino", {}) if mt in sup else {}
    tasks = sorted(ov_tasks)
    out.update(model_type=mt, architectures=getattr(cfg, "architectures", None), supported=bool(tasks), tasks=tasks)
    if not tasks:
        out["closest_supported"] = difflib.get_close_matches(mt, list(sup), n=5, cutoff=0.4)
        out["next_step"] = ("add OpenVINOConfig + model patcher in optimum-intel "
                            "(optimum/exporters/openvino/model_configs.py); start from the closest type")
        rc = 4
    else:
        primary = next((t for t in TASK_PRIORITY if t in ov_tasks), tasks[0])
        p = ov_tasks[primary]
        cls = getattr(p, "func", p)  # registry stores functools.partial(ConfigClass, task=...)
        out["primary_task"], out["config_class"] = primary, getattr(cls, "__name__", str(cls))
        out["transformers"] = check_transformers(cls)
        base = primary.replace("-with-past", "")
        out["extra_pip_deps"] = EXTRA_DEPS.get(base, [])
        if not out["transformers"]["compatible"]:
            rc = 5
print(json.dumps(out, indent=1))
sys.exit(rc)
