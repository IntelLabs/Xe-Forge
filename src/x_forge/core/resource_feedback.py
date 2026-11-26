"""Build provenance and evidence-only resource comparisons for profiling."""

import functools
import hashlib
import inspect
import json
import os
import shlex
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def capture_build_provenance(directory: str | Path):
    """Observe extension builds without changing their sources, flags, or cache keys."""
    import torch
    from torch.utils import cpp_extension

    directory = Path(directory)
    provenance = {
        "torch_version": torch.__version__,
        "environment": {
            name: os.environ[name]
            for name in (
                "CXX",
                "SYCL_CXX",
                "TORCH_XPU_ARCH_LIST",
                "IGC_ExtraOCLOptions",
                "SYCL_PROGRAM_COMPILE_OPTIONS",
                "SYCL_PROGRAM_APPEND_COMPILE_OPTIONS",
                "ONEAPI_DEVICE_SELECTOR",
                "ZE_FLAT_DEVICE_HIERARCHY",
                "UR_L0_V2_DISABLE_ZE_LAUNCH_KERNEL_WITH_ARGS",
                "UR_L0_USE_DRIVER_INORDER_LISTS",
                "IGC_ShaderDumpEnable",
                "IGC_DumpToCustomDir",
                "SYCL_CACHE_PERSISTENT",
                "NEO_CACHE_PERSISTENT",
            )
            if name in os.environ
        },
        "builds": [],
        "device": None,
    }
    originals = {name: getattr(cpp_extension, name) for name in ("load", "load_inline")}

    def observe(loader, loader_name):
        @functools.wraps(loader)
        def wrapped(*args, **kwargs):
            arguments = inspect.signature(loader).bind_partial(*args, **kwargs).arguments
            record = {
                "loader": loader_name,
                "arguments": {
                    name: arguments[name]
                    for name in (
                        "name",
                        "extra_cflags",
                        "extra_sycl_cflags",
                        "extra_cuda_cflags",
                        "extra_ldflags",
                        "extra_include_paths",
                        "build_directory",
                    )
                    if name in arguments
                },
            }
            provenance["builds"].append(record)
            module = loader(*args, **kwargs)
            module_file = getattr(module, "__file__", None)
            if module_file:
                record["module_file"] = str(module_file)
                try:
                    _capture_build_commands(Path(module_file).parent, directory, record)
                except Exception as exc:
                    record["capture_error"] = str(exc)
            return module

        return wrapped

    try:
        for name, loader in originals.items():
            setattr(cpp_extension, name, observe(loader, name))
        yield
    finally:
        for name, loader in originals.items():
            setattr(cpp_extension, name, loader)
        try:
            if torch.xpu.is_available():
                properties = torch.xpu.get_device_properties(torch.xpu.current_device())
                provenance["device"] = {
                    "name": properties.name,
                    "properties": str(properties),
                }
        except Exception as exc:
            provenance["device_error"] = str(exc)
        (directory / "build_provenance.json").write_text(
            json.dumps(provenance, indent=2, default=str)
        )


def _capture_build_commands(build_directory: Path, artifacts: Path, record: dict) -> None:
    ninja_file = build_directory / "build.ninja"
    if not ninja_file.is_file():
        return
    destination = artifacts / "builds" / str(len(list((artifacts / "builds").glob("*"))))
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ninja_file, destination / "build.ninja")
    record["artifacts"] = str(destination.relative_to(artifacts))
    ninja = shutil.which("ninja")
    if ninja is None:
        record["capture_error"] = "ninja unavailable; retained build.ninja only"
        return
    process = subprocess.run(
        [ninja, "-C", str(build_directory), "-t", "compdb"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if process.returncode:
        record["capture_error"] = process.stderr
        return
    commands = json.loads(process.stdout)
    (destination / "compile_commands.json").write_text(json.dumps(commands, indent=2))
    drivers = {}
    for entry in commands:
        tokens = entry.get("arguments") or shlex.split(entry.get("command", ""))
        if not tokens or tokens[0] in drivers:
            continue
        executable = shutil.which(tokens[0])
        if executable is None:
            drivers[tokens[0]] = {"error": "executable not found"}
            continue
        version = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, timeout=10
        )
        drivers[tokens[0]] = {
            "path": executable,
            "version": version.stdout.strip(),
            "returncode": version.returncode,
        }
    record["compile_drivers"] = drivers


def compare_resources(current: dict, parent: dict) -> dict:
    """Compare like-for-like records; never guess which kernels correspond."""
    for field in ("collection_context", "device_identity"):
        if not current.get(field) or not parent.get(field):
            return {"status": f"unverified: missing {field}; no deltas computed"}
        if current[field] != parent[field]:
            return {"status": f"not comparable: {field} changed; no deltas computed"}
    name = current.get("primary_kernel")
    if not name or name != parent.get("primary_kernel"):
        return {"status": "kernel names differ; no automatic mapping or deltas"}
    before = parent.get("kernel_properties", {}).get(name, [])
    after = current.get("kernel_properties", {}).get(name, [])
    if len(before) != 1 or len(after) != 1:
        return {"status": "missing or ambiguous kernel-property records; no deltas computed"}
    changes = {}
    for field in sorted(before[0].keys() | after[0].keys()):
        old, new = before[0].get(field), after[0].get(field)
        change = {"parent": old, "current": new}
        if isinstance(old, str) and isinstance(new, str) and old.isdecimal() and new.isdecimal():
            change["delta"] = int(new) - int(old)
        changes[field] = change
    return {
        "status": "matched workload, device, and kernel name; inspect build changes separately",
        "fields": changes,
    }


def index_assembly(directory: Path) -> dict:
    """Inventory IGC dump artifacts without guessing kernel or sampled-IP mappings."""
    files = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        files.append(
            {
                "path": str(path.relative_to(directory)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "kind": "assembly" if path.suffix.lower() in {".asm", ".visaasm"} else "supporting",
            }
        )
    result = {
        "status": "captured"
        if any(item["kind"] == "assembly" for item in files)
        else "unavailable: no assembly emitted",
        "directory": str(directory),
        "files": files,
        "ip_mapping": "unverified; match the exact kernel binary before correlating sampled addresses",
    }
    (directory.parent / "assembly_manifest.json").write_text(json.dumps(result, indent=2))
    return result
