import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gen_kernel_sbatch.py"


def test_one_sbatch_per_enabled_kernel_variant(tmp_path):
    spec = tmp_path / "k.yaml"
    spec.write_text("default_variant: v\n")
    kernels = [
        {"name": "tk", "repo": "vllm", "dsl": "triton", "spec": str(spec), "variants": ["v"]},
        {"name": "sk", "repo": str(tmp_path), "dsl": "sycl", "spec": str(spec), "variants": ["v"]},
        {
            "name": "off",
            "repo": "vllm",
            "dsl": "triton",
            "spec": str(spec),
            "variants": ["v"],
            "enabled": False,
        },
    ]
    manifest = tmp_path / "kernels.yaml"
    manifest.write_text(yaml.safe_dump({"partition": "p", "kernels": kernels}))
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(SCRIPT), str(manifest), "--out", str(out)], check=True)

    assert sorted(p.name for p in out.glob("*.sbatch")) == ["sk__v.sbatch", "tk__v.sbatch"]
    triton, sycl = (out / "tk__v.sbatch").read_text(), (out / "sk__v.sbatch").read_text()
    assert "--dsl triton" in triton and "--compiler-flags" not in triton
    assert "--dsl sycl" in sycl and '--compiler-flags "$SYCL_COMPILE_FLAGS"' in sycl
    assert f"KERNEL_REPO={tmp_path}\n" in sycl and "/vllm\n" in triton
    for text in (triton, sycl):
        assert "#SBATCH --partition=p" in text and "VARIANT=v" in text and "--reference" not in text
        subprocess.run(["bash", "-n"], input=text, text=True, check=True)
    submit = (out / "submit_all.sh").read_text()
    assert submit.count("sbatch --parsable") == 2 and "--dependency=afterany:$prev" in submit
    subprocess.run(["bash", "-n"], input=submit, text=True, check=True)


def test_entry_compiler_flags_replace_the_default(tmp_path):
    spec = tmp_path / "k.yaml"
    spec.write_text("default_variant: v\n")
    base = {"repo": str(tmp_path), "dsl": "sycl", "spec": str(spec), "variants": ["v"]}
    kernels = [
        {"name": "own", "compiler_flags": "-O3 -I$KERNEL_REPO/csrc", **base},
        {"name": "dflt", **base},
    ]
    manifest = tmp_path / "kernels.yaml"
    manifest.write_text(yaml.safe_dump({"kernels": kernels}))
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(SCRIPT), str(manifest), "--out", str(out)], check=True)

    own, dflt = (out / "own__v.sbatch").read_text(), (out / "dflt__v.sbatch").read_text()
    default = own.index('SYCL_COMPILE_FLAGS="-O2')
    assert own.index('SYCL_COMPILE_FLAGS="-O3 -I$KERNEL_REPO/csrc"\n') > default
    assert "-O3" not in dflt and 'SYCL_COMPILE_FLAGS="-O2' in dflt
    subprocess.run(["bash", "-n"], input=own, text=True, check=True)
