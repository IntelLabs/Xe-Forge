"""An engine's kernel on its own: two builds' check modules side by side (frontend/standalone)."""

from __future__ import annotations

import itertools

import pytest

from x_forge.frontend import standalone

_n = itertools.count()


def _build(root, which, init="", check=None):
    pkg = root / which / "eng"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(init or f"BUILD = {which!r}\n")
    (pkg / "check.py").write_text(check or (
        "from . import BUILD\n"
        "def check(kernel, dims, dtype, rtol, atol, threads=0):\n"
        "    return {'ok': True, 'max_abs': 0.0, 'max_rel': 0.0}\n"
        "def prepare(kernel, dims, dtype, threads=0):\n"
        "    return BUILD\n"))  # fmt: skip
    return pkg


def _alias():
    return f"_xf_test_build{next(_n)}"


def test_two_builds_load_side_by_side_each_its_own(tmp_path):
    a = standalone.load(_alias(), _build(tmp_path, "a"), "eng", "eng.check")
    b = standalone.load(_alias(), _build(tmp_path, "b"), "eng", "eng.check")
    assert a.prepare("k", {}, "float32") == "a" and b.prepare("k", {}, "float32") == "b"


def test_a_package_that_imports_itself_by_name_is_refused(tmp_path, monkeypatch):
    pkg = _build(tmp_path, "a", check="import eng\ndef check(*a, **k): pass\n"
                                      "def prepare(*a, **k): pass\n")  # fmt: skip
    monkeypatch.syspath_prepend(str(pkg.parent))
    with pytest.raises(RuntimeError, match="imports itself by name"):
        standalone.load(_alias(), pkg, "eng", "eng.check")
    import sys

    sys.modules.pop("eng", None)


def test_a_module_without_the_standalone_calls_is_refused(tmp_path):
    pkg = _build(tmp_path, "a", check="def check(*a, **k): pass\n")
    with pytest.raises(RuntimeError, match="offers no prepare"):
        standalone.load(_alias(), pkg, "eng", "eng.check")
    with pytest.raises(RuntimeError, match="not inside the package"):
        standalone.load(_alias(), pkg, "eng", "other.check")


def test_the_slot_swap_cancels_a_slots_bias():
    from x_forge.core.executor import combine_swapped

    # b is twice as fast; the first slot reads 10% slower whatever runs in it
    ab = (2.2, 1.0, [2.2, 2.2])  # a first: a/b = 2 x 1.1
    ba = (1.1, 2.0, [1.1 / 2.0] * 2)  # b first: b/a = 0.5 x 1.1
    a_us, b_us, ratios = combine_swapped(ab, ba)
    assert ratios == pytest.approx([2.0, 2.0])
    assert a_us == pytest.approx((2.2 * 2.0) ** 0.5) and b_us == pytest.approx((1.0 * 1.1) ** 0.5)


def test_a_library_that_binds_a_symbol_once_per_process_is_found(tmp_path):
    import shutil
    import subprocess

    cxx = shutil.which("g++")
    if cxx is None:
        pytest.skip("no g++ here")
    src = tmp_path / "u.cpp"
    src.write_text(
        'inline int table[4] = {1, 2, 3, 4};\nextern "C" int get(int i) { return table[i]; }\n'
    )
    for flags, unique in (([], True), (["-fno-gnu-unique"], False)):
        lib = tmp_path / f"u{len(flags)}.so"
        subprocess.run([cxx, "-std=c++17", "-shared", "-fPIC", *flags, str(src), "-o", str(lib)],
                       check=True)  # fmt: skip
        assert bool(standalone.unique_symbols(lib)) is unique


def test_a_gpu_kernel_is_warmed_up_at_least_the_minimum(monkeypatch):
    monkeypatch.setenv("X_FORGE_KERNEL_WARMUP", "10")
    monkeypatch.setenv("X_FORGE_DEVICE", "cpu")
    assert standalone._executor().warmup_iters == 10
    monkeypatch.setenv("X_FORGE_DEVICE", "xpu")
    ex = standalone._executor()
    assert ex.device == "xpu" and ex.warmup_iters == standalone.DEVICE_MIN_WARMUP
    monkeypatch.setenv("X_FORGE_KERNEL_WARMUP", "300")
    assert standalone._executor().warmup_iters == 300
