"""The Xe-Fuse checkout executor (plan §13.4, §9.6).

Drives the generate → compile → run flow with a fake checkout and a stub compiler,
so the seam is tested without oneAPI, a GPU, or the real repository. What the real
run established (the +3.1% decode win, the -25% prefill loss, the three upstream
bugs) lives in plan.md; what these tests hold is the seam's honesty — which stage
failed is always named, and a missing checkout is an option lost, not a crash.
"""

from __future__ import annotations

import stat

import pytest

from xe_forge.orbit.optimize.fusion import FusionTask
from xe_forge.orbit.optimize.xe_fuse_executor import (
    PRESET_FOR_PATTERN,
    find_sycl_tla,
    find_xe_fuse,
    run_preset,
    run_region,
)


@pytest.fixture
def fake_checkouts(tmp_path, monkeypatch):
    """An Xe-Fuse checkout whose generator writes a cpp, plus a sycl-tla skeleton."""
    fuse = tmp_path / "Xe-Fuse"
    (fuse / "autotune").mkdir(parents=True)
    (fuse / "include").mkdir()
    generator = fuse / "autotune" / "generate_kernel.py"
    generator.write_text(
        "import sys\n"
        "out = sys.argv[sys.argv.index('-o') + 1]\n"
        "open(out, 'w').write('// generated stub kernel\\n')\n"
    )
    tla = tmp_path / "sycl-tla"
    (tla / "include").mkdir(parents=True)
    monkeypatch.setenv("ORBIT_XE_FUSE_DIR", str(fuse))
    monkeypatch.setenv("SYCL_TLA_DIR", str(tla))
    # cmd_fuse holds the device lease around the sweep; tests must contend in a
    # private lease dir, never with a live measurement on this machine (E2).
    monkeypatch.setenv("ORBIT_LEASE_DIR", str(tmp_path / "leases"))
    return fuse, tla


@pytest.fixture
def stub_compiler(tmp_path):
    """A 'compiler' that emits a runnable 'binary' printing a benchmark line."""
    compiler = tmp_path / "fake-icpx"
    compiler.write_text(
        "#!/bin/sh\n"
        "# find the -o argument and write a runnable script there\n"
        "out=''\n"
        "prev=''\n"
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
        'printf \'#!/bin/sh\\necho "K2: [0.766]TFlop/s  (0.3640)ms"\\n\' > "$out"\n'
        'chmod +x "$out"\n'
    )
    compiler.chmod(compiler.stat().st_mode | stat.S_IEXEC)
    return str(compiler)


class TestDiscovery:
    def test_env_override_wins(self, fake_checkouts):
        fuse, tla = fake_checkouts
        assert find_xe_fuse() == fuse
        assert find_sycl_tla() == tla

    def test_absence_is_none_not_an_exception(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ORBIT_XE_FUSE_DIR", str(tmp_path / "nope"))
        monkeypatch.setenv("SYCL_TLA_DIR", str(tmp_path / "nope"))
        monkeypatch.setenv("ORBIT_SYCL_SOURCES", str(tmp_path / "empty"))
        assert find_xe_fuse() is None
        assert find_sycl_tla() is None


class TestRunPreset:
    def test_full_flow_parses_the_measurement(self, fake_checkouts, stub_compiler, tmp_path):
        result = run_preset("k2", 16, 9728, 896, tmp_path / "out", compiler=stub_compiler)
        assert result.ok, result.error
        assert result.ms == pytest.approx(0.364)
        assert result.tflops == pytest.approx(0.766)
        assert result.per_iteration_us == pytest.approx(364.0)
        # The caveat travels with the number: the binary does not verify (upstream
        # --verify is inert), so the note must say correctness is the caller's gate.
        assert any("--verify is inert" in n for n in result.notes)

    def test_missing_checkout_names_what_is_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ORBIT_XE_FUSE_DIR", str(tmp_path / "nope"))
        monkeypatch.setenv("SYCL_TLA_DIR", str(tmp_path / "nope"))
        monkeypatch.setenv("ORBIT_SYCL_SOURCES", str(tmp_path / "empty"))
        result = run_preset("k2", 16, 9728, 896, tmp_path / "out", compiler="cc")
        assert not result.ok
        assert "Xe-Fuse" in result.error and "sycl-tla" in result.error

    def test_generator_failure_is_the_named_stage(self, fake_checkouts, tmp_path):
        fuse, _ = fake_checkouts
        (fuse / "autotune" / "generate_kernel.py").write_text("import sys; sys.exit(3)\n")
        result = run_preset("k2", 16, 9728, 896, tmp_path / "out", compiler="cc")
        assert not result.ok and result.error.startswith("generate failed")

    def test_compile_failure_is_the_named_stage(self, fake_checkouts, tmp_path):
        result = run_preset("k2", 16, 9728, 896, tmp_path / "out", compiler="/bin/false")
        assert not result.ok and result.error.startswith("compile failed")

    def test_unparseable_output_is_refused_not_defaulted(self, fake_checkouts, tmp_path):
        compiler = tmp_path / "weird-icpx"
        compiler.write_text(
            "#!/bin/sh\n"
            "out=''; prev=''\n"
            'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
            "printf '#!/bin/sh\\necho no numbers here\\n' > \"$out\"\n"
            'chmod +x "$out"\n'
        )
        compiler.chmod(compiler.stat().st_mode | stat.S_IEXEC)
        result = run_preset("k2", 16, 9728, 896, tmp_path / "out", compiler=str(compiler))
        assert not result.ok and "could not parse timing" in result.error


class TestRunRegion:
    def test_pattern_maps_to_preset(self):
        assert PRESET_FOR_PATTERN["gemm+activation"] == "k2"
        assert PRESET_FOR_PATTERN["gemm+rmsnorm"] == "k1"

    def test_unknown_pattern_is_an_honest_refusal(self, tmp_path):
        task = FusionTask(region_id="r9", pattern="gemm+mystery", gpu_share=0.1)
        result = run_region(task, (16, 9728, 896), tmp_path / "out")
        assert not result.ok
        assert "gemm+mystery" in result.error and "known" in result.error

    def test_known_pattern_flows_through(self, fake_checkouts, stub_compiler, tmp_path):
        task = FusionTask(region_id="r0", pattern="gemm+activation", gpu_share=0.4)
        result = run_region(task, (16, 9728, 896), tmp_path / "out", compiler=stub_compiler)
        assert result.ok and result.preset == "k2"


class TestAvailabilityFeedsTheRoute:
    def test_checkout_makes_xe_fuse_available(self, fake_checkouts):
        from xe_forge.orbit.analysis.xe_fuse import xe_fuse_available

        assert xe_fuse_available()

    def test_absence_degrades_to_author_not_a_dead_end(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ORBIT_XE_FUSE_DIR", str(tmp_path / "nope"))
        monkeypatch.setenv("SYCL_TLA_DIR", str(tmp_path / "nope"))
        monkeypatch.setenv("ORBIT_SYCL_SOURCES", str(tmp_path / "empty"))
        # With no importable module and no checkout, the region route must fall
        # back to authoring (§13.4) — an option lost, never the path.
        import importlib.util as iu

        if iu.find_spec("xe_fuse") is None:
            from xe_forge.orbit.analysis.xe_fuse import xe_fuse_available
            from xe_forge.orbit.optimize.fusion import FusionExecutor, default_executor

            assert not xe_fuse_available()

            class Region:
                fusion_pattern = "gemm+activation"

            assert default_executor(Region()) is FusionExecutor.AUTHOR


@pytest.fixture
def tile_aware_compiler(tmp_path):
    """A 'compiler' whose binary reports a timing keyed to the tile in its name.

    16x256x32 is scripted fastest, auto middling, everything else slow — so the
    sweep's selection logic is observable without hardware.
    """
    compiler = tmp_path / "tile-icpx"
    compiler.write_text(
        "#!/bin/sh\n"
        "out=''; prev=''\n"
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
        'case "$out" in\n'
        "  *16_256_32*) ms=0.300 ;;\n"
        "  *auto*) ms=0.360 ;;\n"
        "  *) ms=0.500 ;;\n"
        "esac\n"
        'printf "#!/bin/sh\\necho \\"K2: [1.0]TFlop/s  (%s)ms\\"\\n" "$ms" > "$out"\n'
        'chmod +x "$out"\n'
    )
    compiler.chmod(compiler.stat().st_mode | stat.S_IEXEC)
    return str(compiler)


class TestAutotune:
    def test_candidate_tiles_track_problem_m(self):
        from xe_forge.orbit.optimize.xe_fuse_executor import candidate_tiles

        # The measured Wildcat Lake pattern: tile_m follows M, auto always included.
        assert candidate_tiles(16)[0] == "auto"
        assert any(t.startswith("16x") for t in candidate_tiles(16))
        assert any(t.startswith("32x") for t in candidate_tiles(32))
        assert any(t.startswith("128x") for t in candidate_tiles(128))
        # Bounded above: absurd M does not produce an absurd tile.
        assert all(int(t.split("x")[0]) <= 256 for t in candidate_tiles(4096) if t != "auto")

    def test_sweep_picks_the_measured_best_and_keeps_the_rest(
        self, fake_checkouts, tile_aware_compiler, tmp_path
    ):
        from xe_forge.orbit.optimize.xe_fuse_executor import autotune_preset

        sweep = autotune_preset(
            "k2",
            16,
            9728,
            896,
            tmp_path / "out",
            tiles=["auto", "16x256x32", "32x256x32"],
            compiler=tile_aware_compiler,
        )
        assert sweep.best is not None and sweep.best.tile == "16x256x32"
        assert len(sweep.results) == 3  # nothing silently dropped
        rendered = sweep.format()
        assert "BEST" in rendered and "auto" in rendered
        # The sweep beating auto is itself a finding, and the table says so.
        assert "beat the generator's auto pick" in rendered

    def test_failed_tiles_stay_in_the_table(self, fake_checkouts, tmp_path):
        from xe_forge.orbit.optimize.xe_fuse_executor import autotune_preset

        sweep = autotune_preset(
            "k2",
            16,
            9728,
            896,
            tmp_path / "out",
            tiles=["16x256x32"],
            compiler="/bin/false",
        )
        assert sweep.best is None
        assert "compile failed" in sweep.format()
        assert "nothing to choose" in sweep.format()

    def test_region_autotune_maps_pattern_first(
        self, fake_checkouts, tile_aware_compiler, tmp_path
    ):
        from xe_forge.orbit.optimize.xe_fuse_executor import autotune_region

        task = FusionTask(region_id="r0", pattern="gemm+activation", gpu_share=0.4)
        sweep = autotune_region(
            task,
            (16, 9728, 896),
            tmp_path / "out",
            tiles=["auto", "16x256x32"],
            compiler=tile_aware_compiler,
        )
        assert sweep.preset == "k2" and sweep.best.tile == "16x256x32"

    def test_unknown_pattern_sweep_is_an_honest_refusal(self, tmp_path):
        from xe_forge.orbit.optimize.xe_fuse_executor import autotune_region

        task = FusionTask(region_id="r9", pattern="gemm+mystery", gpu_share=0.1)
        sweep = autotune_region(task, (16, 9728, 896), tmp_path / "out")
        assert sweep.best is None and "gemm+mystery" in sweep.results[0].error


class TestCmdFuse:
    """The `xe-orbit fuse` seam: catalog in, sweep out, honesty throughout."""

    def _store_with_region(self, tmp_path, dims):
        from xe_forge.orbit.artifacts import KERNEL_CATALOG, RunStore
        from xe_forge.orbit.models import (
            KernelCatalog,
            KernelRecord,
            RegionRecord,
            ShapeObservation,
        )

        store = RunStore.create(base=tmp_path / ".orbit")
        catalog = KernelCatalog(
            run_id=store.run_id,
            kernels=[
                KernelRecord(
                    id="k0",
                    runtime_name="gemm_kernel_onednn",
                    shapes=[ShapeObservation(dims=dims, count=10)],
                )
            ],
            regions=[
                RegionRecord(
                    id="r0",
                    kernel_ids=["k0"],
                    fusion_pattern="gemm+activation",
                    gpu_time_share=0.4,
                )
            ],
        )
        store.save(KERNEL_CATALOG, catalog)
        return store

    def _args(self, store, **over):
        import argparse

        base = {
            "orbit_dir": str(store.root),
            "run": store.run_id,
            "replay": None,
            "region_id": "r0",
            "shapes": None,
            "tiles": None,
            "iterations": 300,
        }
        base.update(over)
        return argparse.Namespace(**base)

    def _stub_sweep(self, monkeypatch, record):
        import xe_forge.orbit.optimize.xe_fuse_executor as mod

        def fake_run_preset(preset, m, n, k, output_dir, *, tile="auto", **kw):
            record.append((preset, m, n, k, tile))
            result = mod.XeFuseResult(preset=preset, tile=tile, m=m, n=n, k=k)
            result.ms = 0.35 if tile != "auto" else 0.36
            return result

        monkeypatch.setattr(mod, "run_preset", fake_run_preset)

    def test_shapes_derived_from_the_gemm_member(self, fake_checkouts, monkeypatch, tmp_path):
        from xe_forge.orbit.cli import cmd_fuse

        calls = []
        self._stub_sweep(monkeypatch, calls)
        store = self._store_with_region(
            tmp_path, {"a0_d0": 16, "a0_d1": 896, "a1_d0": 9728, "a1_d1": 896}
        )
        code = cmd_fuse(self._args(store, tiles="auto,16x256x32"))
        assert code == 0
        assert calls and calls[0][1:4] == (16, 9728, 896)
        sweep_file = store.run_dir / "experiments" / "r0" / "xe_fuse_sweep.json"
        assert sweep_file.is_file()

    def test_shapes_flag_overrides_derivation(self, fake_checkouts, monkeypatch, tmp_path):
        from xe_forge.orbit.cli import cmd_fuse

        calls = []
        self._stub_sweep(monkeypatch, calls)
        store = self._store_with_region(tmp_path, {})
        assert cmd_fuse(self._args(store, shapes="8x512x256", tiles="auto")) == 0
        assert calls[0][1:4] == (8, 512, 256)

    def test_ambiguous_shapes_ask_rather_than_guess(
        self, fake_checkouts, monkeypatch, tmp_path, capsys
    ):
        from xe_forge.orbit.cli import cmd_fuse

        self._stub_sweep(monkeypatch, [])
        # Square: both weight dims equal K — M/N/K assignment is undecidable.
        store = self._store_with_region(
            tmp_path, {"a0_d0": 896, "a0_d1": 896, "a1_d0": 896, "a1_d1": 896}
        )
        assert cmd_fuse(self._args(store)) == 1
        assert "--shapes" in capsys.readouterr().out

    def test_unknown_region_is_a_named_miss(self, fake_checkouts, tmp_path, capsys):
        from xe_forge.orbit.cli import cmd_fuse

        store = self._store_with_region(tmp_path, {})
        assert cmd_fuse(self._args(store, region_id="r9")) == 1
        assert "r9" in capsys.readouterr().out

    def test_missing_checkouts_refuse_before_any_work(self, monkeypatch, tmp_path, capsys):
        from xe_forge.orbit.cli import cmd_fuse

        monkeypatch.setenv("ORBIT_XE_FUSE_DIR", str(tmp_path / "nope"))
        monkeypatch.setenv("SYCL_TLA_DIR", str(tmp_path / "nope"))
        store = self._store_with_region(tmp_path, {})
        assert cmd_fuse(self._args(store)) == 1
        assert "checkout" in capsys.readouterr().out.lower()
