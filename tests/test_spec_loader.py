"""Tests for spec_loader variant resolution and parsing."""

from xe_forge.core.spec_loader import load_spec_from_string


BASIC_SPEC = """\
inputs:
  X:
    shape: [M, N]
    dtype: float32

ci:
  - params: [X]
    dims: {M: 4, N: 8}
    flop: "2*M*N"

bench-gpu:
  - params: [X]
    dims: {M: 1024, N: 2048}
    flop: "2*M*N"
"""

SPEC_WITH_DEFAULT = """\
inputs:
  X:
    shape: [M, N]
    dtype: float32

default_variant: ci

ci:
  - params: [X]
    dims: {M: 4, N: 8}

bench-gpu:
  - params: [X]
    dims: {M: 1024, N: 2048}
"""

SPEC_WITH_CUSTOM_VARIANTS = """\
inputs:
  X:
    shape: [M, N]
    dtype: float16

default_variant: small

small:
  - params: [X]
    dims: {M: 32, N: 64}
    flop: "2*M*N"

large:
  - params: [X]
    dims: {M: 4096, N: 8192}
    flop: "2*M*N"

bench-gpu:
  - params: [X]
    dims: {M: 1024, N: 2048}
    flop: "2*M*N"
"""

SPEC_WITH_NUMBERED = """\
inputs:
  X:
    shape: [M, N]
    dtype: float16

bench-gpu:
  - params: [X]
    dims: {M: 1024, N: 2048}

bench-gpu-0:
  - params: [X]
    dims: {M: 128, N: 256}

bench-gpu-1:
  - params: [X]
    dims: {M: 512, N: 512}
"""


# --- parse_spec / load_spec_from_string ---


class TestParseBasicSpec:
    def test_base_family_keys_parsed(self):
        spec = load_spec_from_string(BASIC_SPEC)
        assert "ci" in spec.list_variant_keys()
        assert "bench-gpu" in spec.list_variant_keys()

    def test_input_shapes(self):
        spec = load_spec_from_string(BASIC_SPEC)
        assert spec.get_input_shapes("ci") == [(4, 8)]
        assert spec.get_input_shapes("bench-gpu") == [(1024, 2048)]

    def test_unknown_variant_returns_empty(self):
        spec = load_spec_from_string(BASIC_SPEC)
        assert spec.get_input_shapes("nonexistent") == []
        assert spec.get_variant("nonexistent") is None


class TestParseCustomVariants:
    def test_custom_names_listed(self):
        spec = load_spec_from_string(SPEC_WITH_CUSTOM_VARIANTS)
        keys = spec.list_variant_keys()
        assert "small" in keys
        assert "large" in keys
        assert "bench-gpu" in keys

    def test_custom_variant_shapes(self):
        spec = load_spec_from_string(SPEC_WITH_CUSTOM_VARIANTS)
        assert spec.get_input_shapes("small") == [(32, 64)]
        assert spec.get_input_shapes("large") == [(4096, 8192)]


class TestParseNumberedVariants:
    def test_numbered_keys_listed(self):
        spec = load_spec_from_string(SPEC_WITH_NUMBERED)
        keys = spec.list_variant_keys()
        assert "bench-gpu" in keys
        assert "bench-gpu-0" in keys
        assert "bench-gpu-1" in keys

    def test_numbered_variant_shapes(self):
        spec = load_spec_from_string(SPEC_WITH_NUMBERED)
        assert spec.get_input_shapes("bench-gpu-0") == [(128, 256)]
        assert spec.get_input_shapes("bench-gpu-1") == [(512, 512)]


class TestDefaultVariantField:
    def test_default_variant_parsed(self):
        spec = load_spec_from_string(SPEC_WITH_DEFAULT)
        assert spec.default_variant == "ci"

    def test_no_default_variant(self):
        spec = load_spec_from_string(BASIC_SPEC)
        assert spec.default_variant is None


# --- resolve_variant ---


class TestResolveVariant:
    def test_cli_variant_wins(self):
        spec = load_spec_from_string(SPEC_WITH_DEFAULT)
        assert spec.resolve_variant("bench-gpu") == "bench-gpu"

    def test_default_variant_when_no_cli(self):
        spec = load_spec_from_string(SPEC_WITH_DEFAULT)
        assert spec.resolve_variant(None) == "ci"

    def test_fallback_to_bench_gpu(self):
        spec = load_spec_from_string(BASIC_SPEC)
        assert spec.resolve_variant(None) == "bench-gpu"

    def test_cli_overrides_default_variant(self):
        spec = load_spec_from_string(SPEC_WITH_CUSTOM_VARIANTS)
        assert spec.default_variant == "small"
        assert spec.resolve_variant("large") == "large"

    def test_custom_default_variant(self):
        spec = load_spec_from_string(SPEC_WITH_CUSTOM_VARIANTS)
        assert spec.resolve_variant(None) == "small"


# --- load from real YAML files ---


class TestLoadFromFile:
    def test_load_existing_spec(self):
        spec = load_spec_from_string(BASIC_SPEC)
        variant = spec.get_variant("bench-gpu")
        assert variant is not None
        assert variant.dims["M"] == 1024
        assert variant.dims["N"] == 2048
