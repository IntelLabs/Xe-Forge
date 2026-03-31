"""
Spec Loader - Load KernelBench YAML specs for testing.

Parses YAML spec files to get:
- Input shapes and dtypes
- FLOP calculations
- Test configurations (ci, bench-gpu, bench-cpu, bench-xpu)
- Per-variant correctness tolerances (rtol, atol)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from ai_bench.harness.core import (
    InitKey,
    InKey,
    SpecKey,
    VKey,
    get_atol,
    get_rtol,
    get_torch_dtype,
)
from ai_bench.utils import eval_eq

V_BENCH_XPU = "bench-xpu"

__all__ = [
    "SpecKey",
    "InKey",
    "VKey",
    "InitKey",
    "V_BENCH_XPU",
    "eval_eq",
    "get_atol",
    "get_rtol",
    "get_torch_dtype",
    "InputSpec",
    "VariantSpec",
    "KernelSpec",
    "load_spec",
    "load_spec_from_string",
    "parse_spec",
    "get_test_config_from_spec",
]


@dataclass
class InputSpec:
    """Specification for a single input tensor."""

    name: str
    shape_vars: List[str]  # e.g., ["K", "M"]
    dtype: str  # e.g., "float16"


@dataclass
class VariantSpec:
    """Specification for a test variant."""

    params: List[str]  # Input parameter names
    dims: Dict[str, int]  # Dimension values
    flop_formula: Optional[str] = None  # e.g., "2*M*N*K"
    dtype: Optional[str] = None  # Override dtype
    rtol: Optional[float] = None  # Relative tolerance for correctness
    atol: Optional[float] = None  # Absolute tolerance for correctness


@dataclass
class KernelSpec:
    """Complete kernel specification."""

    inputs: Dict[str, InputSpec]
    inits: List[dict] = field(default_factory=list)
    ci: List[VariantSpec] = field(default_factory=list)
    bench_cpu: List[VariantSpec] = field(default_factory=list)
    bench_gpu: List[VariantSpec] = field(default_factory=list)
    bench_xpu: List[VariantSpec] = field(default_factory=list)

    # Stores numbered variants like bench-gpu-0, bench-gpu-1, ...
    # keyed by their exact YAML key so callers can request them by name.
    _named_variants: Dict[str, List[VariantSpec]] = field(
        default_factory=dict, repr=False
    )

    # Base-prefix → attribute name for the four standard families.
    _VARIANT_MAP_KEYS = {
        "ci": "ci",
        "bench-cpu": "bench_cpu",
        "bench-gpu": "bench_gpu",
        "bench-xpu": "bench_xpu",
    }

    def _variants(self, variant_type: str) -> list:
        """
        Return the list of VariantSpec objects for *variant_type*.

        Handles three cases:
          1. Exact named key stored in _named_variants  (e.g. "bench-gpu-3")
          2. Base family key mapped via _VARIANT_MAP_KEYS (e.g. "bench-gpu")
          3. Unknown key → empty list
        """
        # 1. Exact named key (covers bench-gpu-N and the bare bench-gpu)
        if variant_type in self._named_variants:
            return self._named_variants[variant_type]

        # 2. Standard family fallback
        attr = self._VARIANT_MAP_KEYS.get(variant_type)
        if attr is not None:
            return getattr(self, attr)

        return []

    def get_variant(self, variant_type: str = "bench-gpu") -> Optional[VariantSpec]:
        """Get first variant of specified type."""
        vl = self._variants(variant_type)
        return vl[0] if vl else None

    def get_input_shapes(
        self,
        variant_type: str = "bench-gpu",
        variant_index: int = 0,
    ) -> List[Tuple[int, ...]]:
        """Get input shapes for a variant."""
        vl = self._variants(variant_type)
        if not vl or variant_index >= len(vl):
            return []

        variant = vl[variant_index]
        shapes = []
        for param in variant.params:
            if param in self.inputs:
                input_spec = self.inputs[param]
                shape = tuple(variant.dims[dim] for dim in input_spec.shape_vars)
                shapes.append(shape)
        return shapes

    def get_dtype(
        self,
        variant_type: str = "bench-gpu",
        variant_index: int = 0,
    ):
        """Get torch dtype for variant."""
        vl = self._variants(variant_type)

        # Variant-level dtype override
        if vl and variant_index < len(vl):
            variant = vl[variant_index]
            if variant.dtype:
                return get_torch_dtype(variant.dtype)

        # Fall back to first input dtype
        if self.inputs:
            first_input = next(iter(self.inputs.values()))
            return get_torch_dtype(first_input.dtype)

        return get_torch_dtype("float32")

    def get_flop(
        self,
        variant_type: str = "bench-gpu",
        variant_index: int = 0,
    ) -> Optional[float]:
        """Calculate FLOP count for variant."""
        vl = self._variants(variant_type)
        if not vl or variant_index >= len(vl):
            return None

        variant = vl[variant_index]
        if not variant.flop_formula:
            return None

        if isinstance(variant.flop_formula, (int, float)):
            return float(variant.flop_formula)

        # Substitute dimension values into formula, then evaluate via ai_bench
        formula = str(variant.flop_formula)
        for dim, value in variant.dims.items():
            formula = formula.replace(dim, str(value))
        return eval_eq(formula)

    def get_rtol(
        self,
        variant_type: str = "bench-gpu",
        variant_index: int = 0,
    ) -> Optional[float]:
        """Get relative tolerance from variant spec."""
        vl = self._variants(variant_type)
        if not vl or variant_index >= len(vl):
            return None
        return vl[variant_index].rtol

    def get_atol(
        self,
        variant_type: str = "bench-gpu",
        variant_index: int = 0,
    ) -> Optional[float]:
        """Get absolute tolerance from variant spec."""
        vl = self._variants(variant_type)
        if not vl or variant_index >= len(vl):
            return None
        return vl[variant_index].atol

    def create_inputs(
        self,
        variant_type: str = "bench-gpu",
        variant_index: int = 0,
        device: str = "xpu",
    ) -> list:
        """Create input tensors for a variant."""
        shapes = self.get_input_shapes(variant_type, variant_index)
        dtype = self.get_dtype(variant_type, variant_index)
        return [torch.randn(shape, dtype=dtype, device=device) for shape in shapes]

    def get_init_args(
        self,
        variant_type: str = "bench-gpu",
        variant_index: int = 0,
    ) -> list:
        """Resolve Model __init__ arguments from the inits section."""
        if not self.inits:
            return []

        vl = self._variants(variant_type)
        if not vl or variant_index >= len(vl):
            return []

        variant = vl[variant_index]
        args = []
        for init_entry in self.inits:
            for param_name, dim_var in init_entry.items():
                if dim_var in variant.dims:
                    args.append(variant.dims[dim_var])
                else:
                    try:
                        args.append(int(dim_var))
                    except (ValueError, TypeError):
                        try:
                            args.append(float(dim_var))
                        except (ValueError, TypeError):
                            args.append(dim_var)
        return args

    def list_variant_keys(self) -> List[str]:
        """Return all available variant keys (named + standard families)."""
        keys = list(self._named_variants.keys())
        for base_key, attr in self._VARIANT_MAP_KEYS.items():
            if getattr(self, attr) and base_key not in self._named_variants:
                keys.append(base_key)
        return sorted(keys)


def load_spec(path: str | Path) -> KernelSpec:
    """Load kernel spec from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return parse_spec(data)


def load_spec_from_string(yaml_string: str) -> KernelSpec:
    """Load kernel spec from YAML string."""
    data = yaml.safe_load(yaml_string)
    return parse_spec(data)


def _parse_variant_entry(vd: dict) -> VariantSpec:
    """Parse a single variant dict into a VariantSpec."""
    return VariantSpec(
        params=vd.get(VKey.PARAMS, []),
        dims=vd.get(VKey.DIMS, {}),
        flop_formula=vd.get(VKey.FLOP),
        dtype=vd.get(VKey.TYPE),
        rtol=get_rtol(vd) if VKey.RTOL in vd else None,
        atol=get_atol(vd) if VKey.ATOL in vd else None,
    )


def parse_spec(data: dict) -> KernelSpec:
    """Parse spec dictionary into KernelSpec.

    Handles both the canonical base keys (ci, bench-gpu, bench-cpu, bench-xpu)
    and numbered variants such as bench-gpu-0, bench-gpu-1, bench-gpu-17, etc.
    All keys are stored in _named_variants so _variants() can look them up by
    their exact name.
    """
    inputs: Dict[str, InputSpec] = {}
    if SpecKey.INS in data:
        for name, input_data in data[SpecKey.INS].items():
            inputs[name] = InputSpec(
                name=name,
                shape_vars=input_data.get(InKey.SHAPE, []),
                dtype=input_data.get(InKey.TYPE, "float32"),
            )

    inits = data.get(SpecKey.INITS, [])

    # Known base family keys → KernelSpec attribute names
    BASE_FAMILY_KEYS = {
        SpecKey.V_CI: "ci",
        SpecKey.V_BENCH_CPU: "bench_cpu",
        SpecKey.V_BENCH_GPU: "bench_gpu",
        V_BENCH_XPU: "bench_xpu",
    }

    # Prefixes that identify numbered variant families (bench-gpu-N, etc.)
    NUMBERED_PREFIXES = ("bench-gpu-", "bench-cpu-", "bench-xpu-", "ci-")

    ci: List[VariantSpec] = []
    bench_cpu: List[VariantSpec] = []
    bench_gpu: List[VariantSpec] = []
    bench_xpu: List[VariantSpec] = []
    named_variants: Dict[str, List[VariantSpec]] = {}

    for key, value in data.items():
        if not isinstance(value, list):
            continue  # skip inputs, inits, and scalar keys

        parsed = [_parse_variant_entry(vd) for vd in value]

        if key in BASE_FAMILY_KEYS:
            # Standard family key — populate both the attribute and named_variants
            attr_name = BASE_FAMILY_KEYS[key]
            if attr_name == "ci":
                ci = parsed
            elif attr_name == "bench_cpu":
                bench_cpu = parsed
            elif attr_name == "bench_gpu":
                bench_gpu = parsed
            elif attr_name == "bench_xpu":
                bench_xpu = parsed
            named_variants[key] = parsed

        elif any(key.startswith(prefix) for prefix in NUMBERED_PREFIXES):
            # Numbered variant like bench-gpu-0, bench-gpu-17, etc.
            named_variants[key] = parsed

    return KernelSpec(
        inputs=inputs,
        inits=inits,
        ci=ci,
        bench_cpu=bench_cpu,
        bench_gpu=bench_gpu,
        bench_xpu=bench_xpu,
        _named_variants=named_variants,
    )


def get_test_config_from_spec(
    spec_path: str | Path,
    variant_type: str = "bench-gpu",
    variant_index: int = 0,
) -> dict:
    """Load spec and return test configuration dict for optimizer."""
    spec = load_spec(spec_path)
    return {
        "input_shapes": spec.get_input_shapes(variant_type, variant_index),
        "flop": spec.get_flop(variant_type, variant_index),
        "dtype": spec.get_dtype(variant_type, variant_index),
        "rtol": spec.get_rtol(variant_type, variant_index),
        "atol": spec.get_atol(variant_type, variant_index),
    }