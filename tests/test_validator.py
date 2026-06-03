"""Tests for static kernel validation."""

from xe_forge.core.validator import KernelValidator

VALID_1D_SWIZZLED_GRID = """\
import triton
import triton.language as tl

GROUP_SIZE_M = 4


@triton.jit
def kernel():
    pass


class Model:
    pass


def launch():
    grid = lambda META: (triton.cdiv(M, META["BM"]) * triton.cdiv(N, META["BN"]),)
    kernel[grid]()
"""


INVALID_2D_SWIZZLED_GRID = """\
import triton
import triton.language as tl

GROUP_SIZE_M = 4


@triton.jit
def kernel():
    pass


class Model:
    pass


def launch():
    grid = lambda META: (triton.cdiv(M, META["BM"]), triton.cdiv(N, META["BN"]))
    kernel[grid]()
"""


INVALID_2D_TUPLE_SWIZZLED_GRID = """\
import triton
import triton.language as tl

GROUP_SIZE_M = 4


@triton.jit
def kernel():
    pass


class Model:
    pass


def launch():
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 256))
    kernel[grid]()
"""


class TestGridSwizzleValidation:
    def test_1d_grid_with_swizzle_is_allowed(self):
        issues = KernelValidator().validate(VALID_1D_SWIZZLED_GRID, dsl="triton")
        assert all(issue.check_name != "grid_swizzle_conflict" for issue in issues)

    def test_2d_grid_with_swizzle_is_rejected(self):
        issues = KernelValidator().validate(INVALID_2D_SWIZZLED_GRID, dsl="triton")
        assert any(issue.check_name == "grid_swizzle_conflict" for issue in issues)

    def test_2d_tuple_grid_with_swizzle_is_rejected(self):
        issues = KernelValidator().validate(INVALID_2D_TUPLE_SWIZZLED_GRID, dsl="triton")
        assert any(issue.check_name == "grid_swizzle_conflict" for issue in issues)
