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


BLOCK_PTR_KERNEL = """\
import triton
import triton.language as tl


@triton.jit
def kernel(x_ptr, M, K, stride_xm, BM: tl.constexpr, BK: tl.constexpr):
    x_bp = tl.make_block_ptr(x_ptr, (M, K), (stride_xm, 1), (0, 0), (BM, BK), (1, 0))
    for _ in range(0, K, BK):
        x = tl.load(x_bp, boundary_check=(0, 1))
        x_bp = tl.advance(x_bp, (0, BK))


class Model:
    pass
"""


TENSOR_DESCRIPTOR_KERNEL = """\
import triton
import triton.language as tl


@triton.jit
def kernel(x_ptr, M, K, stride_xm, BM: tl.constexpr, BK: tl.constexpr):
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=(M, K), strides=(stride_xm, 1), block_shape=(BM, BK))
    off_k = 0
    for _ in range(0, K, BK):
        x = x_desc.load([0, off_k])
        off_k += BK


class Model:
    pass
"""


DESCRIPTOR_INDEXED_BY_LOOP_VARIABLE = """\
import triton
import triton.language as tl


@triton.jit
def kernel(x_ptr, w_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=(M, K), strides=(K, 1), block_shape=(BM, BK))
    w_desc = tl.make_tensor_descriptor(w_ptr, shape=(K, N), strides=(N, 1), block_shape=(BK, BN))
    for k in range(0, K, BK):
        x = x_desc.load([0, k])
        w = w_desc.load([k, 0])


class Model:
    pass
"""


DESCRIPTOR_WITH_BOUNDARY_CHECK = TENSOR_DESCRIPTOR_KERNEL.replace(
    "x_desc.load([0, off_k])", "x_desc.load([0, off_k], boundary_check=(0, 1))"
)


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


class TestTensorDescriptorValidation:
    def test_block_pointer_api_is_flagged_deprecated(self):
        issues = KernelValidator().validate(BLOCK_PTR_KERNEL, dsl="triton")
        flagged = [i for i in issues if i.check_name == "deprecated_block_ptr"]
        assert len(flagged) == 1
        assert flagged[0].severity == "warning"

    def test_tensor_descriptor_kernel_is_not_flagged(self):
        issues = KernelValidator().validate(TENSOR_DESCRIPTOR_KERNEL, dsl="triton")
        names = {i.check_name for i in issues}
        assert "deprecated_block_ptr" not in names
        assert "descriptor_boundary_check" not in names

    def test_boundary_check_on_descriptor_load_is_rejected(self):
        issues = KernelValidator().validate(DESCRIPTOR_WITH_BOUNDARY_CHECK, dsl="triton")
        assert any(
            i.check_name == "descriptor_boundary_check" and i.severity == "error" for i in issues
        )

    def test_loop_variable_in_contiguous_dim_is_flagged(self):
        issues = KernelValidator().validate(DESCRIPTOR_INDEXED_BY_LOOP_VARIABLE, dsl="triton")
        flagged = [i for i in issues if i.check_name == "descriptor_loop_variable_index"]
        # x_desc.load([0, k]) only: w_desc.load([k, 0]) has k in the row dim
        assert len(flagged) == 1
        assert flagged[0].severity == "warning"

    def test_carried_offset_is_not_flagged(self):
        issues = KernelValidator().validate(TENSOR_DESCRIPTOR_KERNEL, dsl="triton")
        assert all(i.check_name != "descriptor_loop_variable_index" for i in issues)
