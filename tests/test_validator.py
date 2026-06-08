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


# A valid, contract-honouring SYCL stub.
VALID_SYCL_STUB = """\
#include "cutlass/gemm/device/gemm_universal.h"

int main(int argc, const char** argv) {
    std::string input_dir, output_dir;  // read A.bin/B0.bin, write D2.bin
    return 0;
}
"""

# Missing main(), missing IO contract, no cutlass.
BAD_SYCL = """\
#include <vector>

void helper() {}
"""


class TestSyclValidation:
    def test_valid_sycl_stub_is_clean_of_errors(self):
        issues = KernelValidator().validate(VALID_SYCL_STUB, dsl="sycl")
        errors = [i for i in issues if i.severity == "error"]
        assert errors == [], f"unexpected errors: {[e.check_name for e in errors]}"
        # Contract satisfied -> no missing_io_contract warning.
        assert all(i.check_name != "missing_io_contract" for i in issues)

    def test_missing_main_is_error(self):
        issues = KernelValidator().validate(BAD_SYCL, dsl="sycl")
        main_errs = [i for i in issues if i.check_name == "missing_main"]
        assert len(main_errs) == 1
        assert main_errs[0].severity == "error"

    def test_missing_io_contract_is_warning(self):
        issues = KernelValidator().validate(BAD_SYCL, dsl="sycl")
        io_warns = [i for i in issues if i.check_name == "missing_io_contract"]
        assert len(io_warns) == 1
        assert io_warns[0].severity == "warning"

    def test_no_cutlass_include_is_info(self):
        issues = KernelValidator().validate(BAD_SYCL, dsl="sycl")
        infos = [i for i in issues if i.check_name == "no_cutlass_include"]
        assert len(infos) == 1
        assert infos[0].severity == "info"
