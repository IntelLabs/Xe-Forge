"""Dtype utilities shared by spec_loader and executor."""

import torch

# Dtypes that don't support torch.randn (no floating-point RNG kernel).
# Applies to both XPU and CUDA — this is a PyTorch limitation.
_CAST_REQUIRED_DTYPES = {
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
    torch.int8,
    torch.uint8,
}


def make_rand_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: str = "xpu",
) -> torch.Tensor:
    """Create a random tensor, casting when the target dtype lacks an RNG kernel."""
    if dtype in _CAST_REQUIRED_DTYPES:
        return torch.randn(shape, dtype=torch.float32, device=device).to(dtype)
    return torch.randn(shape, dtype=dtype, device=device)
