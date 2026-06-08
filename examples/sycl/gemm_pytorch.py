"""Golden PyTorch reference for a plain bf16 GEMM: D = A @ B0 (f32 accumulate)."""

import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, A, B0):
        # f32 accumulate to match the kernel's ElementAccumulator = float.
        return A.float() @ B0.float()


def get_inputs():
    # Shapes are illustrative; the benchmark harness feeds A/B0 from the .bin
    # files generated for the spec's dims, not from here.
    return [
        torch.randn(1024, 1024, dtype=torch.bfloat16),
        torch.randn(1024, 1024, dtype=torch.bfloat16),
    ]


def get_init_inputs():
    return []
