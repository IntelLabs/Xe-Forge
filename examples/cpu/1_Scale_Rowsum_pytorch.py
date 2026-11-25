"""Reference for 1_Scale_Rowsum: ``y[i] = scale * sum_j x[i, j]`` in PyTorch."""

import torch
import torch.nn as nn

rows = 8192
cols = 8192
scale = 0.5


class Model(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = float(scale)

    def forward(self, x):
        return x.sum(dim=1) * self.scale


def get_inputs():
    return [torch.rand(rows, cols, dtype=torch.float32)]


def get_init_inputs():
    return [scale]
