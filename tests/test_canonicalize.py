"""The reference's canonicalize: compared through, never timed, not up to the trial."""

from __future__ import annotations

import types

import pytest

from xe_forge.core.reference_workload import ReferenceWorkload

REFERENCE = """
import torch
class Model(torch.nn.Module):
    def forward(self, x):
        return x.flip(0)          # rows in some order the op does not promise
{canon}
def get_inputs():
    return [torch.arange(6.0)]
"""
CANON = """
    def canonicalize(self, outputs, x):
        return torch.sort(outputs).values
"""
TRIAL = """
import torch
class Model(torch.nn.Module):
    def forward(self, x):
        return x.roll(2)          # the same rows, another order
    def canonicalize(self, outputs, x):
        return outputs            # a trial's own canonicalize is never used
def get_inputs():
    return [torch.arange(6.0)]
"""


def _compile(code):
    module = types.ModuleType("m")
    exec(code, module.__dict__)
    return module


def _workload(canon: str):
    reference = REFERENCE.format(canon=canon)
    return ReferenceWorkload.prepare(_compile, reference, reference, TRIAL, "cpu", 0.0, 0.0)


def test_outputs_are_compared_in_the_references_canonical_order():
    _workload(CANON).validate()


def test_without_it_a_different_order_is_a_different_answer():
    with pytest.raises(ValueError, match="Output differs"):
        _workload("").validate()
