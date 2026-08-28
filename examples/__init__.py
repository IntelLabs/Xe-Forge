"""Example workloads and reference kernels for Xe-Forge and Xe-Orbit.

This file makes `examples` a regular package on purpose. As a namespace package it was
shadowed by any installed distribution that also ships a top-level `examples` module —
which one of the torch-xpu dependencies does — because Python resolves a regular package
ahead of a namespace portion no matter where each sits on `sys.path`. The symptom was
`python -m examples.kernel_replacement.workload` failing only in the GPU environment.
"""
