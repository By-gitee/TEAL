"""
ARM SVE 稀疏 GEMV 自定义算子封装。

提供：
1. C++ 扩展加载与注册。
2. Python 端 wrapper，便于 torch.compile 使用。
3. 简易延迟测量工具。
"""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.cpp_extension import load

from kernels.compile_wrapper import BaseKernel


ROOT = Path(__file__).resolve().parent
CPP_ROOT = ROOT / "cpp_sve_sparse_gemv"
BUILD_DIR = CPP_ROOT / "_build"
EXT_NAME = "teal_sve_sparse_gemv_ext"


def _extra_cflags() -> list[str]:
    if os.name == "nt":
        # MSVC: enable OpenMP when available
        return ["/std:c++17", "/openmp"]

    flags = ["-std=c++17", "-O3", "-fopenmp"]
    arch = platform.machine().lower()
    if arch in {"aarch64", "arm64"}:
        flags.append("-march=armv8-a+sve")
    return flags


def _extra_ldflags() -> list[str]:
    # GCC/Clang: link OpenMP runtime
    if os.name == "nt":
        return []
    return ["-fopenmp"]


def load_sve_sparse_gemv_extension(
    rebuild: bool = False,
    verbose: bool = False,
) -> Optional[torch.types.ModuleType]:
    """
    编译并加载 C++ 扩展。若算子已注册则跳过重复构建。
    """
    # if (
    #     not rebuild
    #     and hasattr(torch.ops, "teal")
    #     and hasattr(torch.ops.teal, "sve_sparse_gemv")
    # ):
    #     return None

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    return load(
        name=EXT_NAME,
        sources=[str(CPP_ROOT / "sve_sparse_gemv_op.cpp")],
        build_directory=str(BUILD_DIR),
        extra_cflags=_extra_cflags(),
        extra_ldflags=_extra_ldflags(),
        verbose=verbose,
    )


class SVESparseGEMVKernel(BaseKernel):
    """
    torch.compile 兼容的 wrapper。
    """

    def meta(
        self,
        activation: torch.Tensor,
        weight: torch.Tensor,
        nz_row: int,
        nz_col_index: torch.Tensor,
    ) -> torch.Tensor:
        return activation.new_empty((weight.size(1),), device="meta")

    def forward(
        self,
        activation: torch.Tensor,
        weight: torch.Tensor,
        nz_row: int,
        nz_col_index: torch.Tensor,
    ) -> torch.Tensor:
        load_sve_sparse_gemv_extension()
        return torch.ops.teal.sve_sparse_gemv(activation, weight, nz_row, nz_col_index)


def measure_latency(
    fn: Callable[[], torch.Tensor],
    warmup: int = 20,
    iters: int = 100,
) -> float:
    """
    简单的端到端延迟测量（毫秒）。
    - CPU: 使用 time.perf_counter
    - GPU: 在关键点同步以避免异步干扰
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for _ in range(warmup):
        fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / iters
