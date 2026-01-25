"""
ARM SVE CSC 稀疏 GEMM 自定义算子封装。

该算子实现了基于 CSC (Compressed Sparse Column) 格式的稀疏矩阵乘法：
1. 使用 scatter store 将稀疏 activation 转换为 CSC 格式
2. 按 weight 矩阵行进行负载均衡分块
3. 使用稀疏 activation 非零值标量 + SIMD 加载 weight 行数据进行加速

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
CPP_ROOT = ROOT / "cpp_sve_sparse_gemm"
BUILD_DIR = CPP_ROOT / "_build_csc"
EXT_NAME = "teal_sve_csc_gemm_ext"


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


def load_sve_csc_gemm_extension(
    rebuild: bool = False,
    verbose: bool = False,
) -> Optional[torch.types.ModuleType]:
    """
    编译并加载 CSC GEMM C++ 扩展。若算子已注册则跳过重复构建。
    """
    if (
        not rebuild
        and hasattr(torch.ops, "teal_csc")
        and hasattr(torch.ops.teal_csc, "sve_csc_gemm")
    ):
        return None

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    return load(
        name=EXT_NAME,
        sources=[str(CPP_ROOT / "sve_csc_gemm_op.cpp")],
        build_directory=str(BUILD_DIR),
        extra_cflags=_extra_cflags(),
        extra_ldflags=_extra_ldflags(),
        verbose=verbose,
    )


class SVECSCGEMMKernel(BaseKernel):
    """
    torch.compile 兼容的 CSC GEMM wrapper。
    
    该算子将稀疏 activation 转换为 CSC 格式，然后进行矩阵乘法。
    计算策略：
    - 按 weight 矩阵的行（即 activation 的列）进行分块
    - 负载均衡分配到多个线程
    - 使用稀疏 activation 非零值标量 × SIMD 加载的 weight 行向量
    
    Args:
        activation: (M, K) 稀疏激活矩阵（以稠密格式传入）
        weight: (K, N) 稠密权重矩阵
        nz_counts: (2 * num_nz_rows) 格式为 [row_idx, count, row_idx, count, ...]
        nz_col_indices: 扁平化的列索引向量
        ncore: 并行线程数，默认为 0（自动使用 OpenMP 默认线程数）
    
    Returns:
        output: (M, N) 输出矩阵
    """

    def meta(
        self,
        activation: torch.Tensor,
        weight: torch.Tensor,
        nz_counts: torch.Tensor,
        nz_col_indices: torch.Tensor,
        ncore: int = 0,
    ) -> torch.Tensor:
        M = activation.size(0)
        N = weight.size(1)
        return activation.new_empty((M, N), device="meta")

    def forward(
        self,
        activation: torch.Tensor,
        weight: torch.Tensor,
        nz_counts: torch.Tensor,
        nz_col_indices: torch.Tensor,
        ncore: int = 0,
    ) -> torch.Tensor:
        load_sve_csc_gemm_extension()
        return torch.ops.teal_csc.sve_csc_gemm(
            activation, weight, nz_counts, nz_col_indices, ncore
        )


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


# 便捷函数：直接调用算子
def sve_csc_gemm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    nz_counts: torch.Tensor,
    nz_col_indices: torch.Tensor,
    ncore: int = 0,
) -> torch.Tensor:
    """
    直接调用 CSC GEMM 算子的便捷函数。
    
    Args:
        activation: (M, K) 稀疏激活矩阵
        weight: (K, N) 稠密权重矩阵
        nz_counts: (2 * num_nz_rows) 格式为 [row_idx, count, ...]
        nz_col_indices: 扁平化的列索引向量
        ncore: 并行线程数，默认 0（自动）
    
    Returns:
        (M, N) 输出矩阵
    """
    load_sve_csc_gemm_extension()
    return torch.ops.teal_csc.sve_csc_gemm(
        activation, weight, nz_counts, nz_col_indices, ncore
    )
