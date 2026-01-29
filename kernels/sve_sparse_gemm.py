"""
ARM SVE 稀疏 GEMV/GEMM 自定义算子封装。

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
BUILD_DIR = CPP_ROOT / "_build"
EXT_NAME = "teal_sve_sparse_gemm_ext"


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


def load_sve_sparse_gemm_extension(
    rebuild: bool = False,
    verbose: bool = False,
) -> Optional[torch.types.ModuleType]:
    """
    编译并加载 C++ 扩展。若算子已注册则跳过重复构建。
    """
    if (
        not rebuild
        and hasattr(torch.ops, "teal")
        and hasattr(torch.ops.teal, "sve_sparse_gemv")
        and hasattr(torch.ops.teal, "sve_sparse_gemm")
        and hasattr(torch.ops.teal, "sve_sparse_act_csr_gemm")
        and hasattr(torch.ops.teal, "sve_sparse_csr_gemm")
        and hasattr(torch.ops.teal, "sve_sparse_act_direct_gemm")
    ):
        return None

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    return load(
        name=EXT_NAME,
        sources=[
            str(CPP_ROOT / "sve_sparse_gemm_op.cpp"),
            str(CPP_ROOT / "sve_sparse_act_csr_gemm_op.cpp"),
            str(CPP_ROOT / "sve_sparse_csr_gemm_op.cpp"),
            str(CPP_ROOT / "sve_sparse_act_direct_gemm_op.cpp"),
        ],
        build_directory=str(BUILD_DIR),
        extra_cflags=_extra_cflags(),
        extra_ldflags=_extra_ldflags(),
        verbose=verbose,
    )


class SVESparseGEMVKernel(BaseKernel):
    """
    torch.compile 兼容的 GEMV wrapper。
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
        load_sve_sparse_gemm_extension()
        return torch.ops.teal.sve_sparse_gemv(activation, weight, nz_row, nz_col_index)


class SVESparseGEMMKernel(BaseKernel):
    """
    torch.compile 兼容的 GEMM wrapper。
    
    Args:
        activation: (M, K) 稀疏激活矩阵
        weight: (K, N) 密集权重矩阵
        row_offsets: int64 [M+1]，前缀和偏移量，row_offsets[m] 表示第 m 行在 nz_col_indices 中的起始位置
        nz_col_indices: 扁平化的列索引向量（uint32）
    
    Returns:
        output: (M, N) 输出矩阵
    """

    def meta(
        self,
        activation: torch.Tensor,
        weight: torch.Tensor,
        row_offsets: torch.Tensor,
        nz_col_indices: torch.Tensor,
    ) -> torch.Tensor:
        M = activation.size(0)
        N = weight.size(1)
        return activation.new_empty((M, N), device="meta")

    def forward(
        self,
        activation: torch.Tensor,
        weight: torch.Tensor,
        row_offsets: torch.Tensor,
        nz_col_indices: torch.Tensor,
    ) -> torch.Tensor:
        load_sve_sparse_gemm_extension()
        return torch.ops.teal.sve_sparse_gemm(activation, weight, row_offsets, nz_col_indices)


class SVESparseActCSRGEMMKernel(BaseKernel):
    """
    torch.compile 兼容的 CSR activation（gather 构建）GEMM wrapper。

    该算子输入与 `sve_sparse_gemm` 一致，但内部会：
    - gather activation 非零值 -> CSR(values, col_idx, row_ptr)
    - CSR × dense weight 得到输出

    Args:
        activation: (M, K) 稀疏激活矩阵（以稠密格式传入）
        weight: (K, N) 稠密权重矩阵
        row_offsets: int64 [M+1]，前缀和偏移量，row_offsets[m] 表示第 m 行在 nz_col_indices 中的起始位置
        nz_col_indices: 扁平化的列索引向量（uint32/int32）

    Returns:
        output: (M, N) 输出矩阵
    """

    def meta(
        self,
        activation: torch.Tensor,
        weight: torch.Tensor,
        row_offsets: torch.Tensor,
        nz_col_indices: torch.Tensor,
    ) -> torch.Tensor:
        M = activation.size(0)
        N = weight.size(1)
        return activation.new_empty((M, N), device="meta")

    def forward(
        self,
        activation: torch.Tensor,
        weight: torch.Tensor,
        row_offsets: torch.Tensor,
        nz_col_indices: torch.Tensor,
    ) -> torch.Tensor:
        load_sve_sparse_gemm_extension()
        return torch.ops.teal.sve_sparse_act_csr_gemm(
            activation, weight, row_offsets, nz_col_indices
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


class SVESparseCSRGEMMKernel(BaseKernel):
    """
    torch.compile 兼容的 CSR GEMM wrapper（使用 sve_sparse_gemm 风格的计算）。

    该算子输入与 `sve_sparse_gemm` 一致，但内部会：
    - 构建 CSR 格式的 activation
    - 从 CSR values 连续 load，使用 gather load weight 进行计算

    Args:
        activation: (M, K) 稀疏激活矩阵（以稠密格式传入）
        weight: (K, N) 稠密权重矩阵
        row_offsets: int64 [M+1]，前缀和偏移量，row_offsets[m] 表示第 m 行在 nz_col_indices 中的起始位置
        nz_col_indices: 扁平化的列索引向量（uint32/int32）

    Returns:
        output: (M, N) 输出矩阵
    """

    def meta(
        self,
        activation: torch.Tensor,
        weight: torch.Tensor,
        row_offsets: torch.Tensor,
        nz_col_indices: torch.Tensor,
    ) -> torch.Tensor:
        M = activation.size(0)
        N = weight.size(1)
        return activation.new_empty((M, N), device="meta")

    def forward(
        self,
        activation: torch.Tensor,
        weight: torch.Tensor,
        row_offsets: torch.Tensor,
        nz_col_indices: torch.Tensor,
    ) -> torch.Tensor:
        load_sve_sparse_gemm_extension()
        return torch.ops.teal.sve_sparse_csr_gemm(
            activation, weight, row_offsets, nz_col_indices
        )


# 便捷函数：直接调用算子
def sve_sparse_act_csr_gemm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    row_offsets: torch.Tensor,
    nz_col_indices: torch.Tensor,
) -> torch.Tensor:
    """
    直接调用 CSR activation（gather 构建）GEMM 算子的便捷函数。
    """
    load_sve_sparse_gemm_extension()
    return torch.ops.teal.sve_sparse_act_csr_gemm(
        activation, weight, row_offsets, nz_col_indices
    )


def sve_sparse_csr_gemm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    row_offsets: torch.Tensor,
    nz_col_indices: torch.Tensor,
) -> torch.Tensor:
    """
    直接调用 CSR GEMM 算子的便捷函数（使用 sve_sparse_gemm 风格的计算）。
    """
    load_sve_sparse_gemm_extension()
    return torch.ops.teal.sve_sparse_csr_gemm(
        activation, weight, row_offsets, nz_col_indices
    )


class SVESparseActDirectGEMMKernel(BaseKernel):
    """
    torch.compile 兼容的直接 GEMM wrapper（不构建 CSR 格式）。

    该算子输入格式与 `sve_sparse_gemm` 一致（row_offsets + nz_col_indices），但内部会：
    - 直接使用 row_offsets / nz_col_indices + activation 进行 GEMM 计算
    - 不构建 CSR 格式，直接从 activation 取值
    - 计算过程中使用 SIMD（SVE）加速

    Args:
        activation: (M, K) 稀疏激活矩阵（以稠密格式传入）
        weight: (K, N) 稠密权重矩阵
        row_offsets: 前缀和数组，长度为 M+1，表示每行在 nz_col_indices 中的起始位置
        nz_col_indices: 扁平化的列索引向量（uint32/int32）

    Returns:
        output: (M, N) 输出矩阵
    """

    def meta(
        self,
        activation: torch.Tensor,
        weight: torch.Tensor,
        row_offsets: torch.Tensor,
        nz_col_indices: torch.Tensor,
    ) -> torch.Tensor:
        M = activation.size(0)
        N = weight.size(1)
        return activation.new_empty((M, N), device="meta")

    def forward(
        self,
        activation: torch.Tensor,
        weight: torch.Tensor,
        row_offsets: torch.Tensor,
        nz_col_indices: torch.Tensor,
    ) -> torch.Tensor:
        load_sve_sparse_gemm_extension()
        return torch.ops.teal.sve_sparse_act_direct_gemm(
            activation, weight, row_offsets, nz_col_indices
        )


def sve_sparse_act_direct_gemm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    row_offsets: torch.Tensor,
    nz_col_indices: torch.Tensor,
) -> torch.Tensor:
    """
    直接调用直接 GEMM 算子的便捷函数（不构建 CSR 格式）。
    
    Args:
        activation: (M, K) 稀疏激活矩阵（以稠密格式传入）
        weight: (K, N) 稠密权重矩阵
        row_offsets: 前缀和数组，长度为 M+1，表示每行在 nz_col_indices 中的起始位置
        nz_col_indices: 扁平化的列索引向量（uint32/int32）
    
    Returns:
        output: (M, N) 输出矩阵
    """
    load_sve_sparse_gemm_extension()
    return torch.ops.teal.sve_sparse_act_direct_gemm(
        activation, weight, row_offsets, nz_col_indices
    )


# kernels/sve_sparse_gemm.py

import os
import torch
from torch.utils.cpp_extension import load

_row_scan_sve_ext = None

def load_row_scan_sve_extension(verbose: bool = False):
    global _row_scan_sve_ext
    if _row_scan_sve_ext is not None:
        return _row_scan_sve_ext

    this_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_dir = os.path.join(this_dir, "cpp_sve_sparse_gemm")
    src = os.path.join(cpp_dir, "row_scan_sve_op.cpp")

    extra_cflags = [
        "-O3",
        "-fopenmp",
        "-march=armv8-a+sve2",
        "-ffast-math",
    ]
    extra_ldflags = [
        "-fopenmp",
        # 如果你的环境需要，取消下一行注释：
        # "-lgomp",
    ]

    _row_scan_sve_ext = load(
        name="row_scan_sve_ext",
        sources=[src],
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        with_cuda=False,
        verbose=verbose,
    )
    return _row_scan_sve_ext


def row_scan_sve(activation: torch.Tensor, threshold: float, verbose: bool = False):
    ext = load_row_scan_sve_extension(verbose=verbose)
    return ext.row_scan_sve(activation, float(threshold))


# Naive version (parallel but no SVE)
_row_scan_naive_ext = None

def load_row_scan_naive_extension(verbose: bool = False):
    global _row_scan_naive_ext
    if _row_scan_naive_ext is not None:
        return _row_scan_naive_ext

    this_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_dir = os.path.join(this_dir, "cpp_sve_sparse_gemm")
    src = os.path.join(cpp_dir, "row_scan_naive_op.cpp")

    # 朴素版本不需要 SVE 相关标志，但保留 OpenMP 并行支持
    if os.name == "nt":
        # Windows/MSVC
        extra_cflags = ["/std:c++17", "/openmp"]
        extra_ldflags = []
    else:
        # Linux/macOS
        extra_cflags = ["-std=c++17", "-O3", "-fopenmp"]
        extra_ldflags = ["-fopenmp"]

    _row_scan_naive_ext = load(
        name="row_scan_naive_ext",
        sources=[src],
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        with_cuda=False,
        verbose=verbose,
    )
    return _row_scan_naive_ext


def row_scan_naive(activation: torch.Tensor, threshold: float, verbose: bool = False):
    """
    朴素版本的行扫描算子（使用 OpenMP 并行，但不使用 SVE 加速）。
    
    Args:
        activation: (M, K) float32 CPU tensor，激活矩阵
        threshold: float，阈值，绝对值大于等于该值的元素被视为非零
        verbose: bool，是否显示编译信息
    
    Returns:
        tuple: (nz_counts, nz_col_indices, row_offsets)
            - nz_counts: int64 tensor，长度为 2 * num_nz_rows，成对存储 (row_idx, count)
            - nz_col_indices: uint32 tensor，扁平化的列索引向量
            - row_offsets: int64 tensor，长度为 M+1，前缀和偏移量
    """
    ext = load_row_scan_naive_extension(verbose=verbose)
    return ext.row_scan_naive(activation, float(threshold))


# Dense to CSR SVE2 version
_row_scan_sve_csr_ext = None

def load_row_scan_sve_csr_extension(verbose: bool = False):
    global _row_scan_sve_csr_ext
    if _row_scan_sve_csr_ext is not None:
        return _row_scan_sve_csr_ext

    this_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_dir = os.path.join(this_dir, "cpp_sve_sparse_gemm")
    src = os.path.join(cpp_dir, "row_scan_sve_csr_op.cpp")

    if os.name == "nt":
        # Windows/MSVC
        extra_cflags = ["/std:c++17", "/openmp"]
        extra_ldflags = []
    else:
        # Linux/macOS
        extra_cflags = [
            "-std=c++17",
            "-O3",
            "-fopenmp",
            "-march=armv8-a+sve2",
            "-ffast-math",
        ]
        extra_ldflags = ["-fopenmp"]

    _row_scan_sve_csr_ext = load(
        name="row_scan_sve_csr_ext",
        sources=[src],
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        with_cuda=False,
        verbose=verbose,
    )
    return _row_scan_sve_csr_ext


def dense_to_csr_sve2(activation: torch.Tensor, threshold: float, verbose: bool = False):
    """
    Dense activation to CSR format using SVE2 compact (CPU).
    
    Args:
        activation: (M, K) float32 CPU tensor，激活矩阵
        threshold: float，阈值，绝对值大于等于该值的元素被视为非零
        verbose: bool，是否显示编译信息
    
    Returns:
        tuple: (row_offsets, col_idx, values)
            - row_offsets: int64 tensor，长度为 M+1，前缀和偏移量
            - col_idx: uint32 tensor，列索引向量
            - values: float32 tensor，非零元素值
    """
    ext = load_row_scan_sve_csr_extension(verbose=verbose)
    return ext.dense_to_csr_sve2(activation, float(threshold))


# Dense to CSR OpenMP version (no SVE)
_row_scan_omp_csr_ext = None

def load_row_scan_omp_csr_extension(verbose: bool = False):
    global _row_scan_omp_csr_ext
    if _row_scan_omp_csr_ext is not None:
        return _row_scan_omp_csr_ext

    this_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_dir = os.path.join(this_dir, "cpp_sve_sparse_gemm")
    src = os.path.join(cpp_dir, "row_scan_omp_csr_op.cpp")

    if os.name == "nt":
        # Windows/MSVC
        extra_cflags = ["/std:c++17", "/openmp"]
        extra_ldflags = []
    else:
        # Linux/macOS
        extra_cflags = [
            "-std=c++17",
            "-O3",
            "-fopenmp",
            "-ffast-math",
        ]
        extra_ldflags = ["-fopenmp"]

    _row_scan_omp_csr_ext = load(
        name="row_scan_omp_csr_ext",
        sources=[src],
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        with_cuda=False,
        verbose=verbose,
    )
    return _row_scan_omp_csr_ext


def dense_to_csr_omp(activation: torch.Tensor, threshold: float, verbose: bool = False):
    """
    Dense activation to CSR format using OpenMP parallelization (CPU, no SVE).
    
    Args:
        activation: (M, K) float32 CPU tensor，激活矩阵
        threshold: float，阈值，绝对值大于等于该值的元素被视为非零
        verbose: bool，是否显示编译信息
    
    Returns:
        tuple: (row_offsets, col_idx, values)
            - row_offsets: int64 tensor，长度为 M+1，前缀和偏移量
            - col_idx: uint32 tensor，列索引向量
            - values: float32 tensor，非零元素值
    """
    ext = load_row_scan_omp_csr_extension(verbose=verbose)
    return ext.dense_to_csr_omp(activation, float(threshold))
