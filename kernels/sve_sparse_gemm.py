"""
ARM SVE 稀疏 GEMV/GEMM 自定义算子封装。

该模块实现了多种稀疏矩阵乘法格式：
1. iCSR (Implicit CSR) 格式：直接使用 row_offsets + nz_col_indices
2. CSR 格式：构建完整的 CSR 格式进行矩阵乘法
3. CSC (Compressed Sparse Column) 格式：基于 CSC 格式的稀疏矩阵乘法
   - 直接使用输入的 CSC 格式数据 (values, row_indices, col_ptr)
   - 按 weight 矩阵行进行负载均衡分块
   - 使用稀疏 activation 非零值标量 + SIMD 加载 weight 行数据进行加速
4. COO (Coordinate) 格式：基于 COO 格式的稀疏矩阵乘法
   - 直接使用输入的 COO 格式数据 (row_indices, col_indices, values)
   - 要求 row_indices 已按行排序
   - 使用 OpenMP 并行和 SVE 向量化加速
   - COO SVE Gather 版本：使用 SVE gather 指令优化

稀疏化工具：
1. thr_sparsify_to_icsr: 稠密矩阵转换为 iCSR 格式（OpenMP 并行）
2. thr_sparsify_to_icsr_sve: 稠密矩阵转换为 iCSR 格式（SVE 加速）
3. thr_sparsify_to_csr: 稠密矩阵转换为 CSR 格式（OpenMP 并行）
4. thr_sparsify_to_csr_sve: 稠密矩阵转换为 CSR 格式（SVE 加速）
5. thr_sparsify_to_coo: 稠密矩阵转换为 COO 格式（OpenMP 并行）
6. thr_sparsify_to_coo_sve: 稠密矩阵转换为 COO 格式（SVE/SVE2 加速）
7. thr_sparsify_to_csc: 稠密矩阵转换为 CSC 格式（OpenMP 并行）

提供：
1. C++ 扩展加载与注册。
2. Python 端 wrapper，便于 torch.compile 使用。
3. 简易延迟测量工具。
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
from types import ModuleType
from typing import Optional

import torch
from torch.utils.cpp_extension import load

from kernels.compile_wrapper import BaseKernel
from kernels.kernel_utils import measure_latency as measure_latency


ROOT = Path(__file__).resolve().parent
CPP_ROOT = ROOT / "cpp_sve_sparse_gemm"
BUILD_DIR = CPP_ROOT / "_build"
EXT_NAME = "sve_sparse_gemm_ext"


def _extra_cflags() -> list[str]:
    if os.name == "nt":
        # MSVC: enable OpenMP when available
        return ["/std:c++17", "/openmp"]

    flags = ["-std=c++17", "-O3", "-fopenmp"]
    arch = platform.machine().lower()
    if arch in {"aarch64", "arm64"}:
        flags.append("-march=armv8-a+sve2")
    return flags


def _extra_ldflags() -> list[str]:
    # GCC/Clang: link OpenMP runtime
    if os.name == "nt":
        return []
    return ["-fopenmp"]


def load_sve_sparse_gemm_extension(
    rebuild: bool = False,
    verbose: bool = False,
) -> Optional[ModuleType]:
    """
    编译并加载 C++ 扩展。若算子已注册则跳过重复构建。
    """
    if (
        not rebuild
        and hasattr(torch.ops, "sparse_op")
        and hasattr(torch.ops.sparse_op, "sparse_gemv_icsr_sve_gather")
        and hasattr(torch.ops.sparse_op, "sparse_gemm_icsr_sve_gather")
        and hasattr(torch.ops.sparse_op, "sparse_gemm_csr")
        and hasattr(torch.ops.sparse_op, "sparse_gemm_csr_sve_gather")
        and hasattr(torch.ops.sparse_op, "sparse_gemm_icsr")
        and hasattr(torch.ops.sparse_op, "sparse_gemm_csc")
        and hasattr(torch.ops.sparse_op, "sparse_gemm_coo")
        and hasattr(torch.ops.sparse_op, "sparse_gemm_coo_sve_gather")
        and hasattr(torch.ops.sparse_op, "thr_sparsify_to_icsr")
        and hasattr(torch.ops.sparse_op, "thr_sparsify_to_icsr_sve")
        and hasattr(torch.ops.sparse_op, "thr_sparsify_to_icsr_sve_baseline")
        and hasattr(torch.ops.sparse_op, "thr_sparsify_to_csr")
        and hasattr(torch.ops.sparse_op, "thr_sparsify_to_csr_sve")
        and hasattr(torch.ops.sparse_op, "thr_sparsify_to_coo")
        and hasattr(torch.ops.sparse_op, "thr_sparsify_to_coo_sve")
        and hasattr(torch.ops.sparse_op, "thr_sparsify_to_csc")
        and hasattr(torch.ops.sparse_op, "mask_sparsify_to_icsr")
        and hasattr(torch.ops.sparse_op, "mask_sparsify_to_icsr_sve")
        and hasattr(torch.ops.sparse_op, "mask_sparsify_to_csr")
        and hasattr(torch.ops.sparse_op, "mask_sparsify_to_csr_sve")
        and hasattr(torch.ops.sparse_op, "mask_sparsify_to_coo")
        and hasattr(torch.ops.sparse_op, "mask_sparsify_to_coo_sve")
        and hasattr(torch.ops.sparse_op, "mask_sparsify_to_csc_scatter")
    ):
        return None

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    return load(
        name=EXT_NAME,
        sources=[
            str(CPP_ROOT / "sparse_gemm_icsr_sve_gather_op.cpp"),
            str(CPP_ROOT / "sparse_gemm_csr_op.cpp"),
            str(CPP_ROOT / "sparse_gemm_csr_sve_gather_op.cpp"),
            str(CPP_ROOT / "sparse_gemm_icsr_op.cpp"),
            str(CPP_ROOT / "sparse_gemm_csc_op.cpp"),
            str(CPP_ROOT / "sparse_gemm_coo_op.cpp"),
            str(CPP_ROOT / "sparse_gemm_coo_sve_gather_op.cpp"),
            str(CPP_ROOT / "thr_sparsify_to_icsr_op.cpp"),
            str(CPP_ROOT / "thr_sparsify_to_icsr_sve_op.cpp"),
            str(CPP_ROOT / "thr_sparsify_to_icsr_sve_baseline_op.cpp"),
            str(CPP_ROOT / "thr_sparsify_to_csr_op.cpp"),
            str(CPP_ROOT / "thr_sparsify_to_csr_sve_op.cpp"),
            str(CPP_ROOT / "thr_sparsify_to_coo_op.cpp"),
            str(CPP_ROOT / "thr_sparsify_to_coo_sve_op.cpp"),
            str(CPP_ROOT / "thr_sparsify_to_csc_op.cpp"),
            str(CPP_ROOT / "mask_sparsify_to_icsr_op.cpp"),
            str(CPP_ROOT / "mask_sparsify_to_icsr_sve_op.cpp"),
            str(CPP_ROOT / "mask_sparsify_to_csr_op.cpp"),
            str(CPP_ROOT / "mask_sparsify_to_csr_sve_op.cpp"),
            str(CPP_ROOT / "mask_sparsify_to_coo_op.cpp"),
            str(CPP_ROOT / "mask_sparsify_to_coo_sve_op.cpp"),
            str(CPP_ROOT / "mask_sparsify_to_csc_op.cpp"),
            str(CPP_ROOT / "mask_sparsify_to_csc_scatter_op.cpp"),
        ],
        build_directory=str(BUILD_DIR),
        extra_cflags=_extra_cflags(),
        extra_ldflags=_extra_ldflags(),
        verbose=verbose,
    )


class SparseGEMViCSRSVEGatherKernel(BaseKernel):
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
        return torch.ops.sparse_op.sparse_gemv_icsr_sve_gather(activation, weight, nz_row, nz_col_index)


class SparseGEMMiCSRSVEGatherKernel(BaseKernel):
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
        return torch.ops.sparse_op.sparse_gemm_icsr_sve_gather(activation, weight, row_offsets, nz_col_indices)


class SparseGEMMCSRKernel(BaseKernel):
    """
    torch.compile 兼容的 CSR GEMM wrapper。

    Args:
        weight: (K, N) 稠密权重矩阵（float32, contiguous, CPU）
        row_offsets: 1D int64, length = M+1，CSR 的 row_ptr（前缀和）
        nz_col_indices: 1D uint32/int32, length = nnz，CSR 的列索引
        values: 1D float32, length = nnz，CSR 的非零元素值

    Returns:
        output: (M, N) 输出矩阵
    """

    def meta(
        self,
        weight: torch.Tensor,
        row_offsets: torch.Tensor,
        nz_col_indices: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        M = row_offsets.size(0) - 1
        N = weight.size(1)
        return weight.new_empty((M, N), device="meta")

    def forward(
        self,
        weight: torch.Tensor,
        row_offsets: torch.Tensor,
        nz_col_indices: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        load_sve_sparse_gemm_extension()
        return torch.ops.sparse_op.sparse_gemm_csr(
            weight, row_offsets, nz_col_indices, values
        )


class SparseGEMMCSRSVEGatherKernel(BaseKernel):
    """
    torch.compile 兼容的 CSR GEMM wrapper（使用 gather load weight 的实现）。

    Args:
        weight: (K, N) 稠密权重矩阵（float32, contiguous, CPU）
        row_offsets: 1D int64, length = M+1，CSR 的 row_ptr（前缀和）
        nz_col_indices: 1D uint32/int32, length = nnz，CSR 的列索引
        values: 1D float32, length = nnz，CSR 的非零元素值

    Returns:
        output: (M, N) 输出矩阵
    """

    def meta(
        self,
        weight: torch.Tensor,
        row_offsets: torch.Tensor,
        nz_col_indices: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        M = row_offsets.size(0) - 1
        N = weight.size(1)
        return weight.new_empty((M, N), device="meta")

    def forward(
        self,
        weight: torch.Tensor,
        row_offsets: torch.Tensor,
        nz_col_indices: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        load_sve_sparse_gemm_extension()
        return torch.ops.sparse_op.sparse_gemm_csr_sve_gather(
            weight, row_offsets, nz_col_indices, values
        )


class SparseGEMMICSRKernel(BaseKernel):
    """
    torch.compile 兼容的直接 GEMM wrapper（不构建 CSR 格式）。

    该算子输入格式与 `sparse_gemm_icsr_sve_gather` 一致（row_offsets + nz_col_indices），但内部会：
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
        return torch.ops.sparse_op.sparse_gemm_icsr(
            activation, weight, row_offsets, nz_col_indices
        )


class SparseGEMMCSCKernel(BaseKernel):
    """
    torch.compile 兼容的 CSC GEMM wrapper。
    
    该算子直接使用输入的 CSC 格式数据进行矩阵乘法。
    计算策略：
    - 按 weight 矩阵的行（即 activation 的列）进行分块
    - 负载均衡分配到多个线程
    - 使用稀疏 activation 非零值标量 × SIMD 加载的 weight 行向量
    
    Args:
        weight: (K, N) 稠密权重矩阵
        col_ptr: 1D int64, length = K + 1, CSC格式的列指针（前缀和）
        row_indices: 1D uint32, CSC格式的行索引（C++ 侧强制 uint32）
        values: 1D float32, CSC格式的非零元素值
        M: 输出矩阵的行数
        ncore: 并行线程数，默认为 0（自动使用 OpenMP 默认线程数）
    
    Returns:
        output: (M, N) 输出矩阵
    """

    def meta(
        self,
        weight: torch.Tensor,
        col_ptr: torch.Tensor,
        row_indices: torch.Tensor,
        values: torch.Tensor,
        M: int,
        ncore: int = 0,
    ) -> torch.Tensor:
        N = weight.size(1)
        return weight.new_empty((M, N), device="meta")

    def forward(
        self,
        weight: torch.Tensor,
        col_ptr: torch.Tensor,
        row_indices: torch.Tensor,
        values: torch.Tensor,
        M: int,
        ncore: int = 0,
    ) -> torch.Tensor:
        load_sve_sparse_gemm_extension()
        return torch.ops.sparse_op.sparse_gemm_csc(
            weight, col_ptr, row_indices, values, M, ncore
        )


class SparseGEMMCOOKernel(BaseKernel):
    """
    torch.compile 兼容的 COO GEMM wrapper。
    
    该算子直接使用输入的 COO 格式数据进行矩阵乘法。
    计算逻辑：对于每个非零元素 (i, j, val): output[i, :] += val * weight[j, :]
    
    重要：该算子对应的 C++ 签名为：
      sparse_gemm_coo(Tensor weight, Tensor row_indices, Tensor col_indices, Tensor values, int M, int K, int N) -> Tensor
    因此 Python wrapper **必须**额外传入 M（稀疏矩阵行数）。K/N 从 weight 推导并传入 C++。

    Args:
        weight: (K, N) 稠密权重矩阵（float32, contiguous, CPU）
        row_indices: 1D int64, length = nnz, COO格式的行索引（已按行排序）
        col_indices: 1D uint32, length = nnz, COO格式的列索引（C++ 侧要求 uint32）
        values: 1D float32, length = nnz, COO格式的非零元素值（与row_indices对应）
        M: 稀疏矩阵的行数（通常等于生成这些 COO 数据的 activation.size(0)）
    
    Returns:
        output: (M, N) 输出矩阵
    """

    def meta(
        self,
        weight: torch.Tensor,
        row_indices: torch.Tensor,
        col_indices: torch.Tensor,
        values: torch.Tensor,
        M: int,
    ) -> torch.Tensor:
        N = weight.size(1)
        return weight.new_empty((M, N), device="meta")

    def forward(
        self,
        weight: torch.Tensor,
        row_indices: torch.Tensor,
        col_indices: torch.Tensor,
        values: torch.Tensor,
        M: int,
    ) -> torch.Tensor:
        load_sve_sparse_gemm_extension()
        K = int(weight.size(0))
        N = int(weight.size(1))
        # C++ 侧要求 col_indices=uint32；thr_sparsify_to_coo 返回的就是 uint32
        if col_indices.dtype != torch.uint32:
            col_indices = col_indices.to(torch.uint32)
        if not col_indices.is_contiguous():
            col_indices = col_indices.contiguous()
        return torch.ops.sparse_op.sparse_gemm_coo(
            weight, row_indices, col_indices, values, int(M), K, N
        )


class SparseGEMMCOOSVEGatherKernel(BaseKernel):
    """
    torch.compile 兼容的 COO GEMM wrapper（使用 SVE gather load 优化的实现）。
    
    该算子直接使用输入的 COO 格式数据进行矩阵乘法，并使用 SVE gather 指令优化。
    计算过程：
    - 内部将 COO 转换为行索引格式（类似 CSR 的 row_ptr）
    - 使用连续 load 从 values 加载数据
    - 使用 gather load 从 weight 加载数据
    - N 维度分块提高缓存局部性
    
    重要：该算子对应的 C++ 签名为：
      sparse_gemm_coo_sve_gather(Tensor weight, Tensor row_indices, Tensor col_indices, Tensor values, int64 M, int64 K, int64 N) -> Tensor
    因此 Python wrapper **必须**额外传入 M（稀疏矩阵行数）。K/N 从 weight 推导并传入 C++。

    Args:
        weight: (K, N) 稠密权重矩阵（float32, contiguous, CPU）
        row_indices: 1D int64, length = nnz, COO格式的行索引（已按行排序）
        col_indices: 1D uint32, length = nnz, COO格式的列索引（与row_indices对应）
        values: 1D float32, length = nnz, COO格式的非零元素值（与row_indices对应）
        M: 稀疏矩阵的行数（通常等于生成这些 COO 数据的 activation.size(0)）
    
    Returns:
        output: (M, N) 输出矩阵
    """

    def meta(
        self,
        weight: torch.Tensor,
        row_indices: torch.Tensor,
        col_indices: torch.Tensor,
        values: torch.Tensor,
        M: int,
    ) -> torch.Tensor:
        N = weight.size(1)
        return weight.new_empty((M, N), device="meta")

    def forward(
        self,
        weight: torch.Tensor,
        row_indices: torch.Tensor,
        col_indices: torch.Tensor,
        values: torch.Tensor,
        M: int,
    ) -> torch.Tensor:
        load_sve_sparse_gemm_extension()
        K = int(weight.size(0))
        N = int(weight.size(1))
        return torch.ops.sparse_op.sparse_gemm_coo_sve_gather(
            weight, row_indices, col_indices, values, int(M), K, N
        )


def thr_sparsify_to_icsr_sve(activation: torch.Tensor, threshold: float, verbose: bool = False):
    """
    基于阈值的稠密→iCSR 稀疏化（SVE/SVE2 加速版本）。

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
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.thr_sparsify_to_icsr_sve(activation, float(threshold))


def thr_sparsify_to_icsr_sve_baseline(activation: torch.Tensor, threshold: float, verbose: bool = False):
    """
    基于阈值的稠密→iCSR 稀疏化（SVE Baseline版本，不使用SVE2 compact指令）。
    
    该版本作为公平的对比baseline，用于量化SVE2 compact指令的性能收益。
    与SVE2版本的区别仅在于：
    - Pass2中不使用 svcompact_u32 指令，改用手动循环提取满足条件的索引
    - Pass1和其他所有优化（多线程、向量化、循环展开）保持完全一致

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
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.thr_sparsify_to_icsr_sve_baseline(activation, float(threshold))


# Naive version (parallel but no SVE)
def thr_sparsify_to_icsr(activation: torch.Tensor, threshold: float, verbose: bool = False):
    """
    基于阈值的稠密→iCSR 稀疏化（朴素实现：OpenMP 并行，无 SVE）。

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
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.thr_sparsify_to_icsr(activation, float(threshold))


def thr_sparsify_to_csr(activation: torch.Tensor, threshold: float, verbose: bool = False):
    """
    基于阈值的稠密→CSR 稀疏化（OpenMP 并行版本）。

    Args:
        activation: (M, K) float32 CPU tensor，激活矩阵
        threshold: float，阈值，绝对值大于等于该值的元素被视为非零
        verbose: bool，是否显示编译信息
    
    Returns:
        tuple: (row_offsets, nz_col_indices, values)
            - row_offsets: int64 tensor，长度为 M+1，前缀和偏移量
            - nz_col_indices: uint32 tensor，列索引向量
            - values: float32 tensor，非零元素值
    """
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.thr_sparsify_to_csr(activation, float(threshold))


def thr_sparsify_to_csr_sve(activation: torch.Tensor, threshold: float, verbose: bool = False):
    """
    基于阈值的稠密→CSR 稀疏化（SVE/SVE2 加速版本）。

    Args:
        activation: (M, K) float32 CPU tensor，激活矩阵
        threshold: float，阈值，绝对值大于等于该值的元素被视为非零
        verbose: bool，是否显示编译信息
    
    Returns:
        tuple: (row_offsets, nz_col_indices, values)
            - row_offsets: int64 tensor，长度为 M+1，前缀和偏移量
            - nz_col_indices: uint32 tensor，列索引向量
            - values: float32 tensor，非零元素值
    """
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.thr_sparsify_to_csr_sve(activation, float(threshold))


def thr_sparsify_to_coo(activation: torch.Tensor, threshold: float, verbose: bool = False):
    """
    基于阈值的稠密→COO 稀疏化（OpenMP 并行版本）。

    Args:
        activation: (M, K) float32 CPU tensor，激活矩阵
        threshold: float，阈值，绝对值大于等于该值的元素被视为非零
        verbose: bool，是否显示编译信息
    
    Returns:
        tuple: (row_indices, col_indices, values)
            - row_indices: int64 tensor，长度为 nnz，行索引数组（已按行排序）
            - col_indices: uint32 tensor，长度为 nnz，列索引数组
            - values: float32 tensor，长度为 nnz，非零元素值数组
    """
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.thr_sparsify_to_coo(activation, float(threshold))


def thr_sparsify_to_coo_sve(activation: torch.Tensor, threshold: float, verbose: bool = False):
    """
    基于阈值的稠密→COO 稀疏化（SVE/SVE2 加速版本）。

    Args:
        activation: (M, K) float32 CPU tensor，激活矩阵
        threshold: float，阈值，绝对值大于等于该值的元素被视为非零
        verbose: bool，是否显示编译信息
    
    Returns:
        tuple: (row_indices, col_indices, values)
            - row_indices: int64 tensor，长度为 nnz，行索引数组（已按行排序）
            - col_indices: uint32 tensor，长度为 nnz，列索引数组
            - values: float32 tensor，长度为 nnz，非零元素值数组
    """
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.thr_sparsify_to_coo_sve(activation, float(threshold))


def thr_sparsify_to_csc(activation: torch.Tensor, threshold: float, verbose: bool = False):
    """
    基于阈值的稠密→CSC 稀疏化（OpenMP 并行版本）。

    Args:
        activation: (M, K) float32 CPU tensor，激活矩阵
        threshold: float，阈值，绝对值大于等于该值的元素被视为非零
        verbose: bool，是否显示编译信息
    
    Returns:
        tuple: (col_ptr, row_indices, values)
            - col_ptr: int64 tensor，长度为 K+1，列指针数组（前缀和）
            - row_indices: uint32 tensor，长度为 nnz，行索引数组
            - values: float32 tensor，长度为 nnz，非零元素值数组
    """
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.thr_sparsify_to_csc(activation, float(threshold))


def mask_sparsify_to_icsr(mask: torch.Tensor, verbose: bool = False):
    """
    基于 mask 的稠密→iCSR 稀疏化（OpenMP 并行版本）。

    Args:
        mask: (M, K) uint8 CPU tensor，mask 矩阵（非零元素表示有效位置）
        verbose: bool，是否显示编译信息

    Returns:
        tuple: (nz_counts, nz_col_indices, row_offsets)
            - nz_counts: int64 tensor，长度为 2 * num_nz_rows，成对存储 (row_idx, count)
            - nz_col_indices: uint32 tensor，扁平化的列索引向量
            - row_offsets: int64 tensor，长度为 M+1，前缀和偏移量
    """
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.mask_sparsify_to_icsr(mask)


def mask_sparsify_to_icsr_sve(mask: torch.Tensor, verbose: bool = False):
    """
    基于 mask 的稠密→iCSR 稀疏化（SVE/SVE2 加速版本）。

    Args:
        mask: (M, K) uint8 CPU tensor，mask 矩阵（非零元素表示有效位置）
        verbose: bool，是否显示编译信息

    Returns:
        tuple: (nz_counts, nz_col_indices, row_offsets)
            - nz_counts: int64 tensor，长度为 2 * num_nz_rows，成对存储 (row_idx, count)
            - nz_col_indices: uint32 tensor，扁平化的列索引向量
            - row_offsets: int64 tensor，长度为 M+1，前缀和偏移量
    """
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.mask_sparsify_to_icsr_sve(mask)


def mask_sparsify_to_csr(activation: torch.Tensor, mask: torch.Tensor, verbose: bool = False):
    """
    基于 mask 的稠密→CSR 稀疏化（OpenMP 并行版本）。

    Args:
        activation: (M, K) float32 CPU tensor，激活矩阵
        mask: (M, K) uint8 CPU tensor，mask 矩阵（非零元素表示有效位置）
        verbose: bool，是否显示编译信息
    
    Returns:
        tuple: (row_offsets, nz_col_indices, values)
            - row_offsets: int64 tensor，长度为 M+1，前缀和偏移量
            - nz_col_indices: uint32 tensor，列索引向量
            - values: float32 tensor，非零元素值
    """
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.mask_sparsify_to_csr(activation, mask)


def mask_sparsify_to_csr_sve(activation: torch.Tensor, mask: torch.Tensor, verbose: bool = False):
    """
    基于 mask 的稠密→CSR 稀疏化（SVE/SVE2 优化版本）。

    Args:
        activation: (M, K) float32 CPU tensor，激活矩阵
        mask: (M, K) uint8 CPU tensor，mask 矩阵（非零元素表示有效位置）
        verbose: bool，是否显示编译信息
    
    Returns:
        tuple: (row_offsets, nz_col_indices, values)
            - row_offsets: int64 tensor，长度为 M+1，前缀和偏移量
            - nz_col_indices: uint32 tensor，列索引向量
            - values: float32 tensor，非零元素值
    """
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.mask_sparsify_to_csr_sve(activation, mask)


def mask_sparsify_to_coo(activation: torch.Tensor, mask: torch.Tensor, verbose: bool = False):
    """
    基于 mask 的稠密→COO 稀疏化（OpenMP 并行版本）。

    Args:
        activation: (M, K) float32 CPU tensor，激活矩阵
        mask: (M, K) uint8 CPU tensor，mask 矩阵（非零元素表示有效位置）
        verbose: bool，是否显示编译信息
    
    Returns:
        tuple: (row_indices, col_indices, values)
            - row_indices: int64 tensor，长度为 nnz，行索引数组（已按行排序）
            - col_indices: uint32 tensor，长度为 nnz，列索引数组
            - values: float32 tensor，长度为 nnz，非零元素值数组
    """
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.mask_sparsify_to_coo(activation, mask)


def mask_sparsify_to_coo_sve(activation: torch.Tensor, mask: torch.Tensor, verbose: bool = False):
    """
    基于 mask 的稠密→COO 稀疏化（SVE/SVE2 加速版本）。

    Args:
        activation: (M, K) float32 CPU tensor，激活矩阵
        mask: (M, K) uint8 CPU tensor，mask 矩阵（非零元素表示有效位置）
        verbose: bool，是否显示编译信息
    
    Returns:
        tuple: (row_indices, col_indices, values)
            - row_indices: int64 tensor，长度为 nnz，行索引数组（已按行排序）
            - col_indices: uint32 tensor，长度为 nnz，列索引数组
            - values: float32 tensor，长度为 nnz，非零元素值数组
    """
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.mask_sparsify_to_coo_sve(activation, mask)


def mask_sparsify_to_csc(activation: torch.Tensor, mask: torch.Tensor, verbose: bool = False):
    """
    基于 mask 的稠密→CSC 稀疏化（OpenMP 并行版本）。

    Args:
        activation: (M, K) float32 CPU tensor，激活矩阵
        mask: (M, K) uint8 CPU tensor，mask 矩阵（非零元素表示有效位置）
        verbose: bool，是否显示编译信息
    
    Returns:
        tuple: (col_ptr, row_indices, values)
            - col_ptr: int64 tensor，长度为 K+1，列指针数组
            - row_indices: uint32 tensor，长度为 nnz，行索引数组（每列内按行排序）
            - values: float32 tensor，长度为 nnz，非零元素值数组
    """
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.mask_sparsify_to_csc(activation, mask)


def mask_sparsify_to_csc_scatter(activation: torch.Tensor, mask: torch.Tensor, verbose: bool = False):
    """
    基于 mask 的稠密→CSC 稀疏化（SVE scatter store 优化版本）。

    该版本使用 SVE scatter store 指令进行优化，避免了写入冲突。
    在 total_nnz 适合 u32 范围时，使用 SVE scatter store 直接写入每列桶中，
    相比标量版本可以获得更好的性能。

    Args:
        activation: (M, K) float32 CPU tensor，激活矩阵
        mask: (M, K) uint8 CPU tensor，mask 矩阵（非零元素表示有效位置）
        verbose: bool，是否显示编译信息
    
    Returns:
        tuple: (col_ptr, row_indices, values)
            - col_ptr: int64 tensor，长度为 K+1，列指针数组
            - row_indices: uint32 tensor，长度为 nnz，行索引数组（每列内按行排序）
            - values: float32 tensor，长度为 nnz，非零元素值数组
    """
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.mask_sparsify_to_csc_scatter(activation, mask)


# -----------------------------------------------------------------------------
# 兼容层：保持旧导入路径/符号名可用
# -----------------------------------------------------------------------------
# kernels/__init__.py 历史上导出过 SVESparseGEMVKernel / SVESparseGEMMKernel。
# 这里将其映射到当前实现中最常用的 iCSR SVE gather 版本。
SVESparseGEMVKernel = SparseGEMViCSRSVEGatherKernel
SVESparseGEMMKernel = SparseGEMMiCSRSVEGatherKernel

__all__ = [
    # extension loader
    "load_sve_sparse_gemm_extension",
    # latency helper (re-export)
    "measure_latency",
    # legacy aliases
    "SVESparseGEMVKernel",
    "SVESparseGEMMKernel",
    # kernels
    "SparseGEMViCSRSVEGatherKernel",
    "SparseGEMMiCSRSVEGatherKernel",
    "SparseGEMMCSRKernel",
    "SparseGEMMCSRSVEGatherKernel",
    "SparseGEMMICSRKernel",
    "SparseGEMMCSCKernel",
    "SparseGEMMCOOKernel",
    "SparseGEMMCOOSVEGatherKernel",
    # sparsifiers
    "thr_sparsify_to_icsr",
    "thr_sparsify_to_icsr_sve",
    "thr_sparsify_to_icsr_sve_baseline",
    "thr_sparsify_to_csr",
    "thr_sparsify_to_csr_sve",
    "thr_sparsify_to_coo",
    "thr_sparsify_to_coo_sve",
    "thr_sparsify_to_csc",
    "mask_sparsify_to_icsr",
    "mask_sparsify_to_icsr_sve",
    "mask_sparsify_to_csr",
    "mask_sparsify_to_csr_sve",
    "mask_sparsify_to_coo",
    "mask_sparsify_to_coo_sve",
    "mask_sparsify_to_csc",
    "mask_sparsify_to_csc_scatter",
]