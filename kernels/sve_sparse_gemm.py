"""
ARM SVE sparse GEMV/GEMM custom operator bindings.

This module implements multiple sparse matrix multiplication formats:

1. iCSR (Implicit CSR): uses row_offsets + nz_col_indices directly.
2. CSR: builds full CSR and performs matrix multiply.
3. CSC (Compressed Sparse Column): sparse multiply from CSC data
   (values, row_indices, col_ptr), load-balanced by weight rows,
   using scalar sparse activation values and SIMD-loaded weight rows.
4. COO (Coordinate): sparse multiply from COO data (row_indices, col_indices, values),
   with row_indices sorted by row; OpenMP and SVE vectorization;
   COO SVE Gather variant uses SVE gather instructions.

Sparsification utilities:
- thr_sparsify_to_icsr / thr_sparsify_to_icsr_sve: dense -> iCSR (OpenMP / SVE).
- thr_sparsify_to_csr / thr_sparsify_to_csr_sve: dense -> CSR (OpenMP / SVE).
- thr_sparsify_to_coo / thr_sparsify_to_coo_sve: dense -> COO (OpenMP / SVE).
- thr_sparsify_to_csc: dense -> CSC (OpenMP).

Provides: C++ extension load/registration, torch.compile-friendly Python wrappers,
and a simple latency measurement helper.
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

# Op names that must be present to consider the extension already loaded.
_REQUIRED_OPS = (
    "sparse_gemv_icsr_sve_gather",
    "sparse_gemm_icsr_sve_gather",
    "sparse_gemm_csr",
    "sparse_gemm_csr_sve_gather",
    "sparse_gemm_icsr",
    "sparse_gemm_csc",
    "sparse_gemm_coo",
    "sparse_gemm_coo_sve_gather",
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
    "mask_sparsify_to_csc_scatter",
)


def _extra_cflags() -> list[str]:
    if os.name == "nt":
        return ["/std:c++17", "/openmp"]
    flags = ["-std=c++17", "-O3", "-fopenmp"]
    if platform.machine().lower() in {"aarch64", "arm64"}:
        flags.append("-march=armv8-a+sve2")
    return flags


def _extra_ldflags() -> list[str]:
    if os.name == "nt":
        return []
    return ["-fopenmp"]


def load_sve_sparse_gemm_extension(
    rebuild: bool = False,
    verbose: bool = False,
) -> Optional[ModuleType]:
    """Compile and load the C++ extension; skip if ops are already registered."""
    if not rebuild and hasattr(torch.ops, "sparse_op"):
        op = torch.ops.sparse_op
        if all(hasattr(op, name) for name in _REQUIRED_OPS):
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
    """torch.compile-friendly GEMV wrapper (iCSR SVE gather)."""

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
    """torch.compile-friendly GEMM wrapper (iCSR SVE gather).

    Args:
        activation: (M, K) sparse activation matrix.
        weight: (K, N) dense weight matrix.
        row_offsets: int64 [M+1], prefix-sum offsets; row_offsets[m] is start of row m in nz_col_indices.
        nz_col_indices: flattened column indices (uint32).

    Returns:
        output: (M, N) result matrix.
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
    """torch.compile-friendly CSR GEMM wrapper.

    Args:
        weight: (K, N) dense weight matrix (float32, contiguous, CPU).
        row_offsets: 1D int64, length M+1, CSR row_ptr (prefix sum).
        nz_col_indices: 1D uint32/int32, length nnz, CSR column indices.
        values: 1D float32, length nnz, CSR non-zero values.

    Returns:
        output: (M, N) result matrix.
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
    """torch.compile-friendly CSR GEMM wrapper (gather-load weight implementation).

    Args:
        weight: (K, N) dense weight matrix (float32, contiguous, CPU).
        row_offsets: 1D int64, length M+1, CSR row_ptr (prefix sum).
        nz_col_indices: 1D uint32/int32, length nnz, CSR column indices.
        values: 1D float32, length nnz, CSR non-zero values.

    Returns:
        output: (M, N) result matrix.
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
    """torch.compile-friendly GEMM wrapper using row_offsets + nz_col_indices (no CSR build).

    Same input format as sparse_gemm_icsr_sve_gather; uses row_offsets/nz_col_indices and
    activation directly for GEMM with SIMD (SVE), without building full CSR.

    Args:
        activation: (M, K) sparse activation (passed as dense).
        weight: (K, N) dense weight matrix.
        row_offsets: prefix-sum array length M+1, start of each row in nz_col_indices.
        nz_col_indices: flattened column indices (uint32/int32).

    Returns:
        output: (M, N) result matrix.
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
    """torch.compile-friendly CSC GEMM wrapper.

    Uses input CSC data (col_ptr, row_indices, values) directly; load-balanced by
    weight rows, scalar sparse activation values times SIMD-loaded weight rows.

    Args:
        weight: (K, N) dense weight matrix.
        col_ptr: 1D int64, length K+1, CSC column pointers (prefix sum).
        row_indices: 1D uint32, CSC row indices.
        values: 1D float32, CSC non-zero values.
        M: number of output rows.
        ncore: number of threads (0 = OpenMP default).

    Returns:
        output: (M, N) result matrix.
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
    """torch.compile-friendly COO GEMM wrapper.

    Uses COO data directly; for each (i, j, val): output[i, :] += val * weight[j, :].
    C++ signature requires M; K, N are derived from weight and passed through.

    Args:
        weight: (K, N) dense weight matrix (float32, contiguous, CPU).
        row_indices: 1D int64, length nnz, COO row indices (sorted by row).
        col_indices: 1D uint32, length nnz, COO column indices.
        values: 1D float32, length nnz, COO non-zero values.
        M: number of sparse matrix rows (e.g. activation.size(0)).

    Returns:
        output: (M, N) result matrix.
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
        if col_indices.dtype != torch.uint32:
            col_indices = col_indices.to(torch.uint32)
        if not col_indices.is_contiguous():
            col_indices = col_indices.contiguous()
        return torch.ops.sparse_op.sparse_gemm_coo(
            weight, row_indices, col_indices, values, int(M), K, N
        )


class SparseGEMMCOOSVEGatherKernel(BaseKernel):
    """torch.compile-friendly COO GEMM wrapper (SVE gather-load optimized).

    Uses COO data with SVE gather: converts to row-index form (like CSR row_ptr),
    contiguous load from values, gather load from weight, N-dim blocking for cache locality.
    C++ signature requires M; K, N are derived from weight and passed through.

    Args:
        weight: (K, N) dense weight matrix (float32, contiguous, CPU).
        row_indices: 1D int64, length nnz, COO row indices (sorted by row).
        col_indices: 1D uint32, length nnz, COO column indices.
        values: 1D float32, length nnz, COO non-zero values.
        M: number of sparse matrix rows (e.g. activation.size(0)).

    Returns:
        output: (M, N) result matrix.
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
        return torch.ops.sparse_op.sparse_gemm_coo_sve_gather(
            weight, row_indices, col_indices, values, int(M),
            int(weight.size(0)), int(weight.size(1)),
        )


def thr_sparsify_to_icsr_sve(activation: torch.Tensor, threshold: float, verbose: bool = False):
    """Threshold-based dense -> iCSR sparsification (SVE/SVE2 accelerated)."""
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.thr_sparsify_to_icsr_sve(activation, float(threshold))


def thr_sparsify_to_icsr_sve_baseline(activation: torch.Tensor, threshold: float, verbose: bool = False):
    """Threshold-based dense -> iCSR (SVE baseline without SVE2 compact); for comparing compact instruction benefit."""
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.thr_sparsify_to_icsr_sve_baseline(activation, float(threshold))


def thr_sparsify_to_icsr(activation: torch.Tensor, threshold: float, verbose: bool = False):
    """Threshold-based dense -> iCSR (OpenMP parallel, no SVE)."""
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.thr_sparsify_to_icsr(activation, float(threshold))


def thr_sparsify_to_csr(activation: torch.Tensor, threshold: float, verbose: bool = False):
    """Threshold-based dense -> CSR (OpenMP parallel)."""
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.thr_sparsify_to_csr(activation, float(threshold))


def thr_sparsify_to_csr_sve(activation: torch.Tensor, threshold: float, verbose: bool = False):
    """Threshold-based dense -> CSR (SVE/SVE2 accelerated)."""
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.thr_sparsify_to_csr_sve(activation, float(threshold))


def thr_sparsify_to_coo(activation: torch.Tensor, threshold: float, verbose: bool = False):
    """Threshold-based dense -> COO (OpenMP parallel)."""
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.thr_sparsify_to_coo(activation, float(threshold))


def thr_sparsify_to_coo_sve(activation: torch.Tensor, threshold: float, verbose: bool = False):
    """Threshold-based dense -> COO (SVE/SVE2 accelerated)."""
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.thr_sparsify_to_coo_sve(activation, float(threshold))


def thr_sparsify_to_csc(activation: torch.Tensor, threshold: float, verbose: bool = False):
    """Threshold-based dense -> CSC (OpenMP parallel)."""
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.thr_sparsify_to_csc(activation, float(threshold))


def mask_sparsify_to_icsr(mask: torch.Tensor, verbose: bool = False):
    """Mask-based dense -> iCSR (OpenMP parallel)."""
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.mask_sparsify_to_icsr(mask)


def mask_sparsify_to_icsr_sve(mask: torch.Tensor, verbose: bool = False):
    """Mask-based dense -> iCSR (SVE/SVE2 accelerated)."""
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.mask_sparsify_to_icsr_sve(mask)


def mask_sparsify_to_csr(activation: torch.Tensor, mask: torch.Tensor, verbose: bool = False):
    """Mask-based dense -> CSR (OpenMP parallel)."""
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.mask_sparsify_to_csr(activation, mask)


def mask_sparsify_to_csr_sve(activation: torch.Tensor, mask: torch.Tensor, verbose: bool = False):
    """Mask-based dense -> CSR (SVE/SVE2 optimized)."""
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.mask_sparsify_to_csr_sve(activation, mask)


def mask_sparsify_to_coo(activation: torch.Tensor, mask: torch.Tensor, verbose: bool = False):
    """Mask-based dense -> COO (OpenMP parallel)."""
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.mask_sparsify_to_coo(activation, mask)


def mask_sparsify_to_coo_sve(activation: torch.Tensor, mask: torch.Tensor, verbose: bool = False):
    """Mask-based dense -> COO (SVE/SVE2 accelerated)."""
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.mask_sparsify_to_coo_sve(activation, mask)


def mask_sparsify_to_csc(activation: torch.Tensor, mask: torch.Tensor, verbose: bool = False):
    """Mask-based dense -> CSC (OpenMP parallel)."""
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.mask_sparsify_to_csc(activation, mask)


def mask_sparsify_to_csc_scatter(activation: torch.Tensor, mask: torch.Tensor, verbose: bool = False):
    """Mask-based dense -> CSC (SVE scatter-store optimized; avoids write conflicts)."""
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.sparse_op.mask_sparsify_to_csc_scatter(activation, mask)


# Legacy aliases (kernels/__init__.py used to export these names).
SVESparseGEMVKernel = SparseGEMViCSRSVEGatherKernel
SVESparseGEMMKernel = SparseGEMMiCSRSVEGatherKernel

__all__ = [
    "load_sve_sparse_gemm_extension",
    "measure_latency",
    "SVESparseGEMVKernel",
    "SVESparseGEMMKernel",
    "SparseGEMViCSRSVEGatherKernel",
    "SparseGEMMiCSRSVEGatherKernel",
    "SparseGEMMCSRKernel",
    "SparseGEMMCSRSVEGatherKernel",
    "SparseGEMMICSRKernel",
    "SparseGEMMCSCKernel",
    "SparseGEMMCOOKernel",
    "SparseGEMMCOOSVEGatherKernel",
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