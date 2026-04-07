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
- thr_sparsify_to_icsr: dense -> iCSR (OpenMP).
- thr_sparsify_to_icsr_sve: dense -> iCSR (SVE/SVE2).
- thr_sparsify_to_icsr_sve_baseline: dense -> iCSR (SVE baseline, no SVE2 compact).
- thr_sparsify_to_csr / thr_sparsify_to_csr_sve: dense -> CSR (OpenMP / SVE).
- thr_sparsify_to_coo / thr_sparsify_to_coo_sve: dense -> COO (OpenMP / SVE).
- thr_sparsify_to_csc: dense -> CSC (OpenMP).

Provides: C++ extension load/registration, torch.compile-friendly Python wrappers,
and a simple latency measurement helper.
"""

from __future__ import annotations

import csv
import os
import platform
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

import torch
from torch import Tensor
from torch.utils.cpp_extension import load

from kernels.compile_wrapper import BaseKernel
from kernels.kernel_utils import measure_latency as measure_latency

# 自适应稀疏 GEMM 配置
USE_CUSTOM_SPARSE_GEMM = os.getenv("USE_CUSTOM_SPARSE_GEMM", "0") == "1"
DENSE_THRESHOLD = 0.8  # 稀疏度 < 0.8 的行被视为 dense
MIN_DENSE_BLOCK = 4    # 至少连续 4 行才用 dense GEMM
SPARSE_DEBUG = os.getenv("SPARSE_DEBUG", "0") == "1"
PRINT_STATISTICS = False

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

_REQUIRED_DENSE_OPS = (
    "dense_gemm_sve_omp",
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

MKN2sparsity = {
    (1,4096,4096): 0.17,
    (1,4096,11008): 0.13,
    (1,11008,4096): 0.20,
    (128,4096,4096): 0.86,
    (128,4096,11008): 0.82,
    (128,11008,4096): 0.90,
}

DEFAULT_SPARSITY = 0.0



def load_sve_sparse_gemm_extension(
    rebuild: bool = False,
    verbose: bool = False,
) -> Optional[ModuleType]:
    """Compile and load the C++ extension; skip if ops are already registered."""
    if not rebuild:
        sparse_ready = hasattr(torch.ops, "sparse_op") and all(
            hasattr(torch.ops.sparse_op, name) for name in _REQUIRED_OPS
        )
        dense_ready = hasattr(torch.ops, "dense_op") and all(
            hasattr(torch.ops.dense_op, name) for name in _REQUIRED_DENSE_OPS
        )
        if sparse_ready and dense_ready:
            return None

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    return load(
        name=EXT_NAME,
        sources=[
            str(CPP_ROOT / "dense_gemm_sve_omp.cpp"),
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

#==============================Dense Version================================
class DenseGEMV(BaseKernel):
    def meta(self, x: torch.Tensor, W: torch.Tensor, threshold: float, *args, **kwargs) -> torch.Tensor:
        return x.new_empty(x.shape[0], x.shape[1], W.shape[1])
    
    def forward(self, x: torch.Tensor, W: torch.Tensor, threshold: float, *args, **kwargs) -> torch.Tensor:
        x_2d = x.view(-1, x.shape[2])
        return torch.ops.dense_op.dense_gemm_sve_omp(x_2d, W).view(x.shape[0], x.shape[1], W.shape[1])
        # return torch.matmul(x, W)

#============================All SVE Sparsr Version==========================
import numpy as np
from collections import defaultdict
import csv
import atexit
import os

class SVEGatherGEMV(BaseKernel):
    def meta(
        self,
        x: torch.Tensor,
        W: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        return x.new_empty((x.size(0), x.size(1), W.size(1)))

    def forward(
        self,
        x: torch.Tensor,
        W: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        B, M, K = x.shape
        N = W.shape[1]
        x_2d = x.view(-1, K)

        if M > 1:
            split_idx = x_2d.size(0) // 2
            x_dense = x_2d[:split_idx]
            out_dense = torch.ops.dense_op.dense_gemm_sve_omp(x_dense, W)

            x_sparse = x_2d[split_idx:]
            _, nz_col_indices, row_offsets = torch.ops.sparse_op.thr_sparsify_to_icsr_sve(x_sparse, threshold)
            out_sparse = torch.ops.sparse_op.sparse_gemm_icsr_sve_gather(
                x_sparse, W, row_offsets, nz_col_indices
            )

            out = torch.cat([out_dense, out_sparse], dim=0)
            return out.view(B, M, N)
        
        _, nz_col_indices, row_offsets = torch.ops.sparse_op.thr_sparsify_to_icsr(x_2d, threshold)
        out = torch.ops.sparse_op.sparse_gemm_icsr_sve_gather(x_2d, W, row_offsets, nz_col_indices)

        return out.view(B, M, N)

class QKVSVEGatherGEMV(BaseKernel):
    """
    3D activation (B, S, K) -> view to (B*S, K), then sparse GEMV/GEMM, output (B, S, N).
    使用三个不同的 threshold 分别处理 Q、K、V 三部分的 weight。
    """

    def meta(
        self,
        x: torch.Tensor,
        W: torch.Tensor,
        threshold_q: float,
        threshold_k: float,
        threshold_v: float,
        kv_size: int,
        W_q: torch.Tensor,
        W_k: torch.Tensor,
        W_v: torch.Tensor,
    ) -> torch.Tensor:
        return x.new_empty((x.size(0), x.size(1), W.size(1)))

    def forward(
        self,
        x: torch.Tensor,
        W: torch.Tensor,
        threshold_q: float,
        threshold_k: float,
        threshold_v: float,
        kv_size: int,
        W_q: torch.Tensor,
        W_k: torch.Tensor,
        W_v: torch.Tensor,
    ) -> torch.Tensor:
        B, M, K = x.shape
        N = W.size(1)
        x_2d = x.view(-1, K)

        if M > 1:
            split_idx = x_2d.size(0) // 2

            # Q
            x_dense = x_2d[:split_idx]
            out_dense = torch.ops.dense_op.dense_gemm_sve_omp(x_dense, W)

            x_sparse = x_2d[split_idx:]
            _, nz_col_indices_q, row_offsets_q = torch.ops.sparse_op.thr_sparsify_to_icsr_sve(x_sparse, threshold_q)
            out_sparse_q = torch.ops.sparse_op.sparse_gemm_icsr_sve_gather(
                x_sparse, W_q, row_offsets_q, nz_col_indices_q
            )

            _, nz_col_indices_k, row_offsets_k = torch.ops.sparse_op.thr_sparsify_to_icsr_sve(x_sparse, threshold_k)
            out_sparse_k = torch.ops.sparse_op.sparse_gemm_icsr_sve_gather(
                x_sparse, W_k, row_offsets_k, nz_col_indices_k
            )

            _, nz_col_indices_v, row_offsets_v = torch.ops.sparse_op.thr_sparsify_to_icsr_sve(x_sparse, threshold_v)
            out_sparse_v = torch.ops.sparse_op.sparse_gemm_icsr_sve_gather(
                x_sparse, W_v, row_offsets_v, nz_col_indices_v
            )

            out_sparse = torch.cat([out_sparse_q, out_sparse_k, out_sparse_v], dim=1)
            out = torch.cat([out_dense, out_sparse], dim=0)
            return out.view(B, M, N)

        _, nz_col_indices_q, row_offsets_q = torch.ops.sparse_op.thr_sparsify_to_icsr(x_2d, threshold_q)
        _, nz_col_indices_k, row_offsets_k = torch.ops.sparse_op.thr_sparsify_to_icsr(x_2d, threshold_k)
        _, nz_col_indices_v, row_offsets_v = torch.ops.sparse_op.thr_sparsify_to_icsr(x_2d, threshold_v)
        out_q = torch.ops.sparse_op.sparse_gemm_icsr_sve_gather(x_2d, W_q, row_offsets_q, nz_col_indices_q)
        out_k = torch.ops.sparse_op.sparse_gemm_icsr_sve_gather(x_2d, W_k, row_offsets_k, nz_col_indices_k)
        out_v = torch.ops.sparse_op.sparse_gemm_icsr_sve_gather(x_2d, W_v, row_offsets_v, nz_col_indices_v)
        out_2d = torch.cat([out_q, out_k, out_v], dim=1)
        return out_2d.view(B, M, N)

#============================Adaptive Sparse Version=============================
class AdaptiveSVEGatherGEMV(BaseKernel):
    def meta(
        self,
        x: torch.Tensor,
        W: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        return x.new_empty((x.size(0), x.size(1), W.size(1)))

    def forward(
        self,
        x: torch.Tensor,
        W: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        B, M, K = x.shape
        N = W.shape[1]
        x_2d = x.view(-1, K)

        if M > 1:
            split_idx = x_2d.size(0) // 2
            x_dense = x_2d[:split_idx]
            out_dense = torch.ops.dense_op.dense_gemm_sve_omp(x_dense, W)

            x_sparse = x_2d[split_idx:]
            nnz_counts, nz_col_indices, row_offsets = torch.ops.sparse_op.thr_sparsify_to_icsr_sve(x_sparse, threshold)
            if 1-nnz_counts.sum()/K/x_sparse.size(0) < MKN2sparsity.get((M, K, N), DEFAULT_SPARSITY):
                out_sparse = torch.ops.dense_op.dense_gemm_sve_omp(x_sparse, W)
            else:
                out_sparse = torch.ops.sparse_op.sparse_gemm_icsr_sve_gather(
                    x_sparse, W, row_offsets, nz_col_indices
                )

            out = torch.cat([out_dense, out_sparse], dim=0)
            return out.view(B, M, N)
        
        nnz_counts, nz_col_indices, row_offsets = torch.ops.sparse_op.thr_sparsify_to_icsr(x_2d, threshold)
        if 1-nnz_counts[0]/K < MKN2sparsity.get((M, K, N), DEFAULT_SPARSITY):
            out = torch.ops.dense_op.dense_gemm_sve_omp(x_2d, W)
        else:
            out = torch.ops.sparse_op.sparse_gemm_icsr_sve_gather(x_2d, W, row_offsets, nz_col_indices)

        return out.view(B, M, N)

class QKVAdaptiveSVEGatherGEMV(BaseKernel):
    """
    3D activation (B, S, K) -> view to (B*S, K), then sparse GEMV/GEMM, output (B, S, N).
    使用三个不同的 threshold 分别处理 Q、K、V 三部分的 weight。
    """
    def meta(
        self,
        x: torch.Tensor,
        W: torch.Tensor,
        threshold_q: float,
        threshold_k: float,
        threshold_v: float,
        kv_size: int,
        W_q: torch.Tensor,
        W_k: torch.Tensor,
        W_v: torch.Tensor,
    ) -> torch.Tensor:
        return x.new_empty((x.size(0), x.size(1), W.size(1)))

    def forward(
        self,
        x: torch.Tensor,
        W: torch.Tensor,
        threshold_q: float,
        threshold_k: float,
        threshold_v: float,
        kv_size: int,
        W_q: torch.Tensor,
        W_k: torch.Tensor,
        W_v: torch.Tensor,
    ) -> torch.Tensor:
        B, M, K = x.shape
        N = W.size(1)
        x_2d = x.view(-1, K)

        if M > 1:
            split_idx = x_2d.size(0) // 2

            # Q
            x_dense = x_2d[:split_idx]
            out_dense = torch.ops.dense_op.dense_gemm_sve_omp(x_dense, W)

            x_sparse = x_2d[split_idx:]
            nnz_counts_q, nz_col_indices_q, row_offsets_q = torch.ops.sparse_op.thr_sparsify_to_icsr_sve(x_sparse, threshold_q)
            nnz_counts_k, nz_col_indices_k, row_offsets_k = torch.ops.sparse_op.thr_sparsify_to_icsr_sve(x_sparse, threshold_k)
            nnz_counts_v, nz_col_indices_v, row_offsets_v = torch.ops.sparse_op.thr_sparsify_to_icsr_sve(x_sparse, threshold_v)

            sparsity_threshold = MKN2sparsity.get((M, K, W_q.size(1)), DEFAULT_SPARSITY)

            # Q
            if 1-nnz_counts_q.sum() / (K * x_sparse.size(0)) < sparsity_threshold:
                out_sparse_q = torch.ops.dense_op.dense_gemm_sve_omp(x_sparse, W_q)
            else:
                out_sparse_q = torch.ops.sparse_op.sparse_gemm_icsr_sve_gather(
                    x_sparse, W_q, row_offsets_q, nz_col_indices_q
                )
            # K
            if 1-nnz_counts_k.sum() / (K * x_sparse.size(0)) < sparsity_threshold:
                out_sparse_k = torch.ops.dense_op.dense_gemm_sve_omp(x_sparse, W_k)
            else:
                out_sparse_k = torch.ops.sparse_op.sparse_gemm_icsr_sve_gather(
                    x_sparse, W_k, row_offsets_k, nz_col_indices_k
                )
            # V
            if 1-nnz_counts_v.sum() / (K * x_sparse.size(0)) < sparsity_threshold:
                out_sparse_v = torch.ops.dense_op.dense_gemm_sve_omp(x_sparse, W_v)
            else:
                out_sparse_v = torch.ops.sparse_op.sparse_gemm_icsr_sve_gather(
                    x_sparse, W_v, row_offsets_v, nz_col_indices_v
                )

            out_sparse = torch.cat([out_sparse_q, out_sparse_k, out_sparse_v], dim=1)
            out = torch.cat([out_dense, out_sparse], dim=0)
            return out.view(B, M, N)

        nnz_counts_q, nz_col_indices_q, row_offsets_q = torch.ops.sparse_op.thr_sparsify_to_icsr(x_2d, threshold_q)
        nnz_counts_k, nz_col_indices_k, row_offsets_k = torch.ops.sparse_op.thr_sparsify_to_icsr(x_2d, threshold_k)
        nnz_counts_v, nz_col_indices_v, row_offsets_v = torch.ops.sparse_op.thr_sparsify_to_icsr(x_2d, threshold_v)

        sparsity_threshold = MKN2sparsity.get((M, K, W_q.size(1)), DEFAULT_SPARSITY)

        if 1-nnz_counts_q[0] / K < sparsity_threshold:
            out_q = torch.ops.dense_op.dense_gemm_sve_omp(x_2d, W_q)
        else:
            out_q = torch.ops.sparse_op.sparse_gemm_icsr_sve_gather(x_2d, W_q, row_offsets_q, nz_col_indices_q)
        if 1-nnz_counts_k[0] / K < sparsity_threshold:
            out_k = torch.ops.dense_op.dense_gemm_sve_omp(x_2d, W_k)
        else:
            out_k = torch.ops.sparse_op.sparse_gemm_icsr_sve_gather(x_2d, W_k, row_offsets_k, nz_col_indices_k)
        if 1-nnz_counts_v[0] / K < sparsity_threshold:
            out_v = torch.ops.dense_op.dense_gemm_sve_omp(x_2d, W_v)
        else:
            out_v = torch.ops.sparse_op.sparse_gemm_icsr_sve_gather(x_2d, W_v, row_offsets_v, nz_col_indices_v)

        out_2d = torch.cat([out_q, out_k, out_v], dim=1)
        return out_2d.view(B, M, N)


# =========================================================================================================================


#============================Origin Sparse Version===========================
import os
import csv

class StandardSparseGEMV(BaseKernel):

    def meta(
        self,
        x: torch.Tensor,
        W: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        return x.new_empty((x.size(0), x.size(1), W.shape[1]))

    def forward(
        self,
        x: torch.Tensor,
        W: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        _, _, _ = torch.ops.sparse_op.thr_sparsify_to_icsr(torch.zeros(0,0), -1)
        B, M, K = x.shape
        N = W.shape[1]

        x_2d = x.view(-1, K)

        if M > 1:
            split_idx = x_2d.size(0) // 2
            x_dense = x_2d[:split_idx]
            out_dense = torch.ops.dense_op.dense_gemm_sve_omp(x_dense, W)

            x_sparse = x_2d[split_idx:]
            x_thr = torch.where(x_sparse.abs() >= threshold, x_sparse, torch.zeros_like(x_sparse))
            out_sparse = torch.sparse.mm(x_thr.to_sparse_csr(), W)
            out = torch.cat([out_dense, out_sparse], dim=0)
            return out.view(B, M, N)

        x_thr = torch.where(x_2d.abs() >= threshold, x_2d, torch.zeros_like(x_2d))
        x_sp = x_thr.to_sparse_csr()
        out = torch.sparse.mm(x_sp, W)
        return out.view(B, M, N)


class QKVStandardSparseGEMV(BaseKernel):

    def meta(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        threshold_q: float,
        threshold_k: float,
        threshold_v: float,
        kv_size: int,
        *args, **kwargs
    ) -> torch.Tensor:
        return x.new_empty((x.shape[0], x.shape[1], weight.shape[1]))

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        threshold_q: float,
        threshold_k: float,
        threshold_v: float,
        kv_size: int,
        W_q: torch.Tensor,
        W_k: torch.Tensor,
        W_v: torch.Tensor,
    ) -> torch.Tensor:
        _, _, _ = torch.ops.sparse_op.thr_sparsify_to_icsr(torch.zeros(0,0), -1)
        B, M, K = x.shape
        N = weight.size(1)
        x_2d = x.view(-1, K)

        # 如果每个 token 有多个 kv（即 S > 1），则对前一半 dense、后一半 sparse
        if M > 1:
            split_idx = x_2d.size(0) // 2

            # Dense 前半部分
            x_dense = x_2d[:split_idx]
            out_dense = torch.ops.dense_op.dense_gemm_sve_omp(x_dense, weight)

            # Sparse 后半部分
            x_sparse = x_2d[split_idx:]
            x_q_sparse = torch.where(x_sparse.abs() >= threshold_q, x_sparse, torch.zeros_like(x_sparse))
            x_k_sparse = torch.where(x_sparse.abs() >= threshold_k, x_sparse, torch.zeros_like(x_sparse))
            x_v_sparse = torch.where(x_sparse.abs() >= threshold_v, x_sparse, torch.zeros_like(x_sparse))

            out_sparse_q = torch.sparse.mm(x_q_sparse.to_sparse_csr(), W_q)
            out_sparse_k = torch.sparse.mm(x_k_sparse.to_sparse_csr(), W_k)
            out_sparse_v = torch.sparse.mm(x_v_sparse.to_sparse_csr(), W_v)

            out_sparse = torch.cat([out_sparse_q, out_sparse_k, out_sparse_v], dim=1)

            out = torch.cat([out_dense, out_sparse], dim=0)
            return out.view(B, M, N)

        x_q = torch.where(x_2d.abs() >= threshold_q, x_2d, torch.zeros_like(x_2d))
        x_k = torch.where(x_2d.abs() >= threshold_k, x_2d, torch.zeros_like(x_2d))
        x_v = torch.where(x_2d.abs() >= threshold_v, x_2d, torch.zeros_like(x_2d))

        x_q_sp = x_q.to_sparse_csr()
        x_k_sp = x_k.to_sparse_csr()
        x_v_sp = x_v.to_sparse_csr()

        out_q = torch.sparse.mm(x_q_sp, W_q)
        out_k = torch.sparse.mm(x_k_sp, W_k)
        out_v = torch.sparse.mm(x_v_sp, W_v)

        out_2d = torch.cat([out_q, out_k, out_v], dim=1)
        return out_2d.view(B, M, N)

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
        # load_sve_sparse_gemm_extension()
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
        # load_sve_sparse_gemm_extension()
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



# for testing purposes, to see if overhead at 0% is really due to strengthening torch.matmul (seems like it is)
class DenseBaseGEMV(BaseKernel):
    # 用于跟踪已打印的形状组合
    _printed_shapes = set()
    # 统计总调用次数
    _total_calls = 0
    # 统计每种形状的调用次数 {(x_shape, W_shape): count}
    _shape_call_counts = {}
    # 分别统计 prefill (seq_len > 1) 和 decode (seq_len == 1)
    _prefill_calls = 0
    _decode_calls = 0
    _prefill_shape_counts = {}  # prefill 阶段每种形状的调用次数
    _decode_shape_counts = {}   # decode 阶段每种形状的调用次数

    def meta(self, x: torch.Tensor, W: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x.new_empty(x.shape[0], x.shape[1], W.shape[1])
    
    def forward(self, x: torch.Tensor, W: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # 更新总调用次数
        if PRINT_STATISTICS:
            DenseBaseGEMV._total_calls += 1
            
            shape_key = (tuple(x.shape), tuple(W.shape))
            seq_len = x.shape[1] if len(x.shape) >= 2 else 1
            
            # 判断是 prefill 还是 decode
            is_prefill = seq_len > 1
            
            # 更新形状调用次数
            if shape_key not in DenseBaseGEMV._shape_call_counts:
                DenseBaseGEMV._shape_call_counts[shape_key] = 0
            DenseBaseGEMV._shape_call_counts[shape_key] += 1
            
            # 分别统计 prefill 和 decode
            if is_prefill:
                DenseBaseGEMV._prefill_calls += 1
                if shape_key not in DenseBaseGEMV._prefill_shape_counts:
                    DenseBaseGEMV._prefill_shape_counts[shape_key] = 0
                DenseBaseGEMV._prefill_shape_counts[shape_key] += 1
            else:
                DenseBaseGEMV._decode_calls += 1
                if shape_key not in DenseBaseGEMV._decode_shape_counts:
                    DenseBaseGEMV._decode_shape_counts[shape_key] = 0
                DenseBaseGEMV._decode_shape_counts[shape_key] += 1
            
            # 记录遇到的形状（不打印）
            DenseBaseGEMV._printed_shapes.add(shape_key)
        print("DenseBaseGEMV:x.shape=",x.shape, "||W.shape=", W.shape)
        return torch.matmul(x, W)
    
    @classmethod
    def print_statistics(cls):
        """打印统计信息"""
        print(f"\n{'='*80}")
        print(f"[DenseBaseGEMV Statistics]")
        print(f"  Total calls: {cls._total_calls}")
        print(f"    - Prefill (seq_len > 1): {cls._prefill_calls} ({cls._prefill_calls / cls._total_calls * 100:.1f}%)" if cls._total_calls > 0 else "    - Prefill (seq_len > 1): 0")
        print(f"    - Decode (seq_len == 1): {cls._decode_calls} ({cls._decode_calls / cls._total_calls * 100:.1f}%)" if cls._total_calls > 0 else "    - Decode (seq_len == 1): 0")
        print(f"  Total unique shapes: {len(cls._printed_shapes)}")
        
        if cls._prefill_shape_counts:
            print(f"\n  Prefill (seq_len > 1) - Shape call counts:")
            print(f"  {'No.':<5} {'seq_len':<10} {'x.shape':<30} {'W.shape':<30} {'Calls':<10} {'%':<8}")
            print(f"  {'-'*5} {'-'*10} {'-'*30} {'-'*30} {'-'*10} {'-'*8}")
            sorted_prefill = sorted(cls._prefill_shape_counts.items(), key=lambda x: x[1], reverse=True)
            for i, ((x_shape, w_shape), count) in enumerate(sorted_prefill, 1):
                seq_len = x_shape[1] if len(x_shape) >= 2 else 1
                percentage = count / cls._prefill_calls * 100 if cls._prefill_calls > 0 else 0
                print(f"  {i:<5} {seq_len:<10} {str(x_shape):<30} {str(w_shape):<30} {count:<10} {percentage:>6.1f}%")
            print(f"  {'='*5} {'='*10} {'='*30} {'='*30} {'='*10} {'='*8}")
            print(f"  Total prefill calls: {cls._prefill_calls}")
        
        if cls._decode_shape_counts:
            print(f"\n  Decode (seq_len == 1) - Shape call counts:")
            print(f"  {'No.':<5} {'x.shape':<30} {'W.shape':<30} {'Calls':<10} {'%':<8}")
            print(f"  {'-'*5} {'-'*30} {'-'*30} {'-'*10} {'-'*8}")
            sorted_decode = sorted(cls._decode_shape_counts.items(), key=lambda x: x[1], reverse=True)
            for i, ((x_shape, w_shape), count) in enumerate(sorted_decode, 1):
                percentage = count / cls._decode_calls * 100 if cls._decode_calls > 0 else 0
                print(f"  {i:<5} {str(x_shape):<30} {str(w_shape):<30} {count:<10} {percentage:>6.1f}%")
            print(f"  {'='*5} {'='*30} {'='*30} {'='*10} {'='*8}")
            print(f"  Total decode calls: {cls._decode_calls}")
        
        print(f"{'='*80}\n")



# for testing purposes, to see if overhead at 0% is really due to strengthening torch.matmul (seems like it is)
class DenseBaseGEMM(BaseKernel):
    # 用于跟踪已打印的形状组合
    _printed_shapes = set()
    # 统计总调用次数
    _total_calls = 0
    # 统计每种形状的调用次数 {(x_shape, W_shape): count}
    _shape_call_counts = {}
    # 分别统计 prefill (seq_len > 1) 和 decode (seq_len == 1)
    _prefill_calls = 0
    _decode_calls = 0
    _prefill_shape_counts = {}  # prefill 阶段每种形状的调用次数
    _decode_shape_counts = {}   # decode 阶段每种形状的调用次数

    def meta(self, x: torch.Tensor, W: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x.new_empty(x.shape[0], x.shape[1], W.shape[1])
    
    def forward(self, x: torch.Tensor, W: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # 更新总调用次数
        if PRINT_STATISTICS:
            DenseBaseGEMM._total_calls += 1
            
            shape_key = (tuple(x.shape), tuple(W.shape))
            seq_len = x.shape[1] if len(x.shape) >= 2 else 1
            
            # 判断是 prefill 还是 decode
            is_prefill = seq_len > 1
            
            # 更新形状调用次数
            if shape_key not in DenseBaseGEMM._shape_call_counts:
                DenseBaseGEMM._shape_call_counts[shape_key] = 0
            DenseBaseGEMM._shape_call_counts[shape_key] += 1
            
            # 分别统计 prefill 和 decode
            if is_prefill:
                DenseBaseGEMM._prefill_calls += 1
                if shape_key not in DenseBaseGEMM._prefill_shape_counts:
                    DenseBaseGEMM._prefill_shape_counts[shape_key] = 0
                DenseBaseGEMM._prefill_shape_counts[shape_key] += 1
            else:
                DenseBaseGEMM._decode_calls += 1
                if shape_key not in DenseBaseGEMM._decode_shape_counts:
                    DenseBaseGEMM._decode_shape_counts[shape_key] = 0
                DenseBaseGEMM._decode_shape_counts[shape_key] += 1
            
            # 记录遇到的形状（不打印）
            DenseBaseGEMM._printed_shapes.add(shape_key)
        
        print("DenseBaseGEMV:x.shape=",x.shape, "||W.shape=", W.shape)
        return torch.matmul(x, W)
    
    @classmethod
    def print_statistics(cls):
        """打印统计信息"""
        print(f"\n{'='*80}")
        print(f"[DenseBaseGEMM Statistics]")
        print(f"  Total calls: {cls._total_calls}")
        print(f"    - Prefill (seq_len > 1): {cls._prefill_calls} ({cls._prefill_calls / cls._total_calls * 100:.1f}%)" if cls._total_calls > 0 else "    - Prefill (seq_len > 1): 0")
        print(f"    - Decode (seq_len == 1): {cls._decode_calls} ({cls._decode_calls / cls._total_calls * 100:.1f}%)" if cls._total_calls > 0 else "    - Decode (seq_len == 1): 0")
        print(f"  Total unique shapes: {len(cls._printed_shapes)}")
        
        if cls._prefill_shape_counts:
            print(f"\n  Prefill (seq_len > 1) - Shape call counts:")
            print(f"  {'No.':<5} {'seq_len':<10} {'x.shape':<30} {'W.shape':<30} {'Calls':<10} {'%':<8}")
            print(f"  {'-'*5} {'-'*10} {'-'*30} {'-'*30} {'-'*10} {'-'*8}")
            sorted_prefill = sorted(cls._prefill_shape_counts.items(), key=lambda x: x[1], reverse=True)
            for i, ((x_shape, w_shape), count) in enumerate(sorted_prefill, 1):
                seq_len = x_shape[1] if len(x_shape) >= 2 else 1
                percentage = count / cls._prefill_calls * 100 if cls._prefill_calls > 0 else 0
                print(f"  {i:<5} {seq_len:<10} {str(x_shape):<30} {str(w_shape):<30} {count:<10} {percentage:>6.1f}%")
            print(f"  {'='*5} {'='*10} {'='*30} {'='*30} {'='*10} {'='*8}")
            print(f"  Total prefill calls: {cls._prefill_calls}")
        
        if cls._decode_shape_counts:
            print(f"\n  Decode (seq_len == 1) - Shape call counts:")
            print(f"  {'No.':<5} {'x.shape':<30} {'W.shape':<30} {'Calls':<10} {'%':<8}")
            print(f"  {'-'*5} {'-'*30} {'-'*30} {'-'*10} {'-'*8}")
            sorted_decode = sorted(cls._decode_shape_counts.items(), key=lambda x: x[1], reverse=True)
            for i, ((x_shape, w_shape), count) in enumerate(sorted_decode, 1):
                percentage = count / cls._decode_calls * 100 if cls._decode_calls > 0 else 0
                print(f"  {i:<5} {str(x_shape):<30} {str(w_shape):<30} {count:<10} {percentage:>6.1f}%")
            print(f"  {'='*5} {'='*30} {'='*30} {'='*10} {'='*8}")
            print(f"  Total decode calls: {cls._decode_calls}")
        
        print(f"{'='*80}\n")


class DenseGEMMSVEOMPKernel(BaseKernel):
    """torch.compile-friendly dense GEMM wrapper backed by the C++ OpenMP kernel."""

    def meta(
        self,
        activation: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        M = activation.size(0)
        N = weight.size(1)
        return activation.new_empty((M, N), device="meta")

    def forward(
        self,
        activation: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        load_sve_sparse_gemm_extension()
        return torch.ops.dense_op.dense_gemm_sve_omp(activation, weight)


def dense_gemm_sve_omp(
    activation: torch.Tensor,
    weight: torch.Tensor,
    verbose: bool = False,
) -> torch.Tensor:
    """Dense GEMM wrapper for the C++ OpenMP kernel."""
    load_sve_sparse_gemm_extension(verbose=verbose)
    return torch.ops.dense_op.dense_gemm_sve_omp(activation, weight)


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


__all__ = [
    # Configuration
    "USE_CUSTOM_SPARSE_GEMM",
    "DENSE_THRESHOLD",
    "MIN_DENSE_BLOCK",
    "SPARSE_DEBUG",
    # Extension loading
    "load_sve_sparse_gemm_extension",
    "measure_latency",
    # Kernels
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
    "DenseGEMMSVEOMPKernel",
    # Adaptive GEMM/GEMV
    "adaptive_sparse_gemm",
    "adaptive_sparse_gemv",
    "dense_gemm_sve_omp",
    # Sparsification utilities
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