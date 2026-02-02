"""
兼容导入路径：允许通过 `kernels.cpp_sve_sparse_gemm` 导入 Python wrapper。

注意：该目录下的 `.cpp` 文件是 C++ 扩展源码；Python 侧 wrapper 实现在 `kernels.sve_sparse_gemm`。
"""

from kernels.sve_sparse_gemm import (  # noqa: F401
    # extension loader / utils
    load_sve_sparse_gemm_extension,
    measure_latency,
    # iCSR
    SparseGEMMiCSRSVEGatherKernel,
    SparseGEMMICSRKernel,
    thr_sparsify_to_icsr,
    thr_sparsify_to_icsr_sve,
    # CSR
    SparseGEMMCSRKernel,
    SparseGEMMCSRSVEGatherKernel,
    thr_sparsify_to_csr,
    thr_sparsify_to_csr_sve,
    # COO
    SparseGEMMCOOKernel,
    SparseGEMMCOOSVEGatherKernel,
    thr_sparsify_to_coo,
    thr_sparsify_to_coo_sve,
    # CSC
    SparseGEMMCSCKernel,
    thr_sparsify_to_csc,
)

__all__ = [
    "load_sve_sparse_gemm_extension",
    "measure_latency",
    # iCSR
    "SparseGEMMiCSRSVEGatherKernel",
    "SparseGEMMICSRKernel",
    "thr_sparsify_to_icsr",
    "thr_sparsify_to_icsr_sve",
    # CSR
    "SparseGEMMCSRKernel",
    "SparseGEMMCSRSVEGatherKernel",
    "thr_sparsify_to_csr",
    "thr_sparsify_to_csr_sve",
    # COO
    "SparseGEMMCOOKernel",
    "SparseGEMMCOOSVEGatherKernel",
    "thr_sparsify_to_coo",
    "thr_sparsify_to_coo_sve",
    # CSC
    "SparseGEMMCSCKernel",
    "thr_sparsify_to_csc",
]

