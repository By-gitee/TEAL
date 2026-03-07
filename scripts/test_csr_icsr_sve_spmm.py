"""
Comprehensive test script for ARM SVE sparse GEMM operators.

This script tests combinations of sparsify and GEMM operators for different sparse formats (17 in total):

Custom SVE operators (13):
- iCSR: thr_sparsify_to_icsr / thr_sparsify_to_icsr_sve / thr_sparsify_to_icsr_sve_baseline + sparse_gemm_icsr(_sve_gather)
- CSR: thr_sparsify_to_csr(_sve) + sparse_gemm_csr(_sve_gather)
- COO: thr_sparsify_to_coo(_sve) + sparse_gemm_coo(_sve_gather)
- CSC: thr_sparsify_to_csc + sparse_gemm_csc

PyTorch reference implementations (4):
- Dense matmul
- Sparse CSR + sparse.mm
- Sparse CSC + sparse.mm
- Selective load of weight non-zero rows + matmul

Test coverage:
1. Correctness: compare with PyTorch reference
2. Performance: measure latency per combination
3. Speedup: relative to PyTorch dense implementation
4. Ranking: find the fastest operator combination

Usage:
    python -m scripts.test_sparse_formats
    python -m scripts.test_sparse_formats --threshold 0.8 --M 16 --K 512 --N 1024
"""

from __future__ import annotations

import argparse
import csv
import os
import torch
import time
from typing import Any, Dict, List, Tuple

from kernels.cpp_sve_sparse_gemm import (
    # iCSR operators
    SparseGEMMiCSRSVEGatherKernel,
    SparseGEMMICSRKernel,
    thr_sparsify_to_icsr,
    thr_sparsify_to_icsr_sve,
    thr_sparsify_to_icsr_sve_baseline,
    # CSR operators
    SparseGEMMCSRKernel,
    SparseGEMMCSRSVEGatherKernel,
    thr_sparsify_to_csr,
    thr_sparsify_to_csr_sve,
    # COO operators
    SparseGEMMCOOKernel,
    SparseGEMMCOOSVEGatherKernel,
    thr_sparsify_to_coo,
    thr_sparsify_to_coo_sve,
    # CSC operators
    SparseGEMMCSCKernel,
    thr_sparsify_to_csc,
    # Utility
    load_sve_sparse_gemm_extension,
)
from kernels.kernel_utils import measure_latency

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

it_num = 100
warm_times = 100

MV = [1]
KNV = [(4096, 4096)]
thresholds = [0.8]


def _apply_threshold(activation: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Apply threshold to activation: keep values with abs(x) >= threshold, zero the rest.

    Note: thr_sparsify_to_* in cpp_sve_sparse_gemm use abs(x) >= thr; this must match
    to avoid systematic correctness bias (especially when activation has negative values).
    """
    return torch.where(activation.abs() >= threshold, activation, torch.zeros_like(activation))


def _print_ranked_latencies(
    title: str,
    latencies: List[Tuple[str, float]],
    baseline_latency: float | None = None,
) -> None:
    """Print ranking by latency (fastest first), optionally with speedup vs baseline."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if not latencies:
        print("(no data)")
        return

    latencies = sorted(latencies, key=lambda x: x[1])
    if baseline_latency is not None and baseline_latency > 0:
        print(f"{'Rank':<4} {'Item':<64} {'Latency(ms)':<12} {'Speedup':<10}")
        print("-" * 94)
        for rank, (name, latency) in enumerate(latencies, 1):
            speedup = baseline_latency / latency if latency > 0 else 0.0
            print(f"{rank:2d}. {name:64s} {latency:8.4f} ms  {speedup:6.2f}x")
    else:
        print(f"{'Rank':<4} {'Item':<72} {'Latency(ms)':<12}")
        print("-" * 92)
        for rank, (name, latency) in enumerate(latencies, 1):
            print(f"{rank:2d}. {name:72s} {latency:8.4f} ms")

    fastest_name, fastest_latency = latencies[0]
    print("\n" + "-" * 80)
    print(f"⚡ Fastest: {fastest_name}")
    print(f"   Latency: {fastest_latency:.4f} ms")
    if baseline_latency is not None and baseline_latency > 0 and fastest_latency > 0:
        print(f"   Speedup vs baseline: {baseline_latency/fastest_latency:.2f}x")


def _maybe_print_cpu_util(prefix: str, interval_s: float = 0.20) -> None:
    """Print CPU utilization (system + current process), best-effort.

    Requires psutil; skips silently if unavailable (main prints a note once).
    """
    if psutil is None:
        return

    try:
        proc = psutil.Process()
        # Set system baseline, then block-sample proc for same time window
        psutil.cpu_percent(interval=None)
        proc_cpu = proc.cpu_percent(interval=interval_s)
        sys_cpu = psutil.cpu_percent(interval=None)

        cpu_cnt = psutil.cpu_count(logical=True) or 1
        proc_cpu_norm = proc_cpu / cpu_cnt

        mem = proc.memory_info().rss / (1024 * 1024)
        threads = proc.num_threads()

        ts = time.strftime("%H:%M:%S")
        print(
            f"[CPU {ts}] {prefix} | sys={sys_cpu:5.1f}% | proc={proc_cpu:6.1f}% "
            f"(norm={proc_cpu_norm:5.1f}%) | rss={mem:7.1f} MB | thr={threads}"
        )
    except Exception:
        # Do not let monitoring affect test flow
        return


def verify_correctness_flat(
    results: Dict[str, Tuple[torch.Tensor, float]],
    reference: torch.Tensor,
) -> None:
    """Verify correctness of a flat result set (compare each item to reference)."""
    print("\n" + "=" * 80)
    print("Correctness verification (flat result set)")
    print("=" * 80)

    all_passed = True
    for name, (result, _) in results.items():
        max_diff = torch.max(torch.abs(result - reference)).item()
        mean_diff = torch.mean(torch.abs(result - reference)).item()
        is_correct = torch.allclose(result, reference, rtol=1e-4, atol=1e-5)
        status = "✅" if is_correct else "❌"
        print(f"  {status} {name}")
        print(f"      Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
        if not is_correct:
            all_passed = False

    if all_passed:
        print("\n✅ Correctness test passed")
    else:
        print("\n❌ Some results incorrect")


def test_core_gemm_only(
    activation: torch.Tensor,
    activation_thresholded: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
) -> Dict[str, Tuple[torch.Tensor, float]]:
    """Test only core GEMM kernel latency; sparse representation is cached, no sparsify in timing loop."""
    print("\n" + "=" * 80)
    print("Core GEMM-only comparison")
    print("=" * 80)

    results: Dict[str, Tuple[torch.Tensor, float]] = {}

    # Baseline: PyTorch dense matmul (thresholded activation precomputed externally)
    print("\n[GEMM-only][PyTorch] torch.matmul(activation_thresholded, weight)")
    def torch_dense_core():
        return torch.matmul(activation_thresholded, weight)

    lat_torch = measure_latency(torch_dense_core, warmup=warm_times, iters=it_num)
    results["GEMM-only: PyTorch torch.matmul(thresholded, weight)"] = (torch_dense_core(), lat_torch)
    print(f"  Latency: {lat_torch:.4f} ms")

    # iCSR: sparsify once, timing loop runs sparse_gemm only
    print("\n[GEMM-only][iCSR] Pre thr_sparsify_to_icsr")
    nz_counts, nz_col_indices, row_offsets = thr_sparsify_to_icsr(activation, threshold)

    icsr_sve_gather_kernel = SparseGEMMiCSRSVEGatherKernel.initialize(
        name="sparse_gemm_icsr_sve_gather", target="CPU"
    )
    icsr_sve_gather_op = icsr_sve_gather_kernel.operator(compiled=True)

    icsr_kernel = SparseGEMMICSRKernel.initialize(
        name="sparse_gemm_icsr", target="CPU"
    )
    icsr_op = icsr_kernel.operator(compiled=True)

    print("  - sparse_gemm_icsr_sve_gather")
    def icsr_gemm_gather_only():
        return icsr_sve_gather_op(activation, weight, row_offsets, nz_col_indices)

    lat = measure_latency(icsr_gemm_gather_only, warmup=warm_times, iters=it_num)
    results["GEMM-only: iCSR sparse_gemm_icsr_sve_gather (cached indices)"] = (icsr_gemm_gather_only(), lat)
    print(f"    Latency: {lat:.4f} ms")

    print("  - sparse_gemm_icsr")
    def icsr_gemm_only():
        return icsr_op(activation, weight, row_offsets, nz_col_indices)

    lat = measure_latency(icsr_gemm_only, warmup=warm_times, iters=it_num)
    results["GEMM-only: iCSR sparse_gemm_icsr (cached indices)"] = (icsr_gemm_only(), lat)
    print(f"    Latency: {lat:.4f} ms")

    # CSR
    print("\n[GEMM-only][CSR] Pre thr_sparsify_to_csr")
    csr_row_offsets, csr_nz_col_indices, csr_values = thr_sparsify_to_csr(activation, threshold)

    csr_kernel = SparseGEMMCSRKernel.initialize(name="sparse_gemm_csr", target="CPU")
    csr_op = csr_kernel.operator(compiled=True)

    csr_sve_gather_kernel = SparseGEMMCSRSVEGatherKernel.initialize(
        name="sparse_gemm_csr_sve_gather", target="CPU"
    )
    csr_sve_gather_op = csr_sve_gather_kernel.operator(compiled=True)

    print("  - sparse_gemm_csr")
    def csr_gemm_only():
        return csr_op(weight, csr_row_offsets, csr_nz_col_indices, csr_values)

    lat = measure_latency(csr_gemm_only, warmup=warm_times, iters=it_num)
    results["GEMM-only: CSR sparse_gemm_csr (cached values)"] = (csr_gemm_only(), lat)
    print(f"    Latency: {lat:.4f} ms")

    print("  - sparse_gemm_csr_sve_gather")
    def csr_gemm_gather_only():
        return csr_sve_gather_op(weight, csr_row_offsets, csr_nz_col_indices, csr_values)

    lat = measure_latency(csr_gemm_gather_only, warmup=warm_times, iters=it_num)
    results["GEMM-only: CSR sparse_gemm_csr_sve_gather (cached values)"] = (csr_gemm_gather_only(), lat)
    print(f"    Latency: {lat:.4f} ms")

    return results


def test_preprocess_only(
    activation: torch.Tensor,
    threshold: float,
) -> Dict[str, Tuple[Any, float]]:
    """Test only input sparse matrix preprocessing (sparsify/format conversion) latency; no GEMM."""
    print("\n" + "=" * 80)
    print("Preprocess-only comparison")
    print("=" * 80)

    results: Dict[str, Tuple[Any, float]] = {}

    # Baseline 1: threshold only (matches cpp abs(x) >= thr)
    print("\n[Preprocess-only][PyTorch] _apply_threshold")
    def torch_threshold_only():
        return _apply_threshold(activation, threshold=threshold)

    lat = measure_latency(torch_threshold_only, warmup=warm_times, iters=10)
    results["Preprocess-only: PyTorch _apply_threshold(abs>=thr)"] = (torch_threshold_only(), lat)
    print(f"  Latency: {lat:.4f} ms")

    # Baseline 2: threshold + to_sparse_csr (end-to-end PyTorch CSR build)
    print("\n[Preprocess-only][PyTorch] threshold + to_sparse_csr")
    def torch_to_sparse_csr():
        x = _apply_threshold(activation, threshold=threshold)
        sp = x.to_sparse_csr()
        # Touch components so lazy path does not defer work
        _ = sp.crow_indices()
        _ = sp.col_indices()
        _ = sp.values()
        return sp

    lat = measure_latency(torch_to_sparse_csr, warmup=warm_times, iters=10)
    results["Preprocess-only: PyTorch threshold + to_sparse_csr()"] = (torch_to_sparse_csr(), lat)
    print(f"  Latency: {lat:.4f} ms")

    # Custom: iCSR sparsify
    print("\n[Preprocess-only][iCSR] thr_sparsify_to_icsr / thr_sparsify_to_icsr_sve / thr_sparsify_to_icsr_sve_baseline")
    def icsr_pre_1():
        return thr_sparsify_to_icsr(activation, threshold)

    lat = measure_latency(icsr_pre_1, warmup=warm_times, iters=it_num)
    results["Preprocess-only: iCSR thr_sparsify_to_icsr"] = (icsr_pre_1(), lat)
    print(f"  - thr_sparsify_to_icsr: Latency {lat:.4f} ms")

    def icsr_pre_2():
        return thr_sparsify_to_icsr_sve(activation, threshold)

    lat = measure_latency(icsr_pre_2, warmup=warm_times, iters=it_num)
    results["Preprocess-only: iCSR thr_sparsify_to_icsr_sve"] = (icsr_pre_2(), lat)
    print(f"  - thr_sparsify_to_icsr_sve: Latency {lat:.4f} ms")

    # Custom: CSR sparsify
    print("\n[Preprocess-only][CSR] thr_sparsify_to_csr / thr_sparsify_to_csr_sve")
    def csr_pre_1():
        return thr_sparsify_to_csr(activation, threshold)

    lat = measure_latency(csr_pre_1, warmup=warm_times, iters=it_num)
    results["Preprocess-only: CSR thr_sparsify_to_csr"] = (csr_pre_1(), lat)
    print(f"  - thr_sparsify_to_csr: Latency {lat:.4f} ms")

    def csr_pre_2():
        return thr_sparsify_to_csr_sve(activation, threshold)

    lat = measure_latency(csr_pre_2, warmup=warm_times, iters=it_num)
    results["Preprocess-only: CSR thr_sparsify_to_csr_sve"] = (csr_pre_2(), lat)
    print(f"  - thr_sparsify_to_csr_sve: Latency {lat:.4f} ms")


    return results


# =============================================================================
# iCSR format combination tests
# =============================================================================

def test_icsr_combinations(
    activation: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
) -> Dict[str, Tuple[torch.Tensor, float]]:
    """
    Test all combinations of iCSR sparsify and GEMM operators.

    Combinations:
    - thr_sparsify_to_icsr + sparse_gemm_icsr_sve_gather
    - thr_sparsify_to_icsr_sve + sparse_gemm_icsr_sve_gather
    - thr_sparsify_to_icsr + sparse_gemm_icsr
    - thr_sparsify_to_icsr_sve + sparse_gemm_icsr
    - thr_sparsify_to_icsr_sve_baseline + sparse_gemm_icsr_sve_gather
    - thr_sparsify_to_icsr_sve_baseline + sparse_gemm_icsr

    Returns:
        Dict[combo_name, (result, latency)]
    """
    print("\n" + "=" * 80)
    print("Test iCSR format combinations")
    print("=" * 80)

    results = {}

    # Initialize GEMM operators
    icsr_sve_gather_kernel = SparseGEMMiCSRSVEGatherKernel.initialize(
        name="sparse_gemm_icsr_sve_gather", target="CPU"
    )
    icsr_sve_gather_op = icsr_sve_gather_kernel.operator(compiled=True)
    
    icsr_kernel = SparseGEMMICSRKernel.initialize(
        name="sparse_gemm_icsr", target="CPU"
    )
    icsr_op = icsr_kernel.operator(compiled=True)
    
    # Combo 1: thr_sparsify_to_icsr + sparse_gemm_icsr_sve_gather
    print("\n[iCSR-1] thr_sparsify_to_icsr + sparse_gemm_icsr_sve_gather")
    def icsr_combo1():
        nz_counts, nz_col_indices, row_offsets = thr_sparsify_to_icsr(activation, threshold)
        return icsr_sve_gather_op(activation, weight, row_offsets, nz_col_indices)
    
    lat1 = measure_latency(icsr_combo1, warmup=warm_times, iters=it_num)
    result1 = icsr_combo1()
    results["iCSR-1: thr_sparsify_to_icsr + sparse_gemm_icsr_sve_gather"] = (result1, lat1)
    print(f"  Latency: {lat1:.4f} ms")

    # Combo 2: thr_sparsify_to_icsr_sve + sparse_gemm_icsr_sve_gather
    print("\n[iCSR-2] thr_sparsify_to_icsr_sve + sparse_gemm_icsr_sve_gather")
    def icsr_combo2():
        nz_counts, nz_col_indices, row_offsets = thr_sparsify_to_icsr_sve(activation, threshold)
        return icsr_sve_gather_op(activation, weight, row_offsets, nz_col_indices)
    
    lat2 = measure_latency(icsr_combo2, warmup=warm_times, iters=it_num)
    result2 = icsr_combo2()
    results["iCSR-2: thr_sparsify_to_icsr_sve + sparse_gemm_icsr_sve_gather"] = (result2, lat2)
    print(f"  Latency: {lat2:.4f} ms")

    # Combo 3: thr_sparsify_to_icsr + sparse_gemm_icsr
    print("\n[iCSR-3] thr_sparsify_to_icsr + sparse_gemm_icsr")
    def icsr_combo3():
        nz_counts, nz_col_indices, row_offsets = thr_sparsify_to_icsr(activation, threshold)
        return icsr_op(activation, weight, row_offsets, nz_col_indices)
    
    lat3 = measure_latency(icsr_combo3, warmup=warm_times, iters=it_num)
    result3 = icsr_combo3()
    results["iCSR-3: thr_sparsify_to_icsr + sparse_gemm_icsr"] = (result3, lat3)
    print(f"  Latency: {lat3:.4f} ms")

    # Combo 4: thr_sparsify_to_icsr_sve + sparse_gemm_icsr
    print("\n[iCSR-4] thr_sparsify_to_icsr_sve + sparse_gemm_icsr")
    def icsr_combo4():
        nz_counts, nz_col_indices, row_offsets = thr_sparsify_to_icsr_sve(activation, threshold)
        return icsr_op(activation, weight, row_offsets, nz_col_indices)
    
    lat4 = measure_latency(icsr_combo4, warmup=warm_times, iters=it_num)
    result4 = icsr_combo4()
    results["iCSR-4: thr_sparsify_to_icsr_sve + sparse_gemm_icsr"] = (result4, lat4)
    print(f"  Latency: {lat4:.4f} ms")

    return results


# =============================================================================
# CSR format combination tests
# =============================================================================

def test_csr_combinations(
    activation: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
) -> Dict[str, Tuple[torch.Tensor, float]]:
    """
    Test all combinations of CSR sparsify and GEMM operators.

    Combinations:
    - thr_sparsify_to_csr + sparse_gemm_csr
    - thr_sparsify_to_csr_sve + sparse_gemm_csr
    - thr_sparsify_to_csr + sparse_gemm_csr_sve_gather
    - thr_sparsify_to_csr_sve + sparse_gemm_csr_sve_gather

    Returns:
        Dict[combo_name, (result, latency)]
    """
    print("\n" + "=" * 80)
    print("Test CSR format combinations")
    print("=" * 80)

    results = {}

    # Initialize GEMM operators
    csr_kernel = SparseGEMMCSRKernel.initialize(
        name="sparse_gemm_csr", target="CPU"
    )
    csr_op = csr_kernel.operator(compiled=True)
    
    csr_sve_gather_kernel = SparseGEMMCSRSVEGatherKernel.initialize(
        name="sparse_gemm_csr_sve_gather", target="CPU"
    )
    csr_sve_gather_op = csr_sve_gather_kernel.operator(compiled=True)
    
    # Combo 1: thr_sparsify_to_csr + sparse_gemm_csr
    print("\n[CSR-1] thr_sparsify_to_csr + sparse_gemm_csr")
    def csr_combo1():
        row_offsets, nz_col_indices, values = thr_sparsify_to_csr(activation, threshold)
        return csr_op(weight, row_offsets, nz_col_indices, values)
    
    lat1 = measure_latency(csr_combo1, warmup=warm_times, iters=it_num)
    result1 = csr_combo1()
    results["CSR-1: thr_sparsify_to_csr + sparse_gemm_csr"] = (result1, lat1)
    print(f"  Latency: {lat1:.4f} ms")

    # Combo 2: thr_sparsify_to_csr_sve + sparse_gemm_csr
    print("\n[CSR-2] thr_sparsify_to_csr_sve + sparse_gemm_csr")
    def csr_combo2():
        row_offsets, nz_col_indices, values = thr_sparsify_to_csr_sve(activation, threshold)
        return csr_op(weight, row_offsets, nz_col_indices, values)
    
    lat2 = measure_latency(csr_combo2, warmup=warm_times, iters=it_num)
    result2 = csr_combo2()
    results["CSR-2: thr_sparsify_to_csr_sve + sparse_gemm_csr"] = (result2, lat2)
    print(f"  Latency: {lat2:.4f} ms")

    # Combo 3: thr_sparsify_to_csr + sparse_gemm_csr_sve_gather
    print("\n[CSR-3] thr_sparsify_to_csr + sparse_gemm_csr_sve_gather")
    def csr_combo3():
        row_offsets, nz_col_indices, values = thr_sparsify_to_csr(activation, threshold)
        return csr_sve_gather_op(weight, row_offsets, nz_col_indices, values)
    
    lat3 = measure_latency(csr_combo3, warmup=warm_times, iters=it_num)
    result3 = csr_combo3()
    results["CSR-3: thr_sparsify_to_csr + sparse_gemm_csr_sve_gather"] = (result3, lat3)
    print(f"  Latency: {lat3:.4f} ms")

    # Combo 4: thr_sparsify_to_csr_sve + sparse_gemm_csr_sve_gather
    print("\n[CSR-4] thr_sparsify_to_csr_sve + sparse_gemm_csr_sve_gather")
    def csr_combo4():
        row_offsets, nz_col_indices, values = thr_sparsify_to_csr_sve(activation, threshold)
        return csr_sve_gather_op(weight, row_offsets, nz_col_indices, values)
    
    lat4 = measure_latency(csr_combo4, warmup=warm_times, iters=it_num)
    result4 = csr_combo4()
    results["CSR-4: thr_sparsify_to_csr_sve + sparse_gemm_csr_sve_gather"] = (result4, lat4)
    print(f"  Latency: {lat4:.4f} ms")

    return results


# # =============================================================================
# # PyTorch reference implementation tests
# # =============================================================================

def test_pytorch_references(
    activation: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
) -> Dict[str, Tuple[torch.Tensor, float]]:
    """
    Test PyTorch reference implementations.

    Methods:
    - PyTorch dense matmul
    - PyTorch sparse CSR + sparse.mm
    - PyTorch sparse CSC + sparse.mm
    - PyTorch selective load of weight non-zero rows + matmul

    Returns:
        Dict[combo_name, (result, latency)]
    """
    print("\n" + "=" * 80)
    print("Test PyTorch reference implementations")
    print("=" * 80)

    results = {}

    # Enable MKL-DNN
    torch.backends.mkldnn.enabled = True

    # PyTorch dense matmul
    print("\n[PyTorch-1] Dense matmul")
    def pytorch_dense_fn():
        activation_thresholded = _apply_threshold(activation, threshold=threshold)
        return torch.matmul(activation_thresholded, weight)
    
    lat1 = measure_latency(pytorch_dense_fn, warmup=warm_times, iters=it_num)
    result1 = pytorch_dense_fn()
    results["PyTorch-1: Dense matmul"] = (result1, lat1)
    print(f"  Latency: {lat1:.4f} ms")

    # PyTorch sparse CSR + sparse.mm
    print("\n[PyTorch-2] Sparse CSR + sparse.mm")
    def pytorch_sparse_csr_fn():
        activation_thresholded = _apply_threshold(activation, threshold=threshold)
        sp_act = activation_thresholded.to_sparse_csr()
        return torch.sparse.mm(sp_act, weight)
    
    lat2 = measure_latency(pytorch_sparse_csr_fn, warmup=warm_times, iters=it_num)
    result2 = pytorch_sparse_csr_fn()
    results["PyTorch-2: Sparse CSR + sparse.mm"] = (result2, lat2)
    print(f"  Latency: {lat2:.4f} ms")

    # PyTorch selective load of weight non-zero rows + matmul
    print("\n[PyTorch-4] Selective load weight non-zero rows + matmul")
    def pytorch_selective_weight_fn():
        activation_thresholded = _apply_threshold(activation, threshold=threshold)
        M, K = activation_thresholded.shape
        N = weight.shape[1]
        
        # Initialize output
        output = torch.zeros(M, N, dtype=torch.float32)

        # Process each row
        for m in range(M):
            # Find non-zero column indices for this row
            nz_cols = torch.nonzero(activation_thresholded[m], as_tuple=False).flatten()

            if nz_cols.numel() > 0:
                # Select only corresponding non-zero rows of weight
                act_nz = activation_thresholded[m, nz_cols]  # (nnz,)
                weight_nz = weight[nz_cols, :]  # (nnz, N)

                # Matrix multiply: (1, nnz) @ (nnz, N) -> (1, N)
                output[m] = torch.matmul(act_nz.unsqueeze(0), weight_nz).squeeze(0)

        return output

    lat4 = measure_latency(pytorch_selective_weight_fn, warmup=warm_times, iters=it_num)
    result4 = pytorch_selective_weight_fn()
    results["PyTorch-4: Selective load weight non-zero rows + matmul"] = (result4, lat4)
    print(f"  Latency: {lat4:.4f} ms")
    
    return results


# =============================================================================
# Correctness verification and summary
# =============================================================================

def verify_correctness(
    all_results: Dict[str, Dict[str, Tuple[torch.Tensor, float]]],
    reference: torch.Tensor,
) -> None:
    """Verify correctness of all operator combinations."""
    print("\n" + "=" * 80)
    print("Correctness verification")
    print("=" * 80)
    
    all_passed = True
    
    for format_name, results in all_results.items():
        print(f"\n{format_name}:")
        for combo_name, (result, latency) in results.items():
            max_diff = torch.max(torch.abs(result - reference)).item()
            mean_diff = torch.mean(torch.abs(result - reference)).item()
            
            is_correct = torch.allclose(result, reference, rtol=1e-4, atol=1e-5)
            status = "✅" if is_correct else "❌"
            
            print(f"  {status} {combo_name}")
            print(f"      Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")

            if not is_correct:
                all_passed = False

    if all_passed:
        print("\n✅ All combinations passed correctness test")
    else:
        print("\n❌ Some combinations failed correctness test")


def print_performance_summary(
    all_results: Dict[str, Dict[str, Tuple[torch.Tensor, float]]],
) -> None:
    """Print performance comparison summary."""
    print("\n" + "=" * 80)
    print("Performance summary")
    print("=" * 80)

    # Collect all latency data
    all_latencies = []
    pytorch_dense_latency = None

    for format_name, results in all_results.items():
        for combo_name, (result, latency) in results.items():
            all_latencies.append((combo_name, latency, format_name))
            # Use PyTorch dense latency as baseline
            if "PyTorch-1: Dense matmul" in combo_name:
                pytorch_dense_latency = latency

    # Sort by latency
    all_latencies.sort(key=lambda x: x[1])

    print("\nLatency ranking (fastest first):")
    print("-" * 86)
    if pytorch_dense_latency is not None:
        print(f"{'Rank':<4} {'Operator combo':<62} {'Latency(ms)':<12} {'Speedup':<10}")
        print("-" * 86)
        for rank, (combo_name, latency, format_name) in enumerate(all_latencies, 1):
            speedup = pytorch_dense_latency / latency if latency > 0 else 0.0
            # Highlight custom operators (non-PyTorch)
            marker = "🚀" if format_name != "PyTorch" else "📊"
            print(f"{rank:2d}. {marker} {combo_name:60s} {latency:8.4f} ms  {speedup:6.2f}x")
    else:
        for rank, (combo_name, latency, format_name) in enumerate(all_latencies, 1):
            print(f"{rank:2d}. {combo_name:60s} {latency:8.4f} ms")
    
    # Find fastest combination
    fastest_name, fastest_latency, fastest_format = all_latencies[0]
    print("\n" + "=" * 80)
    print(f"⚡ Fastest combination: {fastest_name}")
    print(f"   Latency: {fastest_latency:.4f} ms")
    if pytorch_dense_latency is not None:
        speedup = pytorch_dense_latency / fastest_latency
        print(f"   Speedup vs PyTorch dense: {speedup:.2f}x")
    print("=" * 80)

    # Custom operator stats vs PyTorch
    if pytorch_dense_latency is not None:
        print("\n" + "=" * 80)
        print("Custom operator performance stats")
        print("=" * 80)

        custom_latencies = [(name, lat) for name, lat, fmt in all_latencies if fmt != "PyTorch"]
        if custom_latencies:
            fastest_custom_name, fastest_custom_latency = custom_latencies[0]
            print(f"\nFastest custom operator: {fastest_custom_name}")
            print(f"  Latency: {fastest_custom_latency:.4f} ms")
            print(f"  Speedup vs PyTorch dense: {pytorch_dense_latency/fastest_custom_latency:.2f}x")

            # Count custom operators faster than PyTorch dense
            faster_than_dense = sum(1 for _, lat in custom_latencies if lat < pytorch_dense_latency)
            print(f"\nCustom operators faster than PyTorch dense: {faster_than_dense}/{len(custom_latencies)}")

            # PyTorch non-dense latencies (excluding dense matmul)
            pytorch_nondense_latencies = [(name, lat) for name, lat, fmt in all_latencies
                                          if fmt == "PyTorch" and "Dense matmul" not in name]
            if pytorch_nondense_latencies:
                fastest_pytorch_nondense = min(pytorch_nondense_latencies, key=lambda x: x[1])
                print(f"\nFastest PyTorch non-dense: {fastest_pytorch_nondense[0]}")
                print(f"  Latency: {fastest_pytorch_nondense[1]:.4f} ms")
                faster_than_nondense = sum(1 for _, lat in custom_latencies if lat < fastest_pytorch_nondense[1])
                print(f"\nCustom operators faster than PyTorch fastest non-dense: {faster_than_nondense}/{len(custom_latencies)}")

#         print("=" * 80)


def export_results_to_csv(
    csv_path: str,
    M: int,
    K: int,
    N: int,
    threshold: float,
    sparsity_pct: float,
    reference: torch.Tensor,
    all_results: Dict[str, Dict[str, Tuple[torch.Tensor, float]]],
    gemm_only_results: Dict[str, Tuple[torch.Tensor, float]],
    preprocess_only_results: Dict[str, Tuple[Any, float]],
) -> None:
    """Export all benchmark results (end-to-end / GEMM-only / preprocess-only) to CSV."""
    pytorch_dense_latency: float | None = None
    for _fmt, _res in all_results.items():
        for combo_name, (_, lat) in _res.items():
            if "PyTorch-1: Dense matmul" in combo_name:
                pytorch_dense_latency = lat
                break

    rows: List[Dict[str, Any]] = []
    base = dict(
        M=M, K=K, N=N,
        threshold=threshold,
        sparsity_pct=f"{sparsity_pct:.2f}",
    )

    def _speedup(lat: float) -> str:
        if pytorch_dense_latency and lat > 0:
            return f"{pytorch_dense_latency / lat:.4f}"
        return ""

    def _correctness(result: torch.Tensor) -> tuple[str, str, str]:
        try:
            max_diff = torch.max(torch.abs(result - reference)).item()
            mean_diff = torch.mean(torch.abs(result - reference)).item()
            ok = torch.allclose(result, reference, rtol=1e-4, atol=1e-5)
            return ("PASS" if ok else "FAIL", f"{max_diff:.6e}", f"{mean_diff:.6e}")
        except Exception:
            return ("N/A", "N/A", "N/A")

    # End-to-end combo results
    for fmt_name, results in all_results.items():
        for combo_name, (result, latency) in results.items():
            ok, maxd, meand = _correctness(result)
            rows.append({
                **base,
                "test_type": "end_to_end",
                "category": fmt_name,
                "operator": combo_name,
                "latency_ms": f"{latency:.6f}",
                "speedup_vs_pytorch_dense": _speedup(latency),
                "is_correct": ok,
                "max_diff": maxd,
                "mean_diff": meand,
            })

    # GEMM-only results
    for combo_name, (result, latency) in gemm_only_results.items():
        ok, maxd, meand = _correctness(result)
        rows.append({
            **base,
            "test_type": "gemm_only",
            "category": "GEMM-only",
            "operator": combo_name,
            "latency_ms": f"{latency:.6f}",
            "speedup_vs_pytorch_dense": _speedup(latency),
            "is_correct": ok,
            "max_diff": maxd,
            "mean_diff": meand,
        })

    # Preprocess-only results (output is not GEMM result; no correctness vs reference)
    for combo_name, (_, latency) in preprocess_only_results.items():
        rows.append({
            **base,
            "test_type": "preprocess_only",
            "category": "Preprocess-only",
            "operator": combo_name,
            "latency_ms": f"{latency:.6f}",
            "speedup_vs_pytorch_dense": _speedup(latency),
            "is_correct": "N/A",
            "max_diff": "N/A",
            "mean_diff": "N/A",
        })

    fieldnames = [
        "M", "K", "N", "threshold", "sparsity_pct",
        "test_type", "category", "operator",
        "latency_ms", "speedup_vs_pytorch_dense",
        "is_correct", "max_diff", "mean_diff",
    ]

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"\n[CSV] {len(rows)} rows written to: {csv_path}")


import numpy as np
import torch

def load_matrix(file_path: str, dtype=torch.float32) -> torch.Tensor:
    """Load matrix from a binary file and convert it to a PyTorch tensor."""
    # Load data as numpy array
    data = np.fromfile(file_path, dtype=np.float32)  # Load as float32, adjust dtype if necessary
    # Convert the numpy array to a torch tensor
    return torch.tensor(data, dtype=dtype)

def main() -> None:
    """Run all tests."""
    parser = argparse.ArgumentParser()

    args = parser.parse_args()


    print(f"M: {MV}")
    print(f"K: {KNV}")
    print(f"Threshold: {thresholds}")


    load_sve_sparse_gemm_extension(verbose=False)

    for M in MV:
        for K, N in KNV:
            activation = load_matrix(f"/dev/xvdb/data/activation_{M}_{K}.bin")
            weight = load_matrix(f"/dev/xvdb/data/weight_{K}_{N}.bin")
            activation = activation.reshape(M, K)
            weight = weight.reshape(K, N)
            for threshold in thresholds:
                activation_thresholded = _apply_threshold(activation, threshold=threshold)
                nnz = torch.count_nonzero(activation_thresholded).item()
                sparsity = 100.0 * (1.0 - nnz / (M * K))
                print(f"  - Sparsity: {sparsity:.1f}% ({nnz}/{M}x{K} non-zeros)")

                reference = torch.matmul(activation_thresholded, weight)

                all_results: Dict[str, Dict[str, Tuple[torch.Tensor, float]]] = {}

                # Combo/reference tests (participate in correctness + summary)
                _maybe_print_cpu_util("Start iCSR")
                all_results["iCSR"] = test_icsr_combinations(activation, weight, threshold)
                _maybe_print_cpu_util("End iCSR")
                
                _maybe_print_cpu_util("Start CSR")
                all_results["CSR"] = test_csr_combinations(activation, weight, threshold)
                _maybe_print_cpu_util("End CSR")

                _maybe_print_cpu_util("Start PyTorch reference")
                all_results["PyTorch"] = test_pytorch_references(activation, weight, threshold)
                _maybe_print_cpu_util("End PyTorch reference")


                # Verify correctness + performance summary
                verify_correctness(all_results, reference)
                print_performance_summary(all_results)


                _maybe_print_cpu_util("Start GEMM-only")
                gemm_only_results = test_core_gemm_only(
                    activation=activation,
                    activation_thresholded=activation_thresholded,
                    weight=weight,
                    threshold=threshold,
                )
                _maybe_print_cpu_util("End GEMM-only")
                verify_correctness_flat(gemm_only_results, reference)
                baseline_gemm_latency = gemm_only_results[
                    "GEMM-only: PyTorch torch.matmul(thresholded, weight)"
                ][1]
                _print_ranked_latencies(
                    title="GEMM-only latency ranking (baseline=PyTorch torch.matmul)",
                    latencies=[(k, v[1]) for k, v in gemm_only_results.items()],
                    baseline_latency=baseline_gemm_latency,
                )

        # Extra test 2: Preprocess-only
                _maybe_print_cpu_util("Start Preprocess-only")
                preprocess_only_results = test_preprocess_only(
                    activation=activation,
                    threshold=threshold,
                )
                _maybe_print_cpu_util("End Preprocess-only")
                # Use PyTorch threshold-only as baseline (minimal preprocessing)
                baseline_pre_latency = preprocess_only_results[
                    "Preprocess-only: PyTorch _apply_threshold(abs>=thr)"
                ][1]
                _print_ranked_latencies(
                    title="Preprocess-only latency ranking (baseline=PyTorch _apply_threshold)",
                    latencies=[(k, v[1]) for k, v in preprocess_only_results.items()],
                    baseline_latency=baseline_pre_latency,
                )

                # Export all results to CSV
                csv_name = f"results_{M}_{K}_{N}.csv"
                export_results_to_csv(
                    csv_path=csv_name,
                    M=M, K=K, N=N,
                    threshold=threshold,
                    sparsity_pct=sparsity,
                    reference=reference,
                    all_results=all_results,
                    gemm_only_results=gemm_only_results,
                    preprocess_only_results=preprocess_only_results,
                )


if __name__ == "__main__":
    main()
