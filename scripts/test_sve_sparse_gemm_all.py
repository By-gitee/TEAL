"""
Comprehensive test script for ARM SVE sparse GEMM operators.

This script tests combinations of sparsify and GEMM operators for different sparse formats (17 in total):

Custom SVE operators (13):
- iCSR: thr_sparsify_to_icsr(_sve) + sparse_gemm_icsr(_sve_gather)
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
import torch
import time
from typing import Any, Dict, List, Tuple

from kernels.cpp_sve_sparse_gemm import (
    # iCSR operators
    SparseGEMMiCSRSVEGatherKernel,
    SparseGEMMICSRKernel,
    thr_sparsify_to_icsr,
    thr_sparsify_to_icsr_sve,
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


def _make_random_sparse_activation(
    M: int,
    K: int,
    seed: int,
) -> torch.Tensor:
    """Generate random activation matrix (float32)."""
    g = torch.Generator()
    g.manual_seed(seed)
    x = torch.rand(M, K, dtype=torch.float32, generator=g)
    return x


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


def _parse_selected_tests(raw_tests: List[str] | None) -> List[str]:
    """Parse --tests argument; supports space/comma separation and common aliases.

    Allowed values:
    - all
    - icsr / csr / coo / csc
    - pytorch
    - gemm-only
    - preprocess-only
    """
    if not raw_tests:
        raw_tests = ["all"]

    tokens: List[str] = []
    for item in raw_tests:
        for part in item.split(","):
            part = part.strip()
            if part:
                tokens.append(part)

    normalized: List[str] = []
    for t in tokens:
        tl = t.strip().lower().replace("_", "-")
        if tl in {"all"}:
            normalized.append("all")
            continue
        if tl in {"icsr", "i-csr", "i_csr"}:
            normalized.append("icsr")
            continue
        if tl in {"csr"}:
            normalized.append("csr")
            continue
        if tl in {"coo"}:
            normalized.append("coo")
            continue
        if tl in {"csc"}:
            normalized.append("csc")
            continue
        if tl in {"pytorch", "torch"}:
            normalized.append("pytorch")
            continue
        if tl in {"gemm-only", "gemmonly", "gemm", "core-gemm", "core-gemm-only"}:
            normalized.append("gemm-only")
            continue
        if tl in {"preprocess-only", "preprocessonly", "preprocess", "pre"}:
            normalized.append("preprocess-only")
            continue

        allowed = "all/icsr/csr/coo/csc/pytorch/gemm-only/preprocess-only"
        raise ValueError(f"--tests contains unknown item: {t!r}, allowed: {allowed}")

    if "all" in normalized:
        return ["icsr", "csr", "coo", "csc", "pytorch", "gemm-only", "preprocess-only"]

    # Deduplicate while preserving user order
    seen = set()
    ordered_unique: List[str] = []
    for t in normalized:
        if t not in seen:
            seen.add(t)
            ordered_unique.append(t)
    return ordered_unique


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

    lat_torch = measure_latency(torch_dense_core, warmup=5, iters=100000)
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

    lat = measure_latency(icsr_gemm_gather_only, warmup=5, iters=100000)
    results["GEMM-only: iCSR sparse_gemm_icsr_sve_gather (cached indices)"] = (icsr_gemm_gather_only(), lat)
    print(f"    Latency: {lat:.4f} ms")

    print("  - sparse_gemm_icsr")
    def icsr_gemm_only():
        return icsr_op(activation, weight, row_offsets, nz_col_indices)

    lat = measure_latency(icsr_gemm_only, warmup=5, iters=100000)
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

    lat = measure_latency(csr_gemm_only, warmup=5, iters=100000)
    results["GEMM-only: CSR sparse_gemm_csr (cached values)"] = (csr_gemm_only(), lat)
    print(f"    Latency: {lat:.4f} ms")

    print("  - sparse_gemm_csr_sve_gather")
    def csr_gemm_gather_only():
        return csr_sve_gather_op(weight, csr_row_offsets, csr_nz_col_indices, csr_values)

    lat = measure_latency(csr_gemm_gather_only, warmup=5, iters=100000)
    results["GEMM-only: CSR sparse_gemm_csr_sve_gather (cached values)"] = (csr_gemm_gather_only(), lat)
    print(f"    Latency: {lat:.4f} ms")

    # COO
    print("\n[GEMM-only][COO] Pre thr_sparsify_to_coo")
    coo_row_indices, coo_col_indices, coo_values = thr_sparsify_to_coo(activation, threshold)
    M = int(activation.size(0))

    coo_kernel = SparseGEMMCOOKernel.initialize(name="sparse_gemm_coo", target="CPU")
    coo_op = coo_kernel.operator(compiled=True)

    coo_sve_gather_kernel = SparseGEMMCOOSVEGatherKernel.initialize(
        name="sparse_gemm_coo_sve_gather", target="CPU"
    )
    coo_sve_gather_op = coo_sve_gather_kernel.operator(compiled=True)

    # sparse_gemm_coo expects uint32 col_indices; thr_sparsify_to_coo returns uint32
    print("  - sparse_gemm_coo")
    def coo_gemm_only():
        return coo_op(weight, coo_row_indices, coo_col_indices, coo_values, M)

    lat = measure_latency(coo_gemm_only, warmup=5, iters=100000)
    results["GEMM-only: COO sparse_gemm_coo (cached triplets)"] = (coo_gemm_only(), lat)
    print(f"    Latency: {lat:.4f} ms")

    print("  - sparse_gemm_coo_sve_gather")
    def coo_gemm_gather_only():
        return coo_sve_gather_op(weight, coo_row_indices, coo_col_indices, coo_values, M)

    lat = measure_latency(coo_gemm_gather_only, warmup=5, iters=100000)
    results["GEMM-only: COO sparse_gemm_coo_sve_gather (cached triplets)"] = (coo_gemm_gather_only(), lat)
    print(f"    Latency: {lat:.4f} ms")

    # CSC
    print("\n[GEMM-only][CSC] Pre thr_sparsify_to_csc")
    csc_col_ptr, csc_row_indices, csc_values = thr_sparsify_to_csc(activation, threshold)

    csc_kernel = SparseGEMMCSCKernel.initialize(name="sparse_gemm_csc", target="CPU")
    csc_op = csc_kernel.operator(compiled=True)

    print("  - sparse_gemm_csc")
    def csc_gemm_only():
        return csc_op(weight, csc_col_ptr, csc_row_indices, csc_values, M, 0)

    lat = measure_latency(csc_gemm_only, warmup=5, iters=100000)
    results["GEMM-only: CSC sparse_gemm_csc (cached CSC)"] = (csc_gemm_only(), lat)
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

    lat = measure_latency(torch_threshold_only, warmup=5, iters=10)
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

    lat = measure_latency(torch_to_sparse_csr, warmup=5, iters=10)
    results["Preprocess-only: PyTorch threshold + to_sparse_csr()"] = (torch_to_sparse_csr(), lat)
    print(f"  Latency: {lat:.4f} ms")

    # Baseline 3: threshold + to_sparse_csc (end-to-end PyTorch CSC build)
    print("\n[Preprocess-only][PyTorch] threshold + to_sparse_csc")
    def torch_to_sparse_csc():
        x = _apply_threshold(activation, threshold=threshold)
        sp = x.to_sparse_csc()
        _ = sp.ccol_indices()
        _ = sp.row_indices()
        _ = sp.values()
        return sp

    lat = measure_latency(torch_to_sparse_csc, warmup=5, iters=10)
    results["Preprocess-only: PyTorch threshold + to_sparse_csc()"] = (torch_to_sparse_csc(), lat)
    print(f"  Latency: {lat:.4f} ms")

    # Custom: iCSR sparsify
    print("\n[Preprocess-only][iCSR] thr_sparsify_to_icsr / thr_sparsify_to_icsr_sve")
    def icsr_pre_1():
        return thr_sparsify_to_icsr(activation, threshold)

    lat = measure_latency(icsr_pre_1, warmup=5, iters=100000)
    results["Preprocess-only: iCSR thr_sparsify_to_icsr"] = (icsr_pre_1(), lat)
    print(f"  - thr_sparsify_to_icsr: Latency {lat:.4f} ms")

    def icsr_pre_2():
        return thr_sparsify_to_icsr_sve(activation, threshold)

    lat = measure_latency(icsr_pre_2, warmup=5, iters=100000)
    results["Preprocess-only: iCSR thr_sparsify_to_icsr_sve"] = (icsr_pre_2(), lat)
    print(f"  - thr_sparsify_to_icsr_sve: Latency {lat:.4f} ms")

    # Custom: CSR sparsify
    print("\n[Preprocess-only][CSR] thr_sparsify_to_csr / thr_sparsify_to_csr_sve")
    def csr_pre_1():
        return thr_sparsify_to_csr(activation, threshold)

    lat = measure_latency(csr_pre_1, warmup=5, iters=100000)
    results["Preprocess-only: CSR thr_sparsify_to_csr"] = (csr_pre_1(), lat)
    print(f"  - thr_sparsify_to_csr: Latency {lat:.4f} ms")

    def csr_pre_2():
        return thr_sparsify_to_csr_sve(activation, threshold)

    lat = measure_latency(csr_pre_2, warmup=5, iters=100000)
    results["Preprocess-only: CSR thr_sparsify_to_csr_sve"] = (csr_pre_2(), lat)
    print(f"  - thr_sparsify_to_csr_sve: Latency {lat:.4f} ms")

    # Custom: COO sparsify
    print("\n[Preprocess-only][COO] thr_sparsify_to_coo / thr_sparsify_to_coo_sve")
    def coo_pre_1():
        return thr_sparsify_to_coo(activation, threshold)

    lat = measure_latency(coo_pre_1, warmup=5, iters=100000)
    results["Preprocess-only: COO thr_sparsify_to_coo"] = (coo_pre_1(), lat)
    print(f"  - thr_sparsify_to_coo: Latency {lat:.4f} ms")

    def coo_pre_2():
        return thr_sparsify_to_coo_sve(activation, threshold)

    lat = measure_latency(coo_pre_2, warmup=5, iters=100000)
    results["Preprocess-only: COO thr_sparsify_to_coo_sve"] = (coo_pre_2(), lat)
    print(f"  - thr_sparsify_to_coo_sve: Latency {lat:.4f} ms")

    # Custom: CSC sparsify
    print("\n[Preprocess-only][CSC] thr_sparsify_to_csc")
    def csc_pre():
        return thr_sparsify_to_csc(activation, threshold)

    lat = measure_latency(csc_pre, warmup=5, iters=100000)
    results["Preprocess-only: CSC thr_sparsify_to_csc"] = (csc_pre(), lat)
    print(f"  - thr_sparsify_to_csc: Latency {lat:.4f} ms")

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
    
    lat1 = measure_latency(icsr_combo1, warmup=5, iters=100000)
    result1 = icsr_combo1()
    results["iCSR-1: thr_sparsify_to_icsr + sparse_gemm_icsr_sve_gather"] = (result1, lat1)
    print(f"  Latency: {lat1:.4f} ms")

    # Combo 2: thr_sparsify_to_icsr_sve + sparse_gemm_icsr_sve_gather
    print("\n[iCSR-2] thr_sparsify_to_icsr_sve + sparse_gemm_icsr_sve_gather")
    def icsr_combo2():
        nz_counts, nz_col_indices, row_offsets = thr_sparsify_to_icsr_sve(activation, threshold)
        return icsr_sve_gather_op(activation, weight, row_offsets, nz_col_indices)
    
    lat2 = measure_latency(icsr_combo2, warmup=5, iters=100000)
    result2 = icsr_combo2()
    results["iCSR-2: thr_sparsify_to_icsr_sve + sparse_gemm_icsr_sve_gather"] = (result2, lat2)
    print(f"  Latency: {lat2:.4f} ms")

    # Combo 3: thr_sparsify_to_icsr + sparse_gemm_icsr
    print("\n[iCSR-3] thr_sparsify_to_icsr + sparse_gemm_icsr")
    def icsr_combo3():
        nz_counts, nz_col_indices, row_offsets = thr_sparsify_to_icsr(activation, threshold)
        return icsr_op(activation, weight, row_offsets, nz_col_indices)
    
    lat3 = measure_latency(icsr_combo3, warmup=5, iters=100000)
    result3 = icsr_combo3()
    results["iCSR-3: thr_sparsify_to_icsr + sparse_gemm_icsr"] = (result3, lat3)
    print(f"  Latency: {lat3:.4f} ms")

    # Combo 4: thr_sparsify_to_icsr_sve + sparse_gemm_icsr
    print("\n[iCSR-4] thr_sparsify_to_icsr_sve + sparse_gemm_icsr")
    def icsr_combo4():
        nz_counts, nz_col_indices, row_offsets = thr_sparsify_to_icsr_sve(activation, threshold)
        return icsr_op(activation, weight, row_offsets, nz_col_indices)
    
    lat4 = measure_latency(icsr_combo4, warmup=5, iters=100000)
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
    
    lat1 = measure_latency(csr_combo1, warmup=5, iters=100000)
    result1 = csr_combo1()
    results["CSR-1: thr_sparsify_to_csr + sparse_gemm_csr"] = (result1, lat1)
    print(f"  Latency: {lat1:.4f} ms")

    # Combo 2: thr_sparsify_to_csr_sve + sparse_gemm_csr
    print("\n[CSR-2] thr_sparsify_to_csr_sve + sparse_gemm_csr")
    def csr_combo2():
        row_offsets, nz_col_indices, values = thr_sparsify_to_csr_sve(activation, threshold)
        return csr_op(weight, row_offsets, nz_col_indices, values)
    
    lat2 = measure_latency(csr_combo2, warmup=5, iters=100000)
    result2 = csr_combo2()
    results["CSR-2: thr_sparsify_to_csr_sve + sparse_gemm_csr"] = (result2, lat2)
    print(f"  Latency: {lat2:.4f} ms")

    # Combo 3: thr_sparsify_to_csr + sparse_gemm_csr_sve_gather
    print("\n[CSR-3] thr_sparsify_to_csr + sparse_gemm_csr_sve_gather")
    def csr_combo3():
        row_offsets, nz_col_indices, values = thr_sparsify_to_csr(activation, threshold)
        return csr_sve_gather_op(weight, row_offsets, nz_col_indices, values)
    
    lat3 = measure_latency(csr_combo3, warmup=5, iters=100000)
    result3 = csr_combo3()
    results["CSR-3: thr_sparsify_to_csr + sparse_gemm_csr_sve_gather"] = (result3, lat3)
    print(f"  Latency: {lat3:.4f} ms")

    # Combo 4: thr_sparsify_to_csr_sve + sparse_gemm_csr_sve_gather
    print("\n[CSR-4] thr_sparsify_to_csr_sve + sparse_gemm_csr_sve_gather")
    def csr_combo4():
        row_offsets, nz_col_indices, values = thr_sparsify_to_csr_sve(activation, threshold)
        return csr_sve_gather_op(weight, row_offsets, nz_col_indices, values)
    
    lat4 = measure_latency(csr_combo4, warmup=5, iters=100000)
    result4 = csr_combo4()
    results["CSR-4: thr_sparsify_to_csr_sve + sparse_gemm_csr_sve_gather"] = (result4, lat4)
    print(f"  Latency: {lat4:.4f} ms")

    return results


# =============================================================================
# COO format combination tests
# =============================================================================

def test_coo_combinations(
    activation: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
) -> Dict[str, Tuple[torch.Tensor, float]]:
    """
    Test all combinations of COO sparsify and GEMM operators.

    Combinations:
    - thr_sparsify_to_coo + sparse_gemm_coo
    - thr_sparsify_to_coo_sve + sparse_gemm_coo
    - thr_sparsify_to_coo + sparse_gemm_coo_sve_gather
    - thr_sparsify_to_coo_sve + sparse_gemm_coo_sve_gather

    Returns:
        Dict[combo_name, (result, latency)]
    """
    print("\n" + "=" * 80)
    print("Test COO format combinations")
    print("=" * 80)

    results = {}

    # Initialize GEMM operators
    coo_kernel = SparseGEMMCOOKernel.initialize(
        name="sparse_gemm_coo", target="CPU"
    )
    coo_op = coo_kernel.operator(compiled=True)
    
    coo_sve_gather_kernel = SparseGEMMCOOSVEGatherKernel.initialize(
        name="sparse_gemm_coo_sve_gather", target="CPU"
    )
    coo_sve_gather_op = coo_sve_gather_kernel.operator(compiled=True)
    
    # C++ op signature requires explicit M (sparse matrix row count)
    M = int(activation.size(0))

    # Combo 1: thr_sparsify_to_coo + sparse_gemm_coo
    print("\n[COO-1] thr_sparsify_to_coo + sparse_gemm_coo")
    def coo_combo1():
        row_indices, col_indices, values = thr_sparsify_to_coo(activation, threshold)
        # sparse_gemm_coo expects uint32 col_indices; thr_sparsify_to_coo returns uint32
        return coo_op(weight, row_indices, col_indices, values, M)

    lat1 = measure_latency(coo_combo1, warmup=5, iters=100000)
    result1 = coo_combo1()
    results["COO-1: thr_sparsify_to_coo + sparse_gemm_coo"] = (result1, lat1)
    print(f"  Latency: {lat1:.4f} ms")

    # Combo 2: thr_sparsify_to_coo_sve + sparse_gemm_coo
    print("\n[COO-2] thr_sparsify_to_coo_sve + sparse_gemm_coo")
    def coo_combo2():
        row_indices, col_indices, values = thr_sparsify_to_coo_sve(activation, threshold)
        # sparse_gemm_coo expects uint32 col_indices; thr_sparsify_to_coo_sve returns uint32
        return coo_op(weight, row_indices, col_indices, values, M)

    lat2 = measure_latency(coo_combo2, warmup=5, iters=100000)
    result2 = coo_combo2()
    results["COO-2: thr_sparsify_to_coo_sve + sparse_gemm_coo"] = (result2, lat2)
    print(f"  Latency: {lat2:.4f} ms")

    # Combo 3: thr_sparsify_to_coo + sparse_gemm_coo_sve_gather
    print("\n[COO-3] thr_sparsify_to_coo + sparse_gemm_coo_sve_gather")
    def coo_combo3():
        row_indices, col_indices, values = thr_sparsify_to_coo(activation, threshold)
        # sparse_gemm_coo_sve_gather expects uint32 col_indices; already uint32
        return coo_sve_gather_op(weight, row_indices, col_indices, values, M)

    lat3 = measure_latency(coo_combo3, warmup=5, iters=100000)
    result3 = coo_combo3()
    results["COO-3: thr_sparsify_to_coo + sparse_gemm_coo_sve_gather"] = (result3, lat3)
    print(f"  Latency: {lat3:.4f} ms")

    # Combo 4: thr_sparsify_to_coo_sve + sparse_gemm_coo_sve_gather
    print("\n[COO-4] thr_sparsify_to_coo_sve + sparse_gemm_coo_sve_gather")
    def coo_combo4():
        row_indices, col_indices, values = thr_sparsify_to_coo_sve(activation, threshold)
        # sparse_gemm_coo_sve_gather expects uint32 col_indices; already uint32
        return coo_sve_gather_op(weight, row_indices, col_indices, values, M)

    lat4 = measure_latency(coo_combo4, warmup=5, iters=100000)
    result4 = coo_combo4()
    results["COO-4: thr_sparsify_to_coo_sve + sparse_gemm_coo_sve_gather"] = (result4, lat4)
    print(f"  Latency: {lat4:.4f} ms")

    return results


# =============================================================================
# CSC format combination tests
# =============================================================================

def test_csc_combinations(
    activation: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
) -> Dict[str, Tuple[torch.Tensor, float]]:
    """
    Test all combinations of CSC sparsify and GEMM operators.

    Combination:
    - thr_sparsify_to_csc + sparse_gemm_csc

    Returns:
        Dict[combo_name, (result, latency)]
    """
    print("\n" + "=" * 80)
    print("Test CSC format combinations")
    print("=" * 80)

    results = {}

    # Initialize GEMM operator
    csc_kernel = SparseGEMMCSCKernel.initialize(
        name="sparse_gemm_csc", target="CPU"
    )
    csc_op = csc_kernel.operator(compiled=True)
    
    M = activation.size(0)

    # Combo 1: thr_sparsify_to_csc + sparse_gemm_csc
    print("\n[CSC-1] thr_sparsify_to_csc + sparse_gemm_csc")
    def csc_combo1():
        col_ptr, row_indices, values = thr_sparsify_to_csc(activation, threshold)
        return csc_op(weight, col_ptr, row_indices, values, M, 0)
    
    lat1 = measure_latency(csc_combo1, warmup=5, iters=100000)
    result1 = csc_combo1()
    results["CSC-1: thr_sparsify_to_csc + sparse_gemm_csc"] = (result1, lat1)
    print(f"  Latency: {lat1:.4f} ms")

    return results


# =============================================================================
# PyTorch reference implementation tests
# =============================================================================

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
    
    lat1 = measure_latency(pytorch_dense_fn, warmup=5, iters=100000)
    result1 = pytorch_dense_fn()
    results["PyTorch-1: Dense matmul"] = (result1, lat1)
    print(f"  Latency: {lat1:.4f} ms")

    # PyTorch sparse CSR + sparse.mm
    print("\n[PyTorch-2] Sparse CSR + sparse.mm")
    def pytorch_sparse_csr_fn():
        activation_thresholded = _apply_threshold(activation, threshold=threshold)
        sp_act = activation_thresholded.to_sparse_csr()
        return torch.sparse.mm(sp_act, weight)
    
    lat2 = measure_latency(pytorch_sparse_csr_fn, warmup=5, iters=100000)
    result2 = pytorch_sparse_csr_fn()
    results["PyTorch-2: Sparse CSR + sparse.mm"] = (result2, lat2)
    print(f"  Latency: {lat2:.4f} ms")

    # PyTorch sparse CSC + sparse.mm
    print("\n[PyTorch-3] Sparse CSC + sparse.mm")
    def pytorch_sparse_csc_fn():
        activation_thresholded = _apply_threshold(activation, threshold=threshold)
        sp_act = activation_thresholded.to_sparse_csc()
        return torch.sparse.mm(sp_act, weight)
    
    lat3 = measure_latency(pytorch_sparse_csc_fn, warmup=5, iters=100000)
    result3 = pytorch_sparse_csc_fn()
    results["PyTorch-3: Sparse CSC + sparse.mm"] = (result3, lat3)
    print(f"  Latency: {lat3:.4f} ms")

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

    lat4 = measure_latency(pytorch_selective_weight_fn, warmup=5, iters=100000)
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

        print("=" * 80)


def main() -> None:
    """Run all tests."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--threshold", type=float, default=0.9, help="Sparsification threshold")
    parser.add_argument("--M", type=int, default=1, help="activation rows (M)")
    parser.add_argument("--K", type=int, default=64, help="activation cols / weight rows (K)")
    parser.add_argument("--N", type=int, default=64, help="weight cols (N)")
    parser.add_argument(
        "--tests",
        nargs="+",
        default=["all"],
        help=(
            "Tests to run (space or comma separated): "
            "all/icsr/csr/coo/csc/pytorch/gemm-only/preprocess-only"
        ),
    )
    args = parser.parse_args()

    selected_tests = _parse_selected_tests(args.tests)

    print("=" * 80)
    print("ARM SVE Sparse GEMM Operator Comprehensive Test")
    print("=" * 80)
    print("Config:")
    print(f"  - Seed: {args.seed}")
    print(f"  - Threshold: {args.threshold}")
    print(f"  - Matrix shape: activation ({args.M}, {args.K}), weight ({args.K}, {args.N})")
    print(f"  - Tests: {', '.join(selected_tests)}")

    try:
        if psutil is None:
            print("  - CPU utilization: psutil not available, skipping (optional: pip install psutil)")
        else:
            _maybe_print_cpu_util("Before start")

        # Load extension
        load_sve_sparse_gemm_extension(verbose=False)

        # Generate shared test data
        activation = _make_random_sparse_activation(args.M, args.K, seed=args.seed)
        activation_thresholded = _apply_threshold(activation, threshold=args.threshold)

        # Compute sparsity
        nnz = torch.count_nonzero(activation_thresholded).item()
        sparsity = 100.0 * (1.0 - nnz / (args.M * args.K))
        print(f"  - Sparsity: {sparsity:.1f}% ({nnz}/{args.M * args.K} non-zeros)")

        needs_weight = any(
            t in {"icsr", "csr", "coo", "csc", "pytorch", "gemm-only"} for t in selected_tests
        )
        needs_reference = any(
            t in {"icsr", "csr", "coo", "csc", "pytorch", "gemm-only"} for t in selected_tests
        )

        weight = None
        reference = None
        if needs_weight:
            weight = torch.randn(args.K, args.N, dtype=torch.float32)
        if needs_reference:
            if weight is None:
                raise RuntimeError("Internal error: needs_reference=True but weight not created")
            reference = torch.matmul(activation_thresholded, weight)

        # Combo/reference tests (participate in correctness + summary)
        all_results: Dict[str, Dict[str, Tuple[torch.Tensor, float]]] = {}
        if any(t in {"icsr", "csr", "coo", "csc", "pytorch"} for t in selected_tests):
            if weight is None or reference is None:
                raise RuntimeError("Internal error: weight/reference required but not created")

            if "icsr" in selected_tests:
                _maybe_print_cpu_util("Start iCSR")
                all_results["iCSR"] = test_icsr_combinations(activation, weight, args.threshold)
                _maybe_print_cpu_util("End iCSR")
            if "csr" in selected_tests:
                _maybe_print_cpu_util("Start CSR")
                all_results["CSR"] = test_csr_combinations(activation, weight, args.threshold)
                _maybe_print_cpu_util("End CSR")
            if "coo" in selected_tests:
                _maybe_print_cpu_util("Start COO")
                all_results["COO"] = test_coo_combinations(activation, weight, args.threshold)
                _maybe_print_cpu_util("End COO")
            if "csc" in selected_tests:
                _maybe_print_cpu_util("Start CSC")
                all_results["CSC"] = test_csc_combinations(activation, weight, args.threshold)
                _maybe_print_cpu_util("End CSC")
            if "pytorch" in selected_tests:
                _maybe_print_cpu_util("Start PyTorch reference")
                all_results["PyTorch"] = test_pytorch_references(activation, weight, args.threshold)
                _maybe_print_cpu_util("End PyTorch reference")

            # Verify correctness + performance summary
            verify_correctness(all_results, reference)
            print_performance_summary(all_results)

        # Extra test 1: GEMM-only
        if "gemm-only" in selected_tests:
            if weight is None or reference is None:
                raise RuntimeError("GEMM-only test requires weight/reference but none created (check --tests)")

            _maybe_print_cpu_util("Start GEMM-only")
            gemm_only_results = test_core_gemm_only(
                activation=activation,
                activation_thresholded=activation_thresholded,
                weight=weight,
                threshold=args.threshold,
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
        if "preprocess-only" in selected_tests:
            _maybe_print_cpu_util("Start Preprocess-only")
            preprocess_only_results = test_preprocess_only(
                activation=activation,
                threshold=args.threshold,
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

        print("\n" + "=" * 80)
        print("✅ All tests completed")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
