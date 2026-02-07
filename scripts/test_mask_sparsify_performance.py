"""
Performance test and comparison script for mask_sparsify operators.

This script benchmarks 7 mask-based sparsify operators:
1. mask_sparsify_to_coo - COO format (scalar version)
2. mask_sparsify_to_coo_sve - COO format (SVE-accelerated)
3. mask_sparsify_to_csc - CSC format (scalar version)
4. mask_sparsify_to_csr - CSR format (scalar version)
5. mask_sparsify_to_csr_sve - CSR format (SVE-accelerated)
6. mask_sparsify_to_icsr - iCSR format (scalar version)
7. mask_sparsify_to_icsr_sve - iCSR format (SVE-accelerated)

Test contents:
1. Correctness: ensure all operators produce consistent sparse data
2. Performance: measure latency of each operator
3. Speedup: SVE vs scalar speedup
4. Multi-config: different matrix sizes and sparsity

Usage:
    python -m scripts.test_mask_sparsify_performance
    python -m scripts.test_mask_sparsify_performance --M 16 --K 4096 --sparsity 0.9
    python -m scripts.test_mask_sparsify_performance --test-sizes
"""

from __future__ import annotations

import argparse
import torch
import time
from typing import Any, Dict, List, Tuple

from kernels.cpp_sve_sparse_gemm import (
    mask_sparsify_to_coo,
    mask_sparsify_to_coo_sve,
    mask_sparsify_to_csc,
    mask_sparsify_to_csr,
    mask_sparsify_to_csr_sve,
    mask_sparsify_to_icsr,
    mask_sparsify_to_icsr_sve,
    load_sve_sparse_gemm_extension,
)
from kernels.kernel_utils import measure_latency

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore


def _make_random_mask(
    M: int,
    K: int,
    sparsity: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random mask matrix and corresponding activation matrix.

    Args:
        M: Number of rows.
        K: Number of columns.
        sparsity: Sparsity (0.0-1.0); 0 = dense, 1 = fully sparse.
        seed: Random seed.

    Returns:
        tuple: (activation, mask)
            - activation: (M, K) float32 matrix.
            - mask: (M, K) uint8 matrix; non-zero marks kept positions.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    
    # Random activation
    activation = torch.rand(M, K, dtype=torch.float32, generator=g) * 2.0 - 1.0

    # Random mask (by sparsity)
    mask_prob = torch.rand(M, K, dtype=torch.float32, generator=g)
    mask = (mask_prob >= sparsity).to(torch.uint8)
    
    return activation, mask


def _count_nnz_from_mask(mask: torch.Tensor) -> int:
    """Count number of non-zero elements in mask."""
    return torch.count_nonzero(mask).item()


def _verify_coo_format(
    row_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    activation: torch.Tensor,
    mask: torch.Tensor,
    name: str,
) -> bool:
    """Verify correctness of COO format output."""
    M, K = activation.shape
    nnz = row_indices.size(0)

    # Check length consistency
    if col_indices.size(0) != nnz or values.size(0) != nnz:
        print(f"  ❌ {name}: length mismatch")
        return False

    # Check index range (cast uint32 to int64 for comparison)
    row_idx_i64 = row_indices.to(torch.int64)
    col_idx_i64 = col_indices.to(torch.int64)
    if torch.any(row_idx_i64 < 0) or torch.any(row_idx_i64 >= M):
        print(f"  ❌ {name}: row_indices out of range")
        return False
    if torch.any(col_idx_i64 < 0) or torch.any(col_idx_i64 >= K):
        print(f"  ❌ {name}: col_indices out of range")
        return False

    # Check value correctness
    for i in range(nnz):
        r = row_indices[i].item()
        c = col_indices[i].item()
        v = values[i].item()
        expected = activation[r, c].item()
        if mask[r, c].item() == 0:
            print(f"  ❌ {name}: ({r}, {c}) is 0 in mask but present in COO")
            return False
        if abs(v - expected) > 1e-5:
            print(f"  ❌ {name}: ({r}, {c}) value mismatch: {v} vs {expected}")
            return False
    
    return True


def _verify_csr_format(
    row_offsets: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    activation: torch.Tensor,
    mask: torch.Tensor,
    name: str,
) -> bool:
    """Verify correctness of CSR format output."""
    M, K = activation.shape

    # Check row_offsets length
    if row_offsets.size(0) != M + 1:
        print(f"  ❌ {name}: row_offsets length error")
        return False

    total_nnz = row_offsets[M].item()

    # Check length consistency
    if col_indices.size(0) != total_nnz or values.size(0) != total_nnz:
        print(f"  ❌ {name}: data length mismatch")
        return False

    # Verify each row
    for m in range(M):
        start = row_offsets[m].item()
        end = row_offsets[m + 1].item()

        for idx in range(start, end):
            c = col_indices[idx].item()
            v = values[idx].item()

            if c < 0 or c >= K:
                print(f"  ❌ {name}: row {m} col index out of range")
                return False

            if mask[m, c].item() == 0:
                print(f"  ❌ {name}: ({m}, {c}) is 0 in mask but present in CSR")
                return False

            expected = activation[m, c].item()
            if abs(v - expected) > 1e-5:
                print(f"  ❌ {name}: ({m}, {c}) value mismatch: {v} vs {expected}")
                return False
    
    return True


def _verify_csc_format(
    col_ptr: torch.Tensor,
    row_indices: torch.Tensor,
    values: torch.Tensor,
    activation: torch.Tensor,
    mask: torch.Tensor,
    name: str,
) -> bool:
    """Verify correctness of CSC format output."""
    M, K = activation.shape

    # Check col_ptr length
    if col_ptr.size(0) != K + 1:
        print(f"  ❌ {name}: col_ptr length error")
        return False

    total_nnz = col_ptr[K].item()

    # Check length consistency
    if row_indices.size(0) != total_nnz or values.size(0) != total_nnz:
        print(f"  ❌ {name}: data length mismatch")
        return False

    # Verify each column
    for k in range(K):
        start = col_ptr[k].item()
        end = col_ptr[k + 1].item()

        for idx in range(start, end):
            r = row_indices[idx].item()
            v = values[idx].item()

            if r < 0 or r >= M:
                print(f"  ❌ {name}: col {k} row index out of range")
                return False

            if mask[r, k].item() == 0:
                print(f"  ❌ {name}: ({r}, {k}) is 0 in mask but present in CSC")
                return False

            expected = activation[r, k].item()
            if abs(v - expected) > 1e-5:
                print(f"  ❌ {name}: ({r}, {k}) value mismatch: {v} vs {expected}")
                return False
    
    return True


def _verify_icsr_format(
    nz_counts: torch.Tensor,
    col_indices: torch.Tensor,
    row_offsets: torch.Tensor,
    mask: torch.Tensor,
    name: str,
) -> bool:
    """Verify correctness of iCSR format output."""
    M, K = mask.shape

    # Check row_offsets length
    if row_offsets.size(0) != M + 1:
        print(f"  ❌ {name}: row_offsets length error")
        return False

    total_nnz = row_offsets[M].item()

    # Check col_indices length
    if col_indices.size(0) != total_nnz:
        print(f"  ❌ {name}: col_indices length mismatch")
        return False

    # Check nz_counts format
    if nz_counts.size(0) % 2 != 0:
        print(f"  ❌ {name}: nz_counts length must be even")
        return False

    # Verify each row
    for m in range(M):
        start = row_offsets[m].item()
        end = row_offsets[m + 1].item()

        for idx in range(start, end):
            c = col_indices[idx].item()

            if c < 0 or c >= K:
                print(f"  ❌ {name}: row {m} col index out of range")
                return False

            if mask[m, c].item() == 0:
                print(f"  ❌ {name}: ({m}, {c}) is 0 in mask but present in iCSR")
                return False
    
    return True


def test_correctness(
    activation: torch.Tensor,
    mask: torch.Tensor,
) -> bool:
    """Test correctness of all operators."""
    print("\n" + "=" * 80)
    print("Correctness verification")
    print("=" * 80)

    all_passed = True

    # COO format
    print("\n[COO format]")
    try:
        row_idx_coo, col_idx_coo, val_coo = mask_sparsify_to_coo(activation, mask)
        if _verify_coo_format(row_idx_coo, col_idx_coo, val_coo, activation, mask, "mask_sparsify_to_coo"):
            print("  ✅ mask_sparsify_to_coo")
        else:
            all_passed = False
    except Exception as e:
        print(f"  ❌ mask_sparsify_to_coo: {e}")
        all_passed = False
    
    try:
        row_idx_coo_sve, col_idx_coo_sve, val_coo_sve = mask_sparsify_to_coo_sve(activation, mask)
        if _verify_coo_format(row_idx_coo_sve, col_idx_coo_sve, val_coo_sve, activation, mask, "mask_sparsify_to_coo_sve"):
            print("  ✅ mask_sparsify_to_coo_sve")
        else:
            all_passed = False
    except Exception as e:
        print(f"  ❌ mask_sparsify_to_coo_sve: {e}")
        all_passed = False

    # CSR format
    print("\n[CSR format]")
    try:
        row_off_csr, col_idx_csr, val_csr = mask_sparsify_to_csr(activation, mask)
        if _verify_csr_format(row_off_csr, col_idx_csr, val_csr, activation, mask, "mask_sparsify_to_csr"):
            print("  ✅ mask_sparsify_to_csr")
        else:
            all_passed = False
    except Exception as e:
        print(f"  ❌ mask_sparsify_to_csr: {e}")
        all_passed = False
    
    try:
        row_off_csr_sve, col_idx_csr_sve, val_csr_sve = mask_sparsify_to_csr_sve(activation, mask)
        if _verify_csr_format(row_off_csr_sve, col_idx_csr_sve, val_csr_sve, activation, mask, "mask_sparsify_to_csr_sve"):
            print("  ✅ mask_sparsify_to_csr_sve")
        else:
            all_passed = False
    except Exception as e:
        print(f"  ❌ mask_sparsify_to_csr_sve: {e}")
        all_passed = False

    # CSC format
    print("\n[CSC format]")
    try:
        col_ptr_csc, row_idx_csc, val_csc = mask_sparsify_to_csc(activation, mask)
        if _verify_csc_format(col_ptr_csc, row_idx_csc, val_csc, activation, mask, "mask_sparsify_to_csc"):
            print("  ✅ mask_sparsify_to_csc")
        else:
            all_passed = False
    except Exception as e:
        print(f"  ❌ mask_sparsify_to_csc: {e}")
        all_passed = False

    # iCSR format
    print("\n[iCSR format]")
    try:
        nz_counts_icsr, col_idx_icsr, row_off_icsr = mask_sparsify_to_icsr(mask)
        if _verify_icsr_format(nz_counts_icsr, col_idx_icsr, row_off_icsr, mask, "mask_sparsify_to_icsr"):
            print("  ✅ mask_sparsify_to_icsr")
        else:
            all_passed = False
    except Exception as e:
        print(f"  ❌ mask_sparsify_to_icsr: {e}")
        all_passed = False
    
    try:
        nz_counts_icsr_sve, col_idx_icsr_sve, row_off_icsr_sve = mask_sparsify_to_icsr_sve(mask)
        if _verify_icsr_format(nz_counts_icsr_sve, col_idx_icsr_sve, row_off_icsr_sve, mask, "mask_sparsify_to_icsr_sve"):
            print("  ✅ mask_sparsify_to_icsr_sve")
        else:
            all_passed = False
    except Exception as e:
        print(f"  ❌ mask_sparsify_to_icsr_sve: {e}")
        all_passed = False
    
    if all_passed:
        print("\n✅ All operators passed correctness test")
    else:
        print("\n❌ Some operators failed correctness test")

    return all_passed


def test_performance(
    activation: torch.Tensor,
    mask: torch.Tensor,
    warmup: int = 5,
    iters: int = 100000,
) -> Dict[str, float]:
    """Benchmark all operators."""
    print("\n" + "=" * 80)
    print("Performance test")
    print("=" * 80)

    results: Dict[str, float] = {}

    # COO format
    print("\n[COO format]")
    print("  Testing mask_sparsify_to_coo...")
    lat = measure_latency(lambda: mask_sparsify_to_coo(activation, mask), warmup=warmup, iters=iters)
    results["mask_sparsify_to_coo"] = lat
    print(f"    Latency: {lat:.4f} ms")

    print("  Testing mask_sparsify_to_coo_sve...")
    lat = measure_latency(lambda: mask_sparsify_to_coo_sve(activation, mask), warmup=warmup, iters=iters)
    results["mask_sparsify_to_coo_sve"] = lat
    print(f"    Latency: {lat:.4f} ms")

    # CSR format
    print("\n[CSR format]")
    print("  Testing mask_sparsify_to_csr...")
    lat = measure_latency(lambda: mask_sparsify_to_csr(activation, mask), warmup=warmup, iters=iters)
    results["mask_sparsify_to_csr"] = lat
    print(f"    Latency: {lat:.4f} ms")

    print("  Testing mask_sparsify_to_csr_sve...")
    lat = measure_latency(lambda: mask_sparsify_to_csr_sve(activation, mask), warmup=warmup, iters=iters)
    results["mask_sparsify_to_csr_sve"] = lat
    print(f"    Latency: {lat:.4f} ms")

    # CSC format
    print("\n[CSC format]")
    print("  Testing mask_sparsify_to_csc...")
    lat = measure_latency(lambda: mask_sparsify_to_csc(activation, mask), warmup=warmup, iters=iters)
    results["mask_sparsify_to_csc"] = lat
    print(f"    Latency: {lat:.4f} ms")

    # iCSR format
    print("\n[iCSR format]")
    print("  Testing mask_sparsify_to_icsr...")
    lat = measure_latency(lambda: mask_sparsify_to_icsr(mask), warmup=warmup, iters=iters)
    results["mask_sparsify_to_icsr"] = lat
    print(f"    Latency: {lat:.4f} ms")

    print("  Testing mask_sparsify_to_icsr_sve...")
    lat = measure_latency(lambda: mask_sparsify_to_icsr_sve(mask), warmup=warmup, iters=iters)
    results["mask_sparsify_to_icsr_sve"] = lat
    print(f"    Latency: {lat:.4f} ms")
    
    return results


def print_performance_summary(results: Dict[str, float]) -> None:
    """Print performance comparison summary."""
    print("\n" + "=" * 80)
    print("Performance summary")
    print("=" * 80)

    # Sort by latency
    sorted_results = sorted(results.items(), key=lambda x: x[1])

    print("\nLatency ranking (fastest to slowest):")
    print("-" * 80)
    print(f"{'Rank':<4} {'Operator':<40} {'Latency(ms)':<12} {'Type':<10}")
    print("-" * 80)

    for rank, (name, latency) in enumerate(sorted_results, 1):
        marker = "⚡ SVE" if "sve" in name else "📊 Scalar"
        print(f"{rank:2d}. {name:40s} {latency:8.4f} ms  {marker}")

    # Fastest operator
    fastest_name, fastest_latency = sorted_results[0]
    print("\n" + "-" * 80)
    print(f"⚡ Fastest: {fastest_name}")
    print(f"   Latency: {fastest_latency:.4f} ms")

    # SVE speedup
    print("\n" + "=" * 80)
    print("SVE speedup analysis")
    print("=" * 80)

    comparisons = [
        ("COO", "mask_sparsify_to_coo", "mask_sparsify_to_coo_sve"),
        ("CSR", "mask_sparsify_to_csr", "mask_sparsify_to_csr_sve"),
        ("iCSR", "mask_sparsify_to_icsr", "mask_sparsify_to_icsr_sve"),
    ]

    for format_name, scalar_name, sve_name in comparisons:
        if scalar_name in results and sve_name in results:
            scalar_lat = results[scalar_name]
            sve_lat = results[sve_name]
            speedup = scalar_lat / sve_lat if sve_lat > 0 else 0.0
            print(f"\n{format_name} format:")
            print(f"  Scalar: {scalar_lat:.4f} ms")
            print(f"  SVE:    {sve_lat:.4f} ms")
            print(f"  Speedup: {speedup:.2f}x")


def test_multiple_sizes(
    sparsity: float = 0.9,
    seed: int = 42,
    warmup: int = 5,
    iters: int = 50000,
) -> None:
    """Benchmark multiple matrix sizes."""
    print("\n" + "=" * 80)
    print("Multi-size performance test")
    print("=" * 80)

    # Test configs: (M, K)
    test_configs = [
        (1, 2048),
        (1, 4096),
        (1, 8192),
        (8, 4096),
        (16, 4096),
        (32, 4096),
    ]
    
    all_results: List[Tuple[Tuple[int, int], Dict[str, float]]] = []
    
    for M, K in test_configs:
        print(f"\n{'='*80}")
        print(f"Config: M={M}, K={K}, sparsity={sparsity}")
        print(f"{'='*80}")

        # Generate test data
        activation, mask = _make_random_mask(M, K, sparsity, seed)
        nnz = _count_nnz_from_mask(mask)
        actual_sparsity = 1.0 - (nnz / (M * K))
        print(f"Actual sparsity: {actual_sparsity*100:.1f}% ({nnz}/{M*K} non-zeros)")

        # Performance test
        results = test_performance(activation, mask, warmup=warmup, iters=iters)
        all_results.append(((M, K), results))

    # Summary table
    print("\n" + "=" * 80)
    print("Multi-size latency summary (ms)")
    print("=" * 80)

    # Table header
    algo_names = ["COO", "COO_SVE", "CSR", "CSR_SVE", "CSC", "iCSR", "iCSR_SVE"]
    print(f"\n{'Config':<12}", end="")
    for name in algo_names:
        print(f"{name:>10}", end="")
    print()
    print("-" * (12 + 10 * len(algo_names)))
    
    # Data rows
    for (M, K), results in all_results:
        print(f"({M:2d},{K:5d})", end="  ")
        
        lat_coo = results.get("mask_sparsify_to_coo", 0.0)
        lat_coo_sve = results.get("mask_sparsify_to_coo_sve", 0.0)
        lat_csr = results.get("mask_sparsify_to_csr", 0.0)
        lat_csr_sve = results.get("mask_sparsify_to_csr_sve", 0.0)
        lat_csc = results.get("mask_sparsify_to_csc", 0.0)
        lat_icsr = results.get("mask_sparsify_to_icsr", 0.0)
        lat_icsr_sve = results.get("mask_sparsify_to_icsr_sve", 0.0)
        
        print(f"{lat_coo:10.4f}{lat_coo_sve:10.4f}{lat_csr:10.4f}{lat_csr_sve:10.4f}"
              f"{lat_csc:10.4f}{lat_icsr:10.4f}{lat_icsr_sve:10.4f}")
    
    # Speedup table
    print("\n" + "=" * 80)
    print("SVE speedup summary")
    print("=" * 80)

    print(f"\n{'Config':<12}{'COO':>10}{'CSR':>10}{'iCSR':>10}")
    print("-" * 42)
    
    for (M, K), results in all_results:
        print(f"({M:2d},{K:5d})", end="  ")
        
        # COO speedup
        lat_coo = results.get("mask_sparsify_to_coo", 0.0)
        lat_coo_sve = results.get("mask_sparsify_to_coo_sve", 0.0)
        speedup_coo = lat_coo / lat_coo_sve if lat_coo_sve > 0 else 0.0

        # CSR speedup
        lat_csr = results.get("mask_sparsify_to_csr", 0.0)
        lat_csr_sve = results.get("mask_sparsify_to_csr_sve", 0.0)
        speedup_csr = lat_csr / lat_csr_sve if lat_csr_sve > 0 else 0.0

        # iCSR speedup
        lat_icsr = results.get("mask_sparsify_to_icsr", 0.0)
        lat_icsr_sve = results.get("mask_sparsify_to_icsr_sve", 0.0)
        speedup_icsr = lat_icsr / lat_icsr_sve if lat_icsr_sve > 0 else 0.0
        
        print(f"{speedup_coo:10.2f}x{speedup_csr:10.2f}x{speedup_icsr:10.2f}x")


def main() -> None:
    """Run tests."""
    parser = argparse.ArgumentParser(description="mask_sparsify operator performance test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--M", type=int, default=1, help="Matrix rows")
    parser.add_argument("--K", type=int, default=4096, help="Matrix columns")
    parser.add_argument("--sparsity", type=float, default=0.9, help="Sparsity (0.0-1.0)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=10000, help="Benchmark iterations")
    parser.add_argument("--test-sizes", action="store_true", help="Test multiple matrix sizes")
    parser.add_argument("--skip-correctness", action="store_true", help="Skip correctness test")
    args = parser.parse_args()

    print("=" * 80)
    print("mask_sparsify operator performance test")
    print("=" * 80)
    print("Config:")
    print(f"  - Seed: {args.seed}")
    if not args.test_sizes:
        print(f"  - Matrix size: ({args.M}, {args.K})")
        print(f"  - Sparsity: {args.sparsity}")
    print(f"  - Warmup: {args.warmup}")
    print(f"  - Iters: {args.iters}")

    try:
        # Load extension
        print("\nLoading C++ extension...")
        load_sve_sparse_gemm_extension(verbose=False)
        print("✅ C++ extension loaded")

        if args.test_sizes:
            # Multi-size test
            test_multiple_sizes(
                sparsity=args.sparsity,
                seed=args.seed,
                warmup=args.warmup,
                iters=args.iters,
            )
        else:
            # Single config test
            print(f"\nGenerating test data ({args.M}, {args.K})...")
            activation, mask = _make_random_mask(args.M, args.K, args.sparsity, args.seed)

            # Actual sparsity
            nnz = _count_nnz_from_mask(mask)
            actual_sparsity = 1.0 - (nnz / (args.M * args.K))
            print(f"Actual sparsity: {actual_sparsity*100:.1f}% ({nnz}/{args.M * args.K} non-zeros)")

            # Correctness test
            if not args.skip_correctness:
                if not test_correctness(activation, mask):
                    print("\n⚠️  Correctness test failed; continuing with performance test")

            # Performance test
            results = test_performance(activation, mask, warmup=args.warmup, iters=args.iters)

            # Print summary
            print_performance_summary(results)

        print("\n" + "=" * 80)
        print("✅ All tests completed")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
