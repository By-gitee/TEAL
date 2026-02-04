"""
mask_sparsify 算子性能测试与对比脚本。

本脚本测试7个基于mask的稀疏化算子的性能：
1. mask_sparsify_to_coo - COO格式（标量版本）
2. mask_sparsify_to_coo_sve - COO格式（SVE加速）
3. mask_sparsify_to_csc - CSC格式（标量版本）
4. mask_sparsify_to_csr - CSR格式（标量版本）
5. mask_sparsify_to_csr_sve - CSR格式（SVE加速）
6. mask_sparsify_to_icsr - iCSR格式（标量版本）
7. mask_sparsify_to_icsr_sve - iCSR格式（SVE加速）

测试内容：
1. 正确性验证：确保所有算子输出的稀疏数据一致
2. 性能测试：测量每个算子的延迟
3. 加速比计算：SVE版本相对标量版本的加速比
4. 多种配置测试：不同矩阵尺寸和稀疏度

运行方式:
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
    """生成随机 mask 矩阵和对应的 activation 矩阵。
    
    Args:
        M: 行数
        K: 列数
        sparsity: 稀疏度 (0.0-1.0)，0表示全密集，1表示全稀疏
        seed: 随机种子
        
    Returns:
        tuple: (activation, mask)
            - activation: (M, K) float32 矩阵
            - mask: (M, K) uint8 矩阵，非零元素标记保留位置
    """
    g = torch.Generator()
    g.manual_seed(seed)
    
    # 生成随机 activation
    activation = torch.rand(M, K, dtype=torch.float32, generator=g) * 2.0 - 1.0
    
    # 生成随机 mask（基于稀疏度）
    mask_prob = torch.rand(M, K, dtype=torch.float32, generator=g)
    mask = (mask_prob >= sparsity).to(torch.uint8)
    
    return activation, mask


def _count_nnz_from_mask(mask: torch.Tensor) -> int:
    """统计 mask 中的非零元素数量。"""
    return torch.count_nonzero(mask).item()


def _verify_coo_format(
    row_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    activation: torch.Tensor,
    mask: torch.Tensor,
    name: str,
) -> bool:
    """验证 COO 格式输出的正确性。"""
    M, K = activation.shape
    nnz = row_indices.size(0)
    
    # 检查长度一致性
    if col_indices.size(0) != nnz or values.size(0) != nnz:
        print(f"  ❌ {name}: 长度不一致")
        return False
    
    # 检查索引范围（将 uint32 转为 int64 以支持比较操作）
    row_idx_i64 = row_indices.to(torch.int64)
    col_idx_i64 = col_indices.to(torch.int64)
    if torch.any(row_idx_i64 < 0) or torch.any(row_idx_i64 >= M):
        print(f"  ❌ {name}: row_indices 超出范围")
        return False
    if torch.any(col_idx_i64 < 0) or torch.any(col_idx_i64 >= K):
        print(f"  ❌ {name}: col_indices 超出范围")
        return False
    
    # 检查值的正确性
    for i in range(nnz):
        r = row_indices[i].item()
        c = col_indices[i].item()
        v = values[i].item()
        expected = activation[r, c].item()
        if mask[r, c].item() == 0:
            print(f"  ❌ {name}: ({r}, {c}) 在mask中为0但在COO中出现")
            return False
        if abs(v - expected) > 1e-5:
            print(f"  ❌ {name}: ({r}, {c}) 值不匹配: {v} vs {expected}")
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
    """验证 CSR 格式输出的正确性。"""
    M, K = activation.shape
    
    # 检查 row_offsets 长度
    if row_offsets.size(0) != M + 1:
        print(f"  ❌ {name}: row_offsets 长度错误")
        return False
    
    total_nnz = row_offsets[M].item()
    
    # 检查长度一致性
    if col_indices.size(0) != total_nnz or values.size(0) != total_nnz:
        print(f"  ❌ {name}: 数据长度不一致")
        return False
    
    # 验证每一行
    for m in range(M):
        start = row_offsets[m].item()
        end = row_offsets[m + 1].item()
        
        for idx in range(start, end):
            c = col_indices[idx].item()
            v = values[idx].item()
            
            if c < 0 or c >= K:
                print(f"  ❌ {name}: 行{m} 列索引超出范围")
                return False
            
            if mask[m, c].item() == 0:
                print(f"  ❌ {name}: ({m}, {c}) 在mask中为0但在CSR中出现")
                return False
            
            expected = activation[m, c].item()
            if abs(v - expected) > 1e-5:
                print(f"  ❌ {name}: ({m}, {c}) 值不匹配: {v} vs {expected}")
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
    """验证 CSC 格式输出的正确性。"""
    M, K = activation.shape
    
    # 检查 col_ptr 长度
    if col_ptr.size(0) != K + 1:
        print(f"  ❌ {name}: col_ptr 长度错误")
        return False
    
    total_nnz = col_ptr[K].item()
    
    # 检查长度一致性
    if row_indices.size(0) != total_nnz or values.size(0) != total_nnz:
        print(f"  ❌ {name}: 数据长度不一致")
        return False
    
    # 验证每一列
    for k in range(K):
        start = col_ptr[k].item()
        end = col_ptr[k + 1].item()
        
        for idx in range(start, end):
            r = row_indices[idx].item()
            v = values[idx].item()
            
            if r < 0 or r >= M:
                print(f"  ❌ {name}: 列{k} 行索引超出范围")
                return False
            
            if mask[r, k].item() == 0:
                print(f"  ❌ {name}: ({r}, {k}) 在mask中为0但在CSC中出现")
                return False
            
            expected = activation[r, k].item()
            if abs(v - expected) > 1e-5:
                print(f"  ❌ {name}: ({r}, {k}) 值不匹配: {v} vs {expected}")
                return False
    
    return True


def _verify_icsr_format(
    nz_counts: torch.Tensor,
    col_indices: torch.Tensor,
    row_offsets: torch.Tensor,
    mask: torch.Tensor,
    name: str,
) -> bool:
    """验证 iCSR 格式输出的正确性。"""
    M, K = mask.shape
    
    # 检查 row_offsets 长度
    if row_offsets.size(0) != M + 1:
        print(f"  ❌ {name}: row_offsets 长度错误")
        return False
    
    total_nnz = row_offsets[M].item()
    
    # 检查 col_indices 长度
    if col_indices.size(0) != total_nnz:
        print(f"  ❌ {name}: col_indices 长度不一致")
        return False
    
    # 检查 nz_counts 格式
    if nz_counts.size(0) % 2 != 0:
        print(f"  ❌ {name}: nz_counts 长度应为偶数")
        return False
    
    # 验证每一行
    for m in range(M):
        start = row_offsets[m].item()
        end = row_offsets[m + 1].item()
        
        for idx in range(start, end):
            c = col_indices[idx].item()
            
            if c < 0 or c >= K:
                print(f"  ❌ {name}: 行{m} 列索引超出范围")
                return False
            
            if mask[m, c].item() == 0:
                print(f"  ❌ {name}: ({m}, {c}) 在mask中为0但在iCSR中出现")
                return False
    
    return True


def test_correctness(
    activation: torch.Tensor,
    mask: torch.Tensor,
) -> bool:
    """测试所有算子的正确性。"""
    print("\n" + "=" * 80)
    print("正确性验证")
    print("=" * 80)
    
    all_passed = True
    
    # COO 格式
    print("\n[COO 格式]")
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
    
    # CSR 格式
    print("\n[CSR 格式]")
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
    
    # CSC 格式
    print("\n[CSC 格式]")
    try:
        col_ptr_csc, row_idx_csc, val_csc = mask_sparsify_to_csc(activation, mask)
        if _verify_csc_format(col_ptr_csc, row_idx_csc, val_csc, activation, mask, "mask_sparsify_to_csc"):
            print("  ✅ mask_sparsify_to_csc")
        else:
            all_passed = False
    except Exception as e:
        print(f"  ❌ mask_sparsify_to_csc: {e}")
        all_passed = False
    
    # iCSR 格式
    print("\n[iCSR 格式]")
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
        print("\n✅ 所有算子正确性测试通过")
    else:
        print("\n❌ 部分算子正确性测试失败")
    
    return all_passed


def test_performance(
    activation: torch.Tensor,
    mask: torch.Tensor,
    warmup: int = 5,
    iters: int = 100000,
) -> Dict[str, float]:
    """测试所有算子的性能。"""
    print("\n" + "=" * 80)
    print("性能测试")
    print("=" * 80)
    
    results: Dict[str, float] = {}
    
    # COO 格式
    print("\n[COO 格式]")
    print("  测试 mask_sparsify_to_coo...")
    lat = measure_latency(lambda: mask_sparsify_to_coo(activation, mask), warmup=warmup, iters=iters)
    results["mask_sparsify_to_coo"] = lat
    print(f"    延迟: {lat:.4f} ms")
    
    print("  测试 mask_sparsify_to_coo_sve...")
    lat = measure_latency(lambda: mask_sparsify_to_coo_sve(activation, mask), warmup=warmup, iters=iters)
    results["mask_sparsify_to_coo_sve"] = lat
    print(f"    延迟: {lat:.4f} ms")
    
    # CSR 格式
    print("\n[CSR 格式]")
    print("  测试 mask_sparsify_to_csr...")
    lat = measure_latency(lambda: mask_sparsify_to_csr(activation, mask), warmup=warmup, iters=iters)
    results["mask_sparsify_to_csr"] = lat
    print(f"    延迟: {lat:.4f} ms")
    
    print("  测试 mask_sparsify_to_csr_sve...")
    lat = measure_latency(lambda: mask_sparsify_to_csr_sve(activation, mask), warmup=warmup, iters=iters)
    results["mask_sparsify_to_csr_sve"] = lat
    print(f"    延迟: {lat:.4f} ms")
    
    # CSC 格式
    print("\n[CSC 格式]")
    print("  测试 mask_sparsify_to_csc...")
    lat = measure_latency(lambda: mask_sparsify_to_csc(activation, mask), warmup=warmup, iters=iters)
    results["mask_sparsify_to_csc"] = lat
    print(f"    延迟: {lat:.4f} ms")
    
    # iCSR 格式
    print("\n[iCSR 格式]")
    print("  测试 mask_sparsify_to_icsr...")
    lat = measure_latency(lambda: mask_sparsify_to_icsr(mask), warmup=warmup, iters=iters)
    results["mask_sparsify_to_icsr"] = lat
    print(f"    延迟: {lat:.4f} ms")
    
    print("  测试 mask_sparsify_to_icsr_sve...")
    lat = measure_latency(lambda: mask_sparsify_to_icsr_sve(mask), warmup=warmup, iters=iters)
    results["mask_sparsify_to_icsr_sve"] = lat
    print(f"    延迟: {lat:.4f} ms")
    
    return results


def print_performance_summary(results: Dict[str, float]) -> None:
    """打印性能对比总结。"""
    print("\n" + "=" * 80)
    print("性能对比总结")
    print("=" * 80)
    
    # 按延迟排序
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    
    print("\n延迟排名（从快到慢）：")
    print("-" * 80)
    print(f"{'排名':<4} {'算子名称':<40} {'延迟(ms)':<12} {'标记':<10}")
    print("-" * 80)
    
    for rank, (name, latency) in enumerate(sorted_results, 1):
        marker = "⚡ SVE" if "sve" in name else "📊 标量"
        print(f"{rank:2d}. {name:40s} {latency:8.4f} ms  {marker}")
    
    # 打印最快算子
    fastest_name, fastest_latency = sorted_results[0]
    print("\n" + "-" * 80)
    print(f"⚡ 最快算子: {fastest_name}")
    print(f"   延迟: {fastest_latency:.4f} ms")
    
    # 计算SVE加速比
    print("\n" + "=" * 80)
    print("SVE 加速比分析")
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
            print(f"\n{format_name} 格式:")
            print(f"  标量版本: {scalar_lat:.4f} ms")
            print(f"  SVE版本:  {sve_lat:.4f} ms")
            print(f"  加速比:   {speedup:.2f}x")


def test_multiple_sizes(
    sparsity: float = 0.9,
    seed: int = 42,
    warmup: int = 5,
    iters: int = 50000,
) -> None:
    """测试多种矩阵尺寸的性能。"""
    print("\n" + "=" * 80)
    print("多尺寸性能测试")
    print("=" * 80)
    
    # 测试配置：(M, K)
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
        print(f"测试配置: M={M}, K={K}, 稀疏度={sparsity}")
        print(f"{'='*80}")
        
        # 生成测试数据
        activation, mask = _make_random_mask(M, K, sparsity, seed)
        nnz = _count_nnz_from_mask(mask)
        actual_sparsity = 1.0 - (nnz / (M * K))
        print(f"实际稀疏度: {actual_sparsity*100:.1f}% ({nnz}/{M*K} 非零元素)")
        
        # 性能测试
        results = test_performance(activation, mask, warmup=warmup, iters=iters)
        all_results.append(((M, K), results))
    
    # 打印汇总表格
    print("\n" + "=" * 80)
    print("多尺寸性能汇总（延迟单位：ms）")
    print("=" * 80)
    
    # 表头
    algo_names = ["COO", "COO_SVE", "CSR", "CSR_SVE", "CSC", "iCSR", "iCSR_SVE"]
    print(f"\n{'配置':<12}", end="")
    for name in algo_names:
        print(f"{name:>10}", end="")
    print()
    print("-" * (12 + 10 * len(algo_names)))
    
    # 数据行
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
    
    # 打印加速比表格
    print("\n" + "=" * 80)
    print("SVE 加速比汇总")
    print("=" * 80)
    
    print(f"\n{'配置':<12}{'COO':>10}{'CSR':>10}{'iCSR':>10}")
    print("-" * 42)
    
    for (M, K), results in all_results:
        print(f"({M:2d},{K:5d})", end="  ")
        
        # COO 加速比
        lat_coo = results.get("mask_sparsify_to_coo", 0.0)
        lat_coo_sve = results.get("mask_sparsify_to_coo_sve", 0.0)
        speedup_coo = lat_coo / lat_coo_sve if lat_coo_sve > 0 else 0.0
        
        # CSR 加速比
        lat_csr = results.get("mask_sparsify_to_csr", 0.0)
        lat_csr_sve = results.get("mask_sparsify_to_csr_sve", 0.0)
        speedup_csr = lat_csr / lat_csr_sve if lat_csr_sve > 0 else 0.0
        
        # iCSR 加速比
        lat_icsr = results.get("mask_sparsify_to_icsr", 0.0)
        lat_icsr_sve = results.get("mask_sparsify_to_icsr_sve", 0.0)
        speedup_icsr = lat_icsr / lat_icsr_sve if lat_icsr_sve > 0 else 0.0
        
        print(f"{speedup_coo:10.2f}x{speedup_csr:10.2f}x{speedup_icsr:10.2f}x")


def main() -> None:
    """运行测试"""
    parser = argparse.ArgumentParser(description="mask_sparsify 算子性能测试")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--M", type=int, default=1, help="矩阵行数")
    parser.add_argument("--K", type=int, default=4096, help="矩阵列数")
    parser.add_argument("--sparsity", type=float, default=0.9, help="稀疏度 (0.0-1.0)")
    parser.add_argument("--warmup", type=int, default=5, help="预热迭代次数")
    parser.add_argument("--iters", type=int, default=10000, help="测试迭代次数")
    parser.add_argument("--test-sizes", action="store_true", help="测试多种矩阵尺寸")
    parser.add_argument("--skip-correctness", action="store_true", help="跳过正确性测试")
    args = parser.parse_args()
    
    print("=" * 80)
    print("mask_sparsify 算子性能测试")
    print("=" * 80)
    print(f"配置参数:")
    print(f"  - 随机种子: {args.seed}")
    if not args.test_sizes:
        print(f"  - 矩阵尺寸: ({args.M}, {args.K})")
        print(f"  - 稀疏度: {args.sparsity}")
    print(f"  - 预热迭代: {args.warmup}")
    print(f"  - 测试迭代: {args.iters}")
    
    try:
        # 加载扩展
        print("\n加载 C++ 扩展...")
        load_sve_sparse_gemm_extension(verbose=False)
        print("✅ C++ 扩展加载成功")
        
        if args.test_sizes:
            # 多尺寸测试
            test_multiple_sizes(
                sparsity=args.sparsity,
                seed=args.seed,
                warmup=args.warmup,
                iters=args.iters,
            )
        else:
            # 单一配置测试
            print(f"\n生成测试数据 ({args.M}, {args.K})...")
            activation, mask = _make_random_mask(args.M, args.K, args.sparsity, args.seed)
            
            # 统计实际稀疏度
            nnz = _count_nnz_from_mask(mask)
            actual_sparsity = 1.0 - (nnz / (args.M * args.K))
            print(f"实际稀疏度: {actual_sparsity*100:.1f}% ({nnz}/{args.M * args.K} 非零元素)")
            
            # 正确性测试
            if not args.skip_correctness:
                if not test_correctness(activation, mask):
                    print("\n⚠️  警告：正确性测试未通过，但继续进行性能测试")
            
            # 性能测试
            results = test_performance(activation, mask, warmup=args.warmup, iters=args.iters)
            
            # 打印性能总结
            print_performance_summary(results)
        
        print("\n" + "=" * 80)
        print("✅ 所有测试完成")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
