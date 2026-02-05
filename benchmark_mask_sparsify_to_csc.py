"""
mask_sparsify_to_csc 算子性能对比测试脚本

对比两种实现：
1. mask_sparsify_to_csc - 基础 OpenMP 并行版本（标量写入）
2. mask_sparsify_to_csc_scatter - SVE scatter store 优化版本
"""

import argparse
import time
from typing import Tuple

import torch
import numpy as np

from kernels.sve_sparse_gemm import (
    mask_sparsify_to_csc,
    mask_sparsify_to_csc_scatter,
    measure_latency,
)


def create_test_data(
    M: int, K: int, sparsity: float, seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    创建测试数据。

    Args:
        M: 行数
        K: 列数
        sparsity: 稀疏度（非零元素比例，0-1）
        seed: 随机种子

    Returns:
        activation: (M, K) float32 CPU tensor
        mask: (M, K) uint8 CPU tensor
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 创建激活矩阵
    activation = torch.randn(M, K, dtype=torch.float32)

    # 创建 mask（根据稀疏度随机生成）
    mask_prob = 1.0 - sparsity  # 非零元素的概率
    mask = (torch.rand(M, K) < mask_prob).to(torch.uint8)

    return activation, mask


def verify_correctness(
    col_ptr1: torch.Tensor,
    row_indices1: torch.Tensor,
    values1: torch.Tensor,
    col_ptr2: torch.Tensor,
    row_indices2: torch.Tensor,
    values2: torch.Tensor,
    tolerance: float = 1e-6,
) -> bool:
    """
    验证两种实现的结果是否一致。

    Args:
        col_ptr1, row_indices1, values1: 第一种实现的输出
        col_ptr2, row_indices2, values2: 第二种实现的输出
        tolerance: 浮点数比较容差

    Returns:
        是否一致
    """
    # 检查列指针
    if not torch.equal(col_ptr1, col_ptr2):
        print("❌ col_ptr 不一致")
        return False

    # 检查行索引
    if not torch.equal(row_indices1, row_indices2):
        print("❌ row_indices 不一致")
        return False

    # 检查值（允许浮点误差）
    if not torch.allclose(values1, values2, atol=tolerance):
        print(f"❌ values 不一致，最大差异：{(values1 - values2).abs().max().item()}")
        return False

    return True


def benchmark_single_config(
    M: int,
    K: int,
    sparsity: float,
    warmup: int = 5,
    iters: int = 20,
    verbose: bool = True,
) -> dict:
    """
    对单个配置进行性能测试。

    Args:
        M: 行数
        K: 列数
        sparsity: 稀疏度
        warmup: 预热次数
        iters: 测试迭代次数
        verbose: 是否显示详细信息

    Returns:
        包含测试结果的字典
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"测试配置: M={M}, K={K}, 稀疏度={sparsity:.2%}")
        print(f"{'='*80}")

    # 创建测试数据
    activation, mask = create_test_data(M, K, sparsity)
    nnz_elements = mask.sum().item()
    total_elements = M * K
    actual_sparsity = 1.0 - (nnz_elements / total_elements)

    if verbose:
        print(f"实际非零元素数: {nnz_elements:,} / {total_elements:,}")
        print(f"实际稀疏度: {actual_sparsity:.2%}")

    # 测试基础版本
    if verbose:
        print(f"\n{'─'*80}")
        print("测试 mask_sparsify_to_csc (基础版本)")
        print(f"{'─'*80}")

    col_ptr1, row_indices1, values1 = mask_sparsify_to_csc(activation, mask)

    latency1_ms = measure_latency(
        lambda: mask_sparsify_to_csc(activation, mask),
        warmup=warmup,
        iters=iters,
    )

    if verbose:
        print(f"延迟: {latency1_ms:.3f} ms")
        print(f"输出 CSC 格式: nnz={values1.numel():,}")

    # 测试 SVE scatter 版本
    if verbose:
        print(f"\n{'─'*80}")
        print("测试 mask_sparsify_to_csc_scatter (SVE scatter 优化版本)")
        print(f"{'─'*80}")

    col_ptr2, row_indices2, values2 = mask_sparsify_to_csc_scatter(activation, mask)

    latency2_ms = measure_latency(
        lambda: mask_sparsify_to_csc_scatter(activation, mask),
        warmup=warmup,
        iters=iters,
    )

    if verbose:
        print(f"延迟: {latency2_ms:.3f} ms")
        print(f"输出 CSC 格式: nnz={values2.numel():,}")

    # 验证正确性
    if verbose:
        print(f"\n{'─'*80}")
        print("验证正确性")
        print(f"{'─'*80}")

    is_correct = verify_correctness(
        col_ptr1, row_indices1, values1,
        col_ptr2, row_indices2, values2,
    )

    if verbose:
        if is_correct:
            print("✓ 结果一致")
        else:
            print("✗ 结果不一致！")

    # 计算加速比
    speedup = latency1_ms / latency2_ms if latency2_ms > 0 else 0.0

    # 输出对比结果
    if verbose:
        print(f"\n{'─'*80}")
        print("性能对比")
        print(f"{'─'*80}")
        print(f"{'算子':<40} {'延迟 (ms)':<15} {'加速比':<10}")
        print(f"{'-'*80}")
        print(f"{'mask_sparsify_to_csc (基础)':<40} {latency1_ms:>12.3f}    {'1.00x':>10}")
        print(f"{'mask_sparsify_to_csc_scatter (SVE)':<40} {latency2_ms:>12.3f}    {speedup:>9.2f}x")
        print(f"{'─'*80}")
        
        if speedup > 1.0:
            improvement = (speedup - 1.0) * 100
            print(f"✓ SVE scatter 版本提速 {improvement:.1f}%")
        elif speedup < 1.0:
            degradation = (1.0 - speedup) * 100
            print(f"✗ SVE scatter 版本反而慢了 {degradation:.1f}%")
        else:
            print("性能相当")

    return {
        "M": M,
        "K": K,
        "sparsity": sparsity,
        "actual_sparsity": actual_sparsity,
        "nnz": nnz_elements,
        "latency_base_ms": latency1_ms,
        "latency_scatter_ms": latency2_ms,
        "speedup": speedup,
        "is_correct": is_correct,
    }


def benchmark_multiple_configs(
    configs: list,
    warmup: int = 5,
    iters: int = 20,
) -> list:
    """
    对多个配置进行性能测试。

    Args:
        configs: 配置列表，每个配置为 (M, K, sparsity) 元组
        warmup: 预热次数
        iters: 测试迭代次数

    Returns:
        测试结果列表
    """
    results = []
    for M, K, sparsity in configs:
        result = benchmark_single_config(
            M, K, sparsity,
            warmup=warmup,
            iters=iters,
            verbose=True,
        )
        results.append(result)

    return results


def print_summary(results: list):
    """
    打印汇总结果。

    Args:
        results: 测试结果列表
    """
    print(f"\n{'='*80}")
    print("汇总结果")
    print(f"{'='*80}")
    print(f"{'配置':<25} {'稀疏度':<12} {'基础 (ms)':<15} {'SVE (ms)':<15} {'加速比':<10}")
    print(f"{'-'*80}")

    for r in results:
        config_str = f"M={r['M']}, K={r['K']}"
        sparsity_str = f"{r['actual_sparsity']:.1%}"
        print(
            f"{config_str:<25} {sparsity_str:<12} "
            f"{r['latency_base_ms']:>12.3f}    "
            f"{r['latency_scatter_ms']:>12.3f}    "
            f"{r['speedup']:>9.2f}x"
        )

    print(f"{'='*80}")

    # 计算平均加速比
    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    print(f"平均加速比: {avg_speedup:.2f}x")

    # 检查正确性
    all_correct = all(r["is_correct"] for r in results)
    if all_correct:
        print("✓ 所有测试结果正确")
    else:
        print("✗ 部分测试结果不正确")


def main():
    parser = argparse.ArgumentParser(
        description="mask_sparsify_to_csc 算子性能对比测试"
    )
    parser.add_argument(
        "--M",
        type=int,
        default=None,
        help="矩阵行数（默认：测试多个配置）",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=None,
        help="矩阵列数（默认：测试多个配置）",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=None,
        help="稀疏度（0-1，默认：测试多个配置）",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="预热次数（默认：5）",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="测试迭代次数（默认：20）",
    )

    args = parser.parse_args()

    # 如果指定了 M, K, sparsity，则测试单个配置
    if args.M is not None and args.K is not None and args.sparsity is not None:
        result = benchmark_single_config(
            args.M,
            args.K,
            args.sparsity,
            warmup=args.warmup,
            iters=args.iters,
            verbose=True,
        )
    else:
        # 否则测试多个配置
        print("未指定配置，将测试多个预设配置...\n")

        configs = [
            # (M, K, sparsity)
            # 小规模测试
            (128, 4096, 0.5),
            (128, 4096, 0.7),
            (128, 4096, 0.9),
            # 中等规模测试
            (256, 4096, 0.5),
            (256, 4096, 0.7),
            (256, 4096, 0.9),
            # 大规模测试
            (512, 4096, 0.5),
            (512, 4096, 0.7),
            (512, 4096, 0.9),
            # 更大规模
            (1024, 4096, 0.7),
            (1024, 4096, 0.9),
        ]

        results = benchmark_multiple_configs(
            configs,
            warmup=args.warmup,
            iters=args.iters,
        )

        print_summary(results)


if __name__ == "__main__":
    main()
