"""
ICSR SVE Baseline vs SVE2 Compact 性能对比脚本

该脚本专门用于对比两个 ICSR 稀疏化算子的性能：
1. thr_sparsify_to_icsr_sve_baseline - SVE 版本（不使用 svcompact）
2. thr_sparsify_to_icsr_sve - SVE2 版本（使用 svcompact_u32 指令）

测试重点：
- 量化 SVE2 compact 指令带来的实际性能提升
- 在不同矩阵尺寸和稀疏度下的表现
- 提供详细的性能分析和可视化结果
"""

import torch
import numpy as np
import time
import argparse
from pathlib import Path
import sys
from typing import Tuple, Dict, List

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernels.sve_sparse_gemm import (
    thr_sparsify_to_icsr_sve,
    thr_sparsify_to_icsr_sve_baseline,
    load_sve_sparse_gemm_extension,
)


def generate_sparse_tensor(M: int, K: int, sparsity: float, seed: int = 42) -> torch.Tensor:
    """生成指定稀疏度的张量"""
    torch.manual_seed(seed)
    activation = torch.randn(M, K, dtype=torch.float32)
    mask = torch.rand(M, K) > sparsity
    return activation * mask.float()


def warmup_runs(func, activation: torch.Tensor, threshold: float, n: int = 5):
    """预热运行"""
    for _ in range(n):
        func(activation, threshold)


def measure_performance(
    func,
    activation: torch.Tensor,
    threshold: float,
    repeats: int = 50
) -> Tuple[float, float]:
    """
    测量函数性能
    返回：(平均延迟(秒), 标准差(秒))
    """
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(activation, threshold)
        end = time.perf_counter()
        times.append(end - start)
    
    times = np.array(times)
    return np.mean(times), np.std(times)


def verify_output_consistency(
    activation: torch.Tensor,
    threshold: float
) -> bool:
    """验证两个算子输出的一致性"""
    nz_counts_sve2, col_indices_sve2, row_offsets_sve2 = thr_sparsify_to_icsr_sve(
        activation, threshold
    )
    nz_counts_baseline, col_indices_baseline, row_offsets_baseline = thr_sparsify_to_icsr_sve_baseline(
        activation, threshold
    )
    
    checks = [
        torch.equal(nz_counts_sve2, nz_counts_baseline),
        torch.equal(col_indices_sve2, col_indices_baseline),
        torch.equal(row_offsets_sve2, row_offsets_baseline),
    ]
    
    return all(checks)


def run_single_benchmark(
    M: int,
    K: int,
    threshold: float,
    sparsity: float,
    warmup: int = 5,
    repeats: int = 50,
    verbose: bool = True
) -> Dict:
    """运行单个配置的性能测试"""
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"测试配置: M={M}, K={K}, 阈值={threshold}, 目标稀疏度={sparsity:.1%}")
        print(f"{'='*80}")
    
    # 生成测试数据
    activation = generate_sparse_tensor(M, K, sparsity)
    
    # 计算实际稀疏度
    _, _, row_offsets = thr_sparsify_to_icsr_sve(activation, threshold)
    actual_nnz = row_offsets[-1].item()
    actual_sparsity = 1 - actual_nnz / (M * K)
    
    if verbose:
        print(f"实际非零元素: {actual_nnz:,} / {M*K:,}")
        print(f"实际稀疏度: {actual_sparsity:.2%}\n")
    
    # 预热
    warmup_runs(thr_sparsify_to_icsr_sve, activation, threshold, warmup)
    warmup_runs(thr_sparsify_to_icsr_sve_baseline, activation, threshold, warmup)
    
    # SVE2 Compact 版本测试
    if verbose:
        print("测试 SVE2 Compact 版本...")
    latency_sve2, std_sve2 = measure_performance(
        thr_sparsify_to_icsr_sve,
        activation,
        threshold,
        repeats
    )
    throughput_sve2 = (M * K) / (latency_sve2 * 1e9)  # G元素/秒
    
    if verbose:
        print(f"  延迟: {latency_sve2*1000:.4f} ± {std_sve2*1000:.4f} ms")
        print(f"  吞吐量: {throughput_sve2:.3f} G元素/秒\n")
    
    # SVE Baseline 版本测试
    if verbose:
        print("测试 SVE Baseline 版本（无 compact）...")
    latency_baseline, std_baseline = measure_performance(
        thr_sparsify_to_icsr_sve_baseline,
        activation,
        threshold,
        repeats
    )
    throughput_baseline = (M * K) / (latency_baseline * 1e9)
    
    if verbose:
        print(f"  延迟: {latency_baseline*1000:.4f} ± {std_baseline*1000:.4f} ms")
        print(f"  吞吐量: {throughput_baseline:.3f} G元素/秒\n")
    
    # 计算加速比
    speedup = latency_baseline / latency_sve2
    improvement_pct = (speedup - 1) * 100
    
    if verbose:
        print(f"{'─'*80}")
        print(f"📊 性能对比结果")
        print(f"{'─'*80}")
        print(f"  加速比:         {speedup:.3f}x")
        print(f"  性能提升:       {improvement_pct:.2f}%")
        print(f"  时间节省:       {(latency_baseline-latency_sve2)*1000:.4f} ms")
        print(f"{'─'*80}")
    
    return {
        'M': M,
        'K': K,
        'threshold': threshold,
        'target_sparsity': sparsity,
        'actual_sparsity': actual_sparsity,
        'actual_nnz': actual_nnz,
        'latency_sve2': latency_sve2,
        'std_sve2': std_sve2,
        'latency_baseline': latency_baseline,
        'std_baseline': std_baseline,
        'speedup': speedup,
        'improvement_pct': improvement_pct,
        'throughput_sve2': throughput_sve2,
        'throughput_baseline': throughput_baseline,
    }


def print_summary_table(results: List[Dict]):
    """打印结果汇总表格"""
    print("\n" + "="*100)
    print("📈 测试结果汇总表")
    print("="*100)
    print(f"{'矩阵尺寸':<15} {'实际稀疏度':<12} {'SVE2 (ms)':<13} {'Baseline (ms)':<15} {'加速比':<12} {'性能提升':<12}")
    print("-"*100)
    
    for r in results:
        config = f"{r['M']}×{r['K']}"
        print(
            f"{config:<15} "
            f"{r['actual_sparsity']*100:>6.2f}%      "
            f"{r['latency_sve2']*1000:>9.4f}    "
            f"{r['latency_baseline']*1000:>11.4f}      "
            f"{r['speedup']:>8.3f}x    "
            f"{r['improvement_pct']:>7.2f}%"
        )
    
    print("="*100)


def print_statistics(results: List[Dict]):
    """打印统计分析"""
    speedups = [r['speedup'] for r in results]
    improvements = [r['improvement_pct'] for r in results]
    
    print("\n" + "="*80)
    print("📊 统计分析")
    print("="*80)
    print(f"  测试场景数量:       {len(results)}")
    print(f"  平均加速比:         {np.mean(speedups):.3f}x")
    print(f"  中位数加速比:       {np.median(speedups):.3f}x")
    print(f"  最大加速比:         {np.max(speedups):.3f}x")
    print(f"  最小加速比:         {np.min(speedups):.3f}x")
    print(f"  加速比标准差:       {np.std(speedups):.3f}")
    print(f"  平均性能提升:       {np.mean(improvements):.2f}%")
    print("="*80)


def print_conclusion(results: List[Dict]):
    """打印测试结论"""
    avg_speedup = np.mean([r['speedup'] for r in results])
    
    print("\n" + "="*80)
    print("🎯 测试结论")
    print("="*80)
    
    if avg_speedup >= 1.5:
        status = "✅ 显著性能提升"
        recommendation = "强烈推荐在生产环境中使用 SVE2 优化版本"
    elif avg_speedup >= 1.2:
        status = "✅ 明显性能提升"
        recommendation = "推荐在支持 SVE2 的硬件上使用优化版本"
    elif avg_speedup >= 1.05:
        status = "⚠️  小幅性能提升"
        recommendation = "收益相对有限，可根据实际场景选择"
    else:
        status = "⚠️  收益不明显"
        recommendation = "可能受到其他瓶颈限制，建议进一步分析"
    
    print(f"{status}")
    print(f"  SVE2 compact 指令平均加速比: {avg_speedup:.3f}x")
    print(f"  平均性能提升: {(avg_speedup-1)*100:.2f}%")
    print(f"\n建议: {recommendation}")
    print("="*80)


def run_quick_test():
    """快速测试（少量配置）"""
    print("\n" + "🚀 运行快速测试模式")
    
    test_configs = [
        # (M, K, threshold, sparsity)
        (512, 4096, 0.01, 0.5),
        (1024, 4096, 0.01, 0.7),
        (2048, 4096, 0.01, 0.9),
    ]
    
    results = []
    for M, K, threshold, sparsity in test_configs:
        result = run_single_benchmark(M, K, threshold, sparsity, warmup=3, repeats=20)
        results.append(result)
    
    print_summary_table(results)
    print_statistics(results)
    print_conclusion(results)


def run_comprehensive_test():
    """全面测试（多种配置）"""
    print("\n" + "🔬 运行全面测试模式")
    
    test_configs = [
        # 不同矩阵尺寸
        (128, 2048, 0.01, 0.5),
        (128, 4096, 0.01, 0.5),
        (256, 4096, 0.01, 0.5),
        (512, 4096, 0.01, 0.5),
        (1, 4096, 0.01, 0.5),
        
        # 不同稀疏度
        (128, 4096, 0.01, 0.3),
        (128, 4096, 0.01, 0.5),
        (128, 4096, 0.01, 0.7),
        (128, 4096, 0.01, 0.8),
        (128, 4096, 0.01, 0.9),
        (128, 4096, 0.01, 0.95),
        
        # 不同形状
        (512, 8192, 0.01, 0.7),   # 宽矩阵
        (2048, 2048, 0.01, 0.7),  # 正方形
        (4096, 1024, 0.01, 0.7),  # 高矩阵
    ]
    
    results = []
    for M, K, threshold, sparsity in test_configs:
        result = run_single_benchmark(M, K, threshold, sparsity, warmup=5, repeats=50, verbose=False)
        # 打印简要进度
        print(f"✓ {M}×{K} (稀疏度 {sparsity:.0%}): 加速比 {result['speedup']:.3f}x")
        results.append(result)
    
    print_summary_table(results)
    print_statistics(results)
    print_conclusion(results)


def run_custom_test(M: int, K: int, threshold: float, sparsity: float, 
                    warmup: int, repeats: int):
    """自定义测试"""
    print("\n" + "⚙️  运行自定义测试")
    result = run_single_benchmark(M, K, threshold, sparsity, warmup, repeats)
    print_conclusion([result])


def verify_correctness():
    """验证正确性"""
    print("\n" + "="*80)
    print("🔍 正确性验证")
    print("="*80)
    
    test_cases = [
        (128, 2048, 0.01, 0.5),
        (512, 4096, 0.01, 0.7),
        (1024, 4096, 0.01, 0.9),
    ]
    
    all_passed = True
    for M, K, threshold, sparsity in test_cases:
        activation = generate_sparse_tensor(M, K, sparsity)
        passed = verify_output_consistency(activation, threshold)
        
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {M}×{K} (稀疏度 {sparsity:.0%}): {status}")
        
        if not passed:
            all_passed = False
    
    print("="*80)
    if all_passed:
        print("✅ 所有正确性检查通过！\n")
    else:
        print("❌ 部分正确性检查失败，请检查算子实现！\n")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="ICSR SVE Baseline vs SVE2 Compact 性能对比测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
测试模式:
  quick         - 快速测试（3个配置，用时约1分钟）
  comprehensive - 全面测试（14个配置，用时约5-10分钟）
  custom        - 自定义测试（需要提供参数）
  verify        - 仅运行正确性验证

示例:
  python %(prog)s --mode quick
  python %(prog)s --mode comprehensive
  python %(prog)s --mode custom -M 1024 -K 4096 --sparsity 0.7
  python %(prog)s --mode verify
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='quick',
        choices=['quick', 'comprehensive', 'custom', 'verify'],
        help='测试模式（默认: quick）'
    )
    
    # 自定义模式参数
    parser.add_argument('-M', type=int, default=128, help='矩阵行数（仅 custom 模式）')
    parser.add_argument('-K', type=int, default=4096, help='矩阵列数（仅 custom 模式）')
    parser.add_argument('--threshold', type=float, default=0.01, help='稀疏化阈值（仅 custom 模式）')
    parser.add_argument('--sparsity', type=float, default=0.7, help='目标稀疏度（仅 custom 模式）')
    parser.add_argument('--warmup', type=int, default=5, help='预热次数（仅 custom 模式）')
    parser.add_argument('--repeats', type=int, default=50, help='重复测试次数（仅 custom 模式）')
    
    parser.add_argument('--no-verify', action='store_true', help='跳过正确性验证')
    parser.add_argument('--verbose', action='store_true', help='详细输出（仅 custom 模式）')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("ICSR SVE Baseline vs SVE2 Compact 性能对比测试")
    print("="*80)
    print(f"测试模式: {args.mode}")
    print("="*80)
    
    # 加载扩展
    print("\n加载 C++ 扩展...")
    load_sve_sparse_gemm_extension(verbose=False)
    print("✅ 扩展加载成功")
    
    # 正确性验证（除非指定跳过）
    if not args.no_verify and args.mode != 'verify':
        verify_correctness()
    
    # 运行测试
    if args.mode == 'quick':
        run_quick_test()
    elif args.mode == 'comprehensive':
        run_comprehensive_test()
    elif args.mode == 'custom':
        run_custom_test(
            args.M, args.K, args.threshold, args.sparsity,
            args.warmup, args.repeats
        )
    elif args.mode == 'verify':
        verify_correctness()
    
    print("\n✅ 测试完成！\n")


if __name__ == "__main__":
    main()
