"""
测试 SVE2 compact 指令在 iCSR 稀疏化中的性能收益。

对比两个算子：
1. thr_sparsify_to_icsr_sve - 使用 SVE2 svcompact_u32 指令优化
2. thr_sparsify_to_icsr_sve_baseline - 不使用 SVE2 compact，手动循环提取

两者唯一区别：Pass2 中是否使用 SVE2 compact 指令
其他完全相同：Pass1统计、多线程、向量化、循环展开

这样可以公平地量化 SVE2 compact 指令的实际收益。
"""

import torch
import numpy as np
import time
from pathlib import Path
import sys

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernels.sve_sparse_gemm import (
    thr_sparsify_to_icsr_sve,
    thr_sparsify_to_icsr_sve_baseline,
    load_sve_sparse_gemm_extension,
)


def measure_latency_with_std(func, warmup: int = 5, repeats: int = 20):
    """
    测量函数延迟，返回 (平均值(秒), 标准差(秒))
    """
    # 预热
    for _ in range(warmup):
        func()
    
    # 测量
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)
    
    times = np.array(times)
    return np.mean(times), np.std(times)


def verify_correctness(M: int, K: int, threshold: float, sparsity: float):
    """验证两个算子输出的正确性"""
    print(f"\n{'='*80}")
    print(f"正确性验证: M={M}, K={K}, threshold={threshold:.4f}, 稀疏度={sparsity:.2%}")
    print(f"{'='*80}")
    
    # 生成测试数据：控制稀疏度
    activation = torch.randn(M, K, dtype=torch.float32)
    mask = torch.rand(M, K) > sparsity
    activation = activation * mask.float()
    
    # SVE2 版本
    nz_counts_sve2, col_indices_sve2, row_offsets_sve2 = thr_sparsify_to_icsr_sve(
        activation, threshold
    )
    
    # Baseline 版本
    nz_counts_baseline, col_indices_baseline, row_offsets_baseline = thr_sparsify_to_icsr_sve_baseline(
        activation, threshold
    )
    
    # 验证输出一致性
    assert torch.equal(nz_counts_sve2, nz_counts_baseline), "nz_counts 不匹配！"
    assert torch.equal(row_offsets_sve2, row_offsets_baseline), "row_offsets 不匹配！"
    assert torch.equal(col_indices_sve2, col_indices_baseline), "col_indices 不匹配！"
    
    total_nnz = row_offsets_sve2[-1].item()
    print(f"✅ 输出完全一致！总非零元素: {total_nnz:,}")
    print(f"   实际稀疏度: {1 - total_nnz / (M * K):.2%}")


def benchmark_versions(M: int, K: int, threshold: float, sparsity: float, 
                       warmup: int = 5, repeats: int = 20):
    """性能对比测试"""
    print(f"\n{'='*80}")
    print(f"性能对比: M={M}, K={K}, threshold={threshold:.4f}, 目标稀疏度={sparsity:.2%}")
    print(f"{'='*80}")
    
    # 生成测试数据
    activation = torch.randn(M, K, dtype=torch.float32)
    mask = torch.rand(M, K) > sparsity
    activation = activation * mask.float()
    
    # 预先计算实际稀疏度
    test_nz_counts, _, test_row_offsets = thr_sparsify_to_icsr_sve(activation, threshold)
    actual_nnz = test_row_offsets[-1].item()
    actual_sparsity = 1 - actual_nnz / (M * K)
    print(f"实际非零元素: {actual_nnz:,} ({100-actual_sparsity*100:.2f}%)")
    print(f"实际稀疏度: {actual_sparsity:.2%}")
    
    # SVE2 版本性能测试
    print(f"\n{'─'*80}")
    print("测试 SVE2 Compact 版本...")
    print(f"{'─'*80}")
    latency_sve2, std_sve2 = measure_latency_with_std(
        lambda: thr_sparsify_to_icsr_sve(activation, threshold),
        warmup=warmup,
        repeats=repeats,
    )
    throughput_sve2 = (M * K) / (latency_sve2 * 1e6)  # 元素/秒
    print(f"  延迟: {latency_sve2*1000:.4f} ± {std_sve2*1000:.4f} ms")
    print(f"  吞吐量: {throughput_sve2/1e9:.3f} G元素/秒")
    
    # Baseline 版本性能测试
    print(f"\n{'─'*80}")
    print("测试 SVE Baseline 版本 (不使用 compact)...")
    print(f"{'─'*80}")
    latency_baseline, std_baseline = measure_latency_with_std(
        lambda: thr_sparsify_to_icsr_sve_baseline(activation, threshold),
        warmup=warmup,
        repeats=repeats,
    )
    throughput_baseline = (M * K) / (latency_baseline * 1e6)
    print(f"  延迟: {latency_baseline*1000:.4f} ± {std_baseline*1000:.4f} ms")
    print(f"  吞吐量: {throughput_baseline/1e9:.3f} G元素/秒")
    
    # 性能对比
    speedup = latency_baseline / latency_sve2
    print(f"\n{'='*80}")
    print(f"📊 SVE2 Compact 指令收益分析")
    print(f"{'='*80}")
    print(f"  SVE2 版本延迟:     {latency_sve2*1000:.4f} ms")
    print(f"  Baseline 版本延迟: {latency_baseline*1000:.4f} ms")
    print(f"  加速比:           {speedup:.3f}x")
    print(f"  性能提升:         {(speedup-1)*100:.2f}%")
    print(f"  绝对时间节省:     {(latency_baseline-latency_sve2)*1000:.4f} ms")
    print(f"{'='*80}\n")
    
    return {
        'M': M,
        'K': K,
        'sparsity': actual_sparsity,
        'nnz': actual_nnz,
        'latency_sve2': latency_sve2,
        'latency_baseline': latency_baseline,
        'speedup': speedup,
        'throughput_sve2': throughput_sve2,
        'throughput_baseline': throughput_baseline,
    }


def run_comprehensive_tests():
    """运行全面的对比测试"""
    print("\n" + "="*80)
    print("SVE2 Compact 指令性能收益测试套件")
    print("="*80)
    
    # 加载扩展
    load_sve_sparse_gemm_extension(verbose=True)
    
    # 1. 正确性验证
    print("\n" + "🔍 第一步：正确性验证")
    verify_correctness(M=128, K=4096, threshold=0.01, sparsity=0.5)
    verify_correctness(M=512, K=4096, threshold=0.01, sparsity=0.8)
    verify_correctness(M=1024, K=8192, threshold=0.01, sparsity=0.9)
    
    # 2. 不同矩阵大小测试
    print("\n" + "📏 第二步：不同矩阵大小性能对比")
    results = []
    
    test_configs = [
        # (M, K, threshold, sparsity)
        (256, 4096, 0.01, 0.5),     # 小矩阵，低稀疏度
        (512, 4096, 0.01, 0.7),     # 中矩阵，中等稀疏度
        (1024, 4096, 0.01, 0.8),    # 中矩阵，高稀疏度
        (2048, 4096, 0.01, 0.9),    # 大矩阵，极高稀疏度
        (4096, 4096, 0.01, 0.5),    # 正方形矩阵
        (1024, 8192, 0.01, 0.85),   # 宽矩阵
        (4096, 2048, 0.01, 0.85),   # 高矩阵
    ]
    
    for M, K, threshold, sparsity in test_configs:
        result = benchmark_versions(M, K, threshold, sparsity, warmup=5, repeats=20)
        results.append(result)
    
    # 3. 汇总结果
    print("\n" + "="*80)
    print("📈 测试结果汇总")
    print("="*80)
    print(f"{'配置':<20} {'实际稀疏度':<12} {'SVE2 (ms)':<12} {'Baseline (ms)':<15} {'加速比':<10}")
    print("-"*80)
    
    for r in results:
        config = f"{r['M']}×{r['K']}"
        print(f"{config:<20} {r['sparsity']*100:>6.2f}%     "
              f"{r['latency_sve2']*1000:>8.4f}    "
              f"{r['latency_baseline']*1000:>10.4f}      "
              f"{r['speedup']:>6.3f}x")
    
    # 4. 统计分析
    speedups = [r['speedup'] for r in results]
    print("\n" + "="*80)
    print("📊 统计分析")
    print("="*80)
    print(f"  平均加速比:     {np.mean(speedups):.3f}x")
    print(f"  中位数加速比:   {np.median(speedups):.3f}x")
    print(f"  最小加速比:     {np.min(speedups):.3f}x")
    print(f"  最大加速比:     {np.max(speedups):.3f}x")
    print(f"  标准差:         {np.std(speedups):.3f}")
    print(f"  平均性能提升:   {(np.mean(speedups)-1)*100:.2f}%")
    print("="*80)
    
    # 5. 结论
    avg_speedup = np.mean(speedups)
    print("\n" + "🎯 测试结论")
    print("="*80)
    if avg_speedup > 1.5:
        print(f"✅ SVE2 compact 指令带来显著性能提升（平均 {avg_speedup:.2f}x 加速）")
        print(f"   推荐在生产环境中使用 SVE2 优化版本。")
    elif avg_speedup > 1.2:
        print(f"✅ SVE2 compact 指令带来明显性能提升（平均 {avg_speedup:.2f}x 加速）")
        print(f"   在支持 SVE2 的硬件上推荐使用优化版本。")
    elif avg_speedup > 1.05:
        print(f"⚠️  SVE2 compact 指令带来小幅性能提升（平均 {avg_speedup:.2f}x 加速）")
        print(f"   收益相对有限，可根据实际场景选择。")
    else:
        print(f"⚠️  SVE2 compact 指令收益不明显（平均 {avg_speedup:.2f}x 加速）")
        print(f"   可能受到其他瓶颈限制，建议进一步分析。")
    print("="*80)


if __name__ == "__main__":
    run_comprehensive_tests()
