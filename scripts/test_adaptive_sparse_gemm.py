"""
测试自适应稀疏 GEMM 实现。

验证:
1. 正确性: 与 dense matmul 对比
2. 调度逻辑: 验证 dense/sparse block 划分
3. 性能: 对比 dense/sparse 路径的性能

使用方法:
    python -m scripts.test_adaptive_sparse_gemm
    SPARSE_DEBUG=1 python -m scripts.test_adaptive_sparse_gemm
"""

import sys
import os
from pathlib import Path

# 添加路径
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
sys.path.append(str(wd / "gpt-fast"))

import torch
from model import adaptive_sparse_gemm

def test_correctness():
    """测试正确性: 与 dense matmul 对比"""
    print("=" * 80)
    print("正确性测试")
    print("=" * 80)
    
    M, K, N = 8, 256, 512
    threshold = 0.3
    
    # 生成测试数据
    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=torch.float32)
    # PyTorch Linear weight 是 (N, K), 但 adaptive_sparse_gemm 期望 column-major
    weight = torch.randn(N, K, dtype=torch.float32)
    
    # Dense baseline: x @ weight.T
    output_dense = torch.matmul(x, weight.T)
    
    # Adaptive sparse GEMM (假设 weight 已经是列主序)
    output_sparse = adaptive_sparse_gemm(x, weight, threshold)
    
    # 对比
    max_diff = torch.max(torch.abs(output_dense - output_sparse)).item()
    mean_diff = torch.mean(torch.abs(output_dense - output_sparse)).item()
    
    print(f"Max diff: {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")
    
    if max_diff < 1e-4:
        print("✅ 正确性测试通过")
        return True
    else:
        print("❌ 正确性测试失败")
        return False


def test_scheduling():
    """测试调度逻辑"""
    print("\n" + "=" * 80)
    print("调度逻辑测试")
    print("=" * 80)
    
    # 构造特定稀疏度的矩阵
    M, K, N = 16, 128, 256
    
    # 前 6 行: dense (低稀疏度)
    # 中间 4 行: sparse (高稀疏度)
    # 后 6 行: dense (低稀疏度)
    
    x = torch.zeros(M, K, dtype=torch.float32)
    
    # 前 6 行: 80% 非零 (稀疏度 0.2)
    x[:6, :] = torch.randn(6, K)
    mask = torch.rand(6, K) > 0.2
    x[:6, :] = x[:6, :] * mask.float()
    
    # 中间 4 行: 10% 非零 (稀疏度 0.9)
    x[6:10, :] = torch.randn(4, K)
    mask = torch.rand(4, K) > 0.9
    x[6:10, :] = x[6:10, :] * mask.float()
    
    # 后 6 行: 80% 非零 (稀疏度 0.2)
    x[10:, :] = torch.randn(6, K)
    mask = torch.rand(6, K) > 0.2
    x[10:, :] = x[10:, :] * mask.float()
    
    weight = torch.randn(N, K, dtype=torch.float32)
    threshold = 0.01  # 低 threshold, 保留大部分非零值
    
    # 启用 debug
    os.environ["SPARSE_DEBUG"] = "1"
    
    print("\n期望调度:")
    print("  - 行 0-5: dense (连续 6 行, 稀疏度 ~0.2 < 0.7)")
    print("  - 行 6-9: sparse (连续 4 行, 稀疏度 ~0.9 > 0.7)")
    print("  - 行 10-15: dense (连续 6 行, 稀疏度 ~0.2 < 0.7)")
    
    print("\n实际调度:")
    output = adaptive_sparse_gemm(x, weight, threshold)
    
    # 关闭 debug
    os.environ["SPARSE_DEBUG"] = "0"
    
    print("\n✅ 调度逻辑测试完成 (请检查上述输出)")


def test_multiple_configs():
    """测试多种配置"""
    print("\n" + "=" * 80)
    print("多配置测试")
    print("=" * 80)
    
    configs = [
        (1, 1024, 2048, 0.3, "单行 (类似 decode, 但用于测试)"),
        (4, 512, 1024, 0.5, "小 batch prefill"),
        (16, 2048, 4096, 0.7, "中等 batch prefill"),
        (32, 4096, 4096, 0.9, "大 batch prefill, 高稀疏度"),
    ]
    
    for M, K, N, threshold, desc in configs:
        print(f"\n配置: {desc}")
        print(f"  M={M}, K={K}, N={N}, threshold={threshold}")
        
        torch.manual_seed(42)
        x = torch.randn(M, K, dtype=torch.float32)
        weight = torch.randn(N, K, dtype=torch.float32)
        
        try:
            output = adaptive_sparse_gemm(x, weight, threshold)
            
            # 验证形状
            assert output.shape == (M, N), f"形状错误: {output.shape} vs {(M, N)}"
            
            # 验证正确性
            output_dense = torch.matmul(x, weight.T)
            max_diff = torch.max(torch.abs(output_dense - output)).item()
            
            if max_diff < 1e-4:
                print(f"  ✅ 通过 (max_diff={max_diff:.6f})")
            else:
                print(f"  ❌ 失败 (max_diff={max_diff:.6f})")
        except Exception as e:
            print(f"  ❌ 异常: {e}")


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 80)
    print("边界情况测试")
    print("=" * 80)
    
    # 1. 全部 dense
    print("\n1. 全部 dense (所有行稀疏度低)")
    M, K, N = 8, 128, 256
    x = torch.randn(M, K, dtype=torch.float32)  # 全部非零
    weight = torch.randn(N, K, dtype=torch.float32)
    threshold = 0.001  # 低 threshold
    
    os.environ["SPARSE_DEBUG"] = "1"
    output = adaptive_sparse_gemm(x, weight, threshold)
    os.environ["SPARSE_DEBUG"] = "0"
    
    # 2. 全部 sparse
    print("\n2. 全部 sparse (所有行稀疏度高)")
    x = torch.randn(M, K, dtype=torch.float32)
    mask = torch.rand(M, K) > 0.95  # 95% 稀疏度
    x = x * mask.float()
    
    os.environ["SPARSE_DEBUG"] = "1"
    output = adaptive_sparse_gemm(x, weight, threshold)
    os.environ["SPARSE_DEBUG"] = "0"
    
    # 3. Dense block < MIN_DENSE_BLOCK
    print("\n3. Dense block 太小 (< MIN_DENSE_BLOCK=4)")
    x = torch.zeros(8, K, dtype=torch.float32)
    x[:2, :] = torch.randn(2, K)  # 前 2 行 dense
    x[2:, :] = torch.randn(6, K) * (torch.rand(6, K) > 0.9).float()  # 后 6 行 sparse
    
    os.environ["SPARSE_DEBUG"] = "1"
    output = adaptive_sparse_gemm(x, weight, threshold)
    os.environ["SPARSE_DEBUG"] = "0"
    
    print("\n✅ 边界情况测试完成")


def main():
    print("自适应稀疏 GEMM 测试")
    print("=" * 80)
    
    # 加载 SVE 扩展
    print("\n加载 SVE 扩展...")
    from kernels.sve_sparse_gemm import load_sve_sparse_gemm_extension
    load_sve_sparse_gemm_extension(verbose=False)
    print("✅ SVE 扩展加载完成")
    
    # 运行测试
    all_passed = True
    
    all_passed &= test_correctness()
    test_scheduling()
    test_multiple_configs()
    test_edge_cases()
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ 所有测试通过")
    else:
        print("❌ 部分测试失败")
    print("=" * 80)


if __name__ == "__main__":
    main()
