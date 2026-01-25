"""
SVE CSC GEMM 算子使用示例

展示两种使用方式：
1. 便捷函数：直接调用 sve_csc_gemm()
2. Kernel 类：用于 torch.compile 优化
"""

import sys
from pathlib import Path

import torch
import numpy as np

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernels.sve_csc_gemm import (
    SVECSCGEMMKernel,
    sve_csc_gemm,
    load_sve_csc_gemm_extension,
)


def generate_sparse_data(M=128, K=256, N=512, sparsity=0.9):
    """生成测试数据"""
    torch.manual_seed(42)
    
    # 生成稀疏 activation
    activation = torch.randn(M, K, dtype=torch.float32)
    mask = torch.rand(M, K) > sparsity
    activation = activation * mask.float()
    
    # 生成权重
    weight = torch.randn(K, N, dtype=torch.float32)
    
    # 构建稀疏元数据
    nz_counts_list = []
    nz_col_indices_list = []
    
    for m in range(M):
        row = activation[m]
        nz_mask = row != 0
        nz_indices = torch.where(nz_mask)[0]
        
        if len(nz_indices) > 0:
            nz_counts_list.append(m)
            nz_counts_list.append(len(nz_indices))
            nz_col_indices_list.extend(nz_indices.tolist())
    
    nz_counts = torch.tensor(nz_counts_list, dtype=torch.int64)
    nz_col_indices = torch.tensor(nz_col_indices_list, dtype=torch.uint32)
    
    return activation, weight, nz_counts, nz_col_indices


def example1_simple_usage():
    """示例1: 使用便捷函数"""
    print("=" * 60)
    print("示例1: 使用便捷函数 sve_csc_gemm()")
    print("=" * 60)
    
    activation, weight, nz_counts, nz_col_indices = generate_sparse_data()
    
    # 方法1: 使用默认线程数（自动）
    output1 = sve_csc_gemm(activation, weight, nz_counts, nz_col_indices)
    print(f"输出形状: {output1.shape}")
    
    # 方法2: 指定线程数
    output2 = sve_csc_gemm(activation, weight, nz_counts, nz_col_indices, ncore=4)
    print(f"输出形状（ncore=4）: {output2.shape}")
    
    # 验证结果一致性
    diff = torch.abs(output1 - output2).max().item()
    print(f"两种方式结果差异: {diff:.2e}")
    print()


def example2_kernel_class():
    """示例2: 使用 Kernel 类（支持 torch.compile）"""
    print("=" * 60)
    print("示例2: 使用 SVECSCGEMMKernel 类（支持 torch.compile）")
    print("=" * 60)
    
    # 首先加载扩展
    load_sve_csc_gemm_extension()
    
    # 使用 initialize 方法创建 kernel 实例
    kernel = SVECSCGEMMKernel.initialize(name="sve_csc_gemm", target="CPU")
    
    # 获取算子（可以选择编译或不编译）
    op_compiled = kernel.operator(compiled=True)
    op_eager = kernel.operator(compiled=False)
    
    # 准备数据
    activation, weight, nz_counts, nz_col_indices = generate_sparse_data()
    
    # 使用编译版本
    output_compiled = op_compiled(activation, weight, nz_counts, nz_col_indices, 0)
    print(f"编译版本输出形状: {output_compiled.shape}")
    
    # 使用非编译版本
    output_eager = op_eager(activation, weight, nz_counts, nz_col_indices, 0)
    print(f"非编译版本输出形状: {output_eager.shape}")
    
    # 验证结果一致性
    diff = torch.abs(output_compiled - output_eager).max().item()
    print(f"编译/非编译结果差异: {diff:.2e}")
    print()


def example3_custom_model():
    """示例3: 在自定义模型中使用"""
    print("=" * 60)
    print("示例3: 在 PyTorch 模型中使用")
    print("=" * 60)
    
    class SparseLinearLayer(torch.nn.Module):
        def __init__(self, K, N):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(K, N))
            # 初始化 kernel
            load_sve_csc_gemm_extension()
            self.kernel = SVECSCGEMMKernel.initialize(
                name="sparse_linear", 
                target="CPU"
            )
            self.op = self.kernel.operator(compiled=True)
        
        def forward(self, activation, nz_counts, nz_col_indices):
            """
            Args:
                activation: (M, K) 稀疏激活
                nz_counts: (2 * num_nz_rows) 元数据
                nz_col_indices: 列索引
            """
            return self.op(activation, self.weight, nz_counts, nz_col_indices, 0)
    
    # 创建模型
    model = SparseLinearLayer(K=256, N=512)
    
    # 准备数据
    activation, _, nz_counts, nz_col_indices = generate_sparse_data(
        M=128, K=256, N=512, sparsity=0.9
    )
    
    # 前向传播
    output = model(activation, nz_counts, nz_col_indices)
    print(f"模型输出形状: {output.shape}")
    
    # 验证梯度计算
    loss = output.sum()
    loss.backward()
    print(f"权重梯度形状: {model.weight.grad.shape}")
    print(f"权重梯度范数: {model.weight.grad.norm().item():.4f}")
    print()


def example4_performance_comparison():
    """示例4: 性能对比"""
    print("=" * 60)
    print("示例4: 性能对比")
    print("=" * 60)
    
    import time
    
    activation, weight, nz_counts, nz_col_indices = generate_sparse_data(
        M=512, K=1024, N=2048, sparsity=0.95
    )
    
    print(f"测试配置:")
    print(f"  矩阵尺寸: ({activation.shape[0]}, {activation.shape[1]}) × ({weight.shape[0]}, {weight.shape[1]})")
    print(f"  稀疏度: 95%")
    print(f"  非零元素: {nz_col_indices.numel()}")
    print()
    
    # PyTorch 原生
    warmup = 10
    iters = 50
    
    for _ in range(warmup):
        _ = torch.matmul(activation, weight)
    
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = torch.matmul(activation, weight)
    t1 = time.perf_counter()
    time_pytorch = (t1 - t0) * 1000 / iters
    
    # CSC GEMM
    for _ in range(warmup):
        _ = sve_csc_gemm(activation, weight, nz_counts, nz_col_indices)
    
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = sve_csc_gemm(activation, weight, nz_counts, nz_col_indices)
    t1 = time.perf_counter()
    time_csc = (t1 - t0) * 1000 / iters
    
    print(f"性能结果:")
    print(f"  PyTorch matmul: {time_pytorch:.3f} ms")
    print(f"  CSC GEMM:       {time_csc:.3f} ms")
    print(f"  加速比:         {time_pytorch/time_csc:.2f}x")
    print()


def main():
    print("\n" + "=" * 60)
    print("SVE CSC GEMM 算子使用示例")
    print("=" * 60 + "\n")
    
    # 运行所有示例
    example1_simple_usage()
    example2_kernel_class()
    example3_custom_model()
    example4_performance_comparison()
    
    print("=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
