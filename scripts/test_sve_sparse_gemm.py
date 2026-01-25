"""
ARM SVE 稀疏 GEMM 算子的正确性与性能测试。

运行方式:
    python -m scripts.test_sve_sparse_gemm
"""

from __future__ import annotations

import argparse
import torch

from kernels.sve_sparse_gemm import (
    SVESparseGEMMKernel,
    load_sve_sparse_gemm_extension,
    measure_latency,
)


def _make_random_sparse_activation(
    M: int,
    K: int,
    sparsity: float,
    seed: int,
) -> torch.Tensor:
    """
    生成随机稀疏 activation（float32）。

    sparsity: 稀疏度(0~1)，表示置零比例；0=全非零，1=全为零。
    """
    assert 0.0 <= sparsity <= 1.0
    g = torch.Generator()
    g.manual_seed(seed)

    x = torch.randn(M, K, dtype=torch.float32, generator=g)
    if sparsity <= 0.0:
        return x
    if sparsity >= 1.0:
        return torch.zeros(M, K, dtype=torch.float32)

    keep = (torch.rand(M, K, generator=g) >= sparsity).to(torch.float32)
    return x * keep


def _get_sparse_indices(activation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从稀疏 activation 中提取非零元素的索引信息。
    
    Returns:
        nz_counts: (M,) 每行的非零元素个数
        nz_col_indices: 扁平化的列索引向量
    """
    M, K = activation.shape
    nz_counts = []
    nz_col_indices = []
    
    for m in range(M):
        row = activation[m]
        nz_idx = torch.nonzero(row != 0, as_tuple=False).flatten()
        nz_counts.append(len(nz_idx))
        nz_col_indices.append(nz_idx)
    
    nz_counts = torch.tensor(nz_counts, dtype=torch.int64)
    nz_col_indices = torch.cat(nz_col_indices, dim=0).to(dtype=torch.int64) if len(nz_col_indices) > 0 else torch.tensor([], dtype=torch.int64)
    
    return nz_counts, nz_col_indices

def _get_sparse_indices_uint32(activation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从稀疏 activation 中提取非零元素的索引信息。
    
    Returns:
        nz_counts: (M,) 每行的非零元素个数
        nz_col_indices: 扁平化的列索引向量
    """
    M, K = activation.shape
    nz_counts = []
    nz_col_indices = []
    
    for m in range(M):
        row = activation[m]
        nz_idx = torch.nonzero(row != 0, as_tuple=False).flatten()
        nz_counts.append(len(nz_idx))
        nz_col_indices.append(nz_idx)
    nz_counts = torch.tensor(nz_counts, dtype=torch.int64)
    nz_col_indices = torch.cat(nz_col_indices, dim=0).to(dtype=torch.uint32) if len(nz_col_indices) > 0 else torch.tensor([], dtype=torch.uint32)
    
    return nz_counts, nz_col_indices

def check_correctness(sparsity: float, seed: int) -> None:
    """测试算子的正确性"""
    print("=" * 60)
    print("测试1: 正确性验证")
    print("=" * 60)

    load_sve_sparse_gemm_extension(verbose=False)
    kernel = SVESparseGEMMKernel.initialize(name="sve_sparse_gemm", target="CPU")
    op = kernel.operator(compiled=True)

    # 创建测试数据
    M, K, N = 10, 8, 6
    activation = _make_random_sparse_activation(M, K, sparsity=sparsity, seed=seed)
    weight = torch.randn(K, N, dtype=torch.float32)

    # 获取稀疏索引
    nz_counts, nz_col_indices = _get_sparse_indices_uint32(activation)
    
    # 调用算子
    result_sve = op(activation, weight, nz_counts, nz_col_indices)

    # 参考计算：直接使用 PyTorch 的 matmul
    result_ref = torch.matmul(activation, weight)

    # 比较结果
    max_diff = torch.max(torch.abs(result_sve - result_ref)).item()
    mean_diff = torch.mean(torch.abs(result_sve - result_ref)).item()
    print(f"最大误差: {max_diff:.6e}")
    print(f"平均误差: {mean_diff:.6e}")
    print(f"稀疏度(sparsity): {sparsity:.3f}")
    print(f"总非零元素数: {int(nz_col_indices.numel())}/{M*K} ({100*nz_col_indices.numel()/(M*K):.1f}%)")

    if torch.allclose(result_sve, result_ref, rtol=1e-4, atol=1e-5):
        print("✅ 正确性测试通过")
    else:
        print("❌ 正确性测试失败")
        print(f"SVE结果:\n{result_sve}")
        print(f"参考结果:\n{result_ref}")
        print(f"差异:\n{result_sve - result_ref}")


def test_sparse_pattern() -> None:
    """测试特定稀疏模式"""
    print("\n" + "=" * 60)
    print("测试2: 稀疏模式验证")
    print("=" * 60)

    kernel = SVESparseGEMMKernel.initialize(name="sve_sparse_gemm", target="CPU")
    op = kernel.operator(compiled=True)

    M, K, N = 5, 10, 4
    activation = torch.zeros(M, K, dtype=torch.float32)
    weight = torch.randn(K, N, dtype=torch.float32)

    # 设置稀疏模式：每行不同的非零模式
    activation[0, [0, 2, 4]] = torch.randn(3)
    activation[1, [1, 3, 5, 7]] = torch.randn(4)
    activation[2, [0]] = torch.randn(1)
    # row 3 全零
    activation[4, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] = torch.randn(10)  # 全非零

    nz_counts, nz_col_indices = _get_sparse_indices_uint32(activation)
    result_sve = op(activation, weight, nz_counts, nz_col_indices)

    # 参考计算
    result_ref = torch.matmul(activation, weight)

    if torch.allclose(result_sve, result_ref, rtol=1e-4, atol=1e-5):
        print("✅ 稀疏模式测试通过")
    else:
        print("❌ 稀疏模式测试失败")
        print(f"最大误差: {torch.max(torch.abs(result_sve - result_ref)).item():.6e}")


def test_edge_cases() -> None:
    """测试边界情况"""
    print("\n" + "=" * 60)
    print("测试3: 边界情况")
    print("=" * 60)

    kernel = SVESparseGEMMKernel.initialize(name="sve_sparse_gemm", target="CPU")
    op = kernel.operator(compiled=True)

    # 测试1: 单行单元素
    print("测试3.1: 单行单元素")
    activation = torch.zeros(1, 5, dtype=torch.float32)
    weight = torch.randn(5, 4, dtype=torch.float32)
    activation[0, 2] = 1.5
    nz_counts, nz_col_indices = _get_sparse_indices_uint32(activation)
    result = op(activation, weight, nz_counts, nz_col_indices)
    expected = torch.matmul(activation, weight)
    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-5), "单行单元素测试失败"
    print("  ✅ 通过")

    # 测试2: 全零矩阵
    print("测试3.2: 全零矩阵")
    activation = torch.zeros(3, 4, dtype=torch.float32)
    weight = torch.randn(4, 5, dtype=torch.float32)
    nz_counts, nz_col_indices = _get_sparse_indices_uint32(activation)
    result = op(activation, weight, nz_counts, nz_col_indices)
    expected = torch.zeros(3, 5, dtype=torch.float32)
    assert torch.allclose(result, expected), "全零矩阵测试失败"
    print("  ✅ 通过")

    # 测试3: 全非零矩阵
    print("测试3.3: 全非零矩阵")
    activation = torch.randn(4, 6, dtype=torch.float32)
    weight = torch.randn(6, 8, dtype=torch.float32)
    nz_counts, nz_col_indices = _get_sparse_indices_uint32(activation)
    result = op(activation, weight, nz_counts, nz_col_indices)
    expected = torch.matmul(activation, weight)
    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-5), "全非零矩阵测试失败"
    print("  ✅ 通过")

    # 测试4: 不同行具有不同的非零元素数
    print("测试3.4: 不同行具有不同的非零元素数")
    activation = torch.zeros(5, 8, dtype=torch.float32)
    weight = torch.randn(8, 6, dtype=torch.float32)
    activation[0, :1] = torch.randn(1)  # 1 个非零
    activation[1, :3] = torch.randn(3)  # 3 个非零
    activation[2, :] = torch.randn(8)   # 8 个非零
    activation[3, :0] = torch.randn(0)  # 0 个非零
    activation[4, :5] = torch.randn(5)  # 5 个非零
    nz_counts, nz_col_indices = _get_sparse_indices_uint32(activation)
    result = op(activation, weight, nz_counts, nz_col_indices)
    expected = torch.matmul(activation, weight)
    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-5), "不同非零数测试失败"
    print("  ✅ 通过")


def benchmark_performance(sparsity: float, seed: int) -> None:
    """性能测试"""
    print("\n" + "=" * 60)
    print("测试4: 性能测试")
    print("=" * 60)
    # torch.set_num_threads(1)

    kernel = SVESparseGEMMKernel.initialize(name="sve_sparse_gemm", target="CPU")
    op = kernel.operator(compiled=True)

    # 创建较大的测试数据（模拟实际场景）
    M, K, N = 8, 4096, 11008
    activation = _make_random_sparse_activation(M, K, sparsity=sparsity, seed=seed)
    weight = torch.randn(K, N, dtype=torch.float32)

    nz_counts, nz_col_indices = _get_sparse_indices_uint32(activation)

    # 测试SVE算子性能
    def sve_fn():
        # nz_counts, nz_col_indices = _get_sparse_indices(activation)
        return op(activation, weight, nz_counts, nz_col_indices)

    lat_sve = measure_latency(sve_fn, warmup=10, iters=10)
    print(f"⏱️  SVE GEMM 算子平均延迟: {lat_sve:.4f} ms")
    print(f"   输入形状: activation={activation.shape}, weight={weight.shape}")
    print(f"   稀疏度(sparsity): {sparsity:.3f}")
    # print(f"   非零元素数: {nnz}/{M*K} ({100*nnz/(M*K):.1f}%)")

    # 对比PyTorch标准实现：直接 matmul
    weight = torch.randn(N, K, dtype=torch.float32)

    def pytorch_dense_fn():
        return torch.matmul(activation, weight.T)


    lat_pytorch_dense = measure_latency(pytorch_dense_fn, warmup=10, iters=10)
    print(f"⏱️  PyTorch 稠密 matmul 平均延迟: {lat_pytorch_dense:.4f} ms")
    if lat_sve > 0:
        print(f"   加速比: {lat_pytorch_dense/lat_sve:.2f}x")

    # 对比PyTorch稀疏实现：to_sparse_csr + sparse.mm
    def pytorch_sparse_fn():
        sp_act = activation.to_sparse_csr()
        return torch.sparse.mm(sp_act, weight.T)

    lat_pytorch_sparse = measure_latency(pytorch_sparse_fn, warmup=10, iters=10)
    print(f"⏱️  PyTorch 稀疏 CSR + sparse.mm 平均延迟: {lat_pytorch_sparse:.4f} ms")
    if lat_sve > 0:
        print(f"   加速比: {lat_pytorch_sparse/lat_sve:.2f}x")


def test_direct_torch_ops() -> None:
    """测试直接使用 torch.ops 调用"""
    print("\n" + "=" * 60)
    print("测试5: 直接使用 torch.ops 调用")
    print("=" * 60)

    load_sve_sparse_gemm_extension(verbose=False)

    # 直接使用 torch.ops 调用
    M, K, N = 5, 8, 4
    activation = _make_random_sparse_activation(M, K, sparsity=0.9, seed=0)
    weight = torch.randn(K, N, dtype=torch.float32)

    nz_counts, nz_col_indices = _get_sparse_indices_uint32(activation)
    
    result_direct = torch.ops.teal.sve_sparse_gemm(
        activation, weight, nz_counts, nz_col_indices
    )

    # 参考计算
    result_ref = torch.matmul(activation, weight)

    if torch.allclose(result_direct, result_ref, rtol=1e-4, atol=1e-5):
        print("✅ torch.ops 直接调用测试通过")
    else:
        print("❌ torch.ops 直接调用测试失败")
        print(f"最大误差: {torch.max(torch.abs(result_direct - result_ref)).item():.6e}")


def main() -> None:
    """运行所有测试"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparsity", type=float, default=0.95, help="activation 置零比例(0~1)")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    args = parser.parse_args()

    try:
        check_correctness(sparsity=args.sparsity, seed=args.seed)
        test_sparse_pattern()
        test_edge_cases()
        test_direct_torch_ops()
        benchmark_performance(sparsity=args.sparsity, seed=args.seed + 1)
        print("\n" + "=" * 60)
        print("✅ 所有测试完成")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
