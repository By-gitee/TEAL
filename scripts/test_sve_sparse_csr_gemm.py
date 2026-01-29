"""
ARM SVE CSR（连续 load activation）稀疏 GEMM 算子的正确性与性能测试。

测试输出与测试方式对齐 `scripts/test_sve_sparse_gemm.py`。

运行方式:
    python -m scripts.test_sve_sparse_csr_gemm
"""

from __future__ import annotations

import argparse
import torch

from kernels.sve_sparse_gemm import (
    SVESparseCSRGEMMKernel,
    load_sve_sparse_gemm_extension,
    measure_latency,
    sve_sparse_csr_gemm,
    row_scan_sve,
)


def _make_random_sparse_activation(
    M: int,
    K: int,
    seed: int,
) -> torch.Tensor:
    """
    生成随机 activation 矩阵（float32）。
    注意：此函数仅生成初始矩阵，稀疏化处理在 _get_sparse_indices 函数中通过 threshold 完成。
    """
    g = torch.Generator()
    g.manual_seed(seed)
    x = torch.randn(M, K, dtype=torch.float32, generator=g)
    return x


def _get_sparse_indices_uint32(activation: torch.Tensor, threshold: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从 activation 中提取大于 threshold 的元素的索引信息。
    
    Args:
        activation: 输入的 activation 矩阵
        threshold: 阈值，大于此值的元素将被保留
    
    Returns:
        row_offsets: int64 [M+1]，前缀和偏移量，row_offsets[m] 表示第 m 行在 nz_col_indices 中的起始位置
        nz_col_indices: 扁平化的列索引向量（uint32）
    """
    M, K = activation.shape
    row_offsets = [0]
    nz_col_indices = []
    
    for m in range(M):
        row = activation[m]
        # 与 threshold 比较，保留大于 threshold 的元素
        nz_idx = torch.nonzero(row > threshold, as_tuple=False).flatten()
        nz_col_indices.append(nz_idx.to(dtype=torch.uint32))
        row_offsets.append(row_offsets[-1] + len(nz_idx))
    
    row_offsets_t = torch.tensor(row_offsets, dtype=torch.int64)
    nz_col_indices_t = torch.cat(nz_col_indices, dim=0) if len(nz_col_indices) > 0 else torch.tensor([], dtype=torch.uint32)
    
    return row_offsets_t, nz_col_indices_t


def _apply_threshold(activation: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """
    对 activation 矩阵应用阈值，大于 threshold 的值保留，其余置零。
    
    Args:
        activation: 输入的 activation 矩阵
        threshold: 阈值
    
    Returns:
        应用阈值后的 activation 矩阵
    """
    return torch.where(activation > threshold, activation, torch.zeros_like(activation))


def check_correctness(seed: int, threshold: float = 0.0) -> None:
    """测试算子的正确性"""
    print("=" * 60)
    print("测试1: 正确性验证")
    print("=" * 60)

    load_sve_sparse_gemm_extension(verbose=False)
    kernel = SVESparseCSRGEMMKernel.initialize(name="sve_sparse_csr_gemm", target="CPU")
    op = kernel.operator(compiled=True)

    # 创建测试数据
    M, K, N = 10, 8, 6
    activation = _make_random_sparse_activation(M, K, seed=seed)
    weight = torch.randn(K, N, dtype=torch.float32)

    # 获取稀疏索引（基于 threshold）
    row_offsets, nz_col_indices = _get_sparse_indices_uint32(activation, threshold=threshold)
    
    # 调用算子
    result_sve = op(activation, weight, row_offsets, nz_col_indices)

    # 参考计算：应用 threshold 后使用 PyTorch 的 matmul
    activation_thresholded = _apply_threshold(activation, threshold=threshold)
    result_ref = torch.matmul(activation_thresholded, weight)

    # 比较结果
    max_diff = torch.max(torch.abs(result_sve - result_ref)).item()
    mean_diff = torch.mean(torch.abs(result_sve - result_ref)).item()
    print(f"最大误差: {max_diff:.6e}")
    print(f"平均误差: {mean_diff:.6e}")
    print(f"阈值(threshold): {threshold:.6f}")
    print(f"总非零元素数: {int(nz_col_indices.numel())}/{M*K} ({100*nz_col_indices.numel()/(M*K):.1f}%)")

    if torch.allclose(result_sve, result_ref, rtol=1e-4, atol=1e-5):
        print("✅ 正确性测试通过")
    else:
        print("❌ 正确性测试失败")
        print(f"CSR-GEMM结果:\n{result_sve}")
        print(f"参考结果:\n{result_ref}")
        print(f"差异:\n{result_sve - result_ref}")


def test_sparse_pattern(threshold: float = 0.0) -> None:
    """测试特定稀疏模式"""
    print("\n" + "=" * 60)
    print("测试2: 稀疏模式验证")
    print("=" * 60)

    kernel = SVESparseCSRGEMMKernel.initialize(name="sve_sparse_csr_gemm", target="CPU")
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

    row_offsets, nz_col_indices = _get_sparse_indices_uint32(activation, threshold=threshold)
    result_sve = op(activation, weight, row_offsets, nz_col_indices)

    # 参考计算
    activation_thresholded = _apply_threshold(activation, threshold=threshold)
    result_ref = torch.matmul(activation_thresholded, weight)
    print(f"CSR-GEMM结果:\n{result_sve}")
    print(f"参考结果:\n{result_ref}")
    print(f"差异:\n{result_sve - result_ref}")

    if torch.allclose(result_sve, result_ref, rtol=1e-4, atol=1e-5):
        print("✅ 稀疏模式测试通过")
    else:
        print("❌ 稀疏模式测试失败")
        print(f"最大误差: {torch.max(torch.abs(result_sve - result_ref)).item():.6e}")


def test_edge_cases(threshold: float = 0.0) -> None:
    """测试边界情况"""
    print("\n" + "=" * 60)
    print("测试3: 边界情况")
    print("=" * 60)

    kernel = SVESparseCSRGEMMKernel.initialize(name="sve_sparse_csr_gemm", target="CPU")
    op = kernel.operator(compiled=True)

    # 测试1: 单行单元素
    print("测试3.1: 单行单元素")
    activation = torch.zeros(1, 5, dtype=torch.float32)
    weight = torch.randn(5, 4, dtype=torch.float32)
    activation[0, 2] = 1.5
    row_offsets, nz_col_indices = _get_sparse_indices_uint32(activation, threshold=threshold)
    result = op(activation, weight, row_offsets, nz_col_indices)
    activation_thresholded = _apply_threshold(activation, threshold=threshold)
    expected = torch.matmul(activation_thresholded, weight)
    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-5), "单行单元素测试失败"
    print("  ✅ 通过")

    # 测试2: 全零矩阵
    print("测试3.2: 全零矩阵")
    activation = torch.zeros(3, 4, dtype=torch.float32)
    weight = torch.randn(4, 5, dtype=torch.float32)
    row_offsets, nz_col_indices = _get_sparse_indices_uint32(activation, threshold=threshold)
    result = op(activation, weight, row_offsets, nz_col_indices)
    activation_thresholded = _apply_threshold(activation, threshold=threshold)
    expected = torch.matmul(activation_thresholded, weight)
    assert torch.allclose(result, expected), "全零矩阵测试失败"
    print("  ✅ 通过")

    # 测试3: 全非零矩阵
    print("测试3.3: 全非零矩阵")
    activation = torch.randn(4, 6, dtype=torch.float32)
    weight = torch.randn(6, 8, dtype=torch.float32)
    row_offsets, nz_col_indices = _get_sparse_indices_uint32(activation, threshold=threshold)
    result = op(activation, weight, row_offsets, nz_col_indices)
    activation_thresholded = _apply_threshold(activation, threshold=threshold)
    expected = torch.matmul(activation_thresholded, weight)
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
    row_offsets, nz_col_indices = _get_sparse_indices_uint32(activation, threshold=threshold)
    result = op(activation, weight, row_offsets, nz_col_indices)
    activation_thresholded = _apply_threshold(activation, threshold=threshold)
    expected = torch.matmul(activation_thresholded, weight)
    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-5), "不同非零数测试失败"
    print("  ✅ 通过")


def benchmark_performance(seed: int, threshold: float = 0.0) -> None:
    """性能测试"""
    print("\n" + "=" * 60)
    print("测试4: 性能测试")
    print("=" * 60)
    # torch.set_num_threads(1)

    kernel = SVESparseCSRGEMMKernel.initialize(name="sve_sparse_csr_gemm", target="CPU")
    op = kernel.operator(compiled=True)

    # 创建较大的测试数据（模拟实际场景）
    M, K, N = 1, 4096, 11008
    activation = _make_random_sparse_activation(M, K, seed=seed)
    weight = torch.randn(K, N, dtype=torch.float32)

    # 测试 CSR-GEMM 算子性能
    def csr_fn():
        row_offsets, nz_col_indices = _get_sparse_indices_uint32(activation, threshold=threshold)
        return op(activation, weight, row_offsets, nz_col_indices)

    lat_csr = measure_latency(csr_fn, warmup=10, iters=100)
    print(f"⏱️  CSR(连续 load) GEMM 算子平均延迟: {lat_csr:.4f} ms")
    print(f"   输入形状: activation={activation.shape}, weight={weight.shape}")
    print(f"   阈值(threshold): {threshold:.6f}")

    # 对比PyTorch标准实现：应用 threshold 后 matmul
    torch.backends.mkldnn.enabled = True

    def pytorch_dense_fn():
        activation_thresholded = _apply_threshold(activation, threshold=threshold)
        return torch.matmul(activation_thresholded, weight)

    lat_pytorch_dense = measure_latency(pytorch_dense_fn, warmup=10, iters=100)
    print(f"⏱️  PyTorch 稠密 matmul 平均延迟: {lat_pytorch_dense:.4f} ms")
    if lat_csr > 0:
        print(f"   加速比: {lat_pytorch_dense/lat_csr:.2f}x")

    # 对比PyTorch稀疏实现：to_sparse_csr + sparse.mm
    def pytorch_sparse_fn():
        activation_thresholded = _apply_threshold(activation, threshold=threshold)
        sp_act = activation_thresholded.to_sparse_csr()
        return torch.sparse.mm(sp_act, weight)

    lat_pytorch_sparse = measure_latency(pytorch_sparse_fn, warmup=10, iters=100)
    print(f"⏱️  PyTorch 稀疏 CSR + sparse.mm 平均延迟: {lat_pytorch_sparse:.4f} ms")
    if lat_csr > 0:
        print(f"   加速比: {lat_pytorch_sparse/lat_csr:.2f}x")

    
    # 对比PyTorch稀疏实现：to_sparse_csc + sparse.mm
    def pytorch_sparse_csc_fn():
        activation_thresholded = _apply_threshold(activation, threshold=threshold)
        sp_act = activation_thresholded.to_sparse_csc()
        return torch.sparse.mm(sp_act, weight)

    lat_pytorch_sparse_csc = measure_latency(pytorch_sparse_csc_fn, warmup=10, iters=100)
    print(f"⏱️  PyTorch 稀疏 CSC + sparse.mm 平均延迟: {lat_pytorch_sparse_csc:.4f} ms")
    if lat_csr > 0:
        print(f"   加速比: {lat_pytorch_sparse_csc/lat_csr:.2f}x")

    # 正确性验证
    activation_thresholded = _apply_threshold(activation, threshold=threshold)
    if torch.allclose(csr_fn(), torch.matmul(activation_thresholded, weight), rtol=1e-3, atol=1e-3):
        print("✅ CSR GEMM 算子正确性测试通过")
    else:
        print("❌ CSR GEMM 算子正确性测试失败")
        print(f"CSR结果:\n{csr_fn()}")
        print(f"参考结果:\n{torch.matmul(activation_thresholded, weight)}")
        print(f"差异:\n{csr_fn() - torch.matmul(activation_thresholded, weight)}")


def test_direct_torch_ops(threshold: float = 0.0) -> None:
    """测试直接使用 torch.ops 调用"""
    print("\n" + "=" * 60)
    print("测试5: 直接使用 torch.ops 调用")
    print("=" * 60)

    load_sve_sparse_gemm_extension(verbose=False)

    # 直接使用 torch.ops 调用
    M, K, N = 5, 8, 4
    activation = _make_random_sparse_activation(M, K, seed=0)
    weight = torch.randn(K, N, dtype=torch.float32)

    row_offsets, nz_col_indices = _get_sparse_indices_uint32(activation, threshold=threshold)
    
    result_direct = torch.ops.teal.sve_sparse_csr_gemm(
        activation, weight, row_offsets, nz_col_indices
    )

    # 参考计算：应用 threshold 后 matmul
    activation_thresholded = _apply_threshold(activation, threshold=threshold)
    result_ref = torch.matmul(activation_thresholded, weight)

    if torch.allclose(result_direct, result_ref, rtol=1e-4, atol=1e-5):
        print("✅ torch.ops 直接调用测试通过")
    else:
        print("❌ torch.ops 直接调用测试失败")
        print(f"最大误差: {torch.max(torch.abs(result_direct - result_ref)).item():.6e}")


def main() -> None:
    """运行所有测试"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--threshold", type=float, default=0.8, help="阈值，大于此值的元素将被保留")
    args = parser.parse_args()

    try:
        check_correctness(seed=args.seed, threshold=args.threshold)
        test_sparse_pattern(threshold=args.threshold)
        test_edge_cases(threshold=args.threshold)
        test_direct_torch_ops(threshold=args.threshold)
        benchmark_performance(seed=args.seed + 1, threshold=args.threshold)
        print("\n" + "=" * 60)
        print("✅ 所有测试完成")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
