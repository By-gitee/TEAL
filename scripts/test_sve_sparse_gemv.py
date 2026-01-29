"""
ARM SVE 稀疏 GEMV 算子的正确性与性能测试。

运行方式:
    python -m scripts.test_sve_sparse_gemv
"""

from __future__ import annotations

import argparse
import torch

from kernels.sve_sparse_gemm import (
    SVESparseGEMVKernel,
    load_sve_sparse_gemm_extension,
    measure_latency,
)


def _make_random_sparse_activation(
    M: int,
    K: int,
    seed: int,
) -> torch.Tensor:
    """
    生成随机 activation 矩阵（float32）。
    注意：此函数仅生成初始矩阵，稀疏化处理在 _get_nz_col_index_from_row 函数中通过 threshold 完成。
    """
    g = torch.Generator()
    g.manual_seed(seed)
    x = torch.randn(M, K, dtype=torch.float32, generator=g)
    return x


def _get_nz_col_index_from_row_uint32(activation: torch.Tensor, nz_row: int, threshold: float = 0.0) -> torch.Tensor:
    """
    从 activation 的指定行中提取大于 threshold 的元素的列索引。
    
    Args:
        activation: 输入的 activation 矩阵
        nz_row: 行索引
        threshold: 阈值，大于此值的元素将被保留
    
    Returns:
        列索引向量（uint32）
    """
    row = activation[nz_row]
    idx = torch.nonzero(row > threshold, as_tuple=False).flatten()
    return idx.to(dtype=torch.uint32)

def _get_nz_col_index_from_row_int32(activation: torch.Tensor, nz_row: int, threshold: float = 0.0) -> torch.Tensor:
    """
    从 activation 的指定行中提取大于 threshold 的元素的列索引。
    
    Args:
        activation: 输入的 activation 矩阵
        nz_row: 行索引
        threshold: 阈值，大于此值的元素将被保留
    
    Returns:
        列索引向量（int32）
    """
    row = activation[nz_row]
    idx = torch.nonzero(row > threshold, as_tuple=False).flatten()
    return idx.to(dtype=torch.int32)

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

def _mean_op_output(
    op,
    activation: torch.Tensor,
    weight: torch.Tensor,
    nz_row: int,
    nz_col_index: torch.Tensor,
    repeats: int,
) -> torch.Tensor:
    outs = []
    for _ in range(repeats):
        outs.append(op(activation, weight, nz_row, nz_col_index))
    return torch.stack(outs, dim=0).mean(dim=0)


def check_correctness(seed: int, threshold: float = 0.0, repeats: int = 5) -> None:
    """测试算子的正确性"""
    print("=" * 60)
    print("测试1: 正确性验证")
    print("=" * 60)

    load_sve_sparse_gemm_extension(verbose=False)
    kernel = SVESparseGEMVKernel.initialize(name="sve_sparse_gemv", target="CPU")
    op = kernel.operator(compiled=True)

    # 创建测试数据
    M, K, N = 10, 8, 8
    activation = _make_random_sparse_activation(M, K, seed=seed)
    weight = torch.randn(K, N, dtype=torch.float32)

    # 选择一行和对应的非零列索引（基于 threshold）
    nz_row = 2
    nz_col_index = _get_nz_col_index_from_row_uint32(activation, nz_row, threshold=threshold)
    if nz_col_index.numel() == 0:
        # 避免全零导致的退化情况：强制保留一个非零位置
        nz_col_index = torch.tensor([0], dtype=torch.uint32)
        activation[nz_row, 0] = 1.0

    # 多次调用算子，结果取均值
    result_sve = _mean_op_output(op, activation, weight, nz_row, nz_col_index, repeats=repeats)

    # 参考计算：应用 threshold 后使用 PyTorch 的 matmul
    activation_thresholded = _apply_threshold(activation, threshold=threshold)
    act_row = activation_thresholded[nz_row]
    weight_rows = weight
    result_ref = torch.matmul(act_row, weight_rows)

    # 比较结果
    max_diff = torch.max(torch.abs(result_sve - result_ref)).item()
    print(f"最大误差: {max_diff:.6e}")
    print(f"阈值(threshold): {threshold:.6f}，该行非零数: {int(nz_col_index.numel())}/{K}，重复次数: {repeats}")

    if torch.allclose(result_sve, result_ref, rtol=1e-4, atol=1e-5):
        print("✅ 正确性测试通过")
    else:
        print("❌ 正确性测试失败")
        print(f"SVE结果: {result_sve}")
        print(f"参考结果: {result_ref}")
        print(f"差异: {result_sve - result_ref}")


def test_sparse_pattern(threshold: float = 0.0) -> None:
    """测试稀疏模式"""
    print("\n" + "=" * 60)
    print("测试2: 稀疏模式验证")
    print("=" * 60)

    kernel = SVESparseGEMVKernel.initialize(name="sve_sparse_gemv", target="CPU")
    op = kernel.operator(compiled=True)

    M, K, N = 5, 10, 4
    activation = torch.zeros(M, K, dtype=torch.float32)
    weight = torch.randn(K, N, dtype=torch.float32)

    # 设置稀疏模式：只有特定位置有非零值
    nz_row = 1
    nz_col_index = torch.tensor([0, 2, 4, 6, 8], dtype=torch.int32)
    activation[nz_row, nz_col_index] = torch.randn(len(nz_col_index))
    nz_col_index = nz_col_index.to(dtype=torch.uint32)

    result_sve = op(activation, weight, nz_row, nz_col_index)

    # 参考计算：应用 threshold 后 matmul
    activation_thresholded = _apply_threshold(activation, threshold=threshold)
    act_row = activation_thresholded[nz_row,:]
    weight_rows = weight
    result_ref = torch.matmul(act_row, weight_rows)

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

    kernel = SVESparseGEMVKernel.initialize(name="sve_sparse_gemv", target="CPU")
    op = kernel.operator(compiled=True)

    # 测试1: 单个非零元素
    print("测试3.1: 单个非零元素")
    activation = torch.zeros(3, 5, dtype=torch.float32)
    weight = torch.randn(5, 4, dtype=torch.float32)
    activation[1, 2] = 1.0
    nz_col_index = torch.tensor([2], dtype=torch.uint32)
    result = op(activation, weight, 1, nz_col_index)
    expected = weight[2, :]
    assert torch.allclose(result, expected), "单个元素测试失败"
    print("  ✅ 通过")

    # 测试2: 所有元素都是非零
    print("测试3.2: 所有元素都是非零")
    activation = torch.randn(2, 4, dtype=torch.float32)
    weight = torch.randn(4, 4, dtype=torch.float32)
    nz_col_index = torch.arange(4, dtype=torch.int32)
    nz_col_index = nz_col_index.to(dtype=torch.uint32)
    result = op(activation, weight, 0, nz_col_index)
    expected = torch.matmul(activation[0:1, :], weight).squeeze(0)
    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-5), "全非零测试失败"
    print("  ✅ 通过")

    # 测试3: 包含零值的非零索引
    print("测试3.3: 包含零值的非零索引（应该被过滤）")
    activation = torch.randn(2, 5, dtype=torch.float32)
    weight = torch.randn(5,4, dtype=torch.float32)
    activation[0, 1] = 0.0  # 设置为0
    activation[0, 3] = 0.0  # 设置为0
    nz_col_index = torch.tensor([0, 1, 2, 3, 4], dtype=torch.uint32)
    result = op(activation, weight, 0, nz_col_index)
    # 参考结果应该排除零值
    nz_col_index = nz_col_index.to(dtype=torch.int32)
    act_row = activation[0, nz_col_index]
    mask = act_row != 0.0
    valid_indices = nz_col_index[mask]
    if len(valid_indices) > 0:
        expected = torch.matmul(
            activation[0:1, valid_indices], weight[valid_indices, :]
        ).squeeze(0)
        assert torch.allclose(
            result, expected, rtol=1e-4, atol=1e-5
        ), "零值过滤测试失败"
    print("  ✅ 通过")


def benchmark_performance(seed: int, threshold: float = 0.0) -> None:
    """性能测试"""
    print("\n" + "=" * 60)
    print("测试4: 性能测试")
    print("=" * 60)
    # torch.set_num_threads(6)

    kernel = SVESparseGEMVKernel.initialize(name="sve_sparse_gemv", target="CPU")
    op = kernel.operator(compiled=True)

    # 创建较大的测试数据
    M, K, N = 1, 4096, 11008
    activation = _make_random_sparse_activation(M, K, seed=seed)
    weight = torch.randn(K, N, dtype=torch.float32)

    nz_row = 0
    # 测试SVE算子性能
    def sve_fn():
        nz_col_index = _get_nz_col_index_from_row_uint32(activation, nz_row, threshold=threshold)
        return op(activation, weight, nz_row, nz_col_index)

    lat_sve = measure_latency(sve_fn, warmup=10, iters=100)
    print(f"⏱️  SVE算子平均延迟: {lat_sve:.4f} ms")
    print(f"   输入形状: activation={activation.shape}, weight={weight.shape}")
    nz_col_index = _get_nz_col_index_from_row_uint32(activation, nz_row, threshold=threshold)
    nnz = int(nz_col_index.numel())
    print(f"   阈值(threshold): {threshold:.6f}")
    print(f"   非零元素数: {nnz}/{K} ({100*nnz/K:.1f}%)")

    # 对比PyTorch标准实现：应用 threshold 后 matmul
    def pytorch_fn0():
        act_row = activation[nz_row]
        # 将 threshold 比较和计算过程结合起来：大于 threshold 的值保留，其余置零后计算
        act_row_thresholded = torch.where(act_row > threshold, act_row, torch.zeros_like(act_row))
        return act_row_thresholded @ weight

    lat_pytorch = measure_latency(pytorch_fn0, warmup=10, iters=100)
    print(f"⏱️  PyTorch标准实现平均延迟（只加载对应元素，对应行版本）: {lat_pytorch:.4f} ms")
    if lat_sve > 0:
        print(f"   加速比: {lat_pytorch/lat_sve:.2f}x")


    def pytorch_fn1():
        activation_thresholded = _apply_threshold(activation, threshold=threshold)
        return activation_thresholded[nz_row] @ weight

    lat_pytorch = measure_latency(pytorch_fn1, warmup=10, iters=100)
    print(f"⏱️  PyTorch标准实现平均延迟（应用threshold后matmul版本）: {lat_pytorch:.4f} ms")
    if lat_sve > 0:
        print(f"   加速比: {lat_pytorch/lat_sve:.2f}x")

    # 常用稀疏基线：构造 1xK 的 sparse CSR，然后 sparse_mm 到 dense weight。
    # 注意：把 sparse 张量构造放到计时外，避免把构造开销混进 kernel 时间。
    
    # 对比PyTorch稀疏实现：to_sparse_csc + sparse.mm
    def pytorch_sparse_csc_fn():
        sp_act =  torch.where(activation > threshold, activation, torch.zeros_like(activation)).to_sparse_csc()
        return torch.sparse.mm(sp_act, weight)

    lat_pytorch_sparse_csc = measure_latency(pytorch_sparse_csc_fn, warmup=10, iters=100)
    print(f"⏱️  PyTorch 稀疏 CSC + sparse.mm 平均延迟: {lat_pytorch_sparse_csc:.4f} ms")
    if lat_sve > 0:
        print(f"   加速比: {lat_pytorch_sparse_csc/lat_sve:.2f}x")

    def pytorch_fn2():
        sp_row = torch.where(activation > threshold, activation, torch.zeros_like(activation)).to_sparse_csr()  # 将threshold比较和CSR转换结合
        return torch.sparse.mm(sp_row, weight).squeeze(0)

    lat_pytorch = measure_latency(pytorch_fn2, warmup=10, iters=100)
    print(f"⏱️  PyTorch稀疏基线平均延迟（sparse_csr + sparse.mm）: {lat_pytorch:.4f} ms")
    if lat_sve > 0:
        print(f"   加速比: {lat_pytorch/lat_sve:.2f}x")


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

    nz_row = 2
    nz_col_index = _get_nz_col_index_from_row_uint32(activation, nz_row, threshold=threshold)
    
    result_direct = torch.ops.teal.sve_sparse_gemv(
        activation, weight, nz_row, nz_col_index
    )

    # 参考计算：应用 threshold 后 matmul
    activation_thresholded = _apply_threshold(activation, threshold=threshold)
    act_row = activation_thresholded[nz_row]
    weight_rows = weight
    result_ref = torch.matmul(act_row, weight_rows)

    if torch.allclose(result_direct, result_ref, rtol=1e-4, atol=1e-5):
        print("✅ torch.ops 直接调用测试通过")
    else:
        print("❌ torch.ops 直接调用测试失败")
        print(result_direct)
        print(result_ref)
        print(
            f"最大误差: {torch.max(torch.abs(result_direct - result_ref)).item():.6e}"
        )


def main() -> None:
    """运行所有测试"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.8, help="阈值，大于此值的元素将被保留")
    parser.add_argument("--repeats", type=int, default=5, help="正确性测试重复调用次数并取均值")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    args = parser.parse_args()

    try:
        check_correctness(seed=args.seed, threshold=args.threshold, repeats=args.repeats)
        test_sparse_pattern(threshold=args.threshold)
        test_edge_cases()
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
