"""
ARM SVE CSR 行扫描算子的正确性与性能测试。

运行方式:
    python -m scripts.test_row_scan_sve_csr
"""

from __future__ import annotations

import argparse
import torch

from kernels.sve_sparse_gemm import dense_to_csr_sve2, dense_to_csr_omp, measure_latency


def reference(act: torch.Tensor, thr: float):
    """参考实现：使用 Python 循环计算每行的非零元素索引和值"""
    M, K = act.shape
    ref_row_nnz = []
    ref_indices = []
    ref_values = []
    for m in range(M):
        idx = []
        vals = []
        for k in range(K):
            if abs(float(act[m, k])) >= thr:
                idx.append(k)
                vals.append(float(act[m, k]))
        ref_row_nnz.append(len(idx))
        ref_indices.append(idx)
        ref_values.append(vals)
    return ref_row_nnz, ref_indices, ref_values


def baseline_dense_to_csr(activation: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    使用 PyTorch 内置 API 直接转换为 CSR 格式。
    
    直接使用 .to_sparse_csr() API：
    1. 创建 mask（绝对值 >= threshold）
    2. 创建稀疏 COO 张量
    3. 调用 .to_sparse_csr() 转换为 CSR 格式
    4. 提取 CSR 组件
    
    Returns:
        row_offsets: int64 [M+1]，前缀和偏移量
        col_idx: uint32，列索引向量
        values: float32，非零元素值
    """
    M, K = activation.shape
    
    # 创建 mask：绝对值 >= threshold 的元素
    mask = torch.abs(activation) >= threshold
    
    # 找到非零元素的坐标和值
    coo_indices = torch.nonzero(mask, as_tuple=False).t()  # [2, nnz]
    values_coo = activation[mask]  # [nnz]
    
    # 创建稀疏 COO 张量，然后直接调用 .to_sparse_csr()
    sparse_coo = torch.sparse_coo_tensor(coo_indices, values_coo, (M, K))
    sparse_csr = sparse_coo.to_sparse_csr()
    
    # 提取 CSR 格式的组件
    row_offsets = sparse_csr.crow_indices().to(torch.int64)  # [M+1]
    col_idx = sparse_csr.col_indices().to(torch.uint32)  # [nnz]
    values = sparse_csr.values()  # [nnz]
    
    return row_offsets, col_idx, values


def check_correctness(M: int, K: int, threshold: float, seed: int) -> None:
    """测试算子的正确性"""
    print("=" * 60)
    print("测试1: 正确性验证")
    print("=" * 60)

    torch.manual_seed(seed)
    act = torch.rand(M, K, dtype=torch.float32, device="cpu").contiguous()

    # SVE 实现
    row_offsets_sve, col_idx_sve, values_sve = dense_to_csr_sve2(act, threshold, verbose=False)

    # Baseline 实现（PyTorch）
    row_offsets_baseline, col_idx_baseline, values_baseline = baseline_dense_to_csr(act, threshold)

    # OpenMP baseline 实现
    row_offsets_omp, col_idx_omp, values_omp = dense_to_csr_omp(act, threshold, verbose=False)

    # ---- correctness checks ----
    ref_row_nnz, ref_indices, ref_values = reference(act, threshold)

    # row_offsets prefix sum check
    assert row_offsets_sve.numel() == M + 1, f"row_offsets 长度应为 {M + 1}，实际为 {row_offsets_sve.numel()}"
    assert int(row_offsets_sve[0].item()) == 0, "row_offsets[0] 应为 0"
    assert int(row_offsets_sve[M].item()) == col_idx_sve.numel(), \
        f"row_offsets[M] 应为 {col_idx_sve.numel()}，实际为 {int(row_offsets_sve[M].item())}"
    assert col_idx_sve.numel() == values_sve.numel(), \
        f"col_idx 和 values 长度应相等，实际 col_idx={col_idx_sve.numel()}, values={values_sve.numel()}"

    # per-row slice check (与 reference 比较)
    for m in range(M):
        s = int(row_offsets_sve[m].item())
        e = int(row_offsets_sve[m + 1].item())
        got_idx = col_idx_sve[s:e].tolist()
        got_vals = values_sve[s:e].tolist()
        exp_idx = ref_indices[m]
        exp_vals = ref_values[m]
        
        if got_idx != exp_idx:
            raise AssertionError(
                f"Row {m} col_idx mismatch: got {got_idx[:16]}... len={len(got_idx)} vs exp {exp_idx[:16]}... len={len(exp_idx)}"
            )
        
        # 比较值（允许浮点误差）
        if len(got_vals) != len(exp_vals):
            raise AssertionError(
                f"Row {m} values length mismatch: got {len(got_vals)} vs exp {len(exp_vals)}"
            )
        for i, (got_val, exp_val) in enumerate(zip(got_vals, exp_vals)):
            if abs(got_val - exp_val) > 1e-5:
                raise AssertionError(
                    f"Row {m} value[{i}] mismatch: got {got_val} vs exp {exp_val}"
                )

    # 与 baseline 比较
    col_idx_sve_int64 = col_idx_sve.to(dtype=torch.int64)
    col_idx_baseline_int64 = col_idx_baseline.to(dtype=torch.int64)
    col_idx_omp_int64 = col_idx_omp.to(dtype=torch.int64)
    
    assert torch.equal(row_offsets_sve, row_offsets_baseline), \
        f"SVE 与 PyTorch baseline 的 row_offsets 不匹配"
    assert torch.equal(col_idx_sve_int64, col_idx_baseline_int64), \
        f"SVE 与 PyTorch baseline 的 col_idx 不匹配"
    assert torch.allclose(values_sve, values_baseline, rtol=1e-5, atol=1e-6), \
        f"SVE 与 PyTorch baseline 的 values 不匹配"

    # 与 OpenMP baseline 比较
    assert torch.equal(row_offsets_sve, row_offsets_omp), \
        f"SVE 与 OpenMP baseline 的 row_offsets 不匹配"
    assert torch.equal(col_idx_sve_int64, col_idx_omp_int64), \
        f"SVE 与 OpenMP baseline 的 col_idx 不匹配"
    assert torch.allclose(values_sve, values_omp, rtol=1e-5, atol=1e-6), \
        f"SVE 与 OpenMP baseline 的 values 不匹配"

    print(f"✅ 正确性测试通过")
    print(f"   总非零元素数: {col_idx_sve.numel()}/{M*K} ({100*col_idx_sve.numel()/(M*K):.1f}%)")
    print(f"   阈值(threshold): {threshold:.3f}")
    print(f"   ✅ SVE 结果与 PyTorch baseline 结果一致")
    print(f"   ✅ SVE 结果与 OpenMP baseline 结果一致")


def test_edge_cases() -> None:
    """测试边界情况"""
    print("\n" + "=" * 60)
    print("测试2: 边界情况")
    print("=" * 60)

    threshold = 0.25

    # 测试1: 全零矩阵
    print("测试2.1: 全零矩阵")
    M, K = 10, 20
    act = torch.zeros(M, K, dtype=torch.float32)
    row_offsets, col_idx, values = dense_to_csr_sve2(act, threshold, verbose=False)
    assert col_idx.numel() == 0, "全零矩阵的 col_idx 应为空"
    assert values.numel() == 0, "全零矩阵的 values 应为空"
    assert torch.all(row_offsets == 0), "全零矩阵的 row_offsets 应全为 0"
    print("  ✅ 通过")

    # 测试2: 全非零矩阵（所有元素都大于阈值）
    print("测试2.2: 全非零矩阵")
    M, K = 5, 10
    act = torch.ones(M, K, dtype=torch.float32) * (threshold + 0.1)
    row_offsets, col_idx, values = dense_to_csr_sve2(act, threshold, verbose=False)
    assert col_idx.numel() == M * K, f"全非零矩阵的非零元素数应为 {M*K}"
    assert values.numel() == M * K, f"全非零矩阵的值数量应为 {M*K}"
    assert row_offsets[-1].item() == M * K, f"row_offsets[-1] 应为 {M*K}"
    print("  ✅ 通过")

    # 测试3: 单行单元素
    print("测试2.3: 单行单元素")
    M, K = 1, 1
    act = torch.tensor([[threshold + 0.1]], dtype=torch.float32)
    row_offsets, col_idx, values = dense_to_csr_sve2(act, threshold, verbose=False)
    assert col_idx.numel() == 1, "单行单元素矩阵的非零元素数应为 1"
    assert values.numel() == 1, "单行单元素矩阵的值数量应为 1"
    assert col_idx[0].item() == 0, "列索引应为 0"
    assert abs(values[0].item() - (threshold + 0.1)) < 1e-5, "值应匹配"
    print("  ✅ 通过")

    # 测试4: 不同行具有不同的非零元素数
    print("测试2.4: 不同行具有不同的非零元素数")
    M, K = 5, 10
    act = torch.zeros(M, K, dtype=torch.float32)
    act[0, :1] = threshold + 0.1  # 1 个非零
    act[1, :3] = threshold + 0.1  # 3 个非零
    act[2, :] = threshold + 0.1   # 10 个非零
    act[3, :0] = threshold + 0.1  # 0 个非零（全零行）
    act[4, :5] = threshold + 0.1  # 5 个非零
    row_offsets, col_idx, values = dense_to_csr_sve2(act, threshold, verbose=False)
    ref_row_nnz, ref_indices, ref_values = reference(act, threshold)
    for m in range(M):
        s = int(row_offsets[m].item())
        e = int(row_offsets[m + 1].item())
        assert e - s == ref_row_nnz[m], f"行 {m} 的非零元素数不匹配"
    print("  ✅ 通过")


def benchmark_performance(M: int, K: int, threshold: float, seed: int) -> None:
    """性能测试"""
    print("\n" + "=" * 60)
    print("测试3: 性能测试")
    print("=" * 60)

    torch.manual_seed(seed)
    act = torch.rand(M, K, dtype=torch.float32, device="cpu").contiguous()

    # 测试 SVE 算子性能
    def sve_fn():
        return dense_to_csr_sve2(act, threshold, verbose=False)

    lat_sve = measure_latency(sve_fn, warmup=10, iters=1000)
    print(f"⏱️  SVE dense_to_csr_sve2 算子平均延迟: {lat_sve:.4f} ms")
    print(f"   输入形状: activation={act.shape}")
    print(f"   阈值(threshold): {threshold:.3f}")

    # Baseline 实现性能（PyTorch）
    def baseline_fn():
        return baseline_dense_to_csr(act, threshold)

    lat_baseline = measure_latency(baseline_fn, warmup=10, iters=1000)
    print(f"⏱️  Baseline (PyTorch) 实现平均延迟: {lat_baseline:.4f} ms")

    # OpenMP baseline 实现性能
    def omp_fn():
        return dense_to_csr_omp(act, threshold, verbose=False)

    lat_omp = measure_latency(omp_fn, warmup=10, iters=1000)
    print(f"⏱️  Baseline (OpenMP) 实现平均延迟: {lat_omp:.4f} ms")

    if lat_baseline > 0:
        speedup_pytorch = lat_baseline / lat_sve
        print(f"   相对 PyTorch baseline 的加速比: {speedup_pytorch:.2f}x" + 
              (f" (SVE 快 {speedup_pytorch:.2f}x)" if speedup_pytorch > 1.0 
               else f" (SVE 慢 {1.0/speedup_pytorch:.2f}x)"))
    else:
        print(f"   ⚠️  无法计算加速比（PyTorch baseline 延迟为 0）")

    if lat_omp > 0:
        speedup_omp = lat_omp / lat_sve
        print(f"   相对 OpenMP baseline 的加速比: {speedup_omp:.2f}x" + 
              (f" (SVE 快 {speedup_omp:.2f}x)" if speedup_omp > 1.0 
               else f" (SVE 慢 {1.0/speedup_omp:.2f}x)"))
    else:
        print(f"   ⚠️  无法计算加速比（OpenMP baseline 延迟为 0）")


def main() -> None:
    """运行所有测试"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=128, help="activation 行数")
    parser.add_argument("--K", type=int, default=4096, help="activation 列数")
    parser.add_argument("--threshold", type=float, default=0.6, help="阈值")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    args = parser.parse_args()

    try:
        check_correctness(M=args.M, K=args.K, threshold=args.threshold, seed=args.seed)
        test_edge_cases()
        benchmark_performance(M=args.M, K=args.K, threshold=args.threshold, seed=args.seed + 1)
        print("\n" + "=" * 60)
        print("✅ 所有测试完成")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
