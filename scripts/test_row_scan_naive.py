"""
朴素版本行扫描算子的正确性与性能测试（使用 OpenMP 并行，但不使用 SVE 加速）。

运行方式:
    python -m scripts.test_row_scan_naive
"""

from __future__ import annotations

import argparse
import torch

from kernels.sve_sparse_gemm import row_scan_naive, measure_latency


def reference(act: torch.Tensor, thr: float):
    """参考实现：使用 Python 循环计算每行的非零元素索引"""
    M, K = act.shape
    ref_row_nnz = []
    ref_indices = []
    for m in range(M):
        idx = [k for k in range(K) if abs(float(act[m, k])) >= thr]
        ref_row_nnz.append(len(idx))
        ref_indices.append(idx)
    return ref_row_nnz, ref_indices


def baseline_get_sparse_indices(activation: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从稀疏 activation 中提取非零元素的索引信息（baseline 实现，使用 PyTorch 操作）。
    
    参照 test_sve_sparse_gemm.py 中的实现方式，但使用阈值而非简单的非零检测。
    
    Returns:
        nz_counts: 成对存储的 (row_idx, count)，长度为 2 * num_nz_rows
        nz_col_indices: 扁平化的列索引向量
        row_offsets: 每行的偏移量，长度为 M + 1
    """
    M, K = activation.shape
    nz_pairs = []
    nz_col_indices_list = []
    
    for m in range(M):
        row = activation[m]
        # 使用阈值检测：abs(row) >= threshold
        nz_mask = torch.abs(row) >= threshold
        nz_idx = torch.nonzero(nz_mask, as_tuple=False).flatten()
        if len(nz_idx) > 0:
            nz_pairs.extend([m, len(nz_idx)])
            nz_col_indices_list.append(nz_idx)
    
    nz_counts = torch.tensor(nz_pairs, dtype=torch.int64) if len(nz_pairs) > 0 else torch.tensor([], dtype=torch.int64)
    nz_col_indices = torch.cat(nz_col_indices_list, dim=0).to(dtype=torch.int64) if len(nz_col_indices_list) > 0 else torch.tensor([], dtype=torch.int64)
    
    # 计算 row_offsets
    row_offsets = torch.zeros(M + 1, dtype=torch.int64)
    offset = 0
    for m in range(M):
        row_offsets[m] = offset
        row = activation[m]
        nz_mask = torch.abs(row) >= threshold
        nz_count = torch.sum(nz_mask).item()
        offset += nz_count
    row_offsets[M] = offset
    
    return nz_counts, nz_col_indices, row_offsets


def get_sparse_indices_pytorch_style(activation: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从稀疏 activation 中提取非零元素的索引信息（PyTorch 风格实现，参照 test_sve_sparse_gemm.py）。
    
    使用 abs(row) >= threshold 进行阈值检测，返回三个值：nz_counts, nz_col_indices, row_offsets。
    
    Returns:
        nz_counts: 成对存储的 (row_idx, count)，长度为 2 * num_nz_rows
        nz_col_indices: 扁平化的列索引向量（uint32）
        row_offsets: 每行的偏移量，长度为 M + 1
    """
    M, K = activation.shape
    row_offsets = [0]
    nz_col_indices = []
    nz_pairs = []
    
    for m in range(M):
        row = activation[m]
        # 使用阈值检测：abs(row) >= threshold
        nz_mask = torch.abs(row) >= threshold
        nz_idx = torch.nonzero(nz_mask, as_tuple=False).flatten()
        nz_col_indices.append(nz_idx.to(dtype=torch.uint32))
        nnz = len(nz_idx)
        row_offsets.append(row_offsets[-1] + nnz)
        if nnz > 0:
            nz_pairs.extend([m, nnz])
    
    row_offsets_t = torch.tensor(row_offsets, dtype=torch.int64)
    nz_col_indices_t = torch.cat(nz_col_indices, dim=0) if len(nz_col_indices) > 0 else torch.tensor([], dtype=torch.uint32)
    nz_counts_t = torch.tensor(nz_pairs, dtype=torch.int64) if len(nz_pairs) > 0 else torch.tensor([], dtype=torch.int64)
    
    return nz_counts_t, nz_col_indices_t, row_offsets_t


def check_correctness(M: int, K: int, threshold: float, seed: int) -> None:
    """测试算子的正确性"""
    print("=" * 60)
    print("测试1: 正确性验证")
    print("=" * 60)

    torch.manual_seed(seed)
    act = torch.rand(M, K, dtype=torch.float32, device="cpu").contiguous()

    # Naive 实现
    nz_counts_naive, nz_col_indices_naive, row_offsets_naive = row_scan_naive(act, threshold, verbose=False)

    # Baseline 实现（PyTorch）
    nz_counts_baseline, nz_col_indices_baseline, row_offsets_baseline = baseline_get_sparse_indices(act, threshold)

    # ---- correctness checks ----
    ref_row_nnz, ref_indices = reference(act, threshold)

    # row_offsets prefix sum check
    assert row_offsets_naive.numel() == M + 1, f"row_offsets 长度应为 {M + 1}，实际为 {row_offsets_naive.numel()}"
    assert int(row_offsets_naive[0].item()) == 0, "row_offsets[0] 应为 0"
    assert int(row_offsets_naive[M].item()) == nz_col_indices_naive.numel(), \
        f"row_offsets[M] 应为 {nz_col_indices_naive.numel()}，实际为 {int(row_offsets_naive[M].item())}"

    # per-row slice check (与 reference 比较)
    for m in range(M):
        s = int(row_offsets_naive[m].item())
        e = int(row_offsets_naive[m + 1].item())
        got = nz_col_indices_naive[s:e].tolist()
        exp = ref_indices[m]
        if got != exp:
            raise AssertionError(
                f"Row {m} mismatch: got {got[:16]}... len={len(got)} vs exp {exp[:16]}... len={len(exp)}"
            )

    # nz_counts format check (only nnz>0, increasing rows)
    assert nz_counts_naive.numel() % 2 == 0, "nz_counts 长度应为偶数"
    pairs_naive = nz_counts_naive.view(-1, 2).tolist()
    last_row = -1
    for row, nnz in pairs_naive:
        assert nnz > 0, f"行 {row} 的非零元素数应为正数"
        assert row > last_row, f"行索引应递增，但 {row} <= {last_row}"
        assert nnz == ref_row_nnz[row], f"行 {row} 的非零元素数不匹配：期望 {ref_row_nnz[row]}，实际 {nnz}"
        last_row = row

    # 与 baseline 比较（统一类型：Naive 返回 uint32，baseline 返回 int64）
    nz_col_indices_naive_int64 = nz_col_indices_naive.to(dtype=torch.int64)
    assert torch.equal(nz_counts_naive, nz_counts_baseline), \
        f"Naive 与 baseline 的 nz_counts 不匹配"
    assert torch.equal(nz_col_indices_naive_int64, nz_col_indices_baseline), \
        f"Naive 与 baseline 的 nz_col_indices 不匹配"
    assert torch.equal(row_offsets_naive, row_offsets_baseline), \
        f"Naive 与 baseline 的 row_offsets 不匹配"

    print(f"✅ 正确性测试通过")
    print(f"   总非零元素数: {nz_col_indices_naive.numel()}/{M*K} ({100*nz_col_indices_naive.numel()/(M*K):.1f}%)")
    print(f"   非零行数: {len(pairs_naive)}/{M} ({100*len(pairs_naive)/M:.1f}%)")
    print(f"   阈值(threshold): {threshold:.3f}")
    print(f"   ✅ Naive 结果与 baseline (PyTorch) 结果一致")


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
    nz_counts, nz_col_indices, row_offsets = row_scan_naive(act, threshold, verbose=False)
    assert nz_counts.numel() == 0, "全零矩阵的 nz_counts 应为空"
    assert nz_col_indices.numel() == 0, "全零矩阵的 nz_col_indices 应为空"
    assert torch.all(row_offsets == 0), "全零矩阵的 row_offsets 应全为 0"
    print("  ✅ 通过")

    # 测试2: 全非零矩阵（所有元素都大于阈值）
    print("测试2.2: 全非零矩阵")
    M, K = 5, 10
    act = torch.ones(M, K, dtype=torch.float32) * (threshold + 0.1)
    nz_counts, nz_col_indices, row_offsets = row_scan_naive(act, threshold, verbose=False)
    assert nz_col_indices.numel() == M * K, f"全非零矩阵的非零元素数应为 {M*K}"
    assert row_offsets[-1].item() == M * K, f"row_offsets[-1] 应为 {M*K}"
    print("  ✅ 通过")

    # 测试3: 单行单元素
    print("测试2.3: 单行单元素")
    M, K = 1, 1
    act = torch.tensor([[threshold + 0.1]], dtype=torch.float32)
    nz_counts, nz_col_indices, row_offsets = row_scan_naive(act, threshold, verbose=False)
    assert nz_col_indices.numel() == 1, "单行单元素矩阵的非零元素数应为 1"
    assert nz_col_indices[0].item() == 0, "列索引应为 0"
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
    nz_counts, nz_col_indices, row_offsets = row_scan_naive(act, threshold, verbose=False)
    ref_row_nnz, ref_indices = reference(act, threshold)
    pairs = nz_counts.view(-1, 2).tolist()
    for row, nnz in pairs:
        assert nnz == ref_row_nnz[row], f"行 {row} 的非零元素数不匹配"
    print("  ✅ 通过")


def benchmark_performance(M: int, K: int, threshold: float, seed: int) -> None:
    """性能测试"""
    print("\n" + "=" * 60)
    print("测试3: 性能测试")
    print("=" * 60)

    torch.manual_seed(seed)
    act = torch.rand(M, K, dtype=torch.float32, device="cpu").contiguous()

    # 测试 Naive 算子性能
    def naive_fn():
        return row_scan_naive(act, threshold, verbose=False)

    lat_naive = measure_latency(naive_fn, warmup=10, iters=1000000)
    print(f"⏱️  Naive row_scan 算子平均延迟: {lat_naive:.4f} ms")
    print(f"   输入形状: activation={act.shape}")
    print(f"   阈值(threshold): {threshold:.3f}")

    # # 获取实际非零元素数用于显示
    # _, nz_col_indices, _ = row_scan_naive(act, threshold, verbose=False)
    # print(f"   总非零元素数: {nz_col_indices.numel()}/{M*K} ({100*nz_col_indices.numel()/(M*K):.1f}%)")

    # # 测试 Baseline (PyTorch) 实现性能
    # def baseline_fn():
    #     return baseline_get_sparse_indices(act, threshold)

    # lat_baseline = measure_latency(baseline_fn, warmup=10, iters=50)
    # print(f"\n⏱️  Baseline (PyTorch) 实现平均延迟: {lat_baseline:.4f} ms")
    
    # # 计算加速比
    # if lat_baseline > 0:
    #     speedup = lat_baseline / lat_naive
    #     print(f"   加速比: {speedup:.2f}x")
    #     if speedup > 1.0:
    #         print(f"   ✅ Naive 实现比 Baseline 快 {speedup:.2f}x")
    #     else:
    #         print(f"   ⚠️  Naive 实现比 Baseline 慢 {1.0/speedup:.2f}x")
    # else:
    #     print(f"   ⚠️  无法计算加速比（baseline 延迟为 0）")


def benchmark_indices_generation_comparison(M: int, K: int, threshold: float, seed: int) -> None:
    """
    对比不同方法生成 nz_counts_t, nz_col_indices_t, row_offsets_t 的性能。
    
    对比的方法：
    1. row_scan_naive - 朴素并行实现（OpenMP，无 SVE）
    2. baseline_get_sparse_indices - Baseline PyTorch 实现
    3. get_sparse_indices_pytorch_style - PyTorch 风格实现（参照 test_sve_sparse_gemm.py）
    """
    print("\n" + "=" * 60)
    print("测试4: 索引生成方法性能对比")
    print("=" * 60)

    torch.manual_seed(seed)
    act = torch.rand(M, K, dtype=torch.float32, device="cpu").contiguous()

    # 获取实际非零元素数用于显示
    _, nz_col_indices_ref, _ = row_scan_naive(act, threshold, verbose=False)
    nnz = nz_col_indices_ref.numel()
    print(f"输入形状: activation={act.shape}")
    print(f"阈值(threshold): {threshold:.3f}")
    print(f"总非零元素数: {nnz}/{M*K} ({100*nnz/(M*K):.1f}%)")
    print()

    # 方法1: Naive 实现
    def naive_fn():
        return row_scan_naive(act, threshold, verbose=False)

    lat_naive = measure_latency(naive_fn, warmup=10, iters=10000000)
    print(f"⏱️  方法1 - Naive row_scan 实现（OpenMP 并行，无 SVE）:")
    print(f"   平均延迟: {lat_naive:.4f} ms")

    # 方法2: Baseline PyTorch 实现
    def baseline_fn():
        return baseline_get_sparse_indices(act, threshold)

    lat_baseline = measure_latency(baseline_fn, warmup=10, iters=50)
    print(f"\n⏱️  方法2 - Baseline PyTorch 实现:")
    print(f"   平均延迟: {lat_baseline:.4f} ms")
    if lat_naive > 0:
        speedup_vs_baseline = lat_baseline / lat_naive
        print(f"   相对 Naive 的加速比: {speedup_vs_baseline:.2f}x" + 
              (f" (Naive 快 {speedup_vs_baseline:.2f}x)" if speedup_vs_baseline > 1.0 
               else f" (Naive 慢 {1.0/speedup_vs_baseline:.2f}x)"))

    # 方法3: PyTorch 风格实现（参照 test_sve_sparse_gemm.py）
    def pytorch_style_fn():
        return get_sparse_indices_pytorch_style(act, threshold)

    lat_pytorch_style = measure_latency(pytorch_style_fn, warmup=10, iters=50)
    print(f"\n⏱️  方法3 - PyTorch 风格实现（参照 test_sve_sparse_gemm.py）:")
    print(f"   平均延迟: {lat_pytorch_style:.4f} ms")
    if lat_naive > 0:
        speedup_vs_pytorch_style = lat_pytorch_style / lat_naive
        print(f"   相对 Naive 的加速比: {speedup_vs_pytorch_style:.2f}x" + 
              (f" (Naive 快 {speedup_vs_pytorch_style:.2f}x)" if speedup_vs_pytorch_style > 1.0 
               else f" (Naive 慢 {1.0/speedup_vs_pytorch_style:.2f}x)"))

    # 正确性验证：确保三种方法产生相同的结果
    print(f"\n📋 正确性验证:")
    nz_counts_naive, nz_col_indices_naive, row_offsets_naive = row_scan_naive(act, threshold, verbose=False)
    nz_counts_baseline, nz_col_indices_baseline, row_offsets_baseline = baseline_get_sparse_indices(act, threshold)
    nz_counts_pytorch, nz_col_indices_pytorch, row_offsets_pytorch = get_sparse_indices_pytorch_style(act, threshold)

    # 比较结果（注意 nz_col_indices 的类型可能不同：Naive 返回 uint32，baseline 返回 int64）
    nz_col_indices_naive_int64 = nz_col_indices_naive.to(dtype=torch.int64)
    nz_col_indices_pytorch_int64 = nz_col_indices_pytorch.to(dtype=torch.int64)

    match_naive_baseline = (
        torch.equal(nz_counts_naive, nz_counts_baseline) and
        torch.equal(nz_col_indices_naive_int64, nz_col_indices_baseline) and
        torch.equal(row_offsets_naive, row_offsets_baseline)
    )
    match_naive_pytorch = (
        torch.equal(nz_counts_naive, nz_counts_pytorch) and
        torch.equal(nz_col_indices_naive_int64, nz_col_indices_pytorch_int64) and
        torch.equal(row_offsets_naive, row_offsets_pytorch)
    )

    if match_naive_baseline:
        print(f"   ✅ Naive 与 Baseline 结果一致")
    else:
        print(f"   ❌ Naive 与 Baseline 结果不一致")

    if match_naive_pytorch:
        print(f"   ✅ Naive 与 PyTorch 风格实现结果一致")
    else:
        print(f"   ❌ Naive 与 PyTorch 风格实现结果不一致")

    # 性能总结
    print(f"\n📊 性能总结:")
    latencies = [
        ("Naive row_scan (OpenMP)", lat_naive),
        ("Baseline PyTorch", lat_baseline),
        ("PyTorch 风格", lat_pytorch_style),
    ]
    latencies.sort(key=lambda x: x[1])
    fastest = latencies[0]
    print(f"   最快方法: {fastest[0]} ({fastest[1]:.4f} ms)")
    for name, lat in latencies[1:]:
        if fastest[1] > 0:
            speedup = lat / fastest[1]
            print(f"   {name}: {lat:.4f} ms (慢 {speedup:.2f}x)")


def main() -> None:
    """运行所有测试"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=256, help="activation 行数")
    parser.add_argument("--K", type=int, default=4096, help="activation 列数")
    parser.add_argument("--threshold", type=float, default=0.8, help="阈值")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    args = parser.parse_args()

    try:
        check_correctness(M=args.M, K=args.K, threshold=args.threshold, seed=args.seed)
        test_edge_cases()
        benchmark_performance(M=args.M, K=args.K, threshold=args.threshold, seed=args.seed + 1)
        # benchmark_indices_generation_comparison(M=args.M, K=args.K, threshold=args.threshold, seed=args.seed + 2)
        print("\n" + "=" * 60)
        print("✅ 所有测试完成")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
