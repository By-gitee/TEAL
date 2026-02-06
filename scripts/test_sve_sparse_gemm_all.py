"""
ARM SVE 稀疏 GEMM 算子的综合测试脚本。

本脚本测试不同稀疏格式的稀疏化算子和GEMM算子的组合（共17个）：

自定义 SVE 算子（13个）：
- iCSR 格式组合：thr_sparsify_to_icsr(_sve) + sparse_gemm_icsr(_sve_gather)
- CSR 格式组合：thr_sparsify_to_csr(_sve) + sparse_gemm_csr(_sve_gather)
- COO 格式组合：thr_sparsify_to_coo(_sve) + sparse_gemm_coo(_sve_gather)
- CSC 格式组合：thr_sparsify_to_csc + sparse_gemm_csc

PyTorch 参考实现（4个）：
- 稠密 matmul
- 稀疏 CSR + sparse.mm
- 稀疏 CSC + sparse.mm
- 选择性加载 weight 非零行 + matmul

测试内容：
1. 正确性验证：与PyTorch参考实现比较
2. 性能测试：测量每个组合的延迟
3. 加速比计算：计算相对于PyTorch稠密实现的加速比
4. 性能排名：找出最快的算子组合

运行方式:
    python -m scripts.test_sparse_formats
    python -m scripts.test_sparse_formats --threshold 0.8 --M 16 --K 512 --N 1024
"""

from __future__ import annotations

import argparse
import torch
import time
from typing import Any, Dict, List, Tuple

from kernels.cpp_sve_sparse_gemm import (
    # iCSR 算子
    SparseGEMMiCSRSVEGatherKernel,
    SparseGEMMICSRKernel,
    thr_sparsify_to_icsr,
    thr_sparsify_to_icsr_sve,
    # CSR 算子
    SparseGEMMCSRKernel,
    SparseGEMMCSRSVEGatherKernel,
    thr_sparsify_to_csr,
    thr_sparsify_to_csr_sve,
    # COO 算子
    SparseGEMMCOOKernel,
    SparseGEMMCOOSVEGatherKernel,
    thr_sparsify_to_coo,
    thr_sparsify_to_coo_sve,
    # CSC 算子
    SparseGEMMCSCKernel,
    thr_sparsify_to_csc,
    # 工具函数
    load_sve_sparse_gemm_extension,
)
from kernels.kernel_utils import measure_latency

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore


def _make_random_sparse_activation(
    M: int,
    K: int,
    seed: int,
) -> torch.Tensor:
    """生成随机 activation 矩阵（float32）。"""
    g = torch.Generator()
    g.manual_seed(seed)
    x = torch.rand(M, K, dtype=torch.float32, generator=g)
    return x


def _apply_threshold(activation: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """对 activation 矩阵应用阈值：abs(x) >= threshold 的值保留，其余置零。

    注意：cpp_sve_sparse_gemm 下的 thr_sparsify_to_* 系列算子使用的是 abs(x) >= thr 的判定；
    这里必须保持一致，否则会出现系统性正确性偏差（尤其是 activation 含负值时）。
    """
    return torch.where(activation.abs() >= threshold, activation, torch.zeros_like(activation))


def _print_ranked_latencies(
    title: str,
    latencies: List[Tuple[str, float]],
    baseline_latency: float | None = None,
) -> None:
    """按延迟从快到慢打印排名，可选打印相对 baseline 的加速比。"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if not latencies:
        print("（无数据）")
        return

    latencies = sorted(latencies, key=lambda x: x[1])
    if baseline_latency is not None and baseline_latency > 0:
        print(f"{'排名':<4} {'项目':<64} {'延迟(ms)':<12} {'加速比':<10}")
        print("-" * 94)
        for rank, (name, latency) in enumerate(latencies, 1):
            speedup = baseline_latency / latency if latency > 0 else 0.0
            print(f"{rank:2d}. {name:64s} {latency:8.4f} ms  {speedup:6.2f}x")
    else:
        print(f"{'排名':<4} {'项目':<72} {'延迟(ms)':<12}")
        print("-" * 92)
        for rank, (name, latency) in enumerate(latencies, 1):
            print(f"{rank:2d}. {name:72s} {latency:8.4f} ms")

    fastest_name, fastest_latency = latencies[0]
    print("\n" + "-" * 80)
    print(f"⚡ 最快: {fastest_name}")
    print(f"   延迟: {fastest_latency:.4f} ms")
    if baseline_latency is not None and baseline_latency > 0 and fastest_latency > 0:
        print(f"   相比 baseline 加速比: {baseline_latency/fastest_latency:.2f}x")


def _maybe_print_cpu_util(prefix: str, interval_s: float = 0.20) -> None:
    """打印 CPU 利用率（系统 + 当前进程），best-effort。

    依赖 psutil；若不可用则静默跳过（main 里会打印一次提示）。
    """
    if psutil is None:
        return

    try:
        proc = psutil.Process()
        # 先设置 system 基线，再用 proc 的阻塞采样得到同一时间窗的 system 利用率
        psutil.cpu_percent(interval=None)
        proc_cpu = proc.cpu_percent(interval=interval_s)
        sys_cpu = psutil.cpu_percent(interval=None)

        cpu_cnt = psutil.cpu_count(logical=True) or 1
        proc_cpu_norm = proc_cpu / cpu_cnt

        mem = proc.memory_info().rss / (1024 * 1024)
        threads = proc.num_threads()

        ts = time.strftime("%H:%M:%S")
        print(
            f"[CPU {ts}] {prefix} | sys={sys_cpu:5.1f}% | proc={proc_cpu:6.1f}% "
            f"(norm={proc_cpu_norm:5.1f}%) | rss={mem:7.1f} MB | thr={threads}"
        )
    except Exception:
        # 不让监控影响测试流程
        return


def _parse_selected_tests(raw_tests: List[str] | None) -> List[str]:
    """解析 --tests 参数，支持空格/逗号分隔与常见别名。

    允许项：
    - all
    - icsr / csr / coo / csc
    - pytorch
    - gemm-only
    - preprocess-only
    """
    if not raw_tests:
        raw_tests = ["all"]

    tokens: List[str] = []
    for item in raw_tests:
        for part in item.split(","):
            part = part.strip()
            if part:
                tokens.append(part)

    normalized: List[str] = []
    for t in tokens:
        tl = t.strip().lower().replace("_", "-")
        if tl in {"all"}:
            normalized.append("all")
            continue
        if tl in {"icsr", "i-csr", "i_csr"}:
            normalized.append("icsr")
            continue
        if tl in {"csr"}:
            normalized.append("csr")
            continue
        if tl in {"coo"}:
            normalized.append("coo")
            continue
        if tl in {"csc"}:
            normalized.append("csc")
            continue
        if tl in {"pytorch", "torch"}:
            normalized.append("pytorch")
            continue
        if tl in {"gemm-only", "gemmonly", "gemm", "core-gemm", "core-gemm-only"}:
            normalized.append("gemm-only")
            continue
        if tl in {"preprocess-only", "preprocessonly", "preprocess", "pre"}:
            normalized.append("preprocess-only")
            continue

        allowed = "all/icsr/csr/coo/csc/pytorch/gemm-only/preprocess-only"
        raise ValueError(f"--tests 包含未知项: {t!r}，允许项: {allowed}")

    if "all" in normalized:
        return ["icsr", "csr", "coo", "csc", "pytorch", "gemm-only", "preprocess-only"]

    # 去重并保持用户输入顺序
    seen = set()
    ordered_unique: List[str] = []
    for t in normalized:
        if t not in seen:
            seen.add(t)
            ordered_unique.append(t)
    return ordered_unique


def verify_correctness_flat(
    results: Dict[str, Tuple[torch.Tensor, float]],
    reference: torch.Tensor,
) -> None:
    """验证一组（扁平）结果的正确性（逐项与 reference 对比）。"""
    print("\n" + "=" * 80)
    print("正确性验证（扁平结果集）")
    print("=" * 80)

    all_passed = True
    for name, (result, _) in results.items():
        max_diff = torch.max(torch.abs(result - reference)).item()
        mean_diff = torch.mean(torch.abs(result - reference)).item()
        is_correct = torch.allclose(result, reference, rtol=1e-4, atol=1e-5)
        status = "✅" if is_correct else "❌"
        print(f"  {status} {name}")
        print(f"      最大误差: {max_diff:.6e}, 平均误差: {mean_diff:.6e}")
        if not is_correct:
            all_passed = False

    if all_passed:
        print("\n✅ 正确性测试通过")
    else:
        print("\n❌ 部分结果不正确")


def test_core_gemm_only(
    activation: torch.Tensor,
    activation_thresholded: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
) -> Dict[str, Tuple[torch.Tensor, float]]:
    """只测试“核心乘法（GEMM 内核）”的耗时：稀疏表示先缓存，计时循环里不包含稀疏化。"""
    print("\n" + "=" * 80)
    print("核心乘法（GEMM-only）对比")
    print("=" * 80)

    results: Dict[str, Tuple[torch.Tensor, float]] = {}

    # Baseline: PyTorch dense matmul（不包含 threshold 的代价，thresholded 由外部预先计算）
    print("\n[GEMM-only][PyTorch] torch.matmul(activation_thresholded, weight)")
    def torch_dense_core():
        return torch.matmul(activation_thresholded, weight)

    lat_torch = measure_latency(torch_dense_core, warmup=5, iters=100000)
    results["GEMM-only: PyTorch torch.matmul(thresholded, weight)"] = (torch_dense_core(), lat_torch)
    print(f"  延迟: {lat_torch:.4f} ms")

    # iCSR: 先 sparsify 一次，计时循环只跑 sparse_gemm
    print("\n[GEMM-only][iCSR] 预先 thr_sparsify_to_icsr")
    nz_counts, nz_col_indices, row_offsets = thr_sparsify_to_icsr(activation, threshold)

    icsr_sve_gather_kernel = SparseGEMMiCSRSVEGatherKernel.initialize(
        name="sparse_gemm_icsr_sve_gather", target="CPU"
    )
    icsr_sve_gather_op = icsr_sve_gather_kernel.operator(compiled=True)

    icsr_kernel = SparseGEMMICSRKernel.initialize(
        name="sparse_gemm_icsr", target="CPU"
    )
    icsr_op = icsr_kernel.operator(compiled=True)

    print("  - sparse_gemm_icsr_sve_gather")
    def icsr_gemm_gather_only():
        return icsr_sve_gather_op(activation, weight, row_offsets, nz_col_indices)

    lat = measure_latency(icsr_gemm_gather_only, warmup=5, iters=100000)
    results["GEMM-only: iCSR sparse_gemm_icsr_sve_gather (cached indices)"] = (icsr_gemm_gather_only(), lat)
    print(f"    延迟: {lat:.4f} ms")

    print("  - sparse_gemm_icsr")
    def icsr_gemm_only():
        return icsr_op(activation, weight, row_offsets, nz_col_indices)

    lat = measure_latency(icsr_gemm_only, warmup=5, iters=100000)
    results["GEMM-only: iCSR sparse_gemm_icsr (cached indices)"] = (icsr_gemm_only(), lat)
    print(f"    延迟: {lat:.4f} ms")

    # CSR
    print("\n[GEMM-only][CSR] 预先 thr_sparsify_to_csr")
    csr_row_offsets, csr_nz_col_indices, csr_values = thr_sparsify_to_csr(activation, threshold)

    csr_kernel = SparseGEMMCSRKernel.initialize(name="sparse_gemm_csr", target="CPU")
    csr_op = csr_kernel.operator(compiled=True)

    csr_sve_gather_kernel = SparseGEMMCSRSVEGatherKernel.initialize(
        name="sparse_gemm_csr_sve_gather", target="CPU"
    )
    csr_sve_gather_op = csr_sve_gather_kernel.operator(compiled=True)

    print("  - sparse_gemm_csr")
    def csr_gemm_only():
        return csr_op(weight, csr_row_offsets, csr_nz_col_indices, csr_values)

    lat = measure_latency(csr_gemm_only, warmup=5, iters=100000)
    results["GEMM-only: CSR sparse_gemm_csr (cached values)"] = (csr_gemm_only(), lat)
    print(f"    延迟: {lat:.4f} ms")

    print("  - sparse_gemm_csr_sve_gather")
    def csr_gemm_gather_only():
        return csr_sve_gather_op(weight, csr_row_offsets, csr_nz_col_indices, csr_values)

    lat = measure_latency(csr_gemm_gather_only, warmup=5, iters=100000)
    results["GEMM-only: CSR sparse_gemm_csr_sve_gather (cached values)"] = (csr_gemm_gather_only(), lat)
    print(f"    延迟: {lat:.4f} ms")

    # COO
    print("\n[GEMM-only][COO] 预先 thr_sparsify_to_coo")
    coo_row_indices, coo_col_indices, coo_values = thr_sparsify_to_coo(activation, threshold)
    M = int(activation.size(0))

    coo_kernel = SparseGEMMCOOKernel.initialize(name="sparse_gemm_coo", target="CPU")
    coo_op = coo_kernel.operator(compiled=True)

    coo_sve_gather_kernel = SparseGEMMCOOSVEGatherKernel.initialize(
        name="sparse_gemm_coo_sve_gather", target="CPU"
    )
    coo_sve_gather_op = coo_sve_gather_kernel.operator(compiled=True)

    # sparse_gemm_coo 需要 uint32 col_indices，thr_sparsify_to_coo 已返回 uint32
    print("  - sparse_gemm_coo")
    def coo_gemm_only():
        return coo_op(weight, coo_row_indices, coo_col_indices, coo_values, M)

    lat = measure_latency(coo_gemm_only, warmup=5, iters=100000)
    results["GEMM-only: COO sparse_gemm_coo (cached triplets)"] = (coo_gemm_only(), lat)
    print(f"    延迟: {lat:.4f} ms")

    print("  - sparse_gemm_coo_sve_gather")
    def coo_gemm_gather_only():
        return coo_sve_gather_op(weight, coo_row_indices, coo_col_indices, coo_values, M)

    lat = measure_latency(coo_gemm_gather_only, warmup=5, iters=100000)
    results["GEMM-only: COO sparse_gemm_coo_sve_gather (cached triplets)"] = (coo_gemm_gather_only(), lat)
    print(f"    延迟: {lat:.4f} ms")

    # CSC
    print("\n[GEMM-only][CSC] 预先 thr_sparsify_to_csc")
    csc_col_ptr, csc_row_indices, csc_values = thr_sparsify_to_csc(activation, threshold)

    csc_kernel = SparseGEMMCSCKernel.initialize(name="sparse_gemm_csc", target="CPU")
    csc_op = csc_kernel.operator(compiled=True)

    print("  - sparse_gemm_csc")
    def csc_gemm_only():
        return csc_op(weight, csc_col_ptr, csc_row_indices, csc_values, M, 0)

    lat = measure_latency(csc_gemm_only, warmup=5, iters=100000)
    results["GEMM-only: CSC sparse_gemm_csc (cached CSC)"] = (csc_gemm_only(), lat)
    print(f"    延迟: {lat:.4f} ms")

    return results


def test_preprocess_only(
    activation: torch.Tensor,
    threshold: float,
) -> Dict[str, Tuple[Any, float]]:
    """只测试输入稀疏矩阵处理（稀疏化/格式转换）耗时：不包含任何 GEMM。"""
    print("\n" + "=" * 80)
    print("输入稀疏矩阵处理（Preprocess-only）对比")
    print("=" * 80)

    results: Dict[str, Tuple[Any, float]] = {}

    # Baseline 1: 仅 threshold（对齐 cpp 判定 abs(x) >= thr）
    print("\n[Preprocess-only][PyTorch] _apply_threshold")
    def torch_threshold_only():
        return _apply_threshold(activation, threshold=threshold)

    lat = measure_latency(torch_threshold_only, warmup=5, iters=10)
    results["Preprocess-only: PyTorch _apply_threshold(abs>=thr)"] = (torch_threshold_only(), lat)
    print(f"  延迟: {lat:.4f} ms")

    # Baseline 2: threshold + to_sparse_csr（端到端的 PyTorch CSR 构建）
    print("\n[Preprocess-only][PyTorch] threshold + to_sparse_csr")
    def torch_to_sparse_csr():
        x = _apply_threshold(activation, threshold=threshold)
        sp = x.to_sparse_csr()
        # 访问 components，避免 lazy 路径把工作延后到后续阶段
        _ = sp.crow_indices()
        _ = sp.col_indices()
        _ = sp.values()
        return sp

    lat = measure_latency(torch_to_sparse_csr, warmup=5, iters=10)
    results["Preprocess-only: PyTorch threshold + to_sparse_csr()"] = (torch_to_sparse_csr(), lat)
    print(f"  延迟: {lat:.4f} ms")

    # Baseline 3: threshold + to_sparse_csc（端到端的 PyTorch CSC 构建）
    print("\n[Preprocess-only][PyTorch] threshold + to_sparse_csc")
    def torch_to_sparse_csc():
        x = _apply_threshold(activation, threshold=threshold)
        sp = x.to_sparse_csc()
        _ = sp.ccol_indices()
        _ = sp.row_indices()
        _ = sp.values()
        return sp

    lat = measure_latency(torch_to_sparse_csc, warmup=5, iters=10)
    results["Preprocess-only: PyTorch threshold + to_sparse_csc()"] = (torch_to_sparse_csc(), lat)
    print(f"  延迟: {lat:.4f} ms")

    # Custom: iCSR sparsify
    print("\n[Preprocess-only][iCSR] thr_sparsify_to_icsr / thr_sparsify_to_icsr_sve")
    def icsr_pre_1():
        return thr_sparsify_to_icsr(activation, threshold)

    lat = measure_latency(icsr_pre_1, warmup=5, iters=100000)
    results["Preprocess-only: iCSR thr_sparsify_to_icsr"] = (icsr_pre_1(), lat)
    print(f"  - thr_sparsify_to_icsr: {lat:.4f} ms")

    def icsr_pre_2():
        return thr_sparsify_to_icsr_sve(activation, threshold)

    lat = measure_latency(icsr_pre_2, warmup=5, iters=100000)
    results["Preprocess-only: iCSR thr_sparsify_to_icsr_sve"] = (icsr_pre_2(), lat)
    print(f"  - thr_sparsify_to_icsr_sve: {lat:.4f} ms")

    # Custom: CSR sparsify
    print("\n[Preprocess-only][CSR] thr_sparsify_to_csr / thr_sparsify_to_csr_sve")
    def csr_pre_1():
        return thr_sparsify_to_csr(activation, threshold)

    lat = measure_latency(csr_pre_1, warmup=5, iters=100000)
    results["Preprocess-only: CSR thr_sparsify_to_csr"] = (csr_pre_1(), lat)
    print(f"  - thr_sparsify_to_csr: {lat:.4f} ms")

    def csr_pre_2():
        return thr_sparsify_to_csr_sve(activation, threshold)

    lat = measure_latency(csr_pre_2, warmup=5, iters=100000)
    results["Preprocess-only: CSR thr_sparsify_to_csr_sve"] = (csr_pre_2(), lat)
    print(f"  - thr_sparsify_to_csr_sve: {lat:.4f} ms")

    # Custom: COO sparsify
    print("\n[Preprocess-only][COO] thr_sparsify_to_coo / thr_sparsify_to_coo_sve")
    def coo_pre_1():
        return thr_sparsify_to_coo(activation, threshold)

    lat = measure_latency(coo_pre_1, warmup=5, iters=100000)
    results["Preprocess-only: COO thr_sparsify_to_coo"] = (coo_pre_1(), lat)
    print(f"  - thr_sparsify_to_coo: {lat:.4f} ms")

    def coo_pre_2():
        return thr_sparsify_to_coo_sve(activation, threshold)

    lat = measure_latency(coo_pre_2, warmup=5, iters=100000)
    results["Preprocess-only: COO thr_sparsify_to_coo_sve"] = (coo_pre_2(), lat)
    print(f"  - thr_sparsify_to_coo_sve: {lat:.4f} ms")

    # Custom: CSC sparsify
    print("\n[Preprocess-only][CSC] thr_sparsify_to_csc")
    def csc_pre():
        return thr_sparsify_to_csc(activation, threshold)

    lat = measure_latency(csc_pre, warmup=5, iters=100000)
    results["Preprocess-only: CSC thr_sparsify_to_csc"] = (csc_pre(), lat)
    print(f"  - thr_sparsify_to_csc: {lat:.4f} ms")

    return results


# =============================================================================
# iCSR 格式组合测试
# =============================================================================

def test_icsr_combinations(
    activation: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
) -> Dict[str, Tuple[torch.Tensor, float]]:
    """
    测试 iCSR 格式的稀疏化和GEMM算子的所有组合。
    
    组合：
    - thr_sparsify_to_icsr + sparse_gemm_icsr_sve_gather
    - thr_sparsify_to_icsr_sve + sparse_gemm_icsr_sve_gather
    - thr_sparsify_to_icsr + sparse_gemm_icsr
    - thr_sparsify_to_icsr_sve + sparse_gemm_icsr
    
    Returns:
        Dict[组合名称, (结果, 延迟)]
    """
    print("\n" + "=" * 80)
    print("测试 iCSR 格式组合")
    print("=" * 80)
    
    results = {}
    
    # 初始化 GEMM 算子
    icsr_sve_gather_kernel = SparseGEMMiCSRSVEGatherKernel.initialize(
        name="sparse_gemm_icsr_sve_gather", target="CPU"
    )
    icsr_sve_gather_op = icsr_sve_gather_kernel.operator(compiled=True)
    
    icsr_kernel = SparseGEMMICSRKernel.initialize(
        name="sparse_gemm_icsr", target="CPU"
    )
    icsr_op = icsr_kernel.operator(compiled=True)
    
    # 组合 1: thr_sparsify_to_icsr + sparse_gemm_icsr_sve_gather
    print("\n[iCSR-1] thr_sparsify_to_icsr + sparse_gemm_icsr_sve_gather")
    def icsr_combo1():
        nz_counts, nz_col_indices, row_offsets = thr_sparsify_to_icsr(activation, threshold)
        return icsr_sve_gather_op(activation, weight, row_offsets, nz_col_indices)
    
    lat1 = measure_latency(icsr_combo1, warmup=5, iters=100000)
    result1 = icsr_combo1()
    results["iCSR-1: thr_sparsify_to_icsr + sparse_gemm_icsr_sve_gather"] = (result1, lat1)
    print(f"  延迟: {lat1:.4f} ms")
    
    # 组合 2: thr_sparsify_to_icsr_sve + sparse_gemm_icsr_sve_gather
    print("\n[iCSR-2] thr_sparsify_to_icsr_sve + sparse_gemm_icsr_sve_gather")
    def icsr_combo2():
        nz_counts, nz_col_indices, row_offsets = thr_sparsify_to_icsr_sve(activation, threshold)
        return icsr_sve_gather_op(activation, weight, row_offsets, nz_col_indices)
    
    lat2 = measure_latency(icsr_combo2, warmup=5, iters=100000)
    result2 = icsr_combo2()
    results["iCSR-2: thr_sparsify_to_icsr_sve + sparse_gemm_icsr_sve_gather"] = (result2, lat2)
    print(f"  延迟: {lat2:.4f} ms")
    
    # 组合 3: thr_sparsify_to_icsr + sparse_gemm_icsr
    print("\n[iCSR-3] thr_sparsify_to_icsr + sparse_gemm_icsr")
    def icsr_combo3():
        nz_counts, nz_col_indices, row_offsets = thr_sparsify_to_icsr(activation, threshold)
        return icsr_op(activation, weight, row_offsets, nz_col_indices)
    
    lat3 = measure_latency(icsr_combo3, warmup=5, iters=100000)
    result3 = icsr_combo3()
    results["iCSR-3: thr_sparsify_to_icsr + sparse_gemm_icsr"] = (result3, lat3)
    print(f"  延迟: {lat3:.4f} ms")
    
    # 组合 4: thr_sparsify_to_icsr_sve + sparse_gemm_icsr
    print("\n[iCSR-4] thr_sparsify_to_icsr_sve + sparse_gemm_icsr")
    def icsr_combo4():
        nz_counts, nz_col_indices, row_offsets = thr_sparsify_to_icsr_sve(activation, threshold)
        return icsr_op(activation, weight, row_offsets, nz_col_indices)
    
    lat4 = measure_latency(icsr_combo4, warmup=5, iters=100000)
    result4 = icsr_combo4()
    results["iCSR-4: thr_sparsify_to_icsr_sve + sparse_gemm_icsr"] = (result4, lat4)
    print(f"  延迟: {lat4:.4f} ms")
    
    return results


# =============================================================================
# CSR 格式组合测试
# =============================================================================

def test_csr_combinations(
    activation: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
) -> Dict[str, Tuple[torch.Tensor, float]]:
    """
    测试 CSR 格式的稀疏化和GEMM算子的所有组合。
    
    组合：
    - thr_sparsify_to_csr + sparse_gemm_csr
    - thr_sparsify_to_csr_sve + sparse_gemm_csr
    - thr_sparsify_to_csr + sparse_gemm_csr_sve_gather
    - thr_sparsify_to_csr_sve + sparse_gemm_csr_sve_gather
    
    Returns:
        Dict[组合名称, (结果, 延迟)]
    """
    print("\n" + "=" * 80)
    print("测试 CSR 格式组合")
    print("=" * 80)
    
    results = {}
    
    # 初始化 GEMM 算子
    csr_kernel = SparseGEMMCSRKernel.initialize(
        name="sparse_gemm_csr", target="CPU"
    )
    csr_op = csr_kernel.operator(compiled=True)
    
    csr_sve_gather_kernel = SparseGEMMCSRSVEGatherKernel.initialize(
        name="sparse_gemm_csr_sve_gather", target="CPU"
    )
    csr_sve_gather_op = csr_sve_gather_kernel.operator(compiled=True)
    
    # 组合 1: thr_sparsify_to_csr + sparse_gemm_csr
    print("\n[CSR-1] thr_sparsify_to_csr + sparse_gemm_csr")
    def csr_combo1():
        row_offsets, nz_col_indices, values = thr_sparsify_to_csr(activation, threshold)
        return csr_op(weight, row_offsets, nz_col_indices, values)
    
    lat1 = measure_latency(csr_combo1, warmup=5, iters=100000)
    result1 = csr_combo1()
    results["CSR-1: thr_sparsify_to_csr + sparse_gemm_csr"] = (result1, lat1)
    print(f"  延迟: {lat1:.4f} ms")
    
    # 组合 2: thr_sparsify_to_csr_sve + sparse_gemm_csr
    print("\n[CSR-2] thr_sparsify_to_csr_sve + sparse_gemm_csr")
    def csr_combo2():
        row_offsets, nz_col_indices, values = thr_sparsify_to_csr_sve(activation, threshold)
        return csr_op(weight, row_offsets, nz_col_indices, values)
    
    lat2 = measure_latency(csr_combo2, warmup=5, iters=100000)
    result2 = csr_combo2()
    results["CSR-2: thr_sparsify_to_csr_sve + sparse_gemm_csr"] = (result2, lat2)
    print(f"  延迟: {lat2:.4f} ms")
    
    # 组合 3: thr_sparsify_to_csr + sparse_gemm_csr_sve_gather
    print("\n[CSR-3] thr_sparsify_to_csr + sparse_gemm_csr_sve_gather")
    def csr_combo3():
        row_offsets, nz_col_indices, values = thr_sparsify_to_csr(activation, threshold)
        return csr_sve_gather_op(weight, row_offsets, nz_col_indices, values)
    
    lat3 = measure_latency(csr_combo3, warmup=5, iters=100000)
    result3 = csr_combo3()
    results["CSR-3: thr_sparsify_to_csr + sparse_gemm_csr_sve_gather"] = (result3, lat3)
    print(f"  延迟: {lat3:.4f} ms")
    
    # 组合 4: thr_sparsify_to_csr_sve + sparse_gemm_csr_sve_gather
    print("\n[CSR-4] thr_sparsify_to_csr_sve + sparse_gemm_csr_sve_gather")
    def csr_combo4():
        row_offsets, nz_col_indices, values = thr_sparsify_to_csr_sve(activation, threshold)
        return csr_sve_gather_op(weight, row_offsets, nz_col_indices, values)
    
    lat4 = measure_latency(csr_combo4, warmup=5, iters=100000)
    result4 = csr_combo4()
    results["CSR-4: thr_sparsify_to_csr_sve + sparse_gemm_csr_sve_gather"] = (result4, lat4)
    print(f"  延迟: {lat4:.4f} ms")
    
    return results


# =============================================================================
# COO 格式组合测试
# =============================================================================

def test_coo_combinations(
    activation: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
) -> Dict[str, Tuple[torch.Tensor, float]]:
    """
    测试 COO 格式的稀疏化和GEMM算子的所有组合。
    
    组合：
    - thr_sparsify_to_coo + sparse_gemm_coo
    - thr_sparsify_to_coo_sve + sparse_gemm_coo
    - thr_sparsify_to_coo + sparse_gemm_coo_sve_gather
    - thr_sparsify_to_coo_sve + sparse_gemm_coo_sve_gather
    
    Returns:
        Dict[组合名称, (结果, 延迟)]
    """
    print("\n" + "=" * 80)
    print("测试 COO 格式组合")
    print("=" * 80)
    
    results = {}
    
    # 初始化 GEMM 算子
    coo_kernel = SparseGEMMCOOKernel.initialize(
        name="sparse_gemm_coo", target="CPU"
    )
    coo_op = coo_kernel.operator(compiled=True)
    
    coo_sve_gather_kernel = SparseGEMMCOOSVEGatherKernel.initialize(
        name="sparse_gemm_coo_sve_gather", target="CPU"
    )
    coo_sve_gather_op = coo_sve_gather_kernel.operator(compiled=True)
    
    # C++ 算子签名需要显式传入 M（稀疏矩阵行数）
    M = int(activation.size(0))

    # 组合 1: thr_sparsify_to_coo + sparse_gemm_coo
    print("\n[COO-1] thr_sparsify_to_coo + sparse_gemm_coo")
    def coo_combo1():
        row_indices, col_indices, values = thr_sparsify_to_coo(activation, threshold)
        # sparse_gemm_coo 需要 uint32 的 col_indices，thr_sparsify_to_coo 已返回 uint32
        return coo_op(weight, row_indices, col_indices, values, M)
    
    lat1 = measure_latency(coo_combo1, warmup=5, iters=100000)
    result1 = coo_combo1()
    results["COO-1: thr_sparsify_to_coo + sparse_gemm_coo"] = (result1, lat1)
    print(f"  延迟: {lat1:.4f} ms")
    
    # 组合 2: thr_sparsify_to_coo_sve + sparse_gemm_coo
    print("\n[COO-2] thr_sparsify_to_coo_sve + sparse_gemm_coo")
    def coo_combo2():
        row_indices, col_indices, values = thr_sparsify_to_coo_sve(activation, threshold)
        # sparse_gemm_coo 需要 uint32 的 col_indices，thr_sparsify_to_coo_sve 已返回 uint32
        return coo_op(weight, row_indices, col_indices, values, M)
    
    lat2 = measure_latency(coo_combo2, warmup=5, iters=100000)
    result2 = coo_combo2()
    results["COO-2: thr_sparsify_to_coo_sve + sparse_gemm_coo"] = (result2, lat2)
    print(f"  延迟: {lat2:.4f} ms")
    
    # 组合 3: thr_sparsify_to_coo + sparse_gemm_coo_sve_gather
    print("\n[COO-3] thr_sparsify_to_coo + sparse_gemm_coo_sve_gather")
    def coo_combo3():
        row_indices, col_indices, values = thr_sparsify_to_coo(activation, threshold)
        # sparse_gemm_coo_sve_gather 需要 uint32 的 col_indices，已经是 uint32
        return coo_sve_gather_op(weight, row_indices, col_indices, values, M)
    
    lat3 = measure_latency(coo_combo3, warmup=5, iters=100000)
    result3 = coo_combo3()
    results["COO-3: thr_sparsify_to_coo + sparse_gemm_coo_sve_gather"] = (result3, lat3)
    print(f"  延迟: {lat3:.4f} ms")
    
    # 组合 4: thr_sparsify_to_coo_sve + sparse_gemm_coo_sve_gather
    print("\n[COO-4] thr_sparsify_to_coo_sve + sparse_gemm_coo_sve_gather")
    def coo_combo4():
        row_indices, col_indices, values = thr_sparsify_to_coo_sve(activation, threshold)
        # sparse_gemm_coo_sve_gather 需要 uint32 的 col_indices，已经是 uint32
        return coo_sve_gather_op(weight, row_indices, col_indices, values, M)
    
    lat4 = measure_latency(coo_combo4, warmup=5, iters=100000)
    result4 = coo_combo4()
    results["COO-4: thr_sparsify_to_coo_sve + sparse_gemm_coo_sve_gather"] = (result4, lat4)
    print(f"  延迟: {lat4:.4f} ms")
    
    return results


# =============================================================================
# CSC 格式组合测试
# =============================================================================

def test_csc_combinations(
    activation: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
) -> Dict[str, Tuple[torch.Tensor, float]]:
    """
    测试 CSC 格式的稀疏化和GEMM算子的所有组合。
    
    组合：
    - thr_sparsify_to_csc + sparse_gemm_csc
    
    Returns:
        Dict[组合名称, (结果, 延迟)]
    """
    print("\n" + "=" * 80)
    print("测试 CSC 格式组合")
    print("=" * 80)
    
    results = {}
    
    # 初始化 GEMM 算子
    csc_kernel = SparseGEMMCSCKernel.initialize(
        name="sparse_gemm_csc", target="CPU"
    )
    csc_op = csc_kernel.operator(compiled=True)
    
    M = activation.size(0)
    
    # 组合 1: thr_sparsify_to_csc + sparse_gemm_csc
    print("\n[CSC-1] thr_sparsify_to_csc + sparse_gemm_csc")
    def csc_combo1():
        col_ptr, row_indices, values = thr_sparsify_to_csc(activation, threshold)
        return csc_op(weight, col_ptr, row_indices, values, M, 0)
    
    lat1 = measure_latency(csc_combo1, warmup=5, iters=100000)
    result1 = csc_combo1()
    results["CSC-1: thr_sparsify_to_csc + sparse_gemm_csc"] = (result1, lat1)
    print(f"  延迟: {lat1:.4f} ms")
    
    return results


# =============================================================================
# PyTorch 参考实现测试
# =============================================================================

def test_pytorch_references(
    activation: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
) -> Dict[str, Tuple[torch.Tensor, float]]:
    """
    测试PyTorch的参考实现。
    
    测试方法：
    - PyTorch 稠密 matmul
    - PyTorch 稀疏 CSR + sparse.mm
    - PyTorch 稀疏 CSC + sparse.mm
    - PyTorch 选择性加载 weight 非零行 + matmul
    
    Returns:
        Dict[组合名称, (结果, 延迟)]
    """
    print("\n" + "=" * 80)
    print("测试 PyTorch 参考实现")
    print("=" * 80)
    
    results = {}
    
    # 启用 MKL-DNN
    torch.backends.mkldnn.enabled = True
    
    # PyTorch 稠密 matmul
    print("\n[PyTorch-1] 稠密 matmul")
    def pytorch_dense_fn():
        activation_thresholded = _apply_threshold(activation, threshold=threshold)
        return torch.matmul(activation_thresholded, weight)
    
    lat1 = measure_latency(pytorch_dense_fn, warmup=5, iters=100000)
    result1 = pytorch_dense_fn()
    results["PyTorch-1: 稠密 matmul"] = (result1, lat1)
    print(f"  延迟: {lat1:.4f} ms")
    
    # PyTorch 稀疏 CSR + sparse.mm
    print("\n[PyTorch-2] 稀疏 CSR + sparse.mm")
    def pytorch_sparse_csr_fn():
        activation_thresholded = _apply_threshold(activation, threshold=threshold)
        sp_act = activation_thresholded.to_sparse_csr()
        return torch.sparse.mm(sp_act, weight)
    
    lat2 = measure_latency(pytorch_sparse_csr_fn, warmup=5, iters=100000)
    result2 = pytorch_sparse_csr_fn()
    results["PyTorch-2: 稀疏 CSR + sparse.mm"] = (result2, lat2)
    print(f"  延迟: {lat2:.4f} ms")
    
    # PyTorch 稀疏 CSC + sparse.mm
    print("\n[PyTorch-3] 稀疏 CSC + sparse.mm")
    def pytorch_sparse_csc_fn():
        activation_thresholded = _apply_threshold(activation, threshold=threshold)
        sp_act = activation_thresholded.to_sparse_csc()
        return torch.sparse.mm(sp_act, weight)
    
    lat3 = measure_latency(pytorch_sparse_csc_fn, warmup=5, iters=100000)
    result3 = pytorch_sparse_csc_fn()
    results["PyTorch-3: 稀疏 CSC + sparse.mm"] = (result3, lat3)
    print(f"  延迟: {lat3:.4f} ms")
    
    # PyTorch 选择性加载 weight 非零行 + matmul
    print("\n[PyTorch-4] 选择性加载 weight 非零行 + matmul")
    def pytorch_selective_weight_fn():
        activation_thresholded = _apply_threshold(activation, threshold=threshold)
        M, K = activation_thresholded.shape
        N = weight.shape[1]
        
        # 初始化输出
        output = torch.zeros(M, N, dtype=torch.float32)
        
        # 对每一行进行处理
        for m in range(M):
            # 找出该行的非零列索引
            nz_cols = torch.nonzero(activation_thresholded[m], as_tuple=False).flatten()
            
            if nz_cols.numel() > 0:
                # 只选择 weight 中对应的非零行
                act_nz = activation_thresholded[m, nz_cols]  # (nnz,)
                weight_nz = weight[nz_cols, :]  # (nnz, N)
                
                # 进行矩阵乘法：(1, nnz) @ (nnz, N) -> (1, N)
                output[m] = torch.matmul(act_nz.unsqueeze(0), weight_nz).squeeze(0)
        
        return output
    
    lat4 = measure_latency(pytorch_selective_weight_fn, warmup=5, iters=100000)
    result4 = pytorch_selective_weight_fn()
    results["PyTorch-4: 选择性加载 weight 非零行 + matmul"] = (result4, lat4)
    print(f"  延迟: {lat4:.4f} ms")
    
    return results


# =============================================================================
# 正确性验证和综合比较
# =============================================================================

def verify_correctness(
    all_results: Dict[str, Dict[str, Tuple[torch.Tensor, float]]],
    reference: torch.Tensor,
) -> None:
    """验证所有算子组合的正确性。"""
    print("\n" + "=" * 80)
    print("正确性验证")
    print("=" * 80)
    
    all_passed = True
    
    for format_name, results in all_results.items():
        print(f"\n{format_name}:")
        for combo_name, (result, latency) in results.items():
            max_diff = torch.max(torch.abs(result - reference)).item()
            mean_diff = torch.mean(torch.abs(result - reference)).item()
            
            is_correct = torch.allclose(result, reference, rtol=1e-4, atol=1e-5)
            status = "✅" if is_correct else "❌"
            
            print(f"  {status} {combo_name}")
            print(f"      最大误差: {max_diff:.6e}, 平均误差: {mean_diff:.6e}")
            
            if not is_correct:
                all_passed = False
    
    if all_passed:
        print("\n✅ 所有组合的正确性测试通过")
    else:
        print("\n❌ 部分组合的正确性测试失败")


def print_performance_summary(
    all_results: Dict[str, Dict[str, Tuple[torch.Tensor, float]]],
) -> None:
    """打印性能对比总结。"""
    print("\n" + "=" * 80)
    print("性能对比总结")
    print("=" * 80)
    
    # 收集所有延迟数据
    all_latencies = []
    pytorch_dense_latency = None
    
    for format_name, results in all_results.items():
        for combo_name, (result, latency) in results.items():
            all_latencies.append((combo_name, latency, format_name))
            # 记录 PyTorch 稠密实现的延迟作为基准
            if "PyTorch-1: 稠密 matmul" in combo_name:
                pytorch_dense_latency = latency
    
    # 按延迟排序
    all_latencies.sort(key=lambda x: x[1])
    
    print("\n延迟排名（从快到慢）：")
    print("-" * 86)
    if pytorch_dense_latency is not None:
        print(f"{'排名':<4} {'算子组合':<62} {'延迟(ms)':<12} {'加速比':<10}")
        print("-" * 86)
        for rank, (combo_name, latency, format_name) in enumerate(all_latencies, 1):
            speedup = pytorch_dense_latency / latency if latency > 0 else 0.0
            # 高亮显示自定义算子（非PyTorch）
            marker = "🚀" if format_name != "PyTorch" else "📊"
            print(f"{rank:2d}. {marker} {combo_name:60s} {latency:8.4f} ms  {speedup:6.2f}x")
    else:
        for rank, (combo_name, latency, format_name) in enumerate(all_latencies, 1):
            print(f"{rank:2d}. {combo_name:60s} {latency:8.4f} ms")
    
    # 找出最快的组合
    fastest_name, fastest_latency, fastest_format = all_latencies[0]
    print("\n" + "=" * 80)
    print(f"⚡ 最快的组合: {fastest_name}")
    print(f"   延迟: {fastest_latency:.4f} ms")
    if pytorch_dense_latency is not None:
        speedup = pytorch_dense_latency / fastest_latency
        print(f"   相比PyTorch稠密实现加速比: {speedup:.2f}x")
    print("=" * 80)
    
    # 统计自定义算子 vs PyTorch
    if pytorch_dense_latency is not None:
        print("\n" + "=" * 80)
        print("自定义算子性能统计")
        print("=" * 80)
        
        custom_latencies = [(name, lat) for name, lat, fmt in all_latencies if fmt != "PyTorch"]
        if custom_latencies:
            fastest_custom_name, fastest_custom_latency = custom_latencies[0]
            print(f"\n最快的自定义算子: {fastest_custom_name}")
            print(f"  延迟: {fastest_custom_latency:.4f} ms")
            print(f"  相比PyTorch稠密实现加速比: {pytorch_dense_latency/fastest_custom_latency:.2f}x")
            
            # 统计有多少自定义算子比PyTorch稠密实现快
            faster_than_dense = sum(1 for _, lat in custom_latencies if lat < pytorch_dense_latency)
            print(f"\n比PyTorch稠密实现更快的自定义算子数量: {faster_than_dense}/{len(custom_latencies)}")
            
            # 找出PyTorch非稠密实现的延迟（排除稠密matmul）
            pytorch_nondense_latencies = [(name, lat) for name, lat, fmt in all_latencies 
                                          if fmt == "PyTorch" and "稠密 matmul" not in name]
            if pytorch_nondense_latencies:
                fastest_pytorch_nondense = min(pytorch_nondense_latencies, key=lambda x: x[1])
                print(f"\nPyTorch最快的非稠密实现: {fastest_pytorch_nondense[0]}")
                print(f"  延迟: {fastest_pytorch_nondense[1]:.4f} ms")
                faster_than_nondense = sum(1 for _, lat in custom_latencies if lat < fastest_pytorch_nondense[1])
                print(f"\n比PyTorch最快非稠密实现更快的自定义算子数量: {faster_than_nondense}/{len(custom_latencies)}")
        
        print("=" * 80)


def main() -> None:
    """运行所有测试"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--threshold", type=float, default=0.9, help="稀疏化阈值")
    parser.add_argument("--M", type=int, default=1, help="activation 行数")
    parser.add_argument("--K", type=int, default=64, help="activation 列数 / weight 行数")
    parser.add_argument("--N", type=int, default=64, help="weight 列数")
    parser.add_argument(
        "--tests",
        nargs="+",
        default=["all"],
        help=(
            "选择要运行的测试项（空格或逗号分隔）："
            "all/icsr/csr/coo/csc/pytorch/gemm-only/preprocess-only"
        ),
    )
    args = parser.parse_args()

    selected_tests = _parse_selected_tests(args.tests)
    
    print("=" * 80)
    print("ARM SVE 稀疏 GEMM 算子综合测试")
    print("=" * 80)
    print(f"配置参数:")
    print(f"  - 随机种子: {args.seed}")
    print(f"  - 阈值: {args.threshold}")
    print(f"  - 矩阵尺寸: activation ({args.M}, {args.K}), weight ({args.K}, {args.N})")
    print(f"  - 测试项: {', '.join(selected_tests)}")
    
    try:
        if psutil is None:
            print("  - CPU 利用率: psutil 不可用，跳过 CPU 利用率打印（可选安装: pip install psutil）")
        else:
            _maybe_print_cpu_util("启动前")

        # 加载扩展
        load_sve_sparse_gemm_extension(verbose=False)
        
        # 生成共享的测试数据
        activation = _make_random_sparse_activation(args.M, args.K, seed=args.seed)
        activation_thresholded = _apply_threshold(activation, threshold=args.threshold)

        # 计算稀疏度
        nnz = torch.count_nonzero(activation_thresholded).item()
        sparsity = 100.0 * (1.0 - nnz / (args.M * args.K))
        print(f"  - 稀疏度: {sparsity:.1f}% ({nnz}/{args.M * args.K} 非零元素)")

        needs_weight = any(
            t in {"icsr", "csr", "coo", "csc", "pytorch", "gemm-only"} for t in selected_tests
        )
        needs_reference = any(
            t in {"icsr", "csr", "coo", "csc", "pytorch", "gemm-only"} for t in selected_tests
        )

        weight = None
        reference = None
        if needs_weight:
            weight = torch.randn(args.K, args.N, dtype=torch.float32)
        if needs_reference:
            if weight is None:
                raise RuntimeError("内部错误：needs_reference=True 但 weight 未生成")
            reference = torch.matmul(activation_thresholded, weight)
        
        # 组合/参考实现测试（会参与 correctness + summary）
        all_results: Dict[str, Dict[str, Tuple[torch.Tensor, float]]] = {}
        if any(t in {"icsr", "csr", "coo", "csc", "pytorch"} for t in selected_tests):
            if weight is None or reference is None:
                raise RuntimeError("内部错误：需要 weight/reference 但未生成")

            if "icsr" in selected_tests:
                _maybe_print_cpu_util("开始 iCSR")
                all_results["iCSR"] = test_icsr_combinations(activation, weight, args.threshold)
                _maybe_print_cpu_util("结束 iCSR")
            if "csr" in selected_tests:
                _maybe_print_cpu_util("开始 CSR")
                all_results["CSR"] = test_csr_combinations(activation, weight, args.threshold)
                _maybe_print_cpu_util("结束 CSR")
            if "coo" in selected_tests:
                _maybe_print_cpu_util("开始 COO")
                all_results["COO"] = test_coo_combinations(activation, weight, args.threshold)
                _maybe_print_cpu_util("结束 COO")
            if "csc" in selected_tests:
                _maybe_print_cpu_util("开始 CSC")
                all_results["CSC"] = test_csc_combinations(activation, weight, args.threshold)
                _maybe_print_cpu_util("结束 CSC")
            if "pytorch" in selected_tests:
                _maybe_print_cpu_util("开始 PyTorch 参考")
                all_results["PyTorch"] = test_pytorch_references(activation, weight, args.threshold)
                _maybe_print_cpu_util("结束 PyTorch 参考")

            # 验证正确性 + 性能总结
            verify_correctness(all_results, reference)
            print_performance_summary(all_results)

        # 额外测试 1：只测试核心乘法（GEMM-only）
        if "gemm-only" in selected_tests:
            if weight is None or reference is None:
                raise RuntimeError("GEMM-only 测试需要 weight/reference，但当前未生成（请检查 --tests）")

            _maybe_print_cpu_util("开始 GEMM-only")
            gemm_only_results = test_core_gemm_only(
                activation=activation,
                activation_thresholded=activation_thresholded,
                weight=weight,
                threshold=args.threshold,
            )
            _maybe_print_cpu_util("结束 GEMM-only")
            verify_correctness_flat(gemm_only_results, reference)
            baseline_gemm_latency = gemm_only_results[
                "GEMM-only: PyTorch torch.matmul(thresholded, weight)"
            ][1]
            _print_ranked_latencies(
                title="核心乘法（GEMM-only）延迟排名（baseline=PyTorch torch.matmul）",
                latencies=[(k, v[1]) for k, v in gemm_only_results.items()],
                baseline_latency=baseline_gemm_latency,
            )

        # 额外测试 2：只测试输入稀疏矩阵处理（Preprocess-only）
        if "preprocess-only" in selected_tests:
            _maybe_print_cpu_util("开始 Preprocess-only")
            preprocess_only_results = test_preprocess_only(
                activation=activation,
                threshold=args.threshold,
            )
            _maybe_print_cpu_util("结束 Preprocess-only")
            # 以 PyTorch threshold-only 作为 baseline（最小化的预处理基准）
            baseline_pre_latency = preprocess_only_results[
                "Preprocess-only: PyTorch _apply_threshold(abs>=thr)"
            ][1]
            _print_ranked_latencies(
                title="输入稀疏矩阵处理（Preprocess-only）延迟排名（baseline=PyTorch _apply_threshold）",
                latencies=[(k, v[1]) for k, v in preprocess_only_results.items()],
                baseline_latency=baseline_pre_latency,
            )
        
        print("\n" + "=" * 80)
        print("✅ 所有测试完成")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
