import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch

THRESHOLD = 0.060534358

def generate_extreme_row_sparsity_matrix(M: int, K: int,threshold: float = THRESHOLD, dtype: torch.dtype = torch.float32, device: str = "cpu") -> torch.Tensor:
    """
    构造一个行间稀疏度极其不均匀的矩阵：
    - 前1/4为全稠密
    - 中间1/2为阈值剪枝，带有中等/低稀疏度
    - 后1/4高度稀疏
    """
    torch.manual_seed(42)
    num_dense = M - 1
    # num_middle = M // 2
    num_sparse = 1
    mats = []

    # Dense rows (no pruning)
    dense_rows = torch.rand((num_dense//2, K), dtype=dtype, device=device) * 2 - 1  # values in [-1, 1]
    mats.append(dense_rows)

    # # Medium sparsity rows (prune with threshold)
    # middle_vals = torch.rand((num_middle, K), dtype=dtype, device=device) * 2 - 1
    # # Prune below threshold (simulate realistic GEMM pruning)
    # mask_middle = middle_vals.abs() > threshold
    # middle_vals = middle_vals * mask_middle
    # mats.append(middle_vals)

    # Sparse rows (high pruning, only a few nnzs per row)
    sparse_rows = torch.zeros((num_sparse, K), dtype=dtype, device=device)
    for r in range(num_sparse):
        nnz = max(1, K // 32)  # very sparse (at least 1, up to 3%)
        idx = torch.randperm(K, device=device)[:nnz]
        sparse_rows[r, idx] = torch.randn(nnz, dtype=dtype, device=device) * 2 - 1
    mats.append(sparse_rows)

        # Dense rows (no pruning)
    dense_rows = torch.rand((num_dense-num_dense//2, K), dtype=dtype, device=device) * 2 - 1  # values in [-1, 1]
    mats.append(dense_rows)

    full_matrix = torch.cat(mats, dim=0)
    assert full_matrix.shape == (M, K)
    return full_matrix

def load_csv_matrix(path: Path, dtype: torch.dtype, device: str) -> torch.Tensor:
    # 兼容原API，但优先注释：用本地极不均匀稀疏度生成函数替代
    # return generate_extreme_row_sparsity_matrix(128, 4096, dtype, device, THRESHOLD)
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader):
            if not row:
                continue
            try:
                rows.append([float(value) for value in row])
            except ValueError:
                if row_idx == 0:
                    continue
                raise

    if not rows:
        raise ValueError(f"CSV 文件为空或不包含有效数值: {path}")

    return torch.tensor(rows, dtype=dtype, device=device)


def benchmark_adaptive_gemm(
    x_2d: torch.Tensor,
    w: torch.Tensor,
    threshold: float,
    warmup: int,
    runs: int,
) -> tuple[torch.Tensor, list[float]]:
    from kernels.sve_sparse_gemm import AdaptiveGEMM, load_sve_sparse_gemm_extension

    load_sve_sparse_gemm_extension()

    x = x_2d

    for _ in range(warmup):
        out = AdaptiveGEMM(x, w, threshold)

    times_ms = []
    out = None
    for _ in range(runs):
        start = time.perf_counter()
        out = AdaptiveGEMM(x, w, threshold)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    assert out is not None
    return out, times_ms

def benchmark_dense_gemm(
    x_2d: torch.Tensor,
    w: torch.Tensor,
    warmup: int,
    runs: int,
) -> tuple[torch.Tensor, list[float]]:
    from kernels.sve_sparse_gemm import dense_gemm_sve_omp, load_sve_sparse_gemm_extension

    load_sve_sparse_gemm_extension()

    x = x_2d

    for _ in range(warmup):
        # out = torch.matmul(x, w)
        out = dense_gemm_sve_omp(x, w)

    times_ms = []
    out = None
    for _ in range(runs):
        start = time.perf_counter()
        out = torch.matmul(x, w)
        # out = dense_gemm_sve_omp(x, w)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    assert out is not None
    return out, times_ms


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    parser = argparse.ArgumentParser(
        description="测试 kernels.sve_sparse_gemm.AdaptiveGEMM 的执行时长。"
    )
    parser.add_argument(
        "--x-csv",
        type=Path,
        default=script_dir / "high_sparsity_x.csv",
        help="x 的 CSV 路径，默认读取脚本同目录下的 high_sparsity_x.csv",
    )
    parser.add_argument(
        "--w-csv",
        type=Path,
        default=script_dir / "high_sparsity_w.csv",
        help="W 的 CSV 路径，默认读取脚本同目录下的 high_sparsity_w.csv",
    )
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    x_2d = generate_extreme_row_sparsity_matrix(128,4096,)
    w = load_csv_matrix(args.w_csv, dtype=dtype, device=args.device)

    if x_2d.dim() != 2 or w.dim() != 2:
        raise ValueError("x 和 W 必须都是二维矩阵。")
    if x_2d.size(1) != w.size(0):
        raise ValueError(
            f"矩阵维度不匹配: x.shape={tuple(x_2d.shape)}, W.shape={tuple(w.shape)}"
        )

    out, times_ms = benchmark_adaptive_gemm(
        x_2d=x_2d,
        w=w,
        threshold=args.threshold,
        warmup=args.warmup,
        runs=args.runs,
    )

    out,time_ms = benchmark_dense_gemm(
        x_2d=x_2d,
        w=w,
        warmup=args.warmup,
        runs=args.runs,
    )

    print(f"x_2d shape: {tuple(x_2d.shape)}")
    print(f"W shape: {tuple(w.shape)}")
    print(f"x shape for AdaptiveGEMM: {(1, x_2d.size(0), x_2d.size(1))}")
    print(f"threshold: {args.threshold}")
    print(f"warmup: {args.warmup}, runs: {args.runs}")
    print(f"output shape: {tuple(out.shape)}")
    print(f"avg: {statistics.mean(times_ms):.6f} ms")
    print(f"min: {min(times_ms):.6f} ms")
    print(f"max: {max(times_ms):.6f} ms")

    print(f"avg: {statistics.mean(time_ms):.6f} ms")
    print(f"min: {min(time_ms):.6f} ms")
    print(f"max: {max(time_ms):.6f} ms")


if __name__ == "__main__":
    main()
