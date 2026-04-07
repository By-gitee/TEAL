import torch
import time
import itertools
import csv
import os

from kernels.sve_sparse_gemm import (
    SparseGEMMiCSRSVEGatherKernel,
    thr_sparsify_to_icsr,
    dense_gemm_sve_omp
)

# ==========================
# CPU环境配置（强烈建议固定）
# ==========================
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["OMP_PROC_BIND"] = "true"
os.environ["OMP_PLACES"] = "cores"

torch.set_num_threads(16)

# ==========================
# 参数配置
# ==========================
M_list = [1, 64]
KN_list = [
    (4096, 4096),
    (4096, 11008),
    (11008, 4096),
]

threshold_list = [0.0,0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95,0.99]

B = 1
DTYPE = torch.float32
DEVICE = "cpu"

WARMUP = 5
REPEAT = 1000

OUTPUT = "sve_vs_dense.csv"


# ==========================
# 工具函数
# ==========================
def measure(func, *args):
    # warmup
    for _ in range(WARMUP):
        func(*args)

    start = time.perf_counter()
    for _ in range(REPEAT):
        func(*args)
    end = time.perf_counter()

    return (end - start) / REPEAT * 1000  # ms


def compute_density(x, thr):
    nnz = (x.abs() >= thr).sum().item()
    return nnz / x.numel()



# ==========================
# 主函数
# ==========================
def main():
    print("Loading SVE extension...")

    # 正确的sve_sparse_gemm算子调用方式
    icsr_sve_gather_kernel = SparseGEMMiCSRSVEGatherKernel.initialize(
        name="sparse_gemm_icsr_sve_gather", target="CPU"
    )
    sve_gather_op = icsr_sve_gather_kernel.operator(compiled=True)

    def sve_e2e(x, W, thr):
        x_2d = x.view(-1, x.shape[-1])
        _, col_idx, row_ptr = thr_sparsify_to_icsr(x_2d, thr)
        return sve_gather_op(x_2d, W, row_ptr, col_idx)

    def dense_e2e(x, W, thr):
        x_2d = x.view(-1, x.shape[-1])
        return dense_gemm_sve_omp(x_2d, W)

    results = []

    for M, (K, N) in itertools.product(M_list, KN_list):
        print(f"\n===== M={M}, K={K}, N={N} =====")

        # 随机数据（0~1）
        x = torch.rand(B, M, K, dtype=DTYPE, device=DEVICE)
        W = torch.rand(K, N, dtype=DTYPE, device=DEVICE)

        for thr in threshold_list:
            density = compute_density(x, thr)
            sparsity = 1 - density

            # SVE sparse gather (use new operator interface)
            t_sve = measure(sve_e2e, x, W, thr)

            # Dense SVE GEMM
            t_dense = measure(dense_e2e, x, W, thr)

            speedup = t_dense / t_sve if t_sve > 0 else 0

            print(
                f"[thr={thr:.2f}] "
                f"sparsity={sparsity:.3f} | "
                f"SVE={t_sve:.3f} ms | "
                f"Dense={t_dense:.3f} ms | "
                f"speedup={speedup:.2f}x"
            )

            results.append({
                "M": M,
                "K": K,
                "N": N,
                "threshold": thr,
                "density": density,
                "sparsity": sparsity,
                "sve_ms": t_sve,
                "dense_ms": t_dense,
                "speedup_vs_dense": speedup
            })

    # 写CSV
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()