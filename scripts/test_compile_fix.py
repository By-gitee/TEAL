"""
快速测试编译修复是否成功
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

print("尝试加载 SVE 稀疏 GEMM 扩展...")
from kernels.sve_sparse_gemm import load_sve_sparse_gemm_extension

try:
    load_sve_sparse_gemm_extension(verbose=True, rebuild=True)
    print("\n✅ 编译成功！扩展已加载。")
except Exception as e:
    print(f"\n❌ 编译失败: {e}")
    sys.exit(1)

print("\n测试基本功能...")
import torch
from kernels.sve_sparse_gemm import (
    thr_sparsify_to_icsr_sve,
    thr_sparsify_to_icsr_sve_baseline,
)

# 创建简单测试数据
activation = torch.randn(128, 256, dtype=torch.float32)
threshold = 0.01

print("测试 SVE2 版本...")
result1 = thr_sparsify_to_icsr_sve(activation, threshold)
print(f"  输出: nz_counts.shape={result1[0].shape}, col_indices.shape={result1[1].shape}")

print("测试 Baseline 版本...")
result2 = thr_sparsify_to_icsr_sve_baseline(activation, threshold)
print(f"  输出: nz_counts.shape={result2[0].shape}, col_indices.shape={result2[1].shape}")

print("\n验证输出一致性...")
assert torch.equal(result1[0], result2[0]), "nz_counts 不匹配"
assert torch.equal(result1[1], result2[1]), "col_indices 不匹配"
assert torch.equal(result1[2], result2[2]), "row_offsets 不匹配"

print("✅ 所有测试通过！")
