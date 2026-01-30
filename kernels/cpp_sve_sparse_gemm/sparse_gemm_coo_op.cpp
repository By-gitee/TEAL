#include <torch/extension.h>

#include <cstdint>
#include <omp.h>
#include <vector>
#include <algorithm>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

/**
 * 新算子：直接使用COO格式输入，然后做 COO × dense weight
 *
 * 输入 COO 格式：
 *   - weight: (K, N) float32, contiguous, CPU, 稠密权重矩阵
 *   - row_indices: 1D int64, length = nnz, COO格式的行索引（已按行排序）
 *   - col_indices: 1D int64, length = nnz, COO格式的列索引（与row_indices对应）
 *   - values: 1D float32, length = nnz, COO格式的非零元素值（与row_indices对应）
 *
 * 功能目标：
 *   1) 直接使用输入的 COO 格式 (row_indices, col_indices, values)
 *   2) 计算稀疏矩阵(M, K) × weight(K, N) -> output(M, N)
 *   3) 计算过程中使用 SIMD（SVE）加速
 *   4) 使用 OpenMP 并行加速
 *
 * 计算逻辑：
 *   对于每个非零元素 (i, j, val):
 *     output[i, :] += val * weight[j, :]
 *
 * 备注：
 * - 假设输入的 row_indices 已按行索引排序，col_indices 和 values 与之对应排列
 * - 本文件不包含 `PYBIND11_MODULE`，方便后续与现有扩展多源编译链接（避免重复定义）
 * - 当前 SIMD 路径：在支持 SVE 的平台上启用 SVE 快路径
 */

namespace {

void check_inputs_coo_gemm(
    const torch::Tensor& weight,
    const torch::Tensor& row_indices,
    const torch::Tensor& col_indices,
    const torch::Tensor& values) {
  TORCH_CHECK(weight.device().is_cpu(), "weight must be a CPU tensor");
  TORCH_CHECK(row_indices.device().is_cpu(), "row_indices must be a CPU tensor");
  TORCH_CHECK(col_indices.device().is_cpu(), "col_indices must be a CPU tensor");
  TORCH_CHECK(values.device().is_cpu(), "values must be a CPU tensor");

  TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
  TORCH_CHECK(row_indices.dtype() == torch::kInt64, "row_indices must be int64");
  TORCH_CHECK(col_indices.dtype() == torch::kInt64, "col_indices must be int64");
  TORCH_CHECK(values.dtype() == torch::kFloat32, "values must be float32");

  TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
  TORCH_CHECK(row_indices.dim() == 1, "row_indices must be 1D");
  TORCH_CHECK(col_indices.dim() == 1, "col_indices must be 1D");
  TORCH_CHECK(values.dim() == 1, "values must be 1D");

  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(row_indices.is_contiguous(), "row_indices must be contiguous");
  TORCH_CHECK(col_indices.is_contiguous(), "col_indices must be contiguous");
  TORCH_CHECK(values.is_contiguous(), "values must be contiguous");

  const int64_t nnz = values.size(0);
  TORCH_CHECK(row_indices.size(0) == nnz, "row_indices length must equal values length");
  TORCH_CHECK(col_indices.size(0) == nnz, "col_indices length must equal values length");
  
  const int64_t K = weight.size(0);
  TORCH_CHECK(K > 0, "weight must have positive dimensions");
}

} // namespace

torch::Tensor sparse_gemm_coo(
    torch::Tensor weight,
    torch::Tensor row_indices,
    torch::Tensor col_indices,
    torch::Tensor values) {
  check_inputs_coo_gemm(weight, row_indices, col_indices, values);

  const int64_t K = weight.size(0);
  const int64_t N = weight.size(1);
  const int64_t nnz = values.size(0);

  // 推断稀疏矩阵的行数 M
  const int64_t* row_indices_ptr = row_indices.data_ptr<int64_t>();
  int64_t M = 0;
  if (nnz > 0) {
    M = *std::max_element(row_indices_ptr, row_indices_ptr + nnz) + 1;
  }

  auto output = torch::zeros({M, N}, weight.options());
  if (M == 0 || K == 0 || N == 0 || nnz == 0) {
    return output;
  }

  const float* weight_ptr = weight.data_ptr<float>();
  const int64_t* col_indices_ptr = col_indices.data_ptr<int64_t>();
  const float* values_ptr = values.data_ptr<float>();
  float* out_ptr = output.data_ptr<float>();

  // 由于 row_indices 已按行排序，直接构建行起始位置数组
  // 找到每一行的起始和结束位置
  std::vector<int64_t> row_starts(M + 1, 0);
  for (int64_t i = 0; i < nnz; ++i) {
    int64_t row = row_indices_ptr[i];
    row_starts[row + 1]++;
  }
  // 累加前缀和，得到每行的起始位置
  for (int64_t i = 0; i < M; ++i) {
    row_starts[i + 1] += row_starts[i];
  }

  // 选择 N 分块大小：L1-friendly 且是向量宽度的倍数
  constexpr int64_t BN = 512;
  const int64_t NB = (N + BN - 1) / BN;

  // 二维并行：每个线程负责一个 (m, nb) tile
  #pragma omp parallel for collapse(2) schedule(dynamic)
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t nb = 0; nb < NB; ++nb) {
      const int64_t p0 = row_starts[m];
      const int64_t p1 = row_starts[m + 1];
      if (p0 == p1) continue;

      const int64_t n0 = nb * BN;
      const int64_t n1 = std::min<int64_t>(n0 + BN, N);

      float* out_row = out_ptr + m * N;

#if defined(__ARM_FEATURE_SVE)
      // SVE 路径：对 [n0, n1) 做向量累加
      for (int64_t p = p0; p < p1; ++p) {
        const int64_t k = col_indices_ptr[p];
        const float a = values_ptr[p];
        const float* w_row = weight_ptr + k * N;

        int64_t n = n0;
        for (; n < n1; ) {
          // 以 32-bit lane 计数的谓词：覆盖 [n, n1)
          svbool_t pg = svwhilelt_b32(n, n1);

          // load out / weight，做 FMA，然后 store 回 out
          svfloat32_t ov = svld1_f32(pg, out_row + n);
          svfloat32_t wv = svld1_f32(pg, w_row  + n);
          svfloat32_t rv = svmla_n_f32_m(pg, ov, wv, a);
          svst1_f32(pg, out_row + n, rv);

          n += svcntw(); // 前进一个 VL（以 float32 lane 数）
        }
      }
#else
      // 标量路径：只更新该 N tile
      for (int64_t p = p0; p < p1; ++p) {
        const int64_t k = col_indices_ptr[p];
        const float a = values_ptr[p];
        const float* w_row = weight_ptr + k * N;
        for (int64_t n = n0; n < n1; ++n) {
          out_row[n] += a * w_row[n];
        }
      }
#endif
    }
  }

  return output;
}

// 注册到 torch.ops.sparse_op
TORCH_LIBRARY_FRAGMENT(sparse_op, m) {
  m.def("sparse_gemm_coo(Tensor weight, Tensor row_indices, Tensor col_indices, Tensor values) -> Tensor");
}

TORCH_LIBRARY_IMPL(sparse_op, CPU, m) {
  m.impl("sparse_gemm_coo", sparse_gemm_coo);
}
