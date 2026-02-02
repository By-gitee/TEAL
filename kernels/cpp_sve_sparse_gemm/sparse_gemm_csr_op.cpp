#include <torch/extension.h>

#include <cstdint>
#include <omp.h>
#include <vector>
#include <algorithm>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

/**
 * 新算子：直接使用CSR格式输入，然后做 CSR × dense weight
 *
 * 输入 CSR 格式：
 *   - weight: (K, N) float32, contiguous, CPU
 *   - row_offsets: 1D int64, length = M + 1, CSR格式的row_ptr（前缀和）
 *   - nz_col_indices: 1D uint32/int32, flattened col indices for all non-zeros
 *   - values: 1D float32, CSR格式的非零元素值
 *
 * 功能目标：
 *   1) 直接使用输入的 CSR 格式 (values, col_idx, row_ptr)
 *   2) 使用 CSR 与 weight 做乘法输出 (M, N)
 *   3) 计算过程中使用 SIMD（SVE）加速
 *   4) 在 activation 行间（M 维）并行加速（OpenMP）
 *
 * 备注：
 * - 本文件刻意不包含 `PYBIND11_MODULE`，方便后续与现有扩展多源编译链接（避免重复定义）。
 * - 当前 SIMD 路径沿用仓库现状：仅在 `svcntw()==4` 时启用 SVE 快路径。
 */

namespace {

void check_inputs_csr_gemm(
    const torch::Tensor& weight,
    const torch::Tensor& row_offsets,
    const torch::Tensor& nz_col_indices,
    const torch::Tensor& values) {
  TORCH_CHECK(weight.device().is_cpu(), "weight must be a CPU tensor");
  TORCH_CHECK(row_offsets.device().is_cpu(), "row_offsets must be a CPU tensor");
  TORCH_CHECK(nz_col_indices.device().is_cpu(), "nz_col_indices must be a CPU tensor");
  TORCH_CHECK(values.device().is_cpu(), "values must be a CPU tensor");

  TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
  TORCH_CHECK(row_offsets.dtype() == torch::kInt64, "row_offsets must be int64");
  TORCH_CHECK(
      nz_col_indices.dtype() == torch::kUInt32 || nz_col_indices.dtype() == torch::kInt32,
      "nz_col_indices must be uint32 or int32");
  TORCH_CHECK(values.dtype() == torch::kFloat32, "values must be float32");

  TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
  TORCH_CHECK(row_offsets.dim() == 1, "row_offsets must be 1D");
  TORCH_CHECK(nz_col_indices.dim() == 1, "nz_col_indices must be 1D");
  TORCH_CHECK(values.dim() == 1, "values must be 1D");

  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(row_offsets.is_contiguous(), "row_offsets must be contiguous");
  TORCH_CHECK(nz_col_indices.is_contiguous(), "nz_col_indices must be contiguous");
  TORCH_CHECK(values.is_contiguous(), "values must be contiguous");

  const auto K = weight.size(0);
  const auto M = row_offsets.size(0) - 1;
  TORCH_CHECK(M > 0, "row_offsets length must be at least 2 (M+1 >= 2)");
  TORCH_CHECK(row_offsets.size(0) == M + 1, "row_offsets length must be M+1");
  const int64_t* row_offsets_ptr = row_offsets.data_ptr<int64_t>();
  TORCH_CHECK(row_offsets_ptr[0] == 0, "row_offsets[0] must be 0");
  const int64_t nnz = row_offsets_ptr[M];
  TORCH_CHECK(nnz == nz_col_indices.size(0), 
              "row_offsets[M] must equal nz_col_indices length");
  TORCH_CHECK(nnz == values.size(0), 
              "values length must equal row_offsets[M] (nnz)");
}

} // namespace

torch::Tensor sparse_gemm_csr(
    torch::Tensor weight,
    torch::Tensor row_offsets,
    torch::Tensor nz_col_indices,
    torch::Tensor values) {
  check_inputs_csr_gemm(weight, row_offsets, nz_col_indices, values);

  const int64_t K = weight.size(0);
  const int64_t N = weight.size(1);
  const int64_t M = row_offsets.size(0) - 1;

  auto output = torch::zeros({M, N}, weight.options());
  if (M == 0 || K == 0 || N == 0) {
    return output;
  }

  const float* weight_ptr = weight.data_ptr<float>();
  const int64_t* row_offsets_ptr = row_offsets.data_ptr<int64_t>();
  const uint32_t* indices_ptr = nz_col_indices.data_ptr<uint32_t>();
  const float* values_ptr = values.data_ptr<float>();
  float* out_ptr = output.data_ptr<float>();

  // CSR × dense weight -> dense output
  // 直接使用输入的CSR数据指针，不需要构建CSR结构
  // 选一个 N 分块大小：建议是 L1-friendly 且是 16 的倍数（方便向量化）
  // 经验：256/384/512 都常用。你可以按平台调参。
  const int64_t BN = N/16;

  const int64_t NB = (N + BN - 1) / BN;

  // 二维并行：每个线程负责一个 (m, nb) tile，写 out_row[n0:n1) 无冲突
  #pragma omp parallel for collapse(2) schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t nb = 0; nb < NB; ++nb) {
      const int64_t p0 = row_offsets_ptr[m];
      const int64_t p1 = row_offsets_ptr[m + 1];
      if (p0 == p1) continue;

      const int64_t n0 = nb * BN;
      const int64_t n1 = std::min<int64_t>(n0 + BN, N);

      float* out_row = out_ptr + m * N;

#if defined(__ARM_FEATURE_SVE)
      // SVE 路径：对 [n0, n1) 做向量累加，尾部用谓词处理
      for (int64_t p = p0; p < p1; ++p) {
        const uint32_t k = indices_ptr[p];
        const float a = values_ptr[p];
        const float* w_row = weight_ptr + (int64_t)k * N;

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
        const uint32_t k = indices_ptr[p];
        const float a = values_ptr[p];
        const float* w_row = weight_ptr + (int64_t)k * N;
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
  m.def("sparse_gemm_csr(Tensor weight, Tensor row_offsets, Tensor nz_col_indices, Tensor values) -> Tensor");
}

TORCH_LIBRARY_IMPL(sparse_op, CPU, m) {
  m.impl("sparse_gemm_csr", sparse_gemm_csr);
}
