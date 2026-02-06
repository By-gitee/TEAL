#include <torch/extension.h>

#include <cstdint>
#include <omp.h>
#include <vector>
#include <algorithm>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

/**
 * COO × dense weight
 *
 * Input:
 *   - weight: (K, N) float32 contiguous CPU
 *   - row_indices: 1D int64 length=nnz (sorted by row)
 *   - col_indices: 1D uint32 length=nnz
 *   - values:      1D float32 length=nnz
 *   - M, K, N: matmul shape info for sparse(M,K) x weight(K,N) -> out(M,N)
 *
 * Computation:
 *   For each non-zero (i, k, a):
 *     out[i, :] += a * weight[k, :]
 */

namespace {

static inline void check_inputs_coo_gemm(
    const torch::Tensor& weight,
    const torch::Tensor& row_indices,
    const torch::Tensor& col_indices,
    const torch::Tensor& values,
    int64_t M,
    int64_t K,
    int64_t N) {
  TORCH_CHECK(weight.device().is_cpu(), "weight must be a CPU tensor");
  TORCH_CHECK(row_indices.device().is_cpu(), "row_indices must be a CPU tensor");
  TORCH_CHECK(col_indices.device().is_cpu(), "col_indices must be a CPU tensor");
  TORCH_CHECK(values.device().is_cpu(), "values must be a CPU tensor");

  TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
  TORCH_CHECK(row_indices.dtype() == torch::kInt64, "row_indices must be int64");
  TORCH_CHECK(col_indices.dtype() == torch::kUInt32, "col_indices must be uint32");
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

  TORCH_CHECK(M >= 0 && K >= 0 && N >= 0, "M,K,N must be non-negative");
  TORCH_CHECK(weight.size(0) == K, "weight.size(0) must equal K");
  TORCH_CHECK(weight.size(1) == N, "weight.size(1) must equal N");
}

} // namespace

torch::Tensor sparse_gemm_coo(
    torch::Tensor weight,
    torch::Tensor row_indices,
    torch::Tensor col_indices,
    torch::Tensor values,
    int64_t M,
    int64_t K,
    int64_t N) {
  check_inputs_coo_gemm(weight, row_indices, col_indices, values, M, K, N);

  const int64_t nnz = values.size(0);

  auto output = torch::zeros({M, N}, weight.options());
  if (M == 0 || K == 0 || N == 0 || nnz == 0) {
    return output;
  }

  const float* weight_ptr = weight.data_ptr<float>();
  const int64_t* row_indices_ptr = row_indices.data_ptr<int64_t>();
  const uint32_t* col_indices_ptr = col_indices.data_ptr<uint32_t>();
  const float* values_ptr = values.data_ptr<float>();
  float* out_ptr = output.data_ptr<float>();

  // row_indices is sorted by row: use counting + prefix sum to get [p0, p1) for each row
  std::vector<int64_t> row_starts(M + 1, 0);
  for (int64_t i = 0; i < nnz; ++i) {
    const int64_t row = row_indices_ptr[i];
    row_starts[row + 1]++;
  }
  // Accumulate prefix sum to get the starting position of each row
  for (int64_t i = 0; i < M; ++i) {
    row_starts[i + 1] += row_starts[i];
  }

  // N-dimension blocking (L1-friendly design)
  const int64_t n_block_sz = N/16;
  const int64_t n_block = (N + n_block_sz - 1) / n_block_sz;

  #pragma omp parallel for collapse(2) schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t nb = 0; nb < n_block; ++nb) {
      const int64_t p0 = row_starts[m];
      const int64_t p1 = row_starts[m + 1];
      if (p0 == p1) continue;

      const int64_t n0 = nb * n_block_sz;
      const int64_t n1 = std::min<int64_t>(n0 + n_block_sz, N);

      float* out_row = out_ptr + m * N;

#if defined(__ARM_FEATURE_SVE)
      for (int64_t p = p0; p < p1; ++p) {
        const uint32_t k = col_indices_ptr[p];
        const float a = values_ptr[p];
        const float* w_row = weight_ptr + (int64_t)k * N;

        int64_t n = n0;
        for (; n < n1; ) {
          svbool_t pg = svwhilelt_b32(n, n1);
          svfloat32_t ov = svld1_f32(pg, out_row + n);
          svfloat32_t wv = svld1_f32(pg, w_row  + n);
          svfloat32_t rv = svmla_n_f32_m(pg, ov, wv, a);
          svst1_f32(pg, out_row + n, rv);
          n += svcntw();
        }
      }
#else
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

// Register to torch.ops.sparse_op
TORCH_LIBRARY_FRAGMENT(sparse_op, m) {
  m.def("sparse_gemm_coo(Tensor weight, Tensor row_indices, Tensor col_indices, Tensor values, int M, int K, int N) -> Tensor");
}

TORCH_LIBRARY_IMPL(sparse_op, CPU, m) {
  m.impl("sparse_gemm_coo", sparse_gemm_coo);
}
