#include <torch/extension.h>

#include <cstdint>
#include <omp.h>
#include <vector>
#include <algorithm>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

/**
 * COO × dense weight with SVE gather optimization
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
 *   Optimized with SVE gather load for weight matrix access
 */

namespace {

void check_inputs_coo_sve_gather(
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
  TORCH_CHECK(K == weight.size(0), "K must be equal to weight.size(0)");
  TORCH_CHECK(N == weight.size(1), "N must be equal to weight.size(1)");
}

} // namespace

torch::Tensor sparse_gemm_coo_sve_gather(
    torch::Tensor weight,
    torch::Tensor row_indices,
    torch::Tensor col_indices,
    torch::Tensor values,
    int64_t M,
    int64_t K,
    int64_t N) {
  check_inputs_coo_sve_gather(weight, row_indices, col_indices, values, M, K, N);

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

  // Convert row_indices to row offsets by traversing
  std::vector<int64_t> row_offsets(M + 1, 0);
  for (int64_t i = 0; i < nnz; ++i) {
    const int64_t row = row_indices_ptr[i];
    row_offsets[row + 1]++;
  }
  // Accumulate prefix sum to get the starting position of each row
  for (int64_t i = 0; i < M; ++i) {
    row_offsets[i + 1] += row_offsets[i];
  }

  // Computation path optimized with SVE gather
#if defined(__ARM_FEATURE_SVE)
  const int64_t vl = svcntw();
  const uint32_t N_u32 = (uint32_t)N;

  // N-dimension blocking, following CSR SVE gather strategy
  int64_t n_block_sz = N/16;
  const int64_t n_full = (N / n_block_sz) * n_block_sz;
  const int64_t rem = N - n_full;

  #pragma omp parallel
  {
    // Full blocks: parallelize over rows and N-blocks
    #pragma omp for collapse(2) schedule(static)
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < n_full; n += n_block_sz) {
        const int64_t p0 = row_offsets[m];
        const int64_t p1 = row_offsets[m + 1];
        const int64_t row_nnz = p1 - p0;
        
        if (row_nnz == 0) continue;

        // COO data pointers for this row
        const float* coo_values_ptr = values_ptr + p0;
        const uint32_t* coo_col_idx_ptr = col_indices_ptr + p0;
        float* out_row_ptr = out_ptr + m * N;
        
        std::vector<float> acc(n_block_sz, 0.0f);
        const float* base = weight_ptr + n;

        // Process non-zeros in chunks of vl (SVE vector length)
        for (int64_t i = 0; i < row_nnz; i += vl) {
          const svbool_t pg = svwhilelt_b32(i, row_nnz);
          
          // Continuous load from COO values (vectorized)
          const svfloat32_t sparse_vals = svld1_f32(pg, coo_values_ptr + i);
          
          // Load column indices and compute weight row indices
          const svuint32_t idx = svld1_u32(pg, coo_col_idx_ptr + i);
          const svuint32_t w_index = svmul_n_u32_x(pg, idx, N_u32);

          // Gather load weight and accumulate for each element in the block
          for (int64_t r = 0; r < n_block_sz; r++) {
            const svfloat32_t w_vals = svld1_gather_u32index_f32(pg, base + r, w_index);
            acc[r] += svaddv_f32(pg, svmul_f32_m(pg, sparse_vals, w_vals));
          }
        }

        // Write accumulated results
        for (int64_t r = 0; r < n_block_sz; r++) {
          out_row_ptr[n + r] += acc[r];
        }
      }
    }

    // Handle remainder columns
    if (rem > 0) {
      #pragma omp for schedule(static)
      for (int64_t m = 0; m < M; ++m) {
        const int64_t p0 = row_offsets[m];
        const int64_t p1 = row_offsets[m + 1];
        const int64_t row_nnz = p1 - p0;
        
        if (row_nnz == 0) continue;

        const float* coo_values_ptr = values_ptr + p0;
        const uint32_t* coo_col_idx_ptr = col_indices_ptr + p0;
        float* out_row_ptr = out_ptr + m * N;
        const int64_t n_start = n_full;
        
        std::vector<float> acc(rem, 0.0f);

        for (int64_t i = 0; i < row_nnz; i += vl) {
          const svbool_t pg = svwhilelt_b32(i, row_nnz);
          
          // Continuous load from COO values
          const svfloat32_t sparse_vals = svld1_f32(pg, coo_values_ptr + i);
          
          // Load column indices and compute weight row indices
          const svuint32_t idx = svld1_u32(pg, coo_col_idx_ptr + i);
          const svuint32_t w_index = svmul_n_u32_x(pg, idx, N_u32);

          for (int64_t r = 0; r < rem; ++r) {
            const svfloat32_t w_vals = svld1_gather_u32index_f32(pg, weight_ptr + (n_start + r), w_index);
            acc[r] += svaddv_f32(pg, svmul_f32_m(pg, sparse_vals, w_vals));
          }
        }

        for (int64_t r = 0; r < rem; ++r) {
          out_row_ptr[n_start + r] += acc[r];
        }
      }
    }
  }
#else
  // Scalar fallback path
  for (int64_t m = 0; m < M; ++m) {
    const int64_t p0 = row_offsets[m];
    const int64_t p1 = row_offsets[m + 1];
    float* out_row = out_ptr + m * N;
    
    for (int64_t p = p0; p < p1; ++p) {
      const uint32_t k = col_indices_ptr[p];
      const float a = values_ptr[p];
      const float* w_row = weight_ptr + (int64_t)k * N;
      for (int64_t n = 0; n < N; ++n) {
        out_row[n] += a * w_row[n];
      }
    }
  }
#endif
  return output;
}

// regist to torch.ops.sparse_op
TORCH_LIBRARY_FRAGMENT(sparse_op, m) {
  m.def("sparse_gemm_coo_sve_gather(Tensor weight, Tensor row_indices, Tensor col_indices, Tensor values, int M, int K, int N) -> Tensor");
}

TORCH_LIBRARY_IMPL(sparse_op, CPU, m) {
  m.impl("sparse_gemm_coo_sve_gather", sparse_gemm_coo_sve_gather);
}
