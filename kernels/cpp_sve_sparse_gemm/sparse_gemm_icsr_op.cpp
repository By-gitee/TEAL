#include <torch/extension.h>

#include <cstdint>
#include <omp.h>
#include <vector>
#include <algorithm>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

/**
 * iCSR(index-only CSR) × dense weight
 *
 * Input:
 *   - activation: (M, K) float32 contiguous CPU
 *   - weight: (K, N) float32 contiguous CPU
 *   - row_offsets: 1D int64 length=M+1 (CSR row pointers)
 *   - nz_col_indices: 1D uint32 length=nnz (CSR column indices)
 *
 * Computation:
 *   For each row i and its non-zeros (k, a):
 *     out[i, :] += a * weight[k, :]
 */

namespace {

void check_inputs_icsr_gemm(
    const torch::Tensor& activation,
    const torch::Tensor& weight,
    const torch::Tensor& row_offsets,
    const torch::Tensor& nz_col_indices) {
  TORCH_CHECK(activation.device().is_cpu(), "activation must be a CPU tensor");
  TORCH_CHECK(weight.device().is_cpu(), "weight must be a CPU tensor");
  TORCH_CHECK(row_offsets.device().is_cpu(), "row_offsets must be a CPU tensor");
  TORCH_CHECK(nz_col_indices.device().is_cpu(), "nz_col_indices must be a CPU tensor");

  TORCH_CHECK(activation.dtype() == torch::kFloat32, "activation must be float32");
  TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
  TORCH_CHECK(row_offsets.dtype() == torch::kInt64, "row_offsets must be int64");
  TORCH_CHECK(nz_col_indices.dtype() == torch::kUInt32, "nz_col_indices must be uint32");

  TORCH_CHECK(activation.dim() == 2, "activation must be 2D");
  TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
  TORCH_CHECK(row_offsets.dim() == 1, "row_offsets must be 1D");
  TORCH_CHECK(nz_col_indices.dim() == 1, "nz_col_indices must be 1D");

  TORCH_CHECK(activation.is_contiguous(), "activation must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(row_offsets.is_contiguous(), "row_offsets must be contiguous");
  TORCH_CHECK(nz_col_indices.is_contiguous(), "nz_col_indices must be contiguous");

  const auto M = activation.size(0);
  const auto K = activation.size(1);
  TORCH_CHECK(weight.size(0) == K, "weight K dimension must match activation K");
  TORCH_CHECK(row_offsets.size(0) == M + 1, "row_offsets length must be M+1");
    TORCH_CHECK(row_offsets.data_ptr<int64_t>()[0] == 0, "row_offsets[0] must be 0");
    TORCH_CHECK(row_offsets.data_ptr<int64_t>()[M] == nz_col_indices.numel(), 
                "row_offsets[M] must equal nz_col_indices size");
  }
} // namespace

torch::Tensor sparse_gemm_icsr(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor row_offsets,
    torch::Tensor nz_col_indices) {
  check_inputs_icsr_gemm(activation, weight, row_offsets, nz_col_indices);

  const int64_t M = activation.size(0);
  const int64_t K = activation.size(1);
  const int64_t N = weight.size(1);

  auto output = torch::zeros({M, N}, activation.options());
  if (M == 0 || K == 0 || N == 0) {
    return output;
  }

  const float* act_ptr = activation.data_ptr<float>();
  const float* weight_ptr = weight.data_ptr<float>();
  const int64_t* row_offsets_ptr = row_offsets.data_ptr<int64_t>();
  const uint32_t* indices_ptr = nz_col_indices.data_ptr<uint32_t>();
  float* out_ptr = output.data_ptr<float>();

  const int64_t n_block_sz = N/16;
  
  const int64_t n_block = (N + n_block_sz - 1) / n_block_sz;
  
  // 2D parallelization: each thread handles one (m, nb) tile
  #pragma omp parallel for collapse(2) schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t nb = 0; nb < n_block; ++nb) {
      const int64_t p0 = row_offsets_ptr[m];
      const int64_t p1 = row_offsets_ptr[m + 1];
      if (p0 == p1) continue;
      
      const int64_t n0 = nb * n_block_sz;
      const int64_t n1 = std::min<int64_t>(n0 + n_block_sz, N);
      
      float* out_row = out_ptr + m * N;
      const float* act_row_ptr = act_ptr + m * K;
      
      #if defined(__ARM_FEATURE_SVE)
        // SVE path: vectorized accumulation for [n0, n1) with predicate for tail
        for (int64_t p = p0; p < p1; ++p) {
          const uint32_t k = indices_ptr[p];
          const float a = act_row_ptr[(int64_t)k];
          const float* w_row = weight_ptr + (int64_t)k * N;
          
          int64_t n = n0;
          for (; n < n1; ) {
            // Predicate for 32-bit lanes covering [n, n1)
            svbool_t pg = svwhilelt_b32(n, n1);
            
            // Load output/weight, perform FMA, then store back to output
            svfloat32_t ov = svld1_f32(pg, out_row + n);
            svfloat32_t wv = svld1_f32(pg, w_row  + n);
            svfloat32_t rv = svmla_n_f32_m(pg, ov, wv, a);
            svst1_f32(pg, out_row + n, rv);
            
            n += svcntw(); // Advance by one vector length (in float32 lanes)
          }
        }
      #else
        // Scalar path: update only this N tile
        for (int64_t p = p0; p < p1; ++p) {
          const uint32_t k = indices_ptr[p];
          const float a = act_row_ptr[(int64_t)k];
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

// Register to torch.ops.sparse_op
TORCH_LIBRARY_FRAGMENT(sparse_op, m) {
  m.def("sparse_gemm_icsr(Tensor activation, Tensor weight, Tensor row_offsets, Tensor nz_col_indices) -> Tensor");
}

TORCH_LIBRARY_IMPL(sparse_op, CPU, m) {
  m.impl("sparse_gemm_icsr", sparse_gemm_icsr);
}
