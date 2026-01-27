#include <torch/extension.h>

#include <cstdint>
#include <iostream>
#include <omp.h>
#include <limits>
#include <vector>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif


void check_gemv_inputs(
    const torch::Tensor& activation,
    const torch::Tensor& weight,
    const torch::Tensor& nz_col_index,
    int64_t nz_row) {
  TORCH_CHECK(activation.device().is_cpu(), "activation must be a CPU tensor");
  TORCH_CHECK(weight.device().is_cpu(), "weight must be a CPU tensor");
  TORCH_CHECK(nz_col_index.device().is_cpu(), "nz_col_index must be a CPU tensor");

  TORCH_CHECK(activation.dtype() == torch::kFloat32, "activation must be float32");
  TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
  TORCH_CHECK(nz_col_index.dtype() == torch::kUInt32, "nz_col_index must be uint32");

  TORCH_CHECK(activation.dim() == 2, "activation must be 2D");
  TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
  TORCH_CHECK(nz_col_index.dim() == 1, "nz_col_index must be 1D");

  TORCH_CHECK(activation.is_contiguous(), "activation must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(nz_col_index.is_contiguous(), "nz_col_index must be contiguous");

  const auto M = activation.size(0);
  const auto K = activation.size(1);
  TORCH_CHECK(weight.size(0) == K, "weight K dimension must match activation K");

  TORCH_CHECK(nz_row >= 0 && nz_row < M, "nz_row out of range");
}


torch::Tensor sve_sparse_gemv(
    torch::Tensor activation,
    torch::Tensor weight,
    int64_t nz_row,
    torch::Tensor nz_col_index) {
  check_gemv_inputs(activation, weight, nz_col_index, nz_row);

  const auto K = weight.size(0);
  const auto N = weight.size(1);
  const auto nnz = nz_col_index.numel();

  auto output = torch::zeros({N}, activation.options());
  if (nnz == 0 || N == 0) {
    return output;
  }
  if (K == 0) {
    TORCH_CHECK(nnz == 0, "nz_col_index must be empty when K=0");
    return output;
  }

  const float* act_row_ptr = activation.data_ptr<float>() + nz_row * K;
  const float* weight_ptr = weight.data_ptr<float>();
  const uint32_t* idx_ptr = nz_col_index.data_ptr<uint32_t>();
  float* out_ptr = output.data_ptr<float>();

#if defined(__ARM_FEATURE_SVE)
    const int64_t vl = svcntw();
    const uint32_t N_u32 = static_cast<uint32_t>(N);

  if (vl == 4) {
    // N-dimension blocking (aligned with sve_sparse_gemm's n_block_sz idea)
    int64_t n_block_sz = N / 16;
    if (n_block_sz < 4) {
      n_block_sz = 4;
    }
    const int64_t n_full = (N / n_block_sz) * n_block_sz;
    const int64_t rem = N - n_full;

    #pragma omp parallel
    {
      // Full blocks: parallelize over N-blocks, each block accumulates locally then writes once.
      #pragma omp for schedule(static)
      for (int64_t n = 0; n < n_full; n += n_block_sz) {
        std::vector<float> acc(n_block_sz, 0.0f);
        const float* base = weight_ptr + n;

        for (int64_t i = 0; i < nnz; i += 4) {
          const svbool_t pg = svwhilelt_b32(i, nnz);
          const svuint32_t idx = svld1_u32(pg, idx_ptr + i);
          const svfloat32_t act_vals =
              svld1_gather_u32index_f32(pg, act_row_ptr, idx);
          const svuint32_t w_index = svmul_n_u32_x(pg, idx, N_u32);  // idx * N

          for (int64_t r = 0; r < n_block_sz; ++r) {
            const svfloat32_t w_vals =
                svld1_gather_u32index_f32(pg, base + r, w_index);
            acc[r] += svaddv_f32(pg, svmul_f32_m(pg, act_vals, w_vals));
          }
        }

        for (int64_t r = 0; r < n_block_sz; ++r) {
          out_ptr[n + r] += acc[r];
        }
      }  

      // Tail: one last partial block (small), computed once.
      if (rem > 0) {
        #pragma omp single
        {
          const int64_t n_start = n_full;
          std::vector<float> acc(rem, 0.0f);

          for (int64_t i = 0; i < nnz; i += 4) {
            const svbool_t pg = svwhilelt_b32(i, nnz);
            const svuint32_t idx = svld1_u32(pg, idx_ptr + i);
            const svfloat32_t act_vals =
                svld1_gather_u32index_f32(pg, act_row_ptr, idx);
            const svuint32_t w_index =
                svmul_n_u32_x(pg, idx, N_u32);  // idx * N

            for (int64_t r = 0; r < rem; ++r) {
              const svfloat32_t w_vals = svld1_gather_u32index_f32(
                  pg, weight_ptr + (n_start + r), w_index);
              acc[r] += svaddv_f32(pg, svmul_f32_m(pg, act_vals, w_vals));
            }
          }

          for (int64_t r = 0; r < rem; ++r) {
            out_ptr[n_start + r] += acc[r];
          }
        }
      }
    }  

    return output;
  }
#endif

  // Scalar computation (used when SVE is unavailable or unsafe).
  for (int64_t i = 0; i < nnz; ++i) {
    const uint32_t k = idx_ptr[i];
    const float a = act_row_ptr[k];
    if (a == 0.0f) {
      continue;
    }
    const float* w_row = weight_ptr + k * N;
    for (int64_t n = 0; n < N; ++n) {
      out_ptr[n] += a * w_row[n];
    }
  }

  return output;
}



// Check inputs for sparse GEMM operation
void check_gemm_inputs(
    const torch::Tensor& activation,
    const torch::Tensor& weight,
    const torch::Tensor& nz_counts,
    const torch::Tensor& nz_col_indices) {
  TORCH_CHECK(activation.device().is_cpu(), "activation must be a CPU tensor");
  TORCH_CHECK(weight.device().is_cpu(), "weight must be a CPU tensor");
  TORCH_CHECK(nz_counts.device().is_cpu(), "nz_counts must be a CPU tensor");
  TORCH_CHECK(nz_col_indices.device().is_cpu(), "nz_col_indices must be a CPU tensor");

  TORCH_CHECK(activation.dtype() == torch::kFloat32, "activation must be float32");
  TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
  TORCH_CHECK(nz_counts.dtype() == torch::kInt64, "nz_counts must be int64");
  TORCH_CHECK(nz_col_indices.dtype() == torch::kUInt32, "nz_col_indices must be uint32");

  TORCH_CHECK(activation.dim() == 2, "activation must be 2D");
  TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
  TORCH_CHECK(nz_counts.dim() == 1, "nz_counts must be 1D");
  TORCH_CHECK(nz_col_indices.dim() == 1, "nz_col_indices must be 1D");

  TORCH_CHECK(activation.is_contiguous(), "activation must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(nz_counts.is_contiguous(), "nz_counts must be contiguous");
  TORCH_CHECK(nz_col_indices.is_contiguous(), "nz_col_indices must be contiguous");

  const auto K = activation.size(1);
  TORCH_CHECK(weight.size(0) == K, "weight K dimension must match activation K");
  TORCH_CHECK(nz_counts.size(0) % 2 == 0, "nz_counts length must be even (pairs of row_idx, count)");
}

// Sparse GEMM: (M, K) sparse × (K, N) dense → (M, N)
// nz_counts: pairs of (row_idx, count) for rows with non-zero elements
// nz_col_indices: flattened column indices for all non-zero elements
torch::Tensor sve_sparse_gemm(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor nz_counts,
    torch::Tensor nz_col_indices) {
  // check_gemm_inputs(activation, weight, nz_counts, nz_col_indices);

  const auto M = activation.size(0);
  const auto K = activation.size(1);
  const auto N = weight.size(1);

  auto output = torch::zeros({M, N}, activation.options());
  
  if (M == 0 || N == 0 || K == 0) {
    return output;
  }

  const float* act_ptr = activation.data_ptr<float>();
  const float* weight_ptr = weight.data_ptr<float>();
  const int64_t* counts_ptr = nz_counts.data_ptr<int64_t>();
  const uint32_t* indices_ptr = nz_col_indices.data_ptr<uint32_t>();
  float* out_ptr = output.data_ptr<float>();

  // Parse nz_counts: pairs of (row_idx, count)
  const int64_t num_nz_rows = nz_counts.size(0) / 2;
  
  // Build mapping from row index to offset in nz_col_indices
  std::vector<int64_t> row_indices;
  std::vector<int64_t> row_offsets;
  int64_t cumulative_offset = 0;
  
  for (int64_t i = 0; i < num_nz_rows; ++i) {
    const int64_t row_idx = counts_ptr[2 * i];
    const int64_t count = counts_ptr[2 * i + 1];
    
    // TORCH_CHECK(row_idx >= 0 && row_idx < M, "row_idx out of range");
    // TORCH_CHECK(count >= 0, "count must be non-negative");
    
    row_indices.push_back(row_idx);
    row_offsets.push_back(cumulative_offset);
    cumulative_offset += count;
  }

  // // Verify total non-zero count matches
  // TORCH_CHECK(
  //     nz_col_indices.numel() == cumulative_offset,
  //     "nz_col_indices size must equal sum of counts in nz_counts");
#if defined(__ARM_FEATURE_SVE)
  const int64_t vl = svcntw();
  const uint32_t N_u32 = (uint32_t)N;
  
  if (vl == 4) {
    int64_t n_block_sz = N/16;
    const int64_t n_full = (N / n_block_sz) * n_block_sz;
    const int64_t rem = N - n_full;

    #pragma omp parallel
    {
      #pragma omp for collapse(2) schedule(static)
      for (int64_t row_idx = 0; row_idx < num_nz_rows; ++row_idx) {
        for (int64_t n = 0; n < n_full; n += n_block_sz) {

          const int64_t m = row_indices[row_idx];
          const int64_t nnz = counts_ptr[2 * row_idx + 1];
          if (nnz == 0) continue;

          const float* act_row_ptr = act_ptr + m * K;
          const uint32_t* idx_ptr = indices_ptr + row_offsets[row_idx];

          float* out_row_ptr = out_ptr + m * N;
          std::vector<float> acc(n_block_sz, 0.0f);

          for (int64_t i = 0; i < nnz; i += 4) {
            const svbool_t pg = svwhilelt_b32(i, nnz);
            const svuint32_t idx = svld1_u32(pg, idx_ptr + i);

            // gather load act nonzeros
            const svfloat32_t act_vals = svld1_gather_u32index_f32(pg, act_row_ptr, idx);

            const svuint32_t w_index = svmul_n_u32_x(pg, idx, N_u32);

            const float* base = weight_ptr + n;

            for(int64_t r = 0; r < n_block_sz; r++) {
              const svfloat32_t w_vals = svld1_gather_u32index_f32(pg, base + r, w_index);
              acc[r] += svaddv_f32(pg, svmul_f32_m(pg, act_vals, w_vals));
            }
          }

          for(int64_t r = 0; r < n_block_sz; r++) {
            out_row_ptr[n + r] += acc[r];
          }
        } 
      } 

      if (rem > 0) {
        #pragma omp for schedule(static)
        for (int64_t row_idx = 0; row_idx < num_nz_rows; ++row_idx) {
          const int64_t m = row_indices[row_idx];
          const int64_t nnz = counts_ptr[2 * row_idx + 1];
          if (nnz == 0) continue;

          const float* act_row_ptr = act_ptr + m * K;
          const uint32_t* idx_ptr = indices_ptr + row_offsets[row_idx];
          float* out_row_ptr = out_ptr + m * N;

          const int64_t n_start = n_full;
          
          std::vector<float> acc(rem, 0.0f);

          for (int64_t i = 0; i < nnz; i += 4) {
            const svbool_t pg = svwhilelt_b32(i, nnz);
            const svuint32_t idx = svld1_u32(pg, idx_ptr + i);
            const svfloat32_t act_vals = svld1_gather_u32index_f32(pg, act_row_ptr, idx);
            const svuint32_t w_index = svmul_n_u32_x(pg, idx, N_u32);

            for (int64_t r = 0; r < rem; ++r) {
              const svfloat32_t w_vals = svld1_gather_u32index_f32(pg, weight_ptr + (n_start + r), w_index);
              acc[r] += svaddv_f32(pg, svmul_f32_m(pg, act_vals, w_vals));
            }
          }

          for (int64_t r = 0; r < rem; ++r) {
            out_row_ptr[n_start + r] += acc[r];
          }
        }
      }
    }
  }
#endif

  return output;
}

TORCH_LIBRARY(teal, m) {
  m.def("sve_sparse_gemv(Tensor activation, Tensor weight, int nz_row, Tensor nz_col_index) -> Tensor");
  m.def("sve_sparse_gemm(Tensor activation, Tensor weight, Tensor nz_counts, Tensor nz_col_indices) -> Tensor");
}

TORCH_LIBRARY_IMPL(teal, CPU, m) {
  m.impl("sve_sparse_gemv", sve_sparse_gemv);
  m.impl("sve_sparse_gemm", sve_sparse_gemm);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
