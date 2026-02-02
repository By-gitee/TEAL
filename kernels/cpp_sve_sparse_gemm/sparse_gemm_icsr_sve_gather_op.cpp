#include <torch/extension.h>

#include <cstdint>
#include <iostream>
#include <omp.h>
#include <limits>
#include <vector>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif


void check_sparse_gemv_icsr_sve_gather_inputs(
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


torch::Tensor sparse_gemv_icsr_sve_gather(
    torch::Tensor activation,
    torch::Tensor weight,
    int64_t nz_row,
    torch::Tensor nz_col_index) {
  check_sparse_gemv_icsr_sve_gather_inputs(activation, weight, nz_col_index, nz_row);

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

    // N-dimension blocking (aligned with sparse_gemm_icsr_sve_gather's n_block_sz idea)
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

        for (int64_t i = 0; i < nnz; i += vl) {
          const svbool_t pg = svwhilelt_b32(i, nnz);
          const svuint32_t idx = svld1_u32(pg, idx_ptr + i);
          const svfloat32_t act_vals = svld1_gather_u32index_f32(pg, act_row_ptr, idx);
          const svuint32_t w_index = svmul_n_u32_x(pg, idx, N_u32);  // idx * N

          for (int64_t r = 0; r < n_block_sz; ++r) {
            const svfloat32_t w_vals = svld1_gather_u32index_f32(pg, base + r, w_index);
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

          for (int64_t i = 0; i < nnz; i += vl) {
            const svbool_t pg = svwhilelt_b32(i, nnz);
            const svuint32_t idx = svld1_u32(pg, idx_ptr + i);
            const svfloat32_t act_vals = svld1_gather_u32index_f32(pg, act_row_ptr, idx);
            const svuint32_t w_index = svmul_n_u32_x(pg, idx, N_u32);  // idx * N

            for (int64_t r = 0; r < rem; ++r) {
              const svfloat32_t w_vals = svld1_gather_u32index_f32(pg, weight_ptr + (n_start + r), w_index);
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
void check_sparse_gemm_icsr_sve_gather_inputs(
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
  
  const int64_t* offsets_ptr = row_offsets.data_ptr<int64_t>();
  TORCH_CHECK(offsets_ptr[0] == 0, "row_offsets[0] must be 0");
  TORCH_CHECK(offsets_ptr[M] == nz_col_indices.numel(), 
              "row_offsets[M] must equal nz_col_indices size");
}

// Sparse GEMM: (M, K) sparse × (K, N) dense → (M, N)
// row_offsets: int64 [M+1], prefix sum offsets for each row in nz_col_indices
// nz_col_indices: flattened column indices for all non-zero elements
torch::Tensor sparse_gemm_icsr_sve_gather(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor row_offsets,
    torch::Tensor nz_col_indices) {
  // check_sparse_gemm_icsr_sve_gather_inputs(activation, weight, row_offsets, nz_col_indices);

  const auto M = activation.size(0);
  const auto K = activation.size(1);
  const auto N = weight.size(1);

  auto output = torch::zeros({M, N}, activation.options());
  
  if (M == 0 || N == 0 || K == 0) {
    return output;
  }

  const float* act_ptr = activation.data_ptr<float>();
  const float* weight_ptr = weight.data_ptr<float>();
  const int64_t* offsets_ptr = row_offsets.data_ptr<int64_t>();
  const uint32_t* indices_ptr = nz_col_indices.data_ptr<uint32_t>();
  float* out_ptr = output.data_ptr<float>();
#if defined(__ARM_FEATURE_SVE)
  const int64_t vl = svcntw();
  const uint32_t N_u32 = (uint32_t)N;
  
    int64_t n_block_sz = N/16;
    const int64_t n_full = (N / n_block_sz) * n_block_sz;
    const int64_t rem = N - n_full;

    #pragma omp parallel
    {
      #pragma omp for collapse(2) schedule(static)
      for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < n_full; n += n_block_sz) {
          const int64_t nnz = offsets_ptr[m + 1] - offsets_ptr[m];
          if (nnz == 0) continue;

          const float* act_row_ptr = act_ptr + m * K;
          const uint32_t* idx_ptr = indices_ptr + offsets_ptr[m];
          float* out_row_ptr = out_ptr + m * N;
          std::vector<float> acc(n_block_sz, 0.0f);

          for (int64_t i = 0; i < nnz; i += vl) {
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
        for (int64_t m = 0; m < M; ++m) {
          const int64_t nnz = offsets_ptr[m + 1] - offsets_ptr[m];
          if (nnz == 0) continue;

          const float* act_row_ptr = act_ptr + m * K;
          const uint32_t* idx_ptr = indices_ptr + offsets_ptr[m];
          float* out_row_ptr = out_ptr + m * N;

          const int64_t n_start = n_full;
          
          std::vector<float> acc(rem, 0.0f);

          for (int64_t i = 0; i < nnz; i += vl) {
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
#endif

  return output;
}

TORCH_LIBRARY(sparse_op, m) {
  m.def("sparse_gemv_icsr_sve_gather(Tensor activation, Tensor weight, int nz_row, Tensor nz_col_index) -> Tensor");
  m.def("sparse_gemm_icsr_sve_gather(Tensor activation, Tensor weight, Tensor row_offsets, Tensor nz_col_indices) -> Tensor");
}

TORCH_LIBRARY_IMPL(sparse_op, CPU, m) {
  m.impl("sparse_gemv_icsr_sve_gather", sparse_gemv_icsr_sve_gather);
  m.impl("sparse_gemm_icsr_sve_gather", sparse_gemm_icsr_sve_gather);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}