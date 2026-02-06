#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <omp.h>
#include <vector>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

/**
 * iCSR(index-only CSR) × dense weight with SVE gather optimization
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
 *   Optimized with SVE gather load for both activation and weight matrix access
 */

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

    const int64_t block_sz = std::max<int64_t>(1, N / 16);
    const int64_t n_block = (N + block_sz - 1) / block_sz;

    #pragma omp parallel
    {
      std::vector<float> acc_buf;
      acc_buf.resize((size_t)block_sz);

      #pragma omp for schedule(static)
      for (int64_t nb = 0; nb < n_block; ++nb) {
        const int64_t n0 = nb * block_sz;
        const int64_t n_valid = std::min<int64_t>(block_sz, N - n0);
        float* acc = acc_buf.data();

        std::fill_n(acc, (size_t)n_valid, 0.0f);
        const float* base = weight_ptr + n0;

        for (int64_t i = 0; i < nnz; i += vl) {
          const svbool_t pg = svwhilelt_b32(i, nnz);
          const svuint32_t idx = svld1_u32(pg, idx_ptr + i);
          const svfloat32_t act_vals = svld1_gather_u32index_f32(pg, act_row_ptr, idx);
          const svuint32_t w_index = svmul_n_u32_x(pg, idx, N_u32);

          for (int64_t r = 0; r < n_valid; ++r) {
            const svfloat32_t w_vals = svld1_gather_u32index_f32(pg, base + r, w_index);
            acc[r] += svaddv_f32(pg, svmul_f32_m(pg, act_vals, w_vals));
          }
        }

        float* out_tile = out_ptr + n0;
        for (int64_t r = 0; r < n_valid; ++r) out_tile[r] = acc[r];
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

torch::Tensor sparse_gemm_icsr_sve_gather(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor row_offsets,
    torch::Tensor nz_col_indices) {
  check_sparse_gemm_icsr_sve_gather_inputs(activation, weight, row_offsets, nz_col_indices);

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

  const int64_t block_sz = std::max<int64_t>(1, N / 16);
  const int64_t n_block = (N + block_sz - 1) / block_sz;

  #pragma omp parallel
  {
    std::vector<float> acc_buf;
    acc_buf.resize((size_t)block_sz);

    #pragma omp for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t nb = 0; nb < n_block; ++nb) {
        const int64_t n0 = nb * block_sz;
        const int64_t n_valid = std::min<int64_t>(block_sz, N - n0);
        float* acc = acc_buf.data();

        const int64_t p0 = offsets_ptr[m];
        const int64_t p1 = offsets_ptr[m + 1];
        const int64_t nnz = p1 - p0;

        std::fill_n(acc, (size_t)n_valid, 0.0f);

        if (nnz > 0) {
        const float* act_row_ptr = act_ptr + m * K;
        const uint32_t* idx_ptr = indices_ptr + p0;
        const float* base = weight_ptr + n0;

        for (int64_t i = 0; i < nnz; i += vl) {
          const svbool_t pg = svwhilelt_b32(i, nnz);
          const svuint32_t idx = svld1_u32(pg, idx_ptr + i);
          const svfloat32_t act_vals = svld1_gather_u32index_f32(pg, act_row_ptr, idx);
          const svuint32_t w_index = svmul_n_u32_x(pg, idx, N_u32);

          for (int64_t r = 0; r < n_valid; ++r) {
            const svfloat32_t w_vals = svld1_gather_u32index_f32(pg, base + r, w_index);
            acc[r] += svaddv_f32(pg, svmul_f32_m(pg, act_vals, w_vals));
          }
        }
        }

        float* out_tile = out_ptr + m * N + n0;
        for (int64_t r = 0; r < n_valid; ++r) out_tile[r] = acc[r];
      }
    }
  }
#else
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        out_ptr[m * N + n] = 0.0f;
        for (int64_t i = 0; i < nnz; ++i) {
          const uint32_t k = indices_ptr[i];
          const float a = act_ptr[m * K + k];
          if (a == 0.0f) continue;
          const float* w_row = weight_ptr + k * N;
          out_ptr[m * N + n] += a * w_row[n];
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