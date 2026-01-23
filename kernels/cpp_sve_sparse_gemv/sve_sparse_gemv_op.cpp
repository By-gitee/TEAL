#include <torch/extension.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

//namespace {

void check_inputs(
    const torch::Tensor& activation,
    const torch::Tensor& weight,
    const torch::Tensor& nz_col_index,
    int64_t nz_row) {
  TORCH_CHECK(activation.device().is_cpu(), "activation must be a CPU tensor");
  TORCH_CHECK(weight.device().is_cpu(), "weight must be a CPU tensor");
  TORCH_CHECK(nz_col_index.device().is_cpu(), "nz_col_index must be a CPU tensor");

  TORCH_CHECK(activation.dtype() == torch::kFloat32, "activation must be float32");
  TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
  TORCH_CHECK(nz_col_index.dtype() == torch::kInt64, "nz_col_index must be int64");

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

//}  // namespace

torch::Tensor sve_sparse_gemv(
    torch::Tensor activation,
    torch::Tensor weight,
    int64_t nz_row,
    torch::Tensor nz_col_index) {
  check_inputs(activation, weight, nz_col_index, nz_row);

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

  const auto* act_ptr = activation.data_ptr<float>() + nz_row * K;
  const auto* weight_ptr = weight.data_ptr<float>();
  const auto* idx_ptr = nz_col_index.data_ptr<int64_t>();
  auto* out_ptr = output.data_ptr<float>();

// Data type conversion.
// From int64 to uint32 for SVE instructions.
#if defined(__ARM_FEATURE_SVE)
  std::vector<uint32_t> idx_u32(nnz);
#endif

  for (int64_t i = 0; i < nnz; ++i) {
    const auto col = idx_ptr[i];
    TORCH_CHECK(col >= 0 && col < K, "nz_col_index contains out-of-range value");
#if defined(__ARM_FEATURE_SVE)
    idx_u32[i] = static_cast<uint32_t>(col);
#endif
  }

#if defined(__ARM_FEATURE_SVE)
  // Gather uses u32 *element indices*; fallback to scalar if too large.
  const uint64_t max_act_index = static_cast<uint64_t>(K - 1);
  const uint64_t max_weight_index = static_cast<uint64_t>(K - 1) * static_cast<uint64_t>(N);
  const bool sve_safe =
      max_act_index <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) &&
      max_weight_index <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());

  if (sve_safe) {
    const int64_t vl = svcntw();
    const uint32_t N_u32 = static_cast<uint32_t>(N);

    // for (int64_t n = 0; n < N; ++n) {
    //   svfloat32_t acc_vec = svdup_f32(0.0f);

    //   for (int64_t i = 0; i < nnz; i += vl) {
    //     const svbool_t pg = svwhilelt_b32(i, nnz);
    //     const svuint32_t idx = svld1_u32(pg, idx_u32.data() + i);
    //     // activation[nz_row, idx]
    //     const svfloat32_t act_vals = svld1_gather_u32index_f32(pg, act_ptr, idx);
    //     // weight[idx, n] where weight is row-major [K, N]
    //     const svuint32_t w_index = svmul_n_u32_x(pg, idx, N_u32);  // idx * N
    //     const svfloat32_t w_vals = svld1_gather_u32index_f32(pg, weight_ptr + n, w_index);
    //     acc_vec = svmla_f32_m(pg, acc_vec, act_vals, w_vals);
    //   }

    //   out_ptr[n] = svaddv_f32(svptrue_b32(), acc_vec);
    // }
    bool did_sve = false;
    if (vl == 4) {
      did_sve = true;

      const int64_t n_full = (N / 4) * 4;
      const int64_t rem = N - n_full;  // 0..3

      for (int64_t i = 0; i < nnz; i += 4) {
        // gather load activation values(non-zero values)
        const svbool_t pg = svwhilelt_b32(i, nnz);
        const svuint32_t idx = svld1_u32(pg, idx_u32.data() + i);
        const svfloat32_t act_vals = svld1_gather_u32index_f32(pg, act_ptr, idx);

        const svuint32_t w_index = svmul_n_u32_x(pg, idx, N_u32);  // idx * N
        
        // Main loop: only full 4-column blocks, no per-iter tail checks.
        for (int64_t n = 0; n < n_full; n += 4) {
          const svfloat32_t w_vals0 = svld1_gather_u32index_f32(pg, weight_ptr + (n + 0), w_index);
          const svfloat32_t w_vals1 = svld1_gather_u32index_f32(pg, weight_ptr + (n + 1), w_index);
          const svfloat32_t w_vals2 = svld1_gather_u32index_f32(pg, weight_ptr + (n + 2), w_index);
          const svfloat32_t w_vals3 = svld1_gather_u32index_f32(pg, weight_ptr + (n + 3), w_index);

          const float sum0 = svaddv_f32(pg, svmul_f32_m(pg, act_vals, w_vals0));
          const float sum1 = svaddv_f32(pg, svmul_f32_m(pg, act_vals, w_vals1));
          const float sum2 = svaddv_f32(pg, svmul_f32_m(pg, act_vals, w_vals2));
          const float sum3 = svaddv_f32(pg, svmul_f32_m(pg, act_vals, w_vals3));

          // Full block: scalar update, no tail checks.
          out_ptr[n + 0] += sum0;
          out_ptr[n + 1] += sum1;
          out_ptr[n + 2] += sum2;
          out_ptr[n + 3] += sum3;
        }

        // Tail: one-time per i-block, with a single switch (no per-n checks).
        if (rem) {
          const int64_t n = n_full;
          // rem is in 1..3 here, and (n + t) is always < N for t < rem.
          if (rem >= 1) {
            const svfloat32_t w0 = svld1_gather_u32index_f32(pg, weight_ptr + (n + 0), w_index);
            out_ptr[n + 0] += svaddv_f32(pg, svmul_f32_m(pg, act_vals, w0));
          }
          if (rem >= 2) {
            const svfloat32_t w1 = svld1_gather_u32index_f32(pg, weight_ptr + (n + 1), w_index);
            out_ptr[n + 1] += svaddv_f32(pg, svmul_f32_m(pg, act_vals, w1));
          }
          if (rem >= 3) {
            const svfloat32_t w2 = svld1_gather_u32index_f32(pg, weight_ptr + (n + 2), w_index);
            out_ptr[n + 2] += svaddv_f32(pg, svmul_f32_m(pg, act_vals, w2));
          }
        }
      }
    }
    if (did_sve) {
      return output;
    }
  } else {
    // SVE available but offsets too large, falling back to scalar.
    std::cout << "[SVE Sparse GEMV] Warning: Falling back to scalar implementation. "
              << "Reason: Index exceeds uint32_t range. "
              << "K=" << K << ", N=" << N << ", "
              << "max_act_index=" << max_act_index << ", "
              << "max_weight_index=" << max_weight_index << std::endl;
  }
#else
  // Scalar fallback (also handles non-SVE targets).
  std::cout << "[SVE Sparse GEMV] Info: Using scalar implementation. "
            << "Reason: SVE not available (not compiled with __ARM_FEATURE_SVE). "
            << "K=" << K << ", N=" << N << ", nnz=" << nnz << std::endl;
#endif

  // Scalar computation (used when SVE is unavailable or unsafe).
  for (int64_t i = 0; i < nnz; ++i) {
    const int64_t k = idx_ptr[i];
    const float a = act_ptr[k];
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

TORCH_LIBRARY(teal, m) {
  m.def("sve_sparse_gemv(Tensor activation, Tensor weight, int nz_row, Tensor nz_col_index) -> Tensor");
}

TORCH_LIBRARY_IMPL(teal, CPU, m) {
  m.impl("sve_sparse_gemv", sve_sparse_gemv);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
