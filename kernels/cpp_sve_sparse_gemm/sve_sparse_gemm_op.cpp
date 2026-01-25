#include <torch/extension.h>

#include <cstdint>
#include <iostream>
#include <omp.h>
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
  TORCH_CHECK(nz_col_index.dtype() == torch::kUInt32 || nz_col_index.dtype() == torch::kInt32, "nz_col_index must be uint32 or int32");

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

// Helper function: compute one row of sparse GEMV
// This is extracted from sve_sparse_gemv to be reused in sve_sparse_gemm
void compute_sparse_gemv_row(
    const float* act_row_ptr,
    const float* weight_ptr,
    const uint32_t* idx_ptr,
    int64_t nnz,
    float* out_ptr,
    int64_t K,
    int64_t N) {
  if (nnz == 0 || N == 0) {
    return;
  }

#if defined(__ARM_FEATURE_SVE)
    const int64_t vl = svcntw();
    const uint32_t N_u32 = static_cast<uint32_t>(N);

    if (vl == 4) {
      const int64_t n_full = (N / 4) * 4;
      const int64_t rem = N - n_full;  // 0..3
      for (int64_t i = 0; i < nnz; i += 4) {
        // gather load activation values (non-zero values)
        const svbool_t pg = svwhilelt_b32(i, nnz);
        const svuint32_t idx = svld1_u32(pg, idx_ptr + i);
        const svfloat32_t act_vals = svld1_gather_u32index_f32(pg, act_row_ptr, idx);

        const svuint32_t w_index = svmul_n_u32_x(pg, idx, N_u32);  // idx * N

        // Main loop: only full 4-column blocks, no per-iter tail checks
        #pragma omp parallel for
        for (int64_t n = 0; n < n_full; n += 4) {
          const float* base = weight_ptr + n;

          const svfloat32_t w_vals0 = svld1_gather_u32index_f32(pg, base, w_index);
          const svfloat32_t w_vals1 = svld1_gather_u32index_f32(pg, base + 1, w_index);
          const svfloat32_t w_vals2 = svld1_gather_u32index_f32(pg, base + 2, w_index);
          const svfloat32_t w_vals3 = svld1_gather_u32index_f32(pg, base + 3, w_index);

          const float sum0 = svaddv_f32(pg, svmul_f32_m(pg, act_vals, w_vals0));
          const float sum1 = svaddv_f32(pg, svmul_f32_m(pg, act_vals, w_vals1));
          const float sum2 = svaddv_f32(pg, svmul_f32_m(pg, act_vals, w_vals2));
          const float sum3 = svaddv_f32(pg, svmul_f32_m(pg, act_vals, w_vals3));

          // Full block: scalar update, no tail checks
          out_ptr[n + 0] += sum0;
          out_ptr[n + 1] += sum1;
          out_ptr[n + 2] += sum2;
          out_ptr[n + 3] += sum3;
        }

        // Tail: one-time per i-block, with a single switch (no per-n checks)
        if (rem) {
          const int64_t n = n_full;
          // rem is in 1..3 here, and (n + t) is always < N for t < rem
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
      return;  // SVE path completed
    }
#endif

  // Scalar computation (used when SVE is unavailable or unsafe)
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
}

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

  const float* act_ptr = activation.data_ptr<float>() + nz_row * K;
  const float* weight_ptr = weight.data_ptr<float>();
  const uint32_t* idx_ptr = nz_col_index.data_ptr<uint32_t>();
  float* out_ptr = output.data_ptr<float>();

  // Use the helper function to compute this row
  compute_sparse_gemv_row(act_ptr, weight_ptr, idx_ptr, nnz, out_ptr, K, N);
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
  TORCH_CHECK(nz_col_indices.dtype() == torch::kUInt32 || nz_col_indices.dtype() == torch::kInt32, "nz_col_indices must be uint32 or int32");

  TORCH_CHECK(activation.dim() == 2, "activation must be 2D");
  TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
  TORCH_CHECK(nz_counts.dim() == 1, "nz_counts must be 1D");
  TORCH_CHECK(nz_col_indices.dim() == 1, "nz_col_indices must be 1D");

  TORCH_CHECK(activation.is_contiguous(), "activation must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(nz_counts.is_contiguous(), "nz_counts must be contiguous");
  TORCH_CHECK(nz_col_indices.is_contiguous(), "nz_col_indices must be contiguous");

  const auto M = activation.size(0);
  const auto K = activation.size(1);
  TORCH_CHECK(weight.size(0) == K, "weight K dimension must match activation K");
  TORCH_CHECK(nz_counts.size(0) == M, "nz_counts length must match activation M");
}

// Sparse GEMM: (M, K) sparse × (K, N) dense → (M, N)
// nz_counts: (M,) number of non-zero elements per row
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

  // Compute cumulative offset for each row's indices
  std::vector<int64_t> row_offsets(M + 1, 0);
  for (int64_t i = 0; i < M; ++i) {
    row_offsets[i + 1] = row_offsets[i] + counts_ptr[i];
  }

  // Verify total non-zero count matches
  const int64_t total_nnz = row_offsets[M];
  TORCH_CHECK(
      nz_col_indices.numel() == total_nnz,
      "nz_col_indices size must equal sum of nz_counts");
#if defined(__ARM_FEATURE_SVE)
  const int64_t vl = svcntw();
  const uint32_t N_u32 = (uint32_t)N;
  
  // 你的实现是 vl==4 才走 SVE
  if (vl == 4) {
    int64_t n_block_sz = 4;
    const int64_t n_full = (N / n_block_sz) * n_block_sz;
    const int64_t rem = N - n_full;

    // 一个 parallel 区：避免嵌套
    #pragma omp parallel
    {
      // collapse(2) 同时切 (m, n_block)
      #pragma omp for collapse(2) schedule(static)
      for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < n_full; n += n_block_sz) {

          const int64_t nnz = counts_ptr[m];
          if (nnz == 0) continue;

          const float* act_row_ptr = act_ptr + m * K;
          const uint32_t* idx_ptr = indices_ptr + row_offsets[m];

          // 每个线程只写自己负责的 4 个输出
          float* out_row_ptr = out_ptr + m * N;
          float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

          // === 把你 compute_sparse_gemv_row 的 SVE 主体逻辑搬到这里 ===
          // 注意：这里不再有 omp parallel for！
          for (int64_t i = 0; i < nnz; i += 4) {
            const svbool_t pg = svwhilelt_b32(i, nnz);
            const svuint32_t idx = svld1_u32(pg, idx_ptr + i);

            // gather load act nonzeros
            const svfloat32_t act_vals = svld1_gather_u32index_f32(pg, act_row_ptr, idx);

            // idx * N 作为 weight 的“行起始偏移”
            const svuint32_t w_index = svmul_n_u32_x(pg, idx, N_u32);

            // base 指向 weight 的第 n 列（你原写法）
            const float* base = weight_ptr + n;

            // 你原逻辑：对同一组 idx，分别取 base+0/1/2/3 的 gather
            const svfloat32_t w_vals0 = svld1_gather_u32index_f32(pg, base + 0, w_index);
            const svfloat32_t w_vals1 = svld1_gather_u32index_f32(pg, base + 1, w_index);
            const svfloat32_t w_vals2 = svld1_gather_u32index_f32(pg, base + 2, w_index);
            const svfloat32_t w_vals3 = svld1_gather_u32index_f32(pg, base + 3, w_index);

            // reduction
            acc0 += svaddv_f32(pg, svmul_f32_m(pg, act_vals, w_vals0));
            acc1 += svaddv_f32(pg, svmul_f32_m(pg, act_vals, w_vals1));
            acc2 += svaddv_f32(pg, svmul_f32_m(pg, act_vals, w_vals2));
            acc3 += svaddv_f32(pg, svmul_f32_m(pg, act_vals, w_vals3));
          }

          // 写回（独占，无冲突）
          out_row_ptr[n + 0] += acc0;
          out_row_ptr[n + 1] += acc1;
          out_row_ptr[n + 2] += acc2;
          out_row_ptr[n + 3] += acc3;
        } // n_full blocks
      } // m

      // tail：同样可以并行按 (m) 或 (m, tail) 处理；这里给一个简单按 m 的并行写法
      if (rem > 0) {
        #pragma omp for schedule(static)
        for (int64_t m = 0; m < M; ++m) {
          const int64_t nnz = counts_ptr[m];
          if (nnz == 0) continue;

          const float* act_row_ptr = act_ptr + m * K;
          const uint32_t* idx_ptr = indices_ptr + row_offsets[m];
          float* out_row_ptr = out_ptr + m * N;

          const int64_t n_start = n_full;
          
          // 动态分配累加器数组，适应任意 rem 大小
          std::vector<float> acc(rem, 0.0f);

          // 遍历所有非零元素
          for (int64_t i = 0; i < nnz; i += 4) {
            const svbool_t pg = svwhilelt_b32(i, nnz);
            const svuint32_t idx = svld1_u32(pg, idx_ptr + i);
            const svfloat32_t act_vals = svld1_gather_u32index_f32(pg, act_row_ptr, idx);
            const svuint32_t w_index = svmul_n_u32_x(pg, idx, N_u32);

            // 对 rem 个剩余列进行计算
            for (int64_t r = 0; r < rem; ++r) {
              const svfloat32_t w_vals = svld1_gather_u32index_f32(pg, weight_ptr + (n_start + r), w_index);
              acc[r] += svaddv_f32(pg, svmul_f32_m(pg, act_vals, w_vals));
            }
          }

          // 写回结果
          for (int64_t r = 0; r < rem; ++r) {
            out_row_ptr[n_start + r] += acc[r];
          }
        }
      }
    } // omp parallel
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
