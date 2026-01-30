#include <torch/extension.h>

#include <cstdint>
#include <omp.h>
#include <vector>
#include <algorithm>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

/**
 * 新算子：将稀疏输入组织为 CSR，然后使用类似 sve_sparse_gemm 的计算方式
 *
 * 输入 CSR 格式：
 *   - weight: (K, N) float32, contiguous, CPU
 *   - row_offsets: 1D int64, length = M+1, CSR格式的row_ptr（前缀和）
 *   - nz_col_indices: 1D uint32/int32, flattened col indices for all non-zeros
 *   - values: 1D float32, CSR格式的非零元素值
 *
 * 功能目标：
 *   1) 直接使用输入的 CSR 格式 (values, col_idx, row_ptr)
 *   2) 使用 CSR 格式与 weight 做乘法，计算方式参考 sve_sparse_gemm：
 *      - 从 CSR values 中连续 load 数据（SIMD）
 *      - 根据 CSR col_idx 计算 weight 中需要参与计算的元素索引位置
 *      - 使用 gather load 加载 weight 到寄存器中参与计算
 *   3) 分块和计算逻辑参考 sve_sparse_gemm_op.cpp (190-325)
 *
 * 与 sve_sparse_gemm 的区别：
 *   - 直接接收 CSR 格式的 values，计算时从 CSR values 连续 load
 *   - 而不是直接从原始 activation 矩阵 gather load
 *
 * 备注：
 * - 本文件刻意不包含 `PYBIND11_MODULE`，方便后续与现有扩展多源编译链接（避免重复定义）。
 * - 当前 SIMD 路径沿用仓库现状：仅在 `svcntw()==4` 时启用 SVE 快路径。
 */

namespace {

void check_inputs_csr_sve_gather(
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
  TORCH_CHECK(nz_col_indices.dtype() == torch::kUInt32,"nz_col_indices must be uint32 ");
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

torch::Tensor sparse_gemm_csr_sve_gather(
    torch::Tensor weight,
    torch::Tensor row_offsets,
    torch::Tensor nz_col_indices,
    torch::Tensor values) {
  check_inputs_csr_sve_gather(weight, row_offsets, nz_col_indices, values);

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

  // 直接使用传入的CSR数据指针进行计算，不进行复制
#if defined(__ARM_FEATURE_SVE)
  const int64_t vl = svcntw();
  const uint32_t N_u32 = (uint32_t)N;

    int64_t n_block_sz = N / 16;
    const int64_t n_full = (N / n_block_sz) * n_block_sz;
    const int64_t rem = N - n_full;

    #pragma omp parallel
    {
      // Full blocks: parallelize over rows and N-blocks
      #pragma omp for collapse(2) schedule(static)
      for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < n_full; n += n_block_sz) {
          const int64_t p0 = row_offsets_ptr[m];
          const int64_t p1 = row_offsets_ptr[m + 1];
          const int64_t nnz = p1 - p0;
          
          if (nnz == 0) continue;

          // CSR data pointers for this row
          const float* csr_values_ptr = values_ptr + p0;
          const uint32_t* csr_col_idx_ptr = indices_ptr + p0;
          float* out_row_ptr = out_ptr + m * N;
          
          std::vector<float> acc(n_block_sz, 0.0f);
          const float* base = weight_ptr + n;

          // Process non-zeros in chunks of 4 (SVE vector length)
          for (int64_t i = 0; i < nnz; i += vl) {
            const svbool_t pg = svwhilelt_b32(i, nnz);
            
            // Continuous load from CSR values (instead of gather from activation)
            const svfloat32_t act_vals = svld1_f32(pg, csr_values_ptr + i);
            
            // Load column indices and compute weight indices
            const svuint32_t idx = svld1_u32(pg, csr_col_idx_ptr + i);
            const svuint32_t w_index = svmul_n_u32_x(pg, idx, N_u32);

            // Gather load weight and accumulate for each element in the block
            for (int64_t r = 0; r < n_block_sz; r++) {
              const svfloat32_t w_vals = svld1_gather_u32index_f32(pg, base + r, w_index);
              acc[r] += svaddv_f32(pg, svmul_f32_m(pg, act_vals, w_vals));
            }
          }

          // Write accumulated results
          for (int64_t r = 0; r < n_block_sz; r++) {
            out_row_ptr[n + r] += acc[r];
          }
        }
      }

      // Handle remainder
      if (rem > 0) {
        #pragma omp for schedule(static)
        for (int64_t m = 0; m < M; ++m) {
          const int64_t p0 = row_offsets_ptr[m];
          const int64_t p1 = row_offsets_ptr[m + 1];
          const int64_t nnz = p1 - p0;
          
          if (nnz == 0) continue;

          const float* csr_values_ptr = values_ptr + p0;
          const uint32_t* csr_col_idx_ptr = indices_ptr + p0;
          float* out_row_ptr = out_ptr + m * N;
          const int64_t n_start = n_full;
          
          std::vector<float> acc(rem, 0.0f);

          for (int64_t i = 0; i < nnz; i += vl) {
            const svbool_t pg = svwhilelt_b32(i, nnz);
            
            // Continuous load from CSR values
            const svfloat32_t act_vals = svld1_f32(pg, csr_values_ptr + i);
            
            // Load column indices and compute weight indices
            const svuint32_t idx = svld1_u32(pg, csr_col_idx_ptr + i);
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
#else
  // Scalar fallback path
  for (int64_t m = 0; m < M; ++m) {
    const int64_t p0 = row_offsets_ptr[m];
    const int64_t p1 = row_offsets_ptr[m + 1];
    float* out_row = out_ptr + m * N;
    
    for (int64_t p = p0; p < p1; ++p) {
      const uint32_t k = indices_ptr[p];
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

// 注册到 torch.ops.sparse_op
TORCH_LIBRARY_FRAGMENT(sparse_op, m) {
  m.def("sparse_gemm_csr_sve_gather(Tensor weight, Tensor row_offsets, Tensor nz_col_indices, Tensor values) -> Tensor");
}

TORCH_LIBRARY_IMPL(sparse_op, CPU, m) {
  m.impl("sparse_gemm_csr_sve_gather", sparse_gemm_csr_sve_gather);
}
