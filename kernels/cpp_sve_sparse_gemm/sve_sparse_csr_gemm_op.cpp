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
 * 输入与 `row_scan_sve` 输出一致：
 *   - activation: (M, K) float32, contiguous, CPU
 *   - weight: (K, N) float32, contiguous, CPU
 *   - row_offsets: 1D int64, length = M+1, CSR格式的row_ptr（前缀和）
 *   - nz_col_indices: 1D uint32/int32, flattened col indices for all non-zeros
 *
 * 功能目标：
 *   1) 根据 row_offsets / nz_col_indices + activation 生成 CSR(values, col_idx, row_ptr)
 *   2) 使用 CSR 格式与 weight 做乘法，计算方式参考 sve_sparse_gemm：
 *      - 从 CSR values 中连续 load 数据（SIMD）
 *      - 根据 CSR col_idx 计算 weight 中需要参与计算的元素索引位置
 *      - 使用 gather load 加载 weight 到寄存器中参与计算
 *   3) 分块和计算逻辑参考 sve_sparse_gemm_op.cpp (190-325)
 *
 * 与 sve_sparse_gemm 的区别：
 *   - activation 提前处理成 CSR 格式，计算时从 CSR values 连续 load
 *   - 而不是直接从原始 activation 矩阵 gather load
 *
 * 备注：
 * - 本文件刻意不包含 `PYBIND11_MODULE`，方便后续与现有扩展多源编译链接（避免重复定义）。
 * - 当前 SIMD 路径沿用仓库现状：仅在 `svcntw()==4` 时启用 SVE 快路径。
 */

namespace {

void check_inputs_csr_gemm(
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
  TORCH_CHECK(
      nz_col_indices.dtype() == torch::kUInt32 || nz_col_indices.dtype() == torch::kInt32,
      "nz_col_indices must be uint32 or int32");

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
  const int64_t* row_offsets_ptr = row_offsets.data_ptr<int64_t>();
  TORCH_CHECK(row_offsets_ptr[0] == 0, "row_offsets[0] must be 0");
  TORCH_CHECK(row_offsets_ptr[M] == nz_col_indices.size(0), 
              "row_offsets[M] must equal nz_col_indices length");
}

struct CSRMatrix {
  // 标准 CSR：row_ptr 长度 M+1，values/col_idx 长度 nnz
  std::vector<int64_t> row_ptr;   // size: M + 1
  std::vector<uint32_t> col_idx;  // size: nnz
  std::vector<float> values;      // size: nnz
  int64_t M{0};
  int64_t K{0};
  int64_t nnz{0};
};

CSRMatrix build_csr_from_inputs(
    const float* act_ptr,
    const int64_t* row_offsets_ptr,
    const uint32_t* nz_col_indices_ptr,
    int64_t M,
    int64_t K,
    int64_t nnz_total) {
  CSRMatrix csr;
  csr.M = M;
  csr.K = K;
  csr.nnz = nnz_total;
  csr.row_ptr.resize((size_t)M + 1);
  csr.col_idx.resize((size_t)nnz_total);
  csr.values.resize((size_t)nnz_total);

  // 直接复制 row_offsets 作为 row_ptr
  for (int64_t m = 0; m <= M; ++m) {
    csr.row_ptr[(size_t)m] = row_offsets_ptr[m];
  }

  // 按行填充 CSR col_idx 和 values
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const int64_t p0 = row_offsets_ptr[m];
    const int64_t p1 = row_offsets_ptr[m + 1];
    const int64_t nnz = p1 - p0;
    
    if (nnz == 0) continue;

    const float* act_row_ptr = act_ptr + m * K;
    
    // col_idx: direct copy
    for (int64_t j = 0; j < nnz; ++j) {
      csr.col_idx[(size_t)(p0 + j)] = nz_col_indices_ptr[p0 + j];
    }

    // values: scalar load from activation using col indices
    for (int64_t j = 0; j < nnz; ++j) {
      const uint32_t col = nz_col_indices_ptr[p0 + j];
      csr.values[(size_t)(p0 + j)] = act_row_ptr[(int64_t)col];
    }
  }

  return csr;
}

// CSR × dense weight -> dense output
// 使用类似 sve_sparse_gemm 的计算方式：从 CSR values 连续 load，gather load weight
void csr_gemm_compute(
    const CSRMatrix& csr,
    const float* weight_ptr, // (K, N)
    float* out_ptr,          // (M, N)
    int64_t N) {
  const int64_t M = csr.M;

#if defined(__ARM_FEATURE_SVE)
  const int64_t vl = svcntw();
  const uint32_t N_u32 = (uint32_t)N;

  if (vl == 4) {
    int64_t n_block_sz = N / 16;
    const int64_t n_full = (N / n_block_sz) * n_block_sz;
    const int64_t rem = N - n_full;

    #pragma omp parallel
    {
      // Full blocks: parallelize over rows and N-blocks
      #pragma omp for collapse(2) schedule(static)
      for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < n_full; n += n_block_sz) {
          const int64_t p0 = csr.row_ptr[(size_t)m];
          const int64_t p1 = csr.row_ptr[(size_t)m + 1];
          const int64_t nnz = p1 - p0;
          
          if (nnz == 0) continue;

          // CSR data pointers for this row
          const float* csr_values_ptr = csr.values.data() + p0;
          const uint32_t* csr_col_idx_ptr = csr.col_idx.data() + p0;
          float* out_row_ptr = out_ptr + m * N;
          
          std::vector<float> acc(n_block_sz, 0.0f);
          const float* base = weight_ptr + n;

          // Process non-zeros in chunks of 4 (SVE vector length)
          for (int64_t i = 0; i < nnz; i += 4) {
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
          const int64_t p0 = csr.row_ptr[(size_t)m];
          const int64_t p1 = csr.row_ptr[(size_t)m + 1];
          const int64_t nnz = p1 - p0;
          
          if (nnz == 0) continue;

          const float* csr_values_ptr = csr.values.data() + p0;
          const uint32_t* csr_col_idx_ptr = csr.col_idx.data() + p0;
          float* out_row_ptr = out_ptr + m * N;
          const int64_t n_start = n_full;
          
          std::vector<float> acc(rem, 0.0f);

          for (int64_t i = 0; i < nnz; i += 4) {
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
  }
#else
  // Scalar fallback path
  for (int64_t m = 0; m < M; ++m) {
    const int64_t p0 = csr.row_ptr[(size_t)m];
    const int64_t p1 = csr.row_ptr[(size_t)m + 1];
    float* out_row = out_ptr + m * N;
    
    for (int64_t p = p0; p < p1; ++p) {
      const uint32_t k = csr.col_idx[(size_t)p];
      const float a = csr.values[(size_t)p];
      const float* w_row = weight_ptr + (int64_t)k * N;
      for (int64_t n = 0; n < N; ++n) {
        out_row[n] += a * w_row[n];
      }
    }
  }
#endif
}

} // namespace

torch::Tensor sve_sparse_csr_gemm(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor row_offsets,
    torch::Tensor nz_col_indices) {
  check_inputs_csr_gemm(activation, weight, row_offsets, nz_col_indices);

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

  // nnz_total 从 row_offsets[M] 获取
  const int64_t nnz_total = row_offsets_ptr[M];
  TORCH_CHECK(nz_col_indices.numel() == nnz_total, 
              "nz_col_indices size must equal row_offsets[M]");

  // Step1: build CSR from (row_offsets, nz_col_indices, activation)
  CSRMatrix csr = build_csr_from_inputs(
      act_ptr, row_offsets_ptr, indices_ptr, M, K, nnz_total);

  // Step2: CSR × dense weight (using sve_sparse_gemm-style computation)
  csr_gemm_compute(csr, weight_ptr, out_ptr, N);
  return output;
}

// 注册到 torch.ops.teal
TORCH_LIBRARY_FRAGMENT(teal, m) {
  m.def("sve_sparse_csr_gemm(Tensor activation, Tensor weight, Tensor row_offsets, Tensor nz_col_indices) -> Tensor");
}

TORCH_LIBRARY_IMPL(teal, CPU, m) {
  m.impl("sve_sparse_csr_gemm", sve_sparse_csr_gemm);
}
