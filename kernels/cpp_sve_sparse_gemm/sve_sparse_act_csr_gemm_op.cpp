#include <torch/extension.h>

#include <cstdint>
#include <omp.h>
#include <vector>
#include <algorithm>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

/**
 * 新算子：将稀疏输入组织为 CSR，然后做 CSR × dense weight
 *
 * 输入：
 *   - activation: (M, K) float32, contiguous, CPU
 *   - weight: (K, N) float32, contiguous, CPU
 *   - row_offsets: 1D int64, length = M + 1, prefix sum offsets (from row_scan_sve)
 *   - nz_col_indices: 1D uint32/int32, flattened col indices for all non-zeros
 *
 * 功能目标（开发中）：
 *   1) 根据 row_offsets / nz_col_indices + activation 生成 CSR(values, col_idx, row_ptr)
 *   2) 使用 CSR 与 weight 做乘法输出 (M, N)
 *   3) 计算过程中使用 SIMD（SVE）加速
 *   4) 在 activation 行间（M 维）并行加速（OpenMP）
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
  TORCH_CHECK(row_offsets.size(0) == M + 1, "row_offsets length must be M + 1");
  TORCH_CHECK(row_offsets[0].item<int64_t>() == 0, "row_offsets[0] must be 0");
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
  
  // row_ptr is directly from row_offsets
  csr.row_ptr.assign(row_offsets_ptr, row_offsets_ptr + M + 1);
  csr.col_idx.resize((size_t)nnz_total);
  csr.values.resize((size_t)nnz_total);

  // fill CSR col_idx and values row by row
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const int64_t p0 = row_offsets_ptr[m];
    const int64_t p1 = row_offsets_ptr[m + 1];
    const int64_t nnz_row = p1 - p0;
    if (nnz_row == 0) continue;

    const float* act_row_ptr = act_ptr + m * K;

    // col_idx: direct copy
    for (int64_t j = 0; j < nnz_row; ++j) {
      csr.col_idx[(size_t)(p0 + j)] = nz_col_indices_ptr[p0 + j];
    }

    // values: scalar load from activation using col indices
    for (int64_t j = 0; j < nnz_row; ++j) {
      const uint32_t col = nz_col_indices_ptr[p0 + j];
      csr.values[(size_t)(p0 + j)] = act_row_ptr[(int64_t)col];
    }
  }

  return csr;
}

// CSR × dense weight -> dense output
void csr_gemm_compute(
    const CSRMatrix& csr,
    const float* weight_ptr, // (K, N)
    float* out_ptr,          // (M, N)
    int64_t N) {
      const int64_t M = csr.M;

      // 选一个 N 分块大小：建议是 L1-friendly 且是 16 的倍数（方便向量化）
      // 经验：256/384/512 都常用。你可以按平台调参。
      constexpr int64_t BN = 512;
    
      const int64_t NB = (N + BN - 1) / BN;
    
      // 二维并行：每个线程负责一个 (m, nb) tile，写 out_row[n0:n1) 无冲突
      #pragma omp parallel for collapse(2) schedule(static)
      for (int64_t m = 0; m < M; ++m) {
        for (int64_t nb = 0; nb < NB; ++nb) {
          const int64_t p0 = csr.row_ptr[(size_t)m];
          const int64_t p1 = csr.row_ptr[(size_t)m + 1];
          if (p0 == p1) continue;
    
          const int64_t n0 = nb * BN;
          const int64_t n1 = std::min<int64_t>(n0 + BN, N);
    
          float* out_row = out_ptr + m * N;
    
    #if defined(__ARM_FEATURE_SVE)
          // SVE 路径：对 [n0, n1) 做向量累加，尾部用谓词处理
          for (int64_t p = p0; p < p1; ++p) {
            const uint32_t k = csr.col_idx[(size_t)p];
            const float a = csr.values[(size_t)p];
            const float* w_row = weight_ptr + (int64_t)k * N;
    
            int64_t n = n0;
            for (; n < n1; ) {
              // 以 32-bit lane 计数的谓词：覆盖 [n, n1)
              svbool_t pg = svwhilelt_b32(n, n1);
    
              // load out / weight，做 FMA，然后 store 回 out
              svfloat32_t ov = svld1_f32(pg, out_row + n);
              svfloat32_t wv = svld1_f32(pg, w_row  + n);
              svfloat32_t rv = svmla_n_f32_m(pg, ov, wv, a);
              svst1_f32(pg, out_row + n, rv);
    
              n += svcntw(); // 前进一个 VL（以 float32 lane 数）
            }
          }
    #else
          // 标量路径：只更新该 N tile
          for (int64_t p = p0; p < p1; ++p) {
            const uint32_t k = csr.col_idx[(size_t)p];
            const float a = csr.values[(size_t)p];
            const float* w_row = weight_ptr + (int64_t)k * N;
            for (int64_t n = n0; n < n1; ++n) {
              out_row[n] += a * w_row[n];
            }
          }
    #endif
        }
      }
    }
} // namespace

torch::Tensor sve_sparse_act_csr_gemm(
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

  // nnz_total is the last element of row_offsets
  const int64_t nnz_total = row_offsets_ptr[M];
  TORCH_CHECK(nz_col_indices.numel() == nnz_total, 
              "nz_col_indices size must equal row_offsets[M] (total nnz)");

  // Step1: build CSR from (row_offsets, nz_col_indices, activation)
  CSRMatrix csr = build_csr_from_inputs(
      act_ptr, row_offsets_ptr, indices_ptr, M, K, nnz_total);

  // Step2: CSR × dense weight
  csr_gemm_compute(csr, weight_ptr, out_ptr, N);
  return output;
}

// 注册到 torch.ops.teal
TORCH_LIBRARY_FRAGMENT(teal, m) {
  m.def("sve_sparse_act_csr_gemm(Tensor activation, Tensor weight, Tensor row_offsets, Tensor nz_col_indices) -> Tensor");
}

TORCH_LIBRARY_IMPL(teal, CPU, m) {
  m.impl("sve_sparse_act_csr_gemm", sve_sparse_act_csr_gemm);
}

