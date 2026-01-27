#include <torch/extension.h>

#include <cstdint>
#include <omp.h>
#include <vector>
#include <algorithm>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

/**
 * 新算子：直接使用输入信息进行稀疏 GEMM，不构建 CSR 格式
 *
 * 输入与 `sve_sparse_act_csr_gemm` 一致：
 *   - activation: (M, K) float32, contiguous, CPU
 *   - weight: (K, N) float32, contiguous, CPU
 *   - nz_counts: 1D int64, length = 2 * num_nz_rows, pairs [row_idx, count, ...]
 *   - nz_col_indices: 1D uint32/int32, flattened col indices for all non-zeros
 *
 * 功能目标：
 *   1) 直接使用 nz_counts / nz_col_indices + activation 进行 GEMM 计算
 *   2) 不构建 CSR 格式，直接从 activation 取值
 *   3) 计算过程中使用 SIMD（SVE）加速
 *   4) 在 activation 行间（M 维）并行加速（OpenMP）
 *
 * 备注：
 * - 本文件刻意不包含 `PYBIND11_MODULE`，方便后续与现有扩展多源编译链接（避免重复定义）。
 * - 当前 SIMD 路径沿用仓库现状：仅在 `svcntw()==4` 时启用 SVE 快路径。
 */

namespace {

void check_inputs_direct_gemm(
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
  TORCH_CHECK(
      nz_col_indices.dtype() == torch::kUInt32 || nz_col_indices.dtype() == torch::kInt32,
      "nz_col_indices must be uint32 or int32");

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

// 构建每行的非零元素计数和行指针（用于并行计算）
void build_row_ptr(
    const int64_t* nz_counts_ptr,
    int64_t M,
    int64_t num_nz_rows,
    std::vector<int64_t>& row_ptr) {
  row_ptr.assign((size_t)M + 1, 0);
  
  // row_counts[m] = nnz in row m (missing rows default 0)
  std::vector<int64_t> row_counts((size_t)M, 0);
  for (int64_t i = 0; i < num_nz_rows; ++i) {
    const int64_t row_idx = nz_counts_ptr[2 * i];
    const int64_t count = nz_counts_ptr[2 * i + 1];
    TORCH_CHECK(row_idx >= 0 && row_idx < M, "row_idx out of range");
    TORCH_CHECK(count >= 0, "count must be non-negative");
    TORCH_CHECK(row_counts[(size_t)row_idx] == 0, "duplicate row_idx in nz_counts is not supported yet");
    row_counts[(size_t)row_idx] = count;
  }

  // prefix sum -> row_ptr
  for (int64_t m = 0; m < M; ++m) {
    row_ptr[(size_t)m + 1] = row_ptr[(size_t)m] + row_counts[(size_t)m];
  }
}

// 构建行索引到 nz_col_indices 偏移量的映射（用于快速查找）
void build_row_to_offset_map(
    const int64_t* nz_counts_ptr,
    int64_t num_nz_rows,
    std::vector<int64_t>& row_to_offset,
    std::vector<int64_t>& row_to_count) {
  row_to_offset.clear();
  row_to_count.clear();
  
  int64_t global_offset = 0;
  for (int64_t i = 0; i < num_nz_rows; ++i) {
    const int64_t row_idx = nz_counts_ptr[2 * i];
    const int64_t count = nz_counts_ptr[2 * i + 1];
    
    if (row_to_offset.size() <= (size_t)row_idx) {
      row_to_offset.resize((size_t)row_idx + 1, -1);
      row_to_count.resize((size_t)row_idx + 1, 0);
    }
    row_to_offset[(size_t)row_idx] = global_offset;
    row_to_count[(size_t)row_idx] = count;
    
    global_offset += count;
  }
}

// 直接使用输入信息进行 GEMM 计算（不构建 CSR）
void direct_gemm_compute(
    const float* act_ptr,
    const uint32_t* nz_col_indices_ptr,
    const float* weight_ptr, // (K, N)
    float* out_ptr,          // (M, N)
    int64_t M,
    int64_t K,
    int64_t N,
    const std::vector<int64_t>& row_ptr,
    const std::vector<int64_t>& row_to_offset,
    const std::vector<int64_t>& row_to_count) {
  
  // 选一个 N 分块大小：建议是 L1-friendly 且是 16 的倍数（方便向量化）
  // 经验：256/384/512 都常用。你可以按平台调参。
  constexpr int64_t BN = 512;
  
  const int64_t NB = (N + BN - 1) / BN;
  
  // 二维并行：每个线程负责一个 (m, nb) tile，写 out_row[n0:n1) 无冲突
  #pragma omp parallel for collapse(2) schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t nb = 0; nb < NB; ++nb) {
      const int64_t p0 = row_ptr[(size_t)m];
      const int64_t p1 = row_ptr[(size_t)m + 1];
      if (p0 == p1) continue;
      
      const int64_t n0 = nb * BN;
      const int64_t n1 = std::min<int64_t>(n0 + BN, N);
      
      float* out_row = out_ptr + m * N;
      const float* act_row_ptr = act_ptr + m * K;
      
      // 获取该行在 nz_col_indices 中的偏移量和计数
      int64_t col_indices_offset = -1;
      int64_t count = 0;
      if ((size_t)m < row_to_offset.size() && row_to_offset[(size_t)m] >= 0) {
        col_indices_offset = row_to_offset[(size_t)m];
        count = row_to_count[(size_t)m];
      }
      if (col_indices_offset < 0) continue; // 该行没有非零元素（不应该发生，因为 p0 != p1）
      
      #if defined(__ARM_FEATURE_SVE)
        // SVE 路径：对 [n0, n1) 做向量累加，尾部用谓词处理
        for (int64_t j = 0; j < count; ++j) {
          const uint32_t k = nz_col_indices_ptr[col_indices_offset + j];
          const float a = act_row_ptr[(int64_t)k]; // 直接从 activation 取值
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
        for (int64_t j = 0; j < count; ++j) {
          const uint32_t k = nz_col_indices_ptr[col_indices_offset + j];
          const float a = act_row_ptr[(int64_t)k]; // 直接从 activation 取值
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

torch::Tensor sve_sparse_act_direct_gemm(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor nz_counts,
    torch::Tensor nz_col_indices) {
  check_inputs_direct_gemm(activation, weight, nz_counts, nz_col_indices);

  const int64_t M = activation.size(0);
  const int64_t K = activation.size(1);
  const int64_t N = weight.size(1);

  auto output = torch::zeros({M, N}, activation.options());
  if (M == 0 || K == 0 || N == 0) {
    return output;
  }

  const int64_t num_nz_rows = nz_counts.numel() / 2;
  const float* act_ptr = activation.data_ptr<float>();
  const float* weight_ptr = weight.data_ptr<float>();
  const int64_t* counts_ptr = nz_counts.data_ptr<int64_t>();
  const uint32_t* indices_ptr = nz_col_indices.data_ptr<uint32_t>();
  float* out_ptr = output.data_ptr<float>();

  // sum nnz_total and validate nz_col_indices length
  int64_t nnz_total = 0;
  for (int64_t i = 0; i < num_nz_rows; ++i) {
    const int64_t count = counts_ptr[2 * i + 1];
    TORCH_CHECK(count >= 0, "count must be non-negative");
    nnz_total += count;
  }
  TORCH_CHECK(nz_col_indices.numel() == nnz_total, "nz_col_indices size must equal sum(count) in nz_counts");

  // Step1: 构建 row_ptr 和 row_to_offset/row_to_count 映射（用于并行计算）
  std::vector<int64_t> row_ptr;
  build_row_ptr(counts_ptr, M, num_nz_rows, row_ptr);
  TORCH_CHECK(row_ptr[(size_t)M] == nnz_total, "nnz_total mismatch vs nz_counts sum");
  
  std::vector<int64_t> row_to_offset;
  std::vector<int64_t> row_to_count;
  build_row_to_offset_map(counts_ptr, num_nz_rows, row_to_offset, row_to_count);

  // Step2: 直接使用输入信息进行 GEMM 计算（不构建 CSR）
  direct_gemm_compute(
      act_ptr, indices_ptr, weight_ptr, out_ptr,
      M, K, N, row_ptr, row_to_offset, row_to_count);
  return output;
}

// 注册到 torch.ops.teal
TORCH_LIBRARY_FRAGMENT(teal, m) {
  m.def("sve_sparse_act_direct_gemm(Tensor activation, Tensor weight, Tensor nz_counts, Tensor nz_col_indices) -> Tensor");
}

TORCH_LIBRARY_IMPL(teal, CPU, m) {
  m.impl("sve_sparse_act_direct_gemm", sve_sparse_act_direct_gemm);
}
