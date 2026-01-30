#include <torch/extension.h>

#include <cstdint>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include <omp.h>

// --------- Input checks ----------
static inline void check_thr_sparsify_to_icsr_inputs(const torch::Tensor& activation) {
  TORCH_CHECK(activation.device().is_cpu(), "activation must be a CPU tensor");
  TORCH_CHECK(activation.dtype() == torch::kFloat32, "activation must be float32");
  TORCH_CHECK(activation.dim() == 2, "activation must be 2D [M, K]");
  TORCH_CHECK(activation.is_contiguous(), "activation must be contiguous");
}

// --------- Core: thr_sparsify_to_icsr(activation, threshold) -> (nz_counts, nz_col_indices, row_offsets) ----------
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
thr_sparsify_to_icsr(torch::Tensor activation, double threshold) {
  check_thr_sparsify_to_icsr_inputs(activation);

  const int64_t M = activation.size(0);
  const int64_t K = activation.size(1);
  const float thr = static_cast<float>(threshold);

  const float* act = activation.data_ptr<float>();

  // Pass1 output (temporary): per-row nnz
  std::vector<int32_t> row_nnz((size_t)M, 0);

  // ---------------- Pass1: count nnz per row (parallel) ----------------
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const float* row_ptr = act + m * K;
    int32_t nnz = 0;

    // 朴素方法：逐个元素检查
    for (int64_t k = 0; k < K; ++k) {
      if (std::fabs(row_ptr[k]) >= thr) nnz++;
    }

    row_nnz[(size_t)m] = nnz;
  }

  // --------------- Build row_offsets (prefix sum) ---------------
  // row_offsets: int64 [M+1]
  auto row_offsets_t = torch::empty({M + 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* row_offsets = row_offsets_t.data_ptr<int64_t>();
  row_offsets[0] = 0;
  for (int64_t m = 0; m < M; ++m) {
    row_offsets[m + 1] = row_offsets[m] + (int64_t)row_nnz[(size_t)m];
  }
  const int64_t total_nnz = row_offsets[M];

  // --------------- Build nz_counts (sparse pairs [row, nnz]) ---------------
  // Only rows with nnz>0 are recorded, so nz_counts length is even but not necessarily 2*M.
  int64_t num_nz_rows = 0;
  for (int64_t m = 0; m < M; ++m) {
    if (row_nnz[(size_t)m] > 0) num_nz_rows++;
  }

  auto nz_counts_t = torch::empty({2 * num_nz_rows}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* nz_counts = nz_counts_t.data_ptr<int64_t>();

  {
    int64_t w = 0;
    for (int64_t m = 0; m < M; ++m) {
      const int32_t nnz = row_nnz[(size_t)m];
      if (nnz > 0) {
        nz_counts[w++] = m;
        nz_counts[w++] = (int64_t)nnz;
      }
    }
  }

  // --------------- Allocate nz_col_indices (flattened) ---------------
  auto nz_col_indices_t = torch::empty({total_nnz}, torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCPU));
  uint32_t* out_idx = (uint32_t*)nz_col_indices_t.data_ptr<uint32_t>();

  // ---------------- Pass2: fill nz_col_indices by row_offsets (parallel) ----------------
  // Each row writes into [row_offsets[m], row_offsets[m+1]) exclusively => no atomics.
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const int32_t nnz = row_nnz[(size_t)m];
    if (nnz == 0) continue;

    const float* row_ptr = act + m * K;
    int64_t write_pos = row_offsets[m];

    // 朴素方法：逐个元素检查并写入索引
    for (int64_t k = 0; k < K; ++k) {
      if (std::fabs(row_ptr[k]) >= thr) {
        out_idx[write_pos++] = (uint32_t)k;
      }
    }

#ifndef NDEBUG
    // Debug correctness: ensure written count matches row_nnz
    TORCH_CHECK(write_pos == row_offsets[m] + (int64_t)nnz,
                "RowScan mismatch at row ", m,
                ": wrote ", (write_pos - row_offsets[m]), " expected ", nnz);
#endif
  }

  return {nz_counts_t, nz_col_indices_t, row_offsets_t};
}

// 注册到 PyTorch
// 注意：该文件会与其它算子源文件一起编译到同一个扩展中，
// 因此这里必须使用 TORCH_LIBRARY_FRAGMENT，避免与其它 TU 中的 TORCH_LIBRARY 重复定义冲突。
TORCH_LIBRARY_FRAGMENT(sparse_op, m) {
  m.def("thr_sparsify_to_icsr(Tensor activation, float threshold) -> (Tensor nz_counts, Tensor nz_col_indices, Tensor row_offsets)");
}

TORCH_LIBRARY_IMPL(sparse_op, CPU, m) {
  m.impl("thr_sparsify_to_icsr", thr_sparsify_to_icsr);
}
