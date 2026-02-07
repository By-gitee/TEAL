#include <torch/extension.h>

#include <cstdint>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Fast abs without calling libm (bit trick). Works for IEEE-754 float.
static inline float abs_f32_fast(float x) {
  union {
    float f;
    uint32_t u;
  } v;
  v.f = x;
  v.u &= 0x7FFFFFFFu;
  return v.f;
}

// --------- Input checks ---------
static inline void check_thr_sparsify_to_icsr_inputs(const torch::Tensor& activation) {
  TORCH_CHECK(activation.device().is_cpu(), "activation must be a CPU tensor");
  TORCH_CHECK(activation.dtype() == torch::kFloat32, "activation must be float32");
  TORCH_CHECK(activation.dim() == 2, "activation must be 2D [M, K]");
  TORCH_CHECK(activation.is_contiguous(), "activation must be contiguous");
}

/**
 * thr_sparsify_to_icsr(activation, threshold) -> (nz_counts, nz_col_indices, row_offsets)
 *
 * Converts a dense matrix to ICSR (indexed CSR) sparse format based on threshold.
 *
 * Args:
 *   activation: (M, K) float32 dense matrix
 *   threshold: float threshold, elements with abs(x) >= threshold are kept
 *
 * Returns:
 *   nz_counts: int64 [2*M] placeholder / non-zero count info
 *   nz_col_indices: uint32 [nnz] column index array
 *   row_offsets: int64 [M+1] CSR row pointer array (prefix sum)
 *
 * Implementation strategy:
 *   Pass 1: Count nnz per row (parallel)
 *   Pass 2: Build row_offsets (prefix sum), then fill nz_col_indices by row
 */
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
thr_sparsify_to_icsr(torch::Tensor activation, double threshold) {
  check_thr_sparsify_to_icsr_inputs(activation);

  const int64_t M = activation.size(0);
  const int64_t K = activation.size(1);
  const float thr = static_cast<float>(threshold);
  const float* act = activation.data_ptr<float>();

  // Per-row nnz counts (int64, for prefix sum)
  auto row_nnz_t = torch::empty({M}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* row_nnz = row_nnz_t.data_ptr<int64_t>();

  // ---------------- Pass 1: Count nnz per row (parallel) ----------------
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const float* row_ptr = act + m * K;
    int64_t nnz = 0;

    // Help auto-vectorization: simple loop + simd reduction.
#pragma omp simd reduction(+:nnz)
    for (int64_t k = 0; k < K; ++k) {
      const float ax = abs_f32_fast(row_ptr[k]);
      nnz += (ax >= thr);
    }

    row_nnz[m] = nnz;
  }

  // ---------------- Build row_offsets (prefix sum, int64 [M+1]) ----------------
  std::vector<int64_t> row_offsets(M + 1);
  row_offsets[0] = 0;
  for (int64_t m = 0; m < M; ++m) {
    row_offsets[m + 1] = row_offsets[m] + row_nnz[m];
  }
  const int64_t total_nnz = row_offsets[M];

  // // --------------- Build nz_counts (sparse pairs [row, nnz]) ---------------
  // // Only rows with nnz>0 are recorded, so nz_counts length is even but not necessarily 2*M.
  // int64_t num_nz_rows = 0;
  // for (int64_t m = 0; m < M; ++m) {
  //   if (row_nnz[m] > 0) num_nz_rows++;
  // }

  // auto nz_counts_t = torch::empty({2 * num_nz_rows}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  auto nz_counts_t = torch::empty({2 * M}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  // int64_t* nz_counts = nz_counts_t.data_ptr<int64_t>();

  // {
  //   int64_t w = 0;
  //   for (int64_t m = 0; m < M; ++m) {
  //     const int64_t nnz = row_nnz[m];
  //     if (nnz > 0) {
  //       nz_counts[w++] = m;
  //       nz_counts[w++] = nnz;
  //     }
  //   }
  // }

  // ---------------- Allocate nz_col_indices (flattened, uint32) ----------------
  auto nz_col_indices_t = torch::empty({total_nnz}, torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCPU));
  uint32_t* out_idx = (uint32_t*)nz_col_indices_t.data_ptr<uint32_t>();

  // ---------------- Pass 2: Fill nz_col_indices by row (parallel, no atomics) ----------------
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const int64_t nnz = row_nnz[m];
    if (nnz == 0) continue;

    const float* row_ptr = act + m * K;
    uint32_t* __restrict dst_idx = out_idx + row_offsets[m];
    int64_t write_pos = 0;

    // Full-keep fast path
    if (nnz == K) {
      for (int64_t k2 = 0; k2 < K; ++k2) {
        dst_idx[k2] = (uint32_t)k2;
      }
      write_pos = K;
#ifndef NDEBUG
      TORCH_CHECK(write_pos == nnz, "thr_sparsify_to_icsr: nnz==K fastpath mismatch");
#endif
      continue;
    }

    // Small staging buffer (stack). Batch stores to reduce store traffic and branch pressure.
    const int BUFSZ = 64;
    uint32_t idx_buf[BUFSZ];
    int buf_n = 0;

    int64_t k = 0;

    // mild unroll by 4 to reduce loop overhead; keeps code simple for compiler.
    for (; k + 4 <= K; k += 4) {
      const float x0 = row_ptr[k + 0];
      const float x1 = row_ptr[k + 1];
      const float x2 = row_ptr[k + 2];
      const float x3 = row_ptr[k + 3];

      const bool keep0 = (abs_f32_fast(x0) >= thr);
      const bool keep1 = (abs_f32_fast(x1) >= thr);
      const bool keep2 = (abs_f32_fast(x2) >= thr);
      const bool keep3 = (abs_f32_fast(x3) >= thr);

      if (keep0) { idx_buf[buf_n++] = (uint32_t)(k + 0); }
      if (keep1) { idx_buf[buf_n++] = (uint32_t)(k + 1); }
      if (keep2) { idx_buf[buf_n++] = (uint32_t)(k + 2); }
      if (keep3) { idx_buf[buf_n++] = (uint32_t)(k + 3); }

      // flush if buffer is getting full
      if (buf_n >= BUFSZ - 4) {
        std::memcpy(dst_idx + write_pos, idx_buf, (size_t)buf_n * sizeof(uint32_t));
        write_pos += buf_n;
        buf_n = 0;
      }
    }

    // tail
    for (; k < K; ++k) {
      const float x = row_ptr[k];
      if (abs_f32_fast(x) >= thr) {
        idx_buf[buf_n++] = (uint32_t)k;
        if (buf_n == BUFSZ) {
          std::memcpy(dst_idx + write_pos, idx_buf, (size_t)buf_n * sizeof(uint32_t));
          write_pos += buf_n;
          buf_n = 0;
        }
      }
    }

    // final flush
    if (buf_n) {
      std::memcpy(dst_idx + write_pos, idx_buf, (size_t)buf_n * sizeof(uint32_t));
      write_pos += buf_n;
    }

#ifndef NDEBUG
    // Debug correctness: ensure written count matches row_nnz
    TORCH_CHECK(write_pos == nnz,
                "RowScan mismatch at row ", m,
                ": wrote ", write_pos, " expected ", nnz);
#endif
  }

  // Convert row_offsets to Tensor and return
  torch::Tensor row_offsets_t = torch::empty({M + 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  std::memcpy(row_offsets_t.data_ptr<int64_t>(), row_offsets.data(), (size_t)(M + 1) * sizeof(int64_t));
  return {nz_counts_t, nz_col_indices_t, row_offsets_t};
}

// Register to PyTorch.
// Note: This file is compiled with other operator sources into the same extension.
// Use TORCH_LIBRARY_FRAGMENT to avoid conflicts with TORCH_LIBRARY in other translation units.
TORCH_LIBRARY_FRAGMENT(sparse_op, m) {
  m.def("thr_sparsify_to_icsr(Tensor activation, float threshold) -> (Tensor nz_counts, Tensor nz_col_indices, Tensor row_offsets)");
}

TORCH_LIBRARY_IMPL(sparse_op, CPU, m) {
  m.impl("thr_sparsify_to_icsr", thr_sparsify_to_icsr);
}
