#include <torch/extension.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// --------- Input checks ----------
static inline void check_mask_sparsify_to_icsr_inputs(const torch::Tensor& mask) {
  TORCH_CHECK(mask.device().is_cpu(), "mask must be a CPU tensor");
  TORCH_CHECK(mask.dtype() == torch::kUInt8, "mask must be uint8");
  TORCH_CHECK(mask.dim() == 2, "mask must be 2D [M, K]");
  TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
}

// --------- Core: mask_sparsify_to_icsr(mask) -> (nz_counts, nz_col_indices, row_offsets) ----------
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
mask_sparsify_to_icsr(torch::Tensor mask) {
  check_mask_sparsify_to_icsr_inputs(mask);

  const int64_t M = mask.size(0);
  const int64_t K = mask.size(1);
  const uint8_t* mask_ptr = mask.data_ptr<uint8_t>();

  // Pass1 output (temporary): per-row nnz
  // Use a Tensor here to keep allocation/ownership overhead comparable with the SVE version.
  auto row_nnz_t = torch::empty({M}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* row_nnz = row_nnz_t.data_ptr<int64_t>();

  // ---------------- Pass1: count nnz per row (parallel, no atomics) ----------------
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t m = 0; m < M; ++m) {
    const uint8_t* row_ptr = mask_ptr + m * K;
    int64_t nnz = 0;

    // Help auto-vectorization: simple loop + simd reduction.
    // 对于 uint8_t mask, 非零元素即为有效元素
#ifdef _OPENMP
#pragma omp simd reduction(+:nnz)
#endif
    for (int64_t k = 0; k < K; ++k) {
      nnz += (row_ptr[k] != 0);
    }

    row_nnz[m] = nnz;
  }

  // --------------- Fused Scan: Build row_offsets + keep_prefix simultaneously ---------------
  // row_offsets: int64 [M+1]
  auto row_offsets_t = torch::empty({M + 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* row_offsets = row_offsets_t.data_ptr<int64_t>();
  
  // keep_prefix[m] = number of non-empty rows in [0, m)
  std::vector<int64_t> keep_prefix((size_t)M + 1, 0);
  
  // Fused scan: compute both prefix sums in one pass
  row_offsets[0] = 0;
  keep_prefix[0] = 0;
  for (int64_t m = 0; m < M; ++m) {
    row_offsets[m + 1] = row_offsets[m] + row_nnz[m];
    keep_prefix[(size_t)m + 1] = keep_prefix[(size_t)m] + (row_nnz[m] > 0 ? 1 : 0);
  }
  const int64_t total_nnz = row_offsets[M];
  const int64_t num_nz_rows = keep_prefix[(size_t)M];

  // --------------- Allocate nz_counts (sparse pairs [row, nnz]) ---------------
  // Only rows with nnz>0 are recorded, so nz_counts length is even but not necessarily 2*M.
  auto nz_counts_t = torch::empty({2 * num_nz_rows}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* nz_counts = (num_nz_rows > 0) ? nz_counts_t.data_ptr<int64_t>() : nullptr;

  // --------------- Allocate nz_col_indices (flattened) ---------------
  auto nz_col_indices_t = torch::empty({total_nnz}, torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCPU));
  uint32_t* out_idx = (total_nnz > 0) ? nz_col_indices_t.data_ptr<uint32_t>() : nullptr;

  // ---------------- Pass2: fill nz_col_indices + nz_counts simultaneously (parallel, no atomics) ----------------
  // Each row writes into [row_offsets[m], row_offsets[m+1]) exclusively => no atomics.
  // Each non-empty row also writes to its unique slot in nz_counts => no atomics.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t m = 0; m < M; ++m) {
    const int64_t nnz = row_nnz[m];
    if (nnz == 0) continue;

    // Fill nz_counts for this non-empty row
    const int64_t slot = keep_prefix[(size_t)m];
    nz_counts[slot * 2 + 0] = m;
    nz_counts[slot * 2 + 1] = nnz;

    const uint8_t* row_ptr = mask_ptr + m * K;
    uint32_t* __restrict dst_idx = out_idx + row_offsets[m];
    int64_t write_pos = 0;

    // Full-keep fast path
    if (nnz == K) {
      for (int64_t k2 = 0; k2 < K; ++k2) {
        dst_idx[k2] = (uint32_t)k2;
      }
      write_pos = K;
#ifndef NDEBUG
      TORCH_CHECK(write_pos == nnz, "mask_sparsify_to_icsr: nnz==K fastpath mismatch");
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
      const uint8_t x0 = row_ptr[k + 0];
      const uint8_t x1 = row_ptr[k + 1];
      const uint8_t x2 = row_ptr[k + 2];
      const uint8_t x3 = row_ptr[k + 3];

      const bool keep0 = (x0 != 0);
      const bool keep1 = (x1 != 0);
      const bool keep2 = (x2 != 0);
      const bool keep3 = (x3 != 0);

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
      const uint8_t x = row_ptr[k];
      if (x != 0) {
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

  return {nz_counts_t, nz_col_indices_t, row_offsets_t};
}

// 注册到 PyTorch
// 注意：该文件会与其它算子源文件一起编译到同一个扩展中，
// 因此这里必须使用 TORCH_LIBRARY_FRAGMENT，避免与其它 TU 中的 TORCH_LIBRARY 重复定义冲突。
TORCH_LIBRARY_FRAGMENT(sparse_op, m) {
  m.def("mask_sparsify_to_icsr(Tensor mask) -> (Tensor nz_counts, Tensor nz_col_indices, Tensor row_offsets)");
}

TORCH_LIBRARY_IMPL(sparse_op, CPU, m) {
  m.impl("mask_sparsify_to_icsr", mask_sparsify_to_icsr);
}