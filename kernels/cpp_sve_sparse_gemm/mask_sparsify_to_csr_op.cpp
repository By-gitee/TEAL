#include <torch/extension.h>

#include <cstdint>
#include <vector>
#include <cstring>   // memcpy
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

using torch::Tensor;

static inline void check_mask_sparsify_to_csr_inputs(const Tensor& activation, const Tensor& mask) {
  TORCH_CHECK(activation.device().is_cpu(), "activation must be a CPU tensor");
  TORCH_CHECK(activation.dtype() == torch::kFloat32, "activation must be float32");
  TORCH_CHECK(activation.is_contiguous(), "activation must be contiguous");
  TORCH_CHECK(activation.dim() == 2, "activation must be 2D [M, K]");
  
  TORCH_CHECK(mask.device().is_cpu(), "mask must be a CPU tensor");
  TORCH_CHECK(mask.dtype() == torch::kUInt8, "mask must be uint8");
  TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
  TORCH_CHECK(mask.dim() == 2, "mask must be 2D [M, K]");
  
  TORCH_CHECK(activation.size(0) == mask.size(0) && activation.size(1) == mask.size(1),
              "activation and mask must have the same shape");
}

// mask_sparsify_to_csr(activation, mask) -> (row_offsets, nz_col_indices, values)
static std::tuple<Tensor, Tensor, Tensor>
mask_sparsify_to_csr(torch::Tensor activation, torch::Tensor mask) {
  check_mask_sparsify_to_csr_inputs(activation, mask);

  const int64_t M = activation.size(0);
  const int64_t K = activation.size(1);
  const float* act_ptr = activation.data_ptr<float>();
  const uint8_t* mask_ptr = mask.data_ptr<uint8_t>();

  // counts per row (int64)
  Tensor counts_t = torch::empty({M}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* counts = counts_t.data_ptr<int64_t>();

  // ---------------- Pass1: count nnz per row ----------------
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t m = 0; m < M; ++m) {
    const uint8_t* mask_row = mask_ptr + m * K;
    int64_t nnz = 0;

    // Help auto-vectorization: simple loop + simd reduction.
#if defined(_OPENMP)
#pragma omp simd reduction(+:nnz)
#endif
    for (int64_t k = 0; k < K; ++k) {
      nnz += (mask_row[k] != 0);
    }

    counts[m] = nnz;
  }

  // ---------------- row_offsets prefix sum (M+1) ----------------
  Tensor row_offsets = torch::empty({M + 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* offsets = row_offsets.data_ptr<int64_t>();
  offsets[0] = 0;
  for (int64_t m = 0; m < M; ++m) {
    offsets[m + 1] = offsets[m] + counts[m];
  }
  const int64_t total_nnz = offsets[M];

  // ---------------- Allocate CSR arrays: nz_col_indices + values ----------------
  Tensor nz_col_indices = torch::empty({total_nnz}, torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCPU));
  uint32_t* out_idx = (total_nnz > 0) ? nz_col_indices.data_ptr<uint32_t>() : nullptr;

  Tensor values = torch::empty({total_nnz}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  float* out_val = (total_nnz > 0) ? values.data_ptr<float>() : nullptr;

  // ---------------- Pass2: write nz_col_indices + values ----------------
  // Optimization: per-row small buffer, batch stores to reduce store traffic and branch pressure.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t m = 0; m < M; ++m) {
    const int64_t nnz = counts[m];
    if (nnz == 0) continue;

    const float* act_row = act_ptr + m * K;
    const uint8_t* mask_row = mask_ptr + m * K;
    uint32_t* __restrict dst_idx = out_idx + offsets[m];
    float* __restrict dst_val = out_val + offsets[m];
    int64_t write_pos = 0;

    // Full-keep fast path
    if (nnz == K) {
      // sequential fill
      for (int64_t k = 0; k < K; ++k) {
        dst_idx[k] = (uint32_t)k;
        dst_val[k] = act_row[k];
      }
      write_pos = K;
#ifndef NDEBUG
      TORCH_CHECK(write_pos == nnz, "mask_sparsify_to_csr: nnz==K fastpath mismatch");
#endif
      continue;
    }

    // Small staging buffers (stack). Tune BUFSZ to trade branch vs memcpy overhead.
    // 64 is a good default for K up to ~1e4 and M small.
    const int BUFSZ = 64;
    uint32_t idx_buf[BUFSZ];
    float val_buf[BUFSZ];
    int buf_n = 0;

    int64_t k = 0;

    // mild unroll by 4 to reduce loop overhead; keeps code simple for compiler.
    for (; k + 4 <= K; k += 4) {
      const float x0 = act_row[k + 0];
      const float x1 = act_row[k + 1];
      const float x2 = act_row[k + 2];
      const float x3 = act_row[k + 3];

      const bool keep0 = (mask_row[k + 0] != 0);
      const bool keep1 = (mask_row[k + 1] != 0);
      const bool keep2 = (mask_row[k + 2] != 0);
      const bool keep3 = (mask_row[k + 3] != 0);

      if (keep0) { idx_buf[buf_n] = (uint32_t)(k + 0); val_buf[buf_n] = x0; ++buf_n; }
      if (keep1) { idx_buf[buf_n] = (uint32_t)(k + 1); val_buf[buf_n] = x1; ++buf_n; }
      if (keep2) { idx_buf[buf_n] = (uint32_t)(k + 2); val_buf[buf_n] = x2; ++buf_n; }
      if (keep3) { idx_buf[buf_n] = (uint32_t)(k + 3); val_buf[buf_n] = x3; ++buf_n; }

      // flush if buffer is getting full
      if (buf_n >= BUFSZ - 4) {
        std::memcpy(dst_idx + write_pos, idx_buf, (size_t)buf_n * sizeof(uint32_t));
        std::memcpy(dst_val + write_pos, val_buf, (size_t)buf_n * sizeof(float));
        write_pos += buf_n;
        buf_n = 0;
      }
    }

    // tail
    for (; k < K; ++k) {
      const float x = act_row[k];
      if (mask_row[k] != 0) {
        idx_buf[buf_n] = (uint32_t)k;
        val_buf[buf_n] = x;
        ++buf_n;
        if (buf_n == BUFSZ) {
          std::memcpy(dst_idx + write_pos, idx_buf, (size_t)buf_n * sizeof(uint32_t));
          std::memcpy(dst_val + write_pos, val_buf, (size_t)buf_n * sizeof(float));
          write_pos += buf_n;
          buf_n = 0;
        }
      }
    }

    // final flush
    if (buf_n) {
      std::memcpy(dst_idx + write_pos, idx_buf, (size_t)buf_n * sizeof(uint32_t));
      std::memcpy(dst_val + write_pos, val_buf, (size_t)buf_n * sizeof(float));
      write_pos += buf_n;
    }

#ifndef NDEBUG
    TORCH_CHECK (write_pos == nnz,
                "mask_sparsify_to_csr: write_pos != nnz at row ", m,
                " write_pos=", write_pos, " nnz=", nnz);
#endif
  }

  return {row_offsets, nz_col_indices, values};
}

// 注册到 PyTorch
// 注意：该文件会与其它算子源文件一起编译到同一个扩展中，
// 因此这里必须使用 TORCH_LIBRARY_FRAGMENT，避免与其它 TU 中的 TORCH_LIBRARY 重复定义冲突。
TORCH_LIBRARY_FRAGMENT(sparse_op, m) {
  m.def("mask_sparsify_to_csr(Tensor activation, Tensor mask) -> (Tensor row_offsets, Tensor nz_col_indices, Tensor values)");
}

TORCH_LIBRARY_IMPL(sparse_op, CPU, m) {
  m.impl("mask_sparsify_to_csr", mask_sparsify_to_csr);
}
