// dense_to_csr_omp_op.cpp  (common optimizations, no SVE/SVE2 specifics)
#include <torch/extension.h>

#include <cstdint>
#include <vector>
#include <cmath>
#include <cstring>   // memcpy
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

using torch::Tensor;

static inline void check_inputs_dense_to_csr(const Tensor& activation) {
  TORCH_CHECK(activation.device().is_cpu(), "activation must be a CPU tensor");
  TORCH_CHECK(activation.dtype() == torch::kFloat32, "activation must be float32");
  TORCH_CHECK(activation.is_contiguous(), "activation must be contiguous");
  TORCH_CHECK(activation.dim() == 2, "activation must be 2D [M, K]");
}

// Fast abs without calling libm (bit trick). Works for IEEE-754 float.
static inline float abs_f32_fast(float x) {
  union { float f; uint32_t u; } v;
  v.f = x;
  v.u &= 0x7FFFFFFFu;
  return v.f;
}

// Return: [row_offsets(int64), col_idx(uint32), values(float32)]
static std::vector<Tensor> dense_to_csr_omp(const Tensor& activation, double threshold) {
  check_inputs_dense_to_csr(activation);

  const int64_t M = activation.size(0);
  const int64_t K = activation.size(1);
  const float* act_ptr = activation.data_ptr<float>();
  const float thr = static_cast<float>(threshold);

  // counts per row (int64)
  Tensor counts_t = torch::empty({M}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* counts = counts_t.data_ptr<int64_t>();

  // ---------------- Pass1: count nnz per row ----------------
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t m = 0; m < M; ++m) {
    const float* row = act_ptr + m * K;
    int64_t nnz = 0;

    // Help auto-vectorization: simple loop + simd reduction.
    // Use abs_f32_fast to avoid std::fabs overhead and keep float domain.
#if defined(_OPENMP)
#pragma omp simd reduction(+:nnz)
#endif
    for (int64_t k = 0; k < K; ++k) {
      const float ax = abs_f32_fast(row[k]);
      nnz += (ax >= thr);
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

  // ---------------- Allocate CSR arrays: col_idx + values ----------------
  Tensor col_idx = torch::empty({total_nnz}, torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCPU));
  uint32_t* out_idx = (total_nnz > 0) ? col_idx.data_ptr<uint32_t>() : nullptr;

  Tensor values = torch::empty({total_nnz}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  float* out_val = (total_nnz > 0) ? values.data_ptr<float>() : nullptr;

  // ---------------- Pass2: write col_idx + values ----------------
  // Optimization: per-row small buffer, batch stores to reduce store traffic and branch pressure.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t m = 0; m < M; ++m) {
    const int64_t nnz = counts[m];
    if (nnz == 0) continue;

    const float* row = act_ptr + m * K;
    uint32_t* __restrict dst_idx = out_idx + offsets[m];
    float* __restrict dst_val = out_val + offsets[m];
    int64_t write_pos = 0;

    // Full-keep fast path
    if (nnz == K) {
      // sequential fill
      for (int64_t k = 0; k < K; ++k) {
        dst_idx[k] = (uint32_t)k;
        dst_val[k] = row[k];
      }
#ifndef NDEBUG
      TORCH_CHECK(write_pos + K == nnz, "dense_to_csr_omp: nnz==K fastpath mismatch");
#endif
      continue;
    }

    // Small staging buffers (stack). Tune BUFSZ to trade branch vs memcpy overhead.
    // 64 is a good default for K up to ~1e4 and M small.
    constexpr int BUFSZ = 64;
    uint32_t idx_buf[BUFSZ];
    float val_buf[BUFSZ];
    int buf_n = 0;

    int64_t k = 0;

    // mild unroll by 4 to reduce loop overhead; keeps code simple for compiler.
    for (; k + 4 <= K; k += 4) {
      const float x0 = row[k + 0];
      const float x1 = row[k + 1];
      const float x2 = row[k + 2];
      const float x3 = row[k + 3];

      const bool keep0 = (abs_f32_fast(x0) >= thr);
      const bool keep1 = (abs_f32_fast(x1) >= thr);
      const bool keep2 = (abs_f32_fast(x2) >= thr);
      const bool keep3 = (abs_f32_fast(x3) >= thr);

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
      const float x = row[k];
      if (abs_f32_fast(x) >= thr) {
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
    TORCH_CHECK(write_pos == nnz,
                "dense_to_csr_omp: write_pos != nnz at row ", m,
                " write_pos=", write_pos, " nnz=", nnz);
#endif
  }

  return {row_offsets, col_idx, values};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dense_to_csr_omp", &dense_to_csr_omp,
        "Dense activation to CSR (row_offsets, col_idx, values) using OpenMP + common optimizations (CPU, no SVE2)");
}
