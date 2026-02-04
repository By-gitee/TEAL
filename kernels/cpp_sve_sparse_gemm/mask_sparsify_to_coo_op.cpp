#include <torch/extension.h>

#include <cstdint>
#include <vector>
#include <cstring>   // memcpy
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

using torch::Tensor;

static inline void check_mask_sparsify_to_coo_inputs(const Tensor& activation, const Tensor& mask) {
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

/**
 * mask_sparsify_to_coo(activation, mask) -> (row_indices, col_indices, values)
 * 
 * 根据 mask 将稠密矩阵转换为 COO 格式的稀疏矩阵。
 * 
 * Args:
 *   activation: (M, K) float32 稠密矩阵
 *   mask: (M, K) uint8 掩码矩阵，非零元素标记需要保留的位置
 * 
 * Returns:
 *   row_indices: int64 [nnz] 行索引数组（已按行排序）
 *   col_indices: uint32 [nnz] 列索引数组
 *   values: float32 [nnz] 非零元素值数组
 * 
 * 实现策略：
 *   Pass 1: 并行统计每行的非零元素数量
 *   Pass 2: 计算行偏移（前缀和）
 *   Pass 3: 并行填充 COO 三元组 (row_idx, col_idx, value)
 */
static std::tuple<Tensor, Tensor, Tensor>
mask_sparsify_to_coo(torch::Tensor activation, torch::Tensor mask) {
  check_mask_sparsify_to_coo_inputs(activation, mask);

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
    // Count non-zero mask elements.
#if defined(_OPENMP)
#pragma omp simd reduction(+:nnz)
#endif
    for (int64_t k = 0; k < K; ++k) {
      nnz += (mask_row[k] != 0);
    }

    counts[m] = nnz;
  }

  // ---------------- row_offsets prefix sum (M+1) for internal use ----------------
  // We compute row_offsets to know where each row's data starts in the flat arrays
  std::vector<int64_t> row_offsets(M + 1);
  row_offsets[0] = 0;
  for (int64_t m = 0; m < M; ++m) {
    row_offsets[m + 1] = row_offsets[m] + counts[m];
  }
  const int64_t total_nnz = row_offsets[M];

  // ---------------- Allocate COO arrays: row_indices, col_indices, values ----------------
  Tensor row_indices = torch::empty({total_nnz}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* out_row = (total_nnz > 0) ? row_indices.data_ptr<int64_t>() : nullptr;

  Tensor col_indices = torch::empty({total_nnz}, torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCPU));
  uint32_t* out_col = (total_nnz > 0) ? col_indices.data_ptr<uint32_t>() : nullptr;

  Tensor values = torch::empty({total_nnz}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  float* out_val = (total_nnz > 0) ? values.data_ptr<float>() : nullptr;

  // ---------------- Pass2: write COO triplets (row_idx, col_idx, value) ----------------
  // Optimization: per-row small buffer, batch stores to reduce store traffic and branch pressure.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t m = 0; m < M; ++m) {
    const int64_t nnz = counts[m];
    if (nnz == 0) continue;

    const float* act_row = act_ptr + m * K;
    const uint8_t* mask_row = mask_ptr + m * K;
    int64_t* __restrict dst_row = out_row + row_offsets[m];
    uint32_t* __restrict dst_col = out_col + row_offsets[m];
    float* __restrict dst_val = out_val + row_offsets[m];
    int64_t write_pos = 0;

    // Full-keep fast path: all elements in the row are masked
    if (nnz == K) {
      // sequential fill
      for (int64_t k = 0; k < K; ++k) {
        dst_row[k] = m;
        dst_col[k] = (uint32_t)k;
        dst_val[k] = act_row[k];
      }
      write_pos = K;
#ifndef NDEBUG
      TORCH_CHECK(write_pos == nnz, "mask_sparsify_to_coo: nnz==K fastpath mismatch");
#endif
      continue;
    }

    // Small staging buffers (stack). Tune BUFSZ to trade branch vs memcpy overhead.
    // 64 is a good default for K up to ~1e4 and M small.
    const int BUFSZ = 64;
    int64_t row_buf[BUFSZ];
    uint32_t col_buf[BUFSZ];
    float val_buf[BUFSZ];
    int buf_n = 0;

    int64_t k = 0;

    // mild unroll by 4 to reduce loop overhead; keeps code simple for compiler.
    for (; k + 4 <= K; k += 4) {
      const uint8_t m0 = mask_row[k + 0];
      const uint8_t m1 = mask_row[k + 1];
      const uint8_t m2 = mask_row[k + 2];
      const uint8_t m3 = mask_row[k + 3];

      const bool keep0 = (m0 != 0);
      const bool keep1 = (m1 != 0);
      const bool keep2 = (m2 != 0);
      const bool keep3 = (m3 != 0);

      if (keep0) { row_buf[buf_n] = m; col_buf[buf_n] = (uint32_t)(k + 0); val_buf[buf_n] = act_row[k + 0]; ++buf_n; }
      if (keep1) { row_buf[buf_n] = m; col_buf[buf_n] = (uint32_t)(k + 1); val_buf[buf_n] = act_row[k + 1]; ++buf_n; }
      if (keep2) { row_buf[buf_n] = m; col_buf[buf_n] = (uint32_t)(k + 2); val_buf[buf_n] = act_row[k + 2]; ++buf_n; }
      if (keep3) { row_buf[buf_n] = m; col_buf[buf_n] = (uint32_t)(k + 3); val_buf[buf_n] = act_row[k + 3]; ++buf_n; }

      // flush if buffer is getting full
      if (buf_n >= BUFSZ - 4) {
        std::memcpy(dst_row + write_pos, row_buf, (size_t)buf_n * sizeof(int64_t));
        std::memcpy(dst_col + write_pos, col_buf, (size_t)buf_n * sizeof(uint32_t));
        std::memcpy(dst_val + write_pos, val_buf, (size_t)buf_n * sizeof(float));
        write_pos += buf_n;
        buf_n = 0;
      }
    }

    // tail
    for (; k < K; ++k) {
      if (mask_row[k] != 0) {
        row_buf[buf_n] = m;
        col_buf[buf_n] = (uint32_t)k;
        val_buf[buf_n] = act_row[k];
        ++buf_n;
        if (buf_n == BUFSZ) {
          std::memcpy(dst_row + write_pos, row_buf, (size_t)buf_n * sizeof(int64_t));
          std::memcpy(dst_col + write_pos, col_buf, (size_t)buf_n * sizeof(uint32_t));
          std::memcpy(dst_val + write_pos, val_buf, (size_t)buf_n * sizeof(float));
          write_pos += buf_n;
          buf_n = 0;
        }
      }
    }

    // final flush
    if (buf_n) {
      std::memcpy(dst_row + write_pos, row_buf, (size_t)buf_n * sizeof(int64_t));
      std::memcpy(dst_col + write_pos, col_buf, (size_t)buf_n * sizeof(uint32_t));
      std::memcpy(dst_val + write_pos, val_buf, (size_t)buf_n * sizeof(float));
      write_pos += buf_n;
    }

#ifndef NDEBUG
    TORCH_CHECK (write_pos == nnz,
                "mask_sparsify_to_coo: write_pos != nnz at row ", m,
                " write_pos=", write_pos, " nnz=", nnz);
#endif
  }

  return {row_indices, col_indices, values};
}

// 注册到 PyTorch
// 注意：该文件会与其它算子源文件一起编译到同一个扩展中，
// 因此这里必须使用 TORCH_LIBRARY_FRAGMENT，避免与其它 TU 中的 TORCH_LIBRARY 重复定义冲突。
TORCH_LIBRARY_FRAGMENT(sparse_op, m) {
  m.def("mask_sparsify_to_coo(Tensor activation, Tensor mask) -> (Tensor row_indices, Tensor col_indices, Tensor values)");
}

TORCH_LIBRARY_IMPL(sparse_op, CPU, m) {
  m.impl("mask_sparsify_to_coo", mask_sparsify_to_coo);
}
