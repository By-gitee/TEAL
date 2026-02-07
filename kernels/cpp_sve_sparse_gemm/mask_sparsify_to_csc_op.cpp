#include <torch/extension.h>

#include <cstdint>
#include <vector>
#include <cstring>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

using torch::Tensor;

static inline void check_mask_sparsify_to_csc_inputs(const Tensor& activation, const Tensor& mask) {
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
 * mask_sparsify_to_csc(activation, mask) -> (col_ptr, row_indices, values)
 *
 * Converts dense matrix to CSC (Compressed Sparse Column) format based on mask.
 *
 * Args:
 *   activation: (M, K) float32 dense matrix
 *   mask: (M, K) uint8 mask matrix; non-zero indicates positions to keep
 *
 * Returns:
 *   col_ptr: int64 [K+1] column pointer array
 *   row_indices: uint32 [nnz] row index array (sorted by row within each column)
 *   values: float32 [nnz] non-zero element values
 *
 * Implementation:
 *   Pass 1: Parallel count nnz per column (thread-local counts)
 *   Pass 2: Reduce thread-local counts and build col_ptr (prefix sum)
 *   Pass 3: Parallel fill CSC (row_indices, values)
 */
static std::tuple<Tensor, Tensor, Tensor>
mask_sparsify_to_csc(torch::Tensor activation, torch::Tensor mask) {
  check_mask_sparsify_to_csc_inputs(activation, mask);

  const int64_t M = activation.size(0);
  const int64_t K = activation.size(1);
  const float* act_ptr = activation.data_ptr<float>();
  const uint8_t* mask_ptr = mask.data_ptr<uint8_t>();

  int num_threads = 1;
#ifdef _OPENMP
  num_threads = omp_get_max_threads();
#endif

  // Pass1: thread-local counts per column (u32 is enough; M is small in your workloads)
  std::vector<uint32_t> local_counts((size_t)num_threads * (size_t)K, 0);

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif
    uint32_t* my_counts = local_counts.data() + (size_t)tid * (size_t)K;

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int64_t m = 0; m < M; ++m) {
      const uint8_t* mask_row = mask_ptr + m * K;

      // Vectorized per-column counting:
      // counts[k] += (mask_row[k] != 0) ? 1 : 0
      // Help auto-vectorization with simple loop
#if defined(_OPENMP)
#pragma omp simd
#endif
      for (int64_t k = 0; k < K; ++k) {
        my_counts[k] += (mask_row[k] != 0);
      }
    }
  }

  // Reduce counts to col_counts (int64)
  std::vector<int64_t> col_counts((size_t)K, 0);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t k = 0; k < K; ++k) {
    int64_t sum = 0;
    const size_t kk = (size_t)k;
    for (int t = 0; t < num_threads; ++t) {
      sum += (int64_t)local_counts[(size_t)t * (size_t)K + kk];
    }
    col_counts[kk] = sum;
  }

  // Pass2: build col_ptr (int64)
  Tensor col_ptr = torch::empty({K + 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* ptr = col_ptr.data_ptr<int64_t>();
  ptr[0] = 0;
  for (int64_t k = 0; k < K; ++k) {
    ptr[k + 1] = ptr[k] + col_counts[(size_t)k];
  }
  const int64_t total_nnz = ptr[K];

  Tensor row_indices = torch::empty({total_nnz}, torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCPU));
  uint32_t* out_row = (total_nnz > 0) ? row_indices.data_ptr<uint32_t>() : nullptr;

  Tensor values = torch::empty({total_nnz}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  float* out_val = (total_nnz > 0) ? values.data_ptr<float>() : nullptr;

  // Build per-thread base offsets: thread_base[tid][k] = ptr[k] + sum_{t'<tid} local_counts[t'][k]
  std::vector<int64_t> thread_base((size_t)num_threads * (size_t)K, 0);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t k = 0; k < K; ++k) {
    int64_t base = ptr[k];
    int64_t run = 0;
    const size_t kk = (size_t)k;
    for (int t = 0; t < num_threads; ++t) {
      thread_base[(size_t)t * (size_t)K + kk] = base + run;
      run += (int64_t)local_counts[(size_t)t * (size_t)K + kk];
    }
#ifndef NDEBUG
    TORCH_CHECK(run == col_counts[kk], "mask_sparsify_to_csc: count mismatch at col ", k);
#endif
  }

  // Pass3: write without atomics
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif

    std::vector<int64_t> pos((size_t)K);
    std::memcpy(
        pos.data(),
        thread_base.data() + (size_t)tid * (size_t)K,
        (size_t)K * sizeof(int64_t));

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int64_t m = 0; m < M; ++m) {
      const float* act_row = act_ptr + m * K;
      const uint8_t* mask_row = mask_ptr + m * K;

      for (int64_t k = 0; k < K; ++k) {
        if (mask_row[k] != 0) {
          const float x = act_row[k];
          const int64_t p = pos[(size_t)k]++;
          out_row[p] = (uint32_t)m;
          out_val[p] = x;
        }
      }
    }

#ifndef NDEBUG
    const int64_t* my_base = thread_base.data() + (size_t)tid * (size_t)K;
    const uint32_t* my_cnt = local_counts.data() + (size_t)tid * (size_t)K;
    for (int64_t k2 = 0; k2 < K; ++k2) {
      const size_t kk2 = (size_t)k2;
      TORCH_CHECK(
          pos[kk2] == my_base[kk2] + (int64_t)my_cnt[kk2],
          "mask_sparsify_to_csc: thread write mismatch at tid=", tid, " col=", k2);
    }
#endif
  }

  return {col_ptr, row_indices, values};
}

// Register to PyTorch.
// Note: This file is compiled with other operator sources into the same extension.
// Use TORCH_LIBRARY_FRAGMENT to avoid conflicts with TORCH_LIBRARY in other translation units.
TORCH_LIBRARY_FRAGMENT(sparse_op, m) {
  m.def("mask_sparsify_to_csc(Tensor activation, Tensor mask) -> (Tensor col_ptr, Tensor row_indices, Tensor values)");
}

TORCH_LIBRARY_IMPL(sparse_op, CPU, m) {
  m.impl("mask_sparsify_to_csc", mask_sparsify_to_csc);
}
