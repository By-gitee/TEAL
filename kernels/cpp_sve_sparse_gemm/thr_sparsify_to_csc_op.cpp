#include <torch/extension.h>

#include <cstdint>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

using torch::Tensor;

static inline void check_thr_sparsify_to_csc_inputs(const Tensor& activation) {
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

/**
 * thr_sparsify_to_csc(activation, threshold) -> (col_ptr, row_indices, values)
 *
 * Converts a dense matrix to CSC sparse format based on threshold.
 *
 * Args:
 *   activation: (M, K) float32 dense matrix
 *   threshold: float threshold, elements with abs(x) >= threshold are kept
 *
 * Returns:
 *   col_ptr: int64 [K+1] CSC column pointer array (prefix sum)
 *   row_indices: uint32 [nnz] row index array
 *   values: float32 [nnz] non-zero element value array
 *
 * Implementation strategy:
 *   Pass 1: Thread-local counts per column (optionally SVE vectorized)
 *   Pass 2: Reduce to col_counts, build col_ptr (prefix sum)
 *   Pass 3: Write row_indices and values without atomics (per-thread base offsets)
 */
static std::tuple<Tensor, Tensor, Tensor>
thr_sparsify_to_csc(torch::Tensor activation, double threshold) {
  check_thr_sparsify_to_csc_inputs(activation);

  const int64_t M = activation.size(0);
  const int64_t K = activation.size(1);
  const float* act_ptr = activation.data_ptr<float>();
  const float thr = static_cast<float>(threshold);

  int num_threads = 1;
#ifdef _OPENMP
  num_threads = omp_get_max_threads();
#endif

  // ---------------- Pass 1: Thread-local counts per column (uint32) ----------------
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

#if defined(__ARM_FEATURE_SVE)
    const svuint32_t vone = svdup_u32(1);
    const svuint32_t vzero = svdup_u32(0);
    const svfloat32_t vthr = svdup_f32(thr);
    const int64_t vl = (int64_t)svcntw();  // number of f32 lanes
#endif

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int64_t m = 0; m < M; ++m) {
      const float* row = act_ptr + m * K;

#if defined(__ARM_FEATURE_SVE)
      // Vectorized per-column counting:
      // counts[k:k+vl] += (abs(row[k:k+vl]) >= thr) ? 1 : 0
      int64_t k = 0;
      for (; k < K; k += vl) {
        const svbool_t pg = svwhilelt_b32(k, K);

        // load activation block
        const svfloat32_t x = svld1_f32(pg, row + k);

        // abs + compare
        const svfloat32_t ax = svabs_f32_x(pg, x);
        const svbool_t keep = svcmpge_f32(pg, ax, vthr);

        // load counts, add predicate-as-0/1, store back
        svuint32_t c = svld1_u32(pg, my_counts + k);
        const svuint32_t inc = svsel_u32(keep, vone, vzero);
        c = svadd_u32_x(pg, c, inc);
        svst1_u32(pg, my_counts + k, c);
      }
#else
      // Scalar fallback
      for (int64_t k = 0; k < K; ++k) {
        if (abs_f32_fast(row[k]) >= thr) {
          my_counts[k] += 1;
        }
      }
#endif
    }
  }

  // ---------------- Reduce thread-local counts to col_counts (int64) ----------------
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

  // ---------------- Pass 2: Build col_ptr (int64 [K+1], prefix sum) ----------------
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

  // ---------------- Build per-thread write base offsets ----------------
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
    TORCH_CHECK(run == col_counts[kk], "thr_sparsify_to_csc: count mismatch at col ", k);
#endif
  }

  // ---------------- Pass 3: Write row_indices and values (no atomics) ----------------
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
      const float* row = act_ptr + m * K;

      for (int64_t k = 0; k < K; ++k) {
        const float x = row[k];
        if (abs_f32_fast(x) >= thr) {
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
          "thr_sparsify_to_csc: thread write mismatch at tid=", tid, " col=", k2);
    }
#endif
  }

  return {col_ptr, row_indices, values};
}

// Register to PyTorch.
// Note: This file is compiled with other operator sources into the same extension.
// Use TORCH_LIBRARY_FRAGMENT to avoid conflicts with TORCH_LIBRARY in other translation units.
TORCH_LIBRARY_FRAGMENT(sparse_op, m) {
  m.def("thr_sparsify_to_csc(Tensor activation, float threshold) -> (Tensor col_ptr, Tensor row_indices, Tensor values)");
}

TORCH_LIBRARY_IMPL(sparse_op, CPU, m) {
  m.impl("thr_sparsify_to_csc", thr_sparsify_to_csc);
}
