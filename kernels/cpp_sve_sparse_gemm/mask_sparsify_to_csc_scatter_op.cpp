#include <torch/extension.h>

#include <cstdint>
#include <vector>
#include <cstring>
#include <algorithm>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
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
 * mask_sparsify_to_csc_scatter(activation, mask) -> (col_ptr, row_indices, values)
 *
 * Converts dense matrix to CSC format based on binary mask (SVE optimized).
 * 
 * Args:
 *   activation: (M, K) float32 dense matrix
 *   mask: (M, K) uint8 binary mask, non-zero elements are kept
 * 
 * Returns:
 *   col_ptr: int64 [K+1] CSC column pointer array
 *   row_indices: uint32 [nnz] row index array
 *   values: float32 [nnz] non-zero element value array
 * 
 * Implementation Strategy:
 *   Pass 1: Parallel count nnz per column (thread-private counts with SVE vectorization)
 *     - svld1ub_u32: load uint8 mask and zero-extend to uint32
 *     - svcmpne_u32: compare mask != 0
 *   Pass 2: Reduce counts + build col_ptr via prefix sum
 *   Pass 3: Fill row_indices/values without atomics
 *     - If SVE supported and total_nnz fits u32: use SVE scatter store for column buckets
 */
static std::tuple<Tensor, Tensor, Tensor>
mask_sparsify_to_csc_scatter(torch::Tensor activation, torch::Tensor mask) {
  check_mask_sparsify_to_csc_inputs(activation, mask);

  const int64_t M = activation.size(0);
  const int64_t K = activation.size(1);
  const float* act_ptr = activation.data_ptr<float>();
  const uint8_t* mask_ptr = mask.data_ptr<uint8_t>();

  int num_threads = 1;
#ifdef _OPENMP
  num_threads = omp_get_max_threads();
#endif

  // ================= Pass 1: Count nnz per column (thread-private) =================
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
    const int64_t vl = (int64_t)svcntw();  // number of u32 lanes
#endif

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int64_t m = 0; m < M; ++m) {
      const uint8_t* mask_row = mask_ptr + m * K;

#if defined(__ARM_FEATURE_SVE)
      // SVE vectorized per-column counting: counts[k:k+vl] += (mask[k:k+vl] != 0) ? 1 : 0
      int64_t k = 0;
      for (; k < K; k += vl) {
        const svbool_t pg32 = svwhilelt_b32(k, K);

        svuint32_t m32 = svld1ub_u32(pg32, mask_row + k);
        const svbool_t keep = svcmpne_n_u32(pg32, m32, 0);
        const svuint32_t inc = svsel_u32(keep, vone, vzero);

        svuint32_t c = svld1_u32(pg32, my_counts + k);
        c = svadd_u32_x(pg32, c, inc);
        svst1_u32(pg32, my_counts + k, c);
      }
#else
      // Scalar fallback
      for (int64_t k = 0; k < K; ++k) {
        my_counts[k] += (mask_row[k] != 0);
      }
#endif
    }
  }

  // ================= Reduce thread-local counts to global col_counts =================
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

  // ================= Pass 2: Build col_ptr via prefix sum =================
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

  // ================= Build per-thread base offsets (no atomics) =================
  // thread_base64[tid][k] = ptr[k] + sum_{t'<tid} local_counts[t'][k]
  // For the SVE scatter-store path we additionally use a u32 view (byte offsets = pos*4)
  const bool offsets_fit_u32 = (total_nnz <= (int64_t)std::numeric_limits<uint32_t>::max());
  std::vector<int64_t> thread_base64((size_t)num_threads * (size_t)K, 0);
  std::vector<uint32_t> thread_base_u32;
  if (offsets_fit_u32) {
    thread_base_u32.resize((size_t)num_threads * (size_t)K);
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t k = 0; k < K; ++k) {
    int64_t base64 = ptr[k];
    int64_t run64 = 0;
    const size_t kk = (size_t)k;
    for (int t = 0; t < num_threads; ++t) {
      const int64_t tb64 = base64 + run64;
      thread_base64[(size_t)t * (size_t)K + kk] = tb64;
      if (offsets_fit_u32) {
        thread_base_u32[(size_t)t * (size_t)K + kk] = (uint32_t)tb64;
      }
      run64 += (int64_t)local_counts[(size_t)t * (size_t)K + kk];
    }
#ifndef NDEBUG
    TORCH_CHECK(run64 == col_counts[kk], "mask_sparsify_to_csc: count mismatch at col ", k);
#endif
  }

  // ================= Pass 3: Fill row_indices and values (no atomics) =================
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif

    std::vector<uint32_t> pos_u32;
    std::vector<int64_t> pos64;
    if (offsets_fit_u32) {
      pos_u32.resize((size_t)K);
      std::memcpy(pos_u32.data(),
                  thread_base_u32.data() + (size_t)tid * (size_t)K,
                  (size_t)K * sizeof(uint32_t));
    } else {
      pos64.resize((size_t)K);
      std::memcpy(pos64.data(),
                  thread_base64.data() + (size_t)tid * (size_t)K,
                  (size_t)K * sizeof(int64_t));
    }

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int64_t m = 0; m < M; ++m) {
      const float* act_row = act_ptr + m * K;
      const uint8_t* mask_row = mask_ptr + m * K;

      if (offsets_fit_u32) {
#if defined(__ARM_FEATURE_SVE)
        // --- SVE scatter-store path ---
        //  1) mask -> predicate
        //  2) load values
        //  3) load pos[k]
        //  4) scatter row_id & value into column buckets
        //  5) pos[k]++ for active lanes, store back
        const uint32_t row_id = (uint32_t)m;
        const svuint32_t v_row = svdup_u32(row_id);
        const svuint32_t v_one = svdup_u32(1u);

        const int64_t vlw = svcntw();
        for (int64_t k = 0; k < K; k += vlw) {
          const svbool_t pg32 = svwhilelt_b32(k, K);

          // mask: load bytes and zero-extend to u32 (1 lane == 1 column)
          const svuint32_t m32 = svld1ub_u32(pg32, mask_row + k);
          const svbool_t p_keep = svcmpne_n_u32(pg32, m32, 0u);

          if (svptest_any(pg32, p_keep)) {
            const svfloat32_t v = svld1_f32(pg32, act_row + k);

            // current write cursor per column
            const svuint32_t p = svld1_u32(pg32, pos_u32.data() + (size_t)k);
            // byte offset = p * 4
            const svuint32_t off = svlsl_n_u32_x(pg32, p, 2);

            // scatter store into CSC buckets
            svst1_scatter_u32offset_u32(p_keep, out_row, off, v_row);
            svst1_scatter_u32offset_f32(p_keep, out_val, off, v);

            // advance cursor only for active lanes
            const svuint32_t p_new = svadd_u32_m(p_keep, p, v_one);
            svst1_u32(pg32, pos_u32.data() + (size_t)k, p_new);
          }
        }
#else
        // --- Scalar (no SVE) ---
        for (int64_t k = 0; k < K; ++k) {
          if (mask_row[k] != 0) {
            const uint32_t p = pos_u32[(size_t)k]++;
            out_row[p] = (uint32_t)m;
            out_val[p] = act_row[k];
          }
        }
#endif
      } else {
        // --- Scalar int64 cursor fallback ---
        for (int64_t k = 0; k < K; ++k) {
          if (mask_row[k] != 0) {
            const int64_t p = pos64[(size_t)k]++;
            out_row[p] = (uint32_t)m;
            out_val[p] = act_row[k];
          }
        }
      }
    }

#ifndef NDEBUG
    const uint32_t* my_cnt = local_counts.data() + (size_t)tid * (size_t)K;
    if (offsets_fit_u32) {
      const uint32_t* my_base = thread_base_u32.data() + (size_t)tid * (size_t)K;
      for (int64_t k2 = 0; k2 < K; ++k2) {
        const size_t kk2 = (size_t)k2;
        TORCH_CHECK(pos_u32[kk2] == (uint32_t)(my_base[kk2] + my_cnt[kk2]),
                    "mask_sparsify_to_csc: thread write mismatch at tid=", tid, " col=", k2);
      }
    } else {
      const int64_t* my_base64 = thread_base64.data() + (size_t)tid * (size_t)K;
      for (int64_t k2 = 0; k2 < K; ++k2) {
        const size_t kk2 = (size_t)k2;
        TORCH_CHECK(pos64[kk2] == my_base64[kk2] + (int64_t)my_cnt[kk2],
                    "mask_sparsify_to_csc: thread write mismatch at tid=", tid, " col=", k2);
      }
    }
#endif
  }

  return {col_ptr, row_indices, values};
}

// Register to PyTorch
// Note: This file is compiled with other operator sources into the same extension.
// Use TORCH_LIBRARY_FRAGMENT to avoid conflicts with TORCH_LIBRARY in other translation units.
TORCH_LIBRARY_FRAGMENT(sparse_op, m) {
  m.def("mask_sparsify_to_csc_scatter(Tensor activation, Tensor mask) -> (Tensor col_ptr, Tensor row_indices, Tensor values)");
}

TORCH_LIBRARY_IMPL(sparse_op, CPU, m) {
  m.impl("mask_sparsify_to_csc_scatter", mask_sparsify_to_csc_scatter);
}
