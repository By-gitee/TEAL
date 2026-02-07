#include <torch/extension.h>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

using torch::Tensor;

static inline void check_inputs(const Tensor& activation) {
  TORCH_CHECK(activation.device().is_cpu(), "activation must be CPU tensor");
  TORCH_CHECK(activation.dtype() == torch::kFloat32, "activation must be float32");
  TORCH_CHECK(activation.is_contiguous(), "activation must be contiguous");
  TORCH_CHECK(activation.dim() == 2, "activation must be 2D [M,K]");
}

// GCC-specific optimization attribute, not supported by clang
// #if defined(__GNUC__) && !defined(__clang__)
//   #define NO_VEC __attribute__((optimize("no-tree-vectorize")))
// #else
//   #define NO_VEC
// #endif

// Scalar pack function: extract elements where keep[i] == 1
static inline void pack_u32(
    const uint32_t* idx,
    const uint32_t* keep,
    int lanes,
    uint32_t* dst,
    int64_t& pos) {
  for (int i = 0; i < lanes; ++i) {
    if (keep[i]) dst[pos++] = idx[i];
  }
}

/**
 * thr_sparsify_to_icsr_sve_baseline(activation, threshold) -> (row_offsets, nz_col_indices, nz_values)
 * 
 * Converts dense matrix to ICSR (Indexed CSR) format - baseline version with 2x loop unrolling.
 * Uses scalar pack_u32 function instead of SVE2 compact instructions.
 * 
 * Args:
 *   activation: (M, K) float32 dense matrix
 *   threshold: float threshold, elements with abs(x) >= threshold are kept
 * 
 * Returns:
 *   row_offsets: int64 [M+1] row pointer array
 *   nz_col_indices: uint32 [nnz] column indices
 *   nz_values: float32 [nnz] values
 * 
 * Implementation Strategy:
 *   Pass 1: Count nnz per row with 2x loop unrolling
 *   Pass 2: Extract indices and values using pack_u32 with 2x loop unrolling
 */
std::tuple<Tensor, Tensor, Tensor>
thr_sparsify_to_icsr_sve_baseline(const Tensor& activation, double threshold) {
  check_inputs(activation);

  const int64_t M = activation.size(0);
  const int64_t K = activation.size(1);
  const float thr = (float)threshold;
  const float* act = activation.data_ptr<float>();

  Tensor counts_t = torch::empty({M}, torch::kInt64);
  int64_t* counts = counts_t.data_ptr<int64_t>();

  // ---------------- Pass 1: Count nnz per row ----------------

#ifdef _OPENMP
#pragma omp parallel
#endif
{
#if defined(__ARM_FEATURE_SVE)
  const int64_t vl = svcntw();
  const svfloat32_t vthr = svdup_f32(thr);
#endif

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
  for (int64_t m = 0; m < M; ++m) {
    const float* row = act + m * K;
    int64_t nnz = 0;

#if defined(__ARM_FEATURE_SVE)
    // SVE vectorized counting with 2x loop unrolling
    int64_t k = 0;

    // Main loop: process 2 vector blocks per iteration
    while (k + 2 * vl <= K) {
      // First vector block
      svbool_t pg1 = svptrue_b32();
      svfloat32_t x1 = svld1_f32(pg1, row + k);
      svfloat32_t a1 = svabs_f32_x(pg1, x1);
      svbool_t keep1 = svcmpge_f32(pg1, a1, vthr);
      nnz += svcntp_b32(pg1, keep1);

      // Second vector block
      svbool_t pg2 = svptrue_b32();
      svfloat32_t x2 = svld1_f32(pg2, row + k + vl);
      svfloat32_t a2 = svabs_f32_x(pg2, x2);
      svbool_t keep2 = svcmpge_f32(pg2, a2, vthr);
      nnz += svcntp_b32(pg2, keep2);

      k += 2 * vl;
    }

    // Tail loop: process remaining elements (< 2*vl)
    while (k < K) {
      svbool_t pg = svwhilelt_b32(k, K);
      svfloat32_t x = svld1_f32(pg, row + k);
      svfloat32_t a = svabs_f32_x(pg, x);
      svbool_t keep = svcmpge_f32(pg, a, vthr);
      nnz += svcntp_b32(pg, keep);
      k += vl;
    }
#else
    // Scalar fallback
    for (int64_t k = 0; k < K; ++k) {
      float v = row[k];
      if (v >= thr || v <= -thr) nnz++;
    }
#endif
    counts[m] = nnz;
  }
}

  // ---------------- Compute row offsets (prefix sum) ----------------
  std::vector<int64_t> row_offsets(M + 1);
  row_offsets[0] = 0;
  for (int64_t m = 0; m < M; ++m) {
    row_offsets[m + 1] = row_offsets[m] + counts[m];
  }
  const int64_t total = row_offsets[M];

  Tensor nz_col = torch::empty({total}, torch::kUInt32);
  uint32_t* out = (total > 0) ? nz_col.data_ptr<uint32_t>() : nullptr;

  // ---------------- Pass 2: Extract indices and pack ----------------

#ifdef _OPENMP
#pragma omp parallel
#endif
{
#if defined(__ARM_FEATURE_SVE)
  const int64_t vl = svcntw();
  const svfloat32_t vthr = svdup_f32(thr);

  alignas(64) uint32_t idx[256];
  alignas(64) uint32_t keep[256];
#endif

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
  for (int64_t m = 0; m < M; ++m) {
    int64_t nnz = counts[m];
    if (nnz == 0) continue;

    const float* row = act + m * K;
    uint32_t* dst = out + row_offsets[m];
    int64_t pos = 0;

#if defined(__ARM_FEATURE_SVE)
    // Fast path: all elements kept (nnz == K)
    if (nnz == K) {
      int64_t k = 0;
      while (k < K) {
        svbool_t pg = svwhilelt_b32(k, K);
        svuint32_t vidx = svindex_u32(k, 1);
        svst1_u32(pg, dst + k, vidx);
        k += vl;
      }
      continue;
    }

    int64_t k = 0;
    // Main loop: process 2 vector blocks per iteration for better ILP
    while (k + 2 * vl <= K) {
      // Process first vector block
      svbool_t pg1 = svwhilelt_b32((uint32_t)k, (uint32_t)K);
      svfloat32_t x1 = svld1_f32(pg1, row + k);
      svfloat32_t a1 = svabs_f32_x(pg1, x1);
      svbool_t km1 = svcmpge_f32(pg1, a1, vthr);

      svuint32_t vidx1 = svindex_u32((uint32_t)k, 1);
      svst1_u32(pg1, idx, vidx1);

      svuint32_t ones1 = svdup_u32_z(km1, 1);
      svst1_u32(pg1, keep, ones1);

      pack_u32(idx, keep, vl, dst, pos);

      // Process second vector block
      svbool_t pg2 = svwhilelt_b32((uint32_t)(k + vl), (uint32_t)K);
      svfloat32_t x2 = svld1_f32(pg2, row + k + vl);
      svfloat32_t a2 = svabs_f32_x(pg2, x2);
      svbool_t km2 = svcmpge_f32(pg2, a2, vthr);

      svuint32_t vidx2 = svindex_u32((uint32_t)(k + vl), 1);
      svst1_u32(pg2, idx, vidx2);

      svuint32_t ones2 = svdup_u32_z(km2, 1);
      svst1_u32(pg2, keep, ones2);

      pack_u32(idx, keep, vl, dst, pos);

      k += 2 * vl;
    }

    // Tail loop: process remaining elements (< 2*vl)
    while (k < K) {
      svbool_t pg = svwhilelt_b32((uint32_t)k, (uint32_t)K);
      svfloat32_t x = svld1_f32(pg, row + k);
      svfloat32_t a = svabs_f32_x(pg, x);
      svbool_t km = svcmpge_f32(pg, a, vthr);

      svuint32_t vidx = svindex_u32((uint32_t)k, 1);
      svst1_u32(pg, idx, vidx);

      svuint32_t ones = svdup_u32_z(km, 1);
      svst1_u32(pg, keep, ones);

      int lanes = std::min<int64_t>(vl, K - k);
      pack_u32(idx, keep, lanes, dst, pos);

      k += vl;
    }
#else
    if (nnz == K) {
      int64_t k = 0;
      while (k < K) {
        dst[pos++] = (uint32_t)k;
        k++;
      }
    }
    else {
      for (int64_t k = 0; k < K; ++k) {
        float v = row[k];
        if (v >= thr || v <= -thr) dst[pos++] = (uint32_t)k;
      }
    }
#endif
  }
}

// ---------------- nz_counts (placeholder 2*M) ----------------

// int64_t rows = 0;
// for (int64_t m = 0; m < M; ++m) if (counts[m] > 0) rows++;

Tensor nz_counts = torch::empty({2 * M}, torch::kInt64);
// Tensor nz_counts = torch::empty({2 * rows}, torch::kInt64);
// int64_t* nzp = nz_counts.data_ptr<int64_t>();

// int64_t p = 0;
// for (int64_t m = 0; m < M; ++m) {
//   if (counts[m] > 0) {
//     nzp[p++] = m;
//     nzp[p++] = counts[m];
//   }
// }

  // Return results as tensors
  Tensor row_offsets_t = torch::empty({M + 1}, torch::kInt64);
  std::memcpy(row_offsets_t.data_ptr<int64_t>(), row_offsets.data(), (size_t)(M + 1) * sizeof(int64_t));
  return {nz_counts, nz_col, row_offsets_t};
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("thr_sparsify_to_icsr_sve_baseline",
//         &thr_sparsify_to_icsr_sve_baseline);
// }

// Register to PyTorch.
// Note: This file is compiled with other operator sources into the same extension.
// Use TORCH_LIBRARY_FRAGMENT to avoid conflicts with TORCH_LIBRARY in other translation units.
TORCH_LIBRARY_FRAGMENT(sparse_op, m) {
  m.def("thr_sparsify_to_icsr_sve_baseline(Tensor activation, float threshold) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(sparse_op, CPU, m) {
  m.impl("thr_sparsify_to_icsr_sve_baseline", thr_sparsify_to_icsr_sve_baseline);
}
