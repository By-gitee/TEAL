// thr_sparsify_to_csr_sve_op.cpp
#include <torch/extension.h>

#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

using torch::Tensor;

static inline void check_thr_sparsify_to_csr_sve_inputs(const Tensor& activation) {
  TORCH_CHECK(activation.device().is_cpu(), "activation must be a CPU tensor");
  TORCH_CHECK(activation.dtype() == torch::kFloat32, "activation must be float32");
  TORCH_CHECK(activation.is_contiguous(), "activation must be contiguous");
  TORCH_CHECK(activation.dim() == 2, "activation must be 2D [M, K]");
}

/**
 * thr_sparsify_to_csr_sve(activation, threshold) -> (row_offsets, col_indices, values)
 * 
 * Converts a dense matrix to CSR sparse format based on threshold (SVE/SVE2 accelerated).
 * 
 * Args:
 *   activation: (M, K) float32 dense matrix
 *   threshold: float threshold, elements with abs(x) >= threshold are kept
 * 
 * Returns:
 *   row_offsets: int64 [M+1] CSR row pointer array (prefix sum)
 *   col_indices: uint32 [nnz] column index array
 *   values: float32 [nnz] non-zero element value array
 * 
 * Implementation Strategy (SVE optimized with 2x loop unrolling):
 *   Pass 1: Count non-zero elements per row using SVE vectorization
 *     - svabs_f32_x + svcmpge_f32: vectorized threshold comparison
 *     - svcntp_b32: count elements meeting condition
 *   Pass 2: Compute row offsets (prefix sum) to get CSR row_ptr
 *   Pass 3: Extract and compact CSR data using SVE2 with 2-way unrolling
 *     - Main loop: process 2*vl elements per iteration for better ILP
 *     - svcompact_u32/svcompact_f32: compress sparse data
 *     - Optimization: skip compact for full-keep chunks
 */
static std::tuple<Tensor, Tensor, Tensor> thr_sparsify_to_csr_sve(const Tensor& activation, double threshold) {
  check_thr_sparsify_to_csr_sve_inputs(activation);

  const int64_t M = activation.size(0);
  const int64_t K = activation.size(1);
  const float* act_ptr = activation.data_ptr<float>();
  const float thr = static_cast<float>(threshold);

  // Allocate counts array to store non-zero count per row
  Tensor counts_t = torch::empty({M}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* counts = counts_t.data_ptr<int64_t>();

  // ---------------- Pass 1: Count nnz per row (SVE vectorized) ----------------
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#if defined(__ARM_FEATURE_SVE)
    // Broadcast threshold to vector register
    const svfloat32_t vthr = svdup_f32(thr);
    const int64_t vl = (int64_t)svcntw();  // SVE vector length in elements
#endif

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int64_t m = 0; m < M; ++m) {
      const float* row = act_ptr + m * K;
      int64_t nnz = 0;

#if defined(__ARM_FEATURE_SVE)
      int64_t k = 0;
      // Main loop: process 2 vector blocks per iteration
      while (k + 2 * vl <= K) {
        // First vector block
        svbool_t pg1 = svwhilelt_b32((uint32_t)k, (uint32_t)K);
        svfloat32_t vx1 = svld1_f32(pg1, row + k);
        svfloat32_t vabs1 = svabs_f32_x(pg1, vx1);
        svbool_t keep1 = svcmpge_f32(pg1, vabs1, vthr);
        nnz += (int64_t)svcntp_b32(pg1, keep1);
        
        // Second vector block
        svbool_t pg2 = svwhilelt_b32((uint32_t)(k + vl), (uint32_t)K);
        svfloat32_t vx2 = svld1_f32(pg2, row + k + vl);
        svfloat32_t vabs2 = svabs_f32_x(pg2, vx2);
        svbool_t keep2 = svcmpge_f32(pg2, vabs2, vthr);
        nnz += (int64_t)svcntp_b32(pg2, keep2);
        
        k += 2 * vl;
      }
      
      // Tail loop: process remaining elements (< 2*vl)
      while (k < K) {
        svbool_t pg = svwhilelt_b32((uint32_t)k, (uint32_t)K);
        svfloat32_t vx = svld1_f32(pg, row + k);
        svfloat32_t vabs = svabs_f32_x(pg, vx);
        svbool_t keep = svcmpge_f32(pg, vabs, vthr);
        nnz += (int64_t)svcntp_b32(pg, keep);
        k += svcntw();
      }
#else
      // Scalar fallback
      for (int64_t k = 0; k < K; ++k) {
        const float x = row[k];
        if (x >= thr || x <= -thr) nnz++;
      }
#endif
      counts[m] = nnz;
    }
  }

  // ---------------- Pass 2: Compute row offsets via prefix sum (CSR row_ptr) ----------------
  std::vector<int64_t> row_offsets(M + 1);
  row_offsets[0] = 0;
  for (int64_t m = 0; m < M; ++m) {
    row_offsets[m + 1] = row_offsets[m] + counts[m];
  }
  const int64_t total_nnz = row_offsets[M];

  // ---------------- Allocate output CSR arrays ----------------
  Tensor col_idx = torch::empty({total_nnz}, torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCPU));
  uint32_t* out_idx = (total_nnz > 0) ? col_idx.data_ptr<uint32_t>() : nullptr;

  Tensor values = torch::empty({total_nnz}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  float* out_val = (total_nnz > 0) ? values.data_ptr<float>() : nullptr;

  // ---------------- Pass 3: Extract and write CSR data (SVE2 with 2x unrolling) ----------------
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#if defined(__ARM_FEATURE_SVE)
    const svfloat32_t vthr = svdup_f32(thr);
    const int64_t vl = (int64_t)svcntw();
#endif

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int64_t m = 0; m < M; ++m) {
      const int64_t nnz = counts[m];
      if (nnz == 0) continue;

      const float* row = act_ptr + m * K;
      uint32_t* dst_idx = out_idx + row_offsets[m];
      float* dst_val = out_val + row_offsets[m];
      int64_t write_pos = 0;

#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_SVE2)
      // Fast path: all elements kept (nnz == K)
      if (nnz == K) {
        int64_t k = 0;
        while (k < K) {
          const svbool_t pg = svwhilelt_b32(k, K);  // Generate predicate
          const svuint32_t vidx = svindex_u32((uint32_t)k, 1);
          const svfloat32_t vx = svld1_f32(pg, row + k);
          svst1_u32(pg, dst_idx + k, vidx);  // Store column indices
          svst1_f32(pg, dst_val + k, vx);  // Store values
          k += vl;
        }
        continue;
      }

      int64_t k = 0;
      // 2x loop unrolling for better instruction-level parallelism
      while (k + 2 * vl <= K) {
        // First vector block
        svbool_t pg1 = svwhilelt_b32((uint32_t)k, (uint32_t)K);
        svfloat32_t vx1 = svld1_f32(pg1, row + k);
        svfloat32_t vabs1 = svabs_f32_x(pg1, vx1);
        svbool_t keep1 = svcmpge_f32(pg1, vabs1, vthr);  // Predicate: |x| >= threshold
        int64_t n_keep1 = (int64_t)svcntp_b32(pg1, keep1);
        if (n_keep1 > 0) {
          // Optimization: if all elements are kept in this chunk, skip compact
          if ((int64_t)n_keep1 == vl) {
            const svuint32_t vidx1 = svindex_u32((uint32_t)k, 1);
            svst1_u32(pg1, dst_idx + write_pos, vidx1);  // Direct write
            svst1_f32(pg1, dst_val + write_pos, vx1);  // Direct write
            write_pos += vl;
          } else {
            svuint32_t vidx1 = svindex_u32((uint32_t)k, 1);
            svuint32_t packed_idx1 = svcompact_u32(keep1, vidx1);
            svfloat32_t packed_val1 = svcompact_f32(keep1, vx1);
            svbool_t pg_out1 = svwhilelt_b32((uint32_t)0, (uint32_t)n_keep1);
            svst1_u32(pg_out1, dst_idx + write_pos, packed_idx1);
            svst1_f32(pg_out1, dst_val + write_pos, packed_val1);
            write_pos += n_keep1;
          }
        }

        // Process second vector block
        svbool_t pg2 = svwhilelt_b32((uint32_t)(k + vl), (uint32_t)K);
        svfloat32_t vx2 = svld1_f32(pg2, row + k + vl);
        svfloat32_t vabs2 = svabs_f32_x(pg2, vx2);
        svbool_t keep2 = svcmpge_f32(pg2, vabs2, vthr);
        int64_t n_keep2 = (int64_t)svcntp_b32(pg2, keep2);
        if (n_keep2 > 0) {
          // Check if all elements are kept in this chunk
          if (n_keep2 == vl) {
            const svuint32_t vidx2 = svindex_u32((uint32_t)(k + vl), 1);
            svst1_u32(pg2, dst_idx + write_pos, vidx2);
            svst1_f32(pg2, dst_val + write_pos, vx2);
            write_pos += vl;
          } else {
            svuint32_t vidx2 = svindex_u32((uint32_t)(k + vl), 1);
            svuint32_t packed_idx2 = svcompact_u32(keep2, vidx2);
            svfloat32_t packed_val2 = svcompact_f32(keep2, vx2);
            svbool_t pg_out2 = svwhilelt_b32((uint32_t)0, (uint32_t)n_keep2);
            svst1_u32(pg_out2, dst_idx + write_pos, packed_idx2);
            svst1_f32(pg_out2, dst_val + write_pos, packed_val2);
            write_pos += n_keep2;
          }
        }

        k += 2 * vl;
      }

      // Process remaining elements (< 2*vl)
      while (k < K) {
        const svbool_t pg = svwhilelt_b32((uint32_t)k, (uint32_t)K);
        const svfloat32_t vx = svld1_f32(pg, row + k);
        const svfloat32_t vabs = svabs_f32_x(pg, vx);
        const svbool_t keep = svcmpge_f32(pg, vabs, vthr);

        const uint32_t n_keep = (uint32_t)svcntp_b32(pg, keep);
        if (n_keep > 0) {
          int64_t chunk_len = K - k;
          if (chunk_len > vl) chunk_len = vl;

          // If all kept in this chunk, skip compact
          if ((int64_t)n_keep == chunk_len) {
            const svuint32_t vidx = svindex_u32((uint32_t)k, 1);
            svst1_u32(pg, dst_idx + write_pos, vidx);
            svst1_f32(pg, dst_val + write_pos, vx);
            write_pos += chunk_len;
          } else {
            const svuint32_t vidx = svindex_u32((uint32_t)k, 1);
            const svuint32_t packed_idx = svcompact_u32(keep, vidx);
            const svfloat32_t packed_val = svcompact_f32(keep, vx);

            const svbool_t pg_out = svwhilelt_b32((uint32_t)0, n_keep);
            svst1_u32(pg_out, dst_idx + write_pos, packed_idx);
            svst1_f32(pg_out, dst_val + write_pos, packed_val);
            write_pos += (int64_t)n_keep;
          }
        }

        k += vl;
      }

#ifndef NDEBUG
      TORCH_CHECK(write_pos == nnz,
                  "thr_sparsify_to_csr_sve: write_pos != nnz at row ", m,
                  " write_pos=", write_pos, " nnz=", nnz);
#endif

#else
      // Scalar fallback
      if (nnz == K) {
        for (uint32_t k = 0; k < (uint32_t)K; ++k) {
          dst_idx[write_pos] = k;
          dst_val[write_pos] = row[k];
          ++write_pos;
        }
      } 
      else {
      for (int64_t k = 0; k < K; ++k) {
        const float x = row[k];
        if (x >= thr || x <= -thr) {
            dst_idx[write_pos] = (uint32_t)k;
            dst_val[write_pos] = x;
            ++write_pos;
          }
        }
      }
#endif
    }
  }

  // 将 row_offsets 转为 Tensor 再返回
  Tensor row_offsets_t = torch::empty({M + 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  std::memcpy(row_offsets_t.data_ptr<int64_t>(), row_offsets.data(), (size_t)(M + 1) * sizeof(int64_t));
  return {row_offsets_t, col_idx, values};
}

// Register to PyTorch
// Note: This file is compiled with other operator sources into the same extension.
// Use TORCH_LIBRARY_FRAGMENT to avoid conflicts with TORCH_LIBRARY in other translation units.
TORCH_LIBRARY_FRAGMENT(sparse_op, m) {
  m.def("thr_sparsify_to_csr_sve(Tensor activation, float threshold) -> (Tensor row_offsets, Tensor nz_col_indices, Tensor values)");
}

TORCH_LIBRARY_IMPL(sparse_op, CPU, m) {
  m.impl("thr_sparsify_to_csr_sve", thr_sparsify_to_csr_sve);
}
