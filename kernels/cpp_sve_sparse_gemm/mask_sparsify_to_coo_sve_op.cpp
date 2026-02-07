// mask_sparsify_to_coo_sve_op.cpp
// Mask-based COO sparsify operator with SVE/SVE2 optimization.
#include <torch/extension.h>

#include <cstdint>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

using torch::Tensor;

static inline void check_mask_sparsify_to_coo_sve_inputs(const Tensor& activation, const Tensor& mask) {
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
 * mask_sparsify_to_coo_sve(activation, mask) -> (row_indices, col_indices, values)
 *
 * Converts dense matrix to COO format based on mask (SVE/SVE2 accelerated).
 *
 * Args:
 *   activation: (M, K) float32 dense matrix
 *   mask: (M, K) uint8 mask matrix; non-zero indicates positions to keep
 *
 * Returns:
 *   row_indices: int64 [nnz] row index array (sorted by row)
 *   col_indices: uint32 [nnz] column index array
 *   values: float32 [nnz] non-zero element values
 *
 * Implementation (SVE optimized):
 *   Pass 1: SVE vectorized count of nnz per row (mask-based)
 *     - svld1_u8: vector load uint8 mask
 *     - svcmpne_n_u8: vector compare mask != 0
 *     - svcntp_b8: count lanes where condition holds
 *   Pass 2: Compute row offsets (prefix sum)
 *   Pass 3: Fill COO triplets using SVE2 compact
 *     - svld1ub_u32: load uint8 mask zero-extended to uint32
 *     - svindex_u32: generate index sequence
 *     - svcompact_u32/svcompact_f32: compact lanes where mask is non-zero
 *     - Optimization: full-keep chunks skip compact and write directly
 */
static std::tuple<Tensor, Tensor, Tensor> mask_sparsify_to_coo_sve(const Tensor& activation, const Tensor& mask) {
  check_mask_sparsify_to_coo_sve_inputs(activation, mask);

  const int64_t M = activation.size(0);
  const int64_t K = activation.size(1);
  const float* act_ptr = activation.data_ptr<float>();
  const uint8_t* mask_ptr = mask.data_ptr<uint8_t>();

  // counts per row (int64)
  Tensor counts_t = torch::empty({M}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* counts = counts_t.data_ptr<int64_t>();

  // ---------------- Pass1: count nnz per row (SVE vectorized) ----------------
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#if defined(__ARM_FEATURE_SVE)
    const size_t vl_u8 = svcntb();   // uint8_t vector length in bytes
#endif

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int64_t m = 0; m < M; ++m) {
      const uint8_t* mask_row = mask_ptr + m * K;
      int64_t nnz = 0;

#if defined(__ARM_FEATURE_SVE)
      // Vectorized nnz count using uint8; 4x unroll (uint8 VL is 4x float32 VL)
      int64_t k = 0;
      while (k + 4 * (int64_t)vl_u8 <= K) {
        // 1st vector chunk
        svbool_t pg1 = svwhilelt_b8((uint64_t)k, (uint64_t)K);
        svuint8_t v1 = svld1_u8(pg1, mask_row + k);
        svbool_t keep1 = svcmpne_n_u8(pg1, v1, 0);
        nnz += (int64_t)svcntp_b8(pg1, keep1);
        
        // 2nd vector chunk
        svbool_t pg2 = svwhilelt_b8((uint64_t)(k + vl_u8), (uint64_t)K);
        svuint8_t v2 = svld1_u8(pg2, mask_row + k + vl_u8);
        svbool_t keep2 = svcmpne_n_u8(pg2, v2, 0);
        nnz += (int64_t)svcntp_b8(pg2, keep2);
        
        // 3rd vector chunk
        svbool_t pg3 = svwhilelt_b8((uint64_t)(k + 2 * vl_u8), (uint64_t)K);
        svuint8_t v3 = svld1_u8(pg3, mask_row + k + 2 * vl_u8);
        svbool_t keep3 = svcmpne_n_u8(pg3, v3, 0);
        nnz += (int64_t)svcntp_b8(pg3, keep3);
        
        // 4th vector chunk
        svbool_t pg4 = svwhilelt_b8((uint64_t)(k + 3 * vl_u8), (uint64_t)K);
        svuint8_t v4 = svld1_u8(pg4, mask_row + k + 3 * vl_u8);
        svbool_t keep4 = svcmpne_n_u8(pg4, v4, 0);
        nnz += (int64_t)svcntp_b8(pg4, keep4);
        
        k += 4 * vl_u8;
      }
      // Handle remainder
      while (k < K) {
        svbool_t pg = svwhilelt_b8((uint64_t)k, (uint64_t)K);
        svuint8_t v = svld1_u8(pg, mask_row + k);
        svbool_t keep = svcmpne_n_u8(pg, v, 0);
        nnz += (int64_t)svcntp_b8(pg, keep);
        k += vl_u8;
      }
#else
      // Scalar fallback
      for (int64_t k = 0; k < K; ++k) {
        if (mask_row[k] != 0) nnz++;
      }
#endif
      counts[m] = nnz;
    }
  }

  // ---------------- row_offsets prefix sum (M+1) for internal use ----------------
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

  // ---------------- Pass2: compact write COO triplets (row, col, value) using SVE2 ----------------
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#if defined(__ARM_FEATURE_SVE)
    const int64_t vl = (int64_t)svcntw();  // uint32_t/float32 vector length in lanes
#endif

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int64_t m = 0; m < M; ++m) {
      const int64_t nnz = counts[m];
      if (nnz == 0) continue;

      const float* act_row = act_ptr + m * K;
      const uint8_t* mask_row = mask_ptr + m * K;
      int64_t* dst_row = out_row + row_offsets[m];
      uint32_t* dst_col = out_col + row_offsets[m];
      float* dst_val = out_val + row_offsets[m];
      int64_t write_pos = 0;

#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_SVE2)
      // Full-keep fast path: nnz == K (all elements in mask)
      if (nnz == K) {
        int64_t k = 0;
        while (k < K) {
          const svbool_t pg = svwhilelt_b32((uint32_t)k, (uint32_t)K);
          const svuint32_t vidx = svindex_u32((uint32_t)k, 1);
          const svfloat32_t vx = svld1_f32(pg, act_row + k);
          
          // Fill row indices (all elements belong to current row m)
          const int64_t chunk_len = (K - k < vl) ? (K - k) : vl;
          for (int64_t i = 0; i < chunk_len; ++i) {
            dst_row[k + i] = m;
          }
          
          svst1_u32(pg, dst_col + k, vidx);
          svst1_f32(pg, dst_val + k, vx);
          k += vl;
        }
        continue;
      }

      // General path: use SVE2 compact to compress data; 2-way unroll.
      int64_t k = 0;
      while (k + 2 * vl <= K) {
        // Vector block 1
        const svbool_t pg1 = svwhilelt_b32((uint32_t)k, (uint32_t)K);
        const svuint32_t vmask1 = svld1ub_u32(pg1, mask_row + k);
        const svbool_t keep1 = svcmpne_n_u32(pg1, vmask1, 0);
        const svfloat32_t vx1 = svld1_f32(pg1, act_row + k);
        const uint32_t n_keep1 = (uint32_t)svcntp_b32(pg1, keep1);
        if (n_keep1) {
          int64_t chunk_len1 = K - k;
          if (chunk_len1 > vl) chunk_len1 = vl;
          if ((int64_t)n_keep1 == chunk_len1) {
            const svuint32_t vidx1 = svindex_u32((uint32_t)k, 1);
            for (int64_t i = 0; i < chunk_len1; ++i) dst_row[write_pos + i] = m;
            svst1_u32(pg1, dst_col + write_pos, vidx1);
            svst1_f32(pg1, dst_val + write_pos, vx1);
            write_pos += chunk_len1;
          } else {
            const svuint32_t vidx1 = svindex_u32((uint32_t)k, 1);
            const svuint32_t packed_col1 = svcompact_u32(keep1, vidx1);
            const svfloat32_t packed_val1 = svcompact_f32(keep1, vx1);
            const svbool_t pg_out1 = svwhilelt_b32((uint32_t)0, n_keep1);
            for (uint32_t i = 0; i < n_keep1; ++i) dst_row[write_pos + i] = m;
            svst1_u32(pg_out1, dst_col + write_pos, packed_col1);
            svst1_f32(pg_out1, dst_val + write_pos, packed_val1);
            write_pos += (int64_t)n_keep1;
          }
        }

        // Vector block 2
        const svbool_t pg2 = svwhilelt_b32((uint32_t)(k + vl), (uint32_t)K);
        const svuint32_t vmask2 = svld1ub_u32(pg2, mask_row + k + vl);
        const svbool_t keep2 = svcmpne_n_u32(pg2, vmask2, 0);
        const svfloat32_t vx2 = svld1_f32(pg2, act_row + k + vl);
        const uint32_t n_keep2 = (uint32_t)svcntp_b32(pg2, keep2);
        if (n_keep2) {
          int64_t chunk_len2 = K - (k + vl);
          if (chunk_len2 > vl) chunk_len2 = vl;
          if ((int64_t)n_keep2 == chunk_len2) {
            const svuint32_t vidx2 = svindex_u32((uint32_t)(k + vl), 1);
            for (int64_t i = 0; i < chunk_len2; ++i) dst_row[write_pos + i] = m;
            svst1_u32(pg2, dst_col + write_pos, vidx2);
            svst1_f32(pg2, dst_val + write_pos, vx2);
            write_pos += chunk_len2;
          } else {
            const svuint32_t vidx2 = svindex_u32((uint32_t)(k + vl), 1);
            const svuint32_t packed_col2 = svcompact_u32(keep2, vidx2);
            const svfloat32_t packed_val2 = svcompact_f32(keep2, vx2);
            const svbool_t pg_out2 = svwhilelt_b32((uint32_t)0, n_keep2);
            for (uint32_t i = 0; i < n_keep2; ++i) dst_row[write_pos + i] = m;
            svst1_u32(pg_out2, dst_col + write_pos, packed_col2);
            svst1_f32(pg_out2, dst_val + write_pos, packed_val2);
            write_pos += (int64_t)n_keep2;
          }
        }
        k += 2 * vl;
      }
      // Remainder
      while (k < K) {
        const svbool_t pg = svwhilelt_b32((uint32_t)k, (uint32_t)K);
        const svuint32_t vmask = svld1ub_u32(pg, mask_row + k);
        const svbool_t keep = svcmpne_n_u32(pg, vmask, 0);
        const svfloat32_t vx = svld1_f32(pg, act_row + k);
        const uint32_t n_keep = (uint32_t)svcntp_b32(pg, keep);
        if (n_keep) {
          int64_t chunk_len = K - k;
          if (chunk_len > vl) chunk_len = vl;
          if ((int64_t)n_keep == chunk_len) {
            const svuint32_t vidx = svindex_u32((uint32_t)k, 1);
            for (int64_t i = 0; i < chunk_len; ++i) dst_row[write_pos + i] = m;
            svst1_u32(pg, dst_col + write_pos, vidx);
            svst1_f32(pg, dst_val + write_pos, vx);
            write_pos += chunk_len;
          } else {
            const svuint32_t vidx = svindex_u32((uint32_t)k, 1);
            const svuint32_t packed_col = svcompact_u32(keep, vidx);
            const svfloat32_t packed_val = svcompact_f32(keep, vx);
            const svbool_t pg_out = svwhilelt_b32((uint32_t)0, n_keep);
            for (uint32_t i = 0; i < n_keep; ++i) dst_row[write_pos + i] = m;
            svst1_u32(pg_out, dst_col + write_pos, packed_col);
            svst1_f32(pg_out, dst_val + write_pos, packed_val);
            write_pos += (int64_t)n_keep;
          }
        }
        k += vl;
      }

#ifndef NDEBUG
      TORCH_CHECK(write_pos == nnz,
                  "mask_sparsify_to_coo_sve: write_pos != nnz at row ", m,
                  " write_pos=", write_pos, " nnz=", nnz);
#endif

#else
      // Scalar fallback
      for (int64_t k = 0; k < K; ++k) {
        if (mask_row[k] != 0) {
          dst_row[write_pos] = m;
          dst_col[write_pos] = (uint32_t)k;
          dst_val[write_pos] = act_row[k];
          ++write_pos;
        }
      }
#endif
    }
  }

  return {row_indices, col_indices, values};
}

// Register to PyTorch.
// Note: This file is compiled with other operator sources into the same extension.
// Use TORCH_LIBRARY_FRAGMENT to avoid conflicts with TORCH_LIBRARY in other translation units.
TORCH_LIBRARY_FRAGMENT(sparse_op, m) {
  m.def("mask_sparsify_to_coo_sve(Tensor activation, Tensor mask) -> (Tensor row_indices, Tensor col_indices, Tensor values)");
}

TORCH_LIBRARY_IMPL(sparse_op, CPU, m) {
  m.impl("mask_sparsify_to_coo_sve", mask_sparsify_to_coo_sve);
}
