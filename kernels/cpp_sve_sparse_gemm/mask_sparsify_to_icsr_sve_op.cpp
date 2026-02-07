// mask_sparsify_to_icsr_sve_op.cpp
// Mask-based ICSR sparsify (indices only) with SVE/SVE2 optimization.

#include <torch/extension.h>
#include <cstdint>
#include <vector>
#include <numeric>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

using torch::Tensor;

static inline void check_inputs(const Tensor& mask) {
  TORCH_CHECK(mask.device().is_cpu(), "mask must be a CPU tensor");
  TORCH_CHECK(mask.dtype() == torch::kUInt8, "mask must be uint8");
  TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
  TORCH_CHECK(mask.dim() == 2, "mask must be 2D [M, K]");
}

std::tuple<Tensor, Tensor, Tensor> mask_sparsify_to_icsr_sve(const Tensor& mask) {
  check_inputs(mask);

  const int64_t M = mask.size(0);
  const int64_t K = mask.size(1);
  const uint8_t* mask_ptr = mask.data_ptr<uint8_t>();

  // Per-row nnz counts (int64) for prefix sum.
  Tensor counts_t = torch::empty({M}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* counts = counts_t.data_ptr<int64_t>();

  // ---------------- Pass1: count nnz per row ----------------
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#if defined(__ARM_FEATURE_SVE)
    const size_t vl = svcntb();  // uint8_t vector length (dynamic VL)
#endif

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int64_t m = 0; m < M; ++m) {
      const uint8_t* row = mask_ptr + m * K;
      int64_t nnz = 0;
#if defined(__ARM_FEATURE_SVE)
      int64_t k = 0;
      // 4x unroll (uint8_t VL is 4x float32 VL).
      while (k + 4 * (int64_t)vl <= K) {
        // Vector block 1
        svbool_t pg1 = svwhilelt_b8((uint64_t)k, (uint64_t)K);
        svuint8_t v1 = svld1_u8(pg1, row + k);
        svbool_t keep1 = svcmpne_n_u8(pg1, v1, 0);  // mask != 0
        nnz += (int64_t)svcntp_b8(pg1, keep1);
        
        // Vector block 2
        svbool_t pg2 = svwhilelt_b8((uint64_t)(k + vl), (uint64_t)K);
        svuint8_t v2 = svld1_u8(pg2, row + k + vl);
        svbool_t keep2 = svcmpne_n_u8(pg2, v2, 0);
        nnz += (int64_t)svcntp_b8(pg2, keep2);
        
        // Vector block 3
        svbool_t pg3 = svwhilelt_b8((uint64_t)(k + 2 * vl), (uint64_t)K);
        svuint8_t v3 = svld1_u8(pg3, row + k + 2 * vl);
        svbool_t keep3 = svcmpne_n_u8(pg3, v3, 0);
        nnz += (int64_t)svcntp_b8(pg3, keep3);
        
        // Vector block 4
        svbool_t pg4 = svwhilelt_b8((uint64_t)(k + 3 * vl), (uint64_t)K);
        svuint8_t v4 = svld1_u8(pg4, row + k + 3 * vl);
        svbool_t keep4 = svcmpne_n_u8(pg4, v4, 0);
        nnz += (int64_t)svcntp_b8(pg4, keep4);
        
        k += 4 * vl;
      }
      // Remainder (< 4*vl)
      while (k < K) {
        svbool_t pg = svwhilelt_b8((uint64_t)k, (uint64_t)K);
        svuint8_t v = svld1_u8(pg, row + k);
        svbool_t keep = svcmpne_n_u8(pg, v, 0);
        nnz += (int64_t)svcntp_b8(pg, keep);
        k += svcntb();
      }
#else
      // Scalar fallback.
      for (int64_t k = 0; k < K; ++k) {
        if (row[k] != 0) {
          nnz++;
        }
      }
#endif
      counts[m] = nnz;
    }
  }

  // ---------------- Row prefix sum (row_offsets, length M+1) ----------------
  std::vector<int64_t> row_offsets(M + 1);
  // Tensor row_offsets = torch::empty({M + 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  // int64_t* offsets = row_offsets.data_ptr<int64_t>();
  row_offsets[0] = 0;
  for (int64_t m = 0; m < M; ++m) {
    row_offsets[m + 1] = row_offsets[m] + counts[m];
  }
  const int64_t total_nnz = row_offsets[M];

  // ---------------- Allocate nz_col_indices (uint32, flattened) ----------------
  Tensor nz_col_indices = torch::empty({total_nnz}, torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCPU));
  uint32_t* out_idx = (total_nnz > 0 ? nz_col_indices.data_ptr<uint32_t>() : nullptr);

  // ---------------- Pass2: compact write col indices per row ----------------
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_SVE2)
    const size_t vl_u8 = svcntb();   // uint8_t vector length
    const size_t vl_u32 = svcntw();  // uint32_t vector length
#endif

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int64_t m = 0; m < M; ++m) {
      const int64_t nnz = counts[m];
      if (nnz == 0) continue;

      const uint8_t* row = mask_ptr + m * K;
      uint32_t* dst = out_idx + row_offsets[m];
      int64_t write_pos = 0;
#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_SVE2)
      if (nnz == K) {
        // Full row: output [0..K-1] directly.
        int64_t kk = 0;
        while (kk < K) {
          svbool_t pg = svwhilelt_b32((uint32_t)kk, (uint32_t)K);
          svuint32_t vidx = svindex_u32((uint32_t)kk, 1);
          svst1_u32(pg, dst + kk, vidx);
          kk += svcntw();
        }
        write_pos = nnz;
      } else {
        // SVE2 path: compress with svcompact; process in uint32 lanes to match output.
        int64_t k = 0;
        while (k + 2 * (int64_t)vl_u32 <= K) {
          // Vector block 1 (uint32)
          svbool_t pg1 = svwhilelt_b32((uint32_t)k, (uint32_t)K);
          svuint32_t vmask1_u32 = svld1ub_u32(pg1, row + k);
          svbool_t keep1 = svcmpne_n_u32(pg1, vmask1_u32, 0);
          int64_t n1 = (int64_t)svcntp_b32(pg1, keep1);
          if (n1 > 0) {
            svuint32_t vidx1 = svindex_u32((uint32_t)k, 1);
            svuint32_t packed1 = svcompact_u32(keep1, vidx1);
            svbool_t pg_out1 = svwhilelt_b32((uint32_t)0, (uint32_t)n1);
            svst1_u32(pg_out1, dst + write_pos, packed1);
            write_pos += n1;
          }
          
          // Vector block 2 (uint32)
          svbool_t pg2 = svwhilelt_b32((uint32_t)(k + vl_u32), (uint32_t)K);
          svuint32_t vmask2_u32 = svld1ub_u32(pg2, row + k + vl_u32);
          svbool_t keep2 = svcmpne_n_u32(pg2, vmask2_u32, 0);
          int64_t n2 = (int64_t)svcntp_b32(pg2, keep2);
          if (n2 > 0) {
            svuint32_t vidx2 = svindex_u32((uint32_t)(k + vl_u32), 1);
            svuint32_t packed2 = svcompact_u32(keep2, vidx2);
            svbool_t pg_out2 = svwhilelt_b32((uint32_t)0, (uint32_t)n2);
            svst1_u32(pg_out2, dst + write_pos, packed2);
            write_pos += n2;
          }
          k += 2 * vl_u32;
        }
        // Remainder
        while (k < K) {
          svbool_t pg = svwhilelt_b32((uint32_t)k, (uint32_t)K);
          svuint32_t vmask_u32 = svld1ub_u32(pg, row + k);
          svbool_t keep = svcmpne_n_u32(pg, vmask_u32, 0);
          int64_t n_keep = (int64_t)svcntp_b32(pg, keep);
          if (n_keep > 0) {
            svuint32_t vidx = svindex_u32((uint32_t)k, 1);
            svuint32_t packed = svcompact_u32(keep, vidx);
            svbool_t pg_out = svwhilelt_b32((uint32_t)0, (uint32_t)n_keep);
            svst1_u32(pg_out, dst + write_pos, packed);
            write_pos += n_keep;
          }
          k += svcntw();
        }
      }
#else
      // Scalar fallback.
      if (nnz == K) {
        for (uint32_t k = 0; k < (uint32_t)K; ++k) {
          dst[write_pos++] = k;
        }
      } else {
        for (int64_t k = 0; k < K; ++k) {
          if (row[k] != 0) {
            dst[write_pos++] = (uint32_t)k;
          }
        }
      }
#endif
    }
  }

  // nz_counts placeholder (shape 2*M); actual sparse [row, nnz] layout can be built from row_offsets.
  Tensor nz_counts = torch::empty({2 * M}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

  Tensor row_offsets_t = torch::empty({M + 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  std::memcpy(row_offsets_t.data_ptr<int64_t>(), row_offsets.data(), (size_t)(M + 1) * sizeof(int64_t));
  return std::make_tuple(nz_counts, nz_col_indices, row_offsets_t);
}

// Register to PyTorch.
// Note: This file is compiled with other operator sources into the same extension.
// Use TORCH_LIBRARY_FRAGMENT to avoid conflicts with TORCH_LIBRARY in other translation units.
TORCH_LIBRARY_FRAGMENT(sparse_op, m) {
  m.def("mask_sparsify_to_icsr_sve(Tensor mask) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(sparse_op, CPU, m) {
  m.impl("mask_sparsify_to_icsr_sve", mask_sparsify_to_icsr_sve);
}
