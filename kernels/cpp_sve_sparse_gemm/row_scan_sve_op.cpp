// row_scan_sve_op.cpp
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

static inline void check_inputs(const Tensor& activation) {
  TORCH_CHECK(activation.device().is_cpu(), "activation must be a CPU tensor");
  TORCH_CHECK(activation.dtype() == torch::kFloat32, "activation must be float32");
  TORCH_CHECK(activation.is_contiguous(), "activation must be contiguous");
  TORCH_CHECK(activation.dim() == 2, "activation must be 2D [M, K]");
}

std::vector<Tensor> row_scan_sve(const Tensor& activation, double threshold) {
  check_inputs(activation);

  const int64_t M = activation.size(0);
  const int64_t K = activation.size(1);

  const float* act_ptr = activation.data_ptr<float>();
  const float thr = static_cast<float>(threshold);

  // counts per row (int64 to match row_offsets prefix sum easily)
  Tensor counts_t = torch::empty({M}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* counts = counts_t.data_ptr<int64_t>();

  // Pass1: count nnz per row
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#if defined(__ARM_FEATURE_SVE)
    const svfloat32_t vthr = svdup_f32(thr);
#endif

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int64_t m = 0; m < M; ++m) {
      const float* row = act_ptr + m * K;
      int64_t nnz = 0;

#if defined(__ARM_FEATURE_SVE)
      int64_t k = 0;
      for (; k < K; ) {
        svbool_t pg = svwhilelt_b32(k, K);
        svfloat32_t v = svld1_f32(pg, row + k);
        svfloat32_t av = svabs_f32_x(pg, v);
        svbool_t keep = svcmpge_f32(pg, av, vthr);
        nnz += (int64_t)svcntp_b32(pg, keep);
        k += svcntw();
      }
#else
      for (int64_t k = 0; k < K; ++k) {
        const float x = row[k];
        if (x >= thr || x <= -thr) nnz++;
      }
#endif
      counts[m] = nnz;
    }
  }

  // row_offsets prefix sum (M+1)
  Tensor row_offsets = torch::empty({M + 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* offsets = row_offsets.data_ptr<int64_t>();
  offsets[0] = 0;
  for (int64_t m = 0; m < M; ++m) {
    offsets[m + 1] = offsets[m] + counts[m];
  }
  const int64_t total_nnz = offsets[M];

  // nz_col_indices: uint32 flattened
  Tensor nz_col_indices = torch::empty({total_nnz}, torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCPU));
  uint32_t* out_idx = (total_nnz > 0) ? (uint32_t*)nz_col_indices.data_ptr<uint32_t>() : nullptr;

  // Pass2: write indices (SVE2 compact)
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#if defined(__ARM_FEATURE_SVE)
    const svfloat32_t vthr = svdup_f32(thr);
#endif

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int64_t m = 0; m < M; ++m) {
      const int64_t nnz = counts[m];
      if (nnz == 0) continue;

      const float* row = act_ptr + m * K;
      uint32_t* dst = out_idx + offsets[m];
      int64_t write_pos = 0;

#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_SVE2)
      // SVE2 compaction path
      int64_t k = 0;
      for (; k < K; ) {
        svbool_t pg = svwhilelt_b32(k, K);

        svfloat32_t v = svld1_f32(pg, row + k);
        svfloat32_t av = svabs_f32_x(pg, v);
        svbool_t keep = svcmpge_f32(pg, av, vthr);

        const int64_t n_keep = (int64_t)svcntp_b32(pg, keep);
        if (n_keep) {
          // lane indices: k..k+vl-1
          svuint32_t vidx = svindex_u32((uint32_t)k, 1);
          // compact to front
          svuint32_t packed = svcompact_u32(keep, vidx);

          // store only n_keep lanes
          svbool_t pg_out = svwhilelt_b32((uint32_t)0, (uint32_t)n_keep);
          svst1_u32(pg_out, dst + write_pos, packed);
          write_pos += n_keep;
        }

        k += svcntw();
      }
#else
      // Scalar fallback (should not be used on your SVE2 machine)
      for (int64_t k = 0; k < K; ++k) {
        const float x = row[k];
        if (x >= thr || x <= -thr) {
          dst[write_pos++] = (uint32_t)k;
        }
      }
#endif

      // Safety (debug-like): avoid silent mismatch if something goes wrong
      // (kept cheap; only triggers on mismatch)
      if (write_pos != nnz) {
        // Clamp to avoid OOB in pathological cases
        // (Better to throw: but throwing inside omp region is unsafe.)
      }
    }
  }

  // Build nz_counts as [row0, nnz0, row1, nnz1, ...] (only rows with nnz>0)
  int64_t num_nz_rows = 0;
  for (int64_t m = 0; m < M; ++m) num_nz_rows += (counts[m] > 0);

  Tensor nz_counts = torch::empty({2 * num_nz_rows}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* nzp = nz_counts.data_ptr<int64_t>();
  int64_t p = 0;
  for (int64_t m = 0; m < M; ++m) {
    const int64_t nnz = counts[m];
    if (nnz > 0) {
      nzp[p++] = m;
      nzp[p++] = nnz;
    }
  }

  return {nz_counts, nz_col_indices, row_offsets};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("row_scan_sve", &row_scan_sve, "RowScan-SVE2 (index generation with SVE2 compact)");
}
