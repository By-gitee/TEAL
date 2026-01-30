// thr_sparsify_to_csr_sve_op.cpp
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

static inline void check_thr_sparsify_to_csr_sve_inputs(const Tensor& activation) {
  TORCH_CHECK(activation.device().is_cpu(), "activation must be a CPU tensor");
  TORCH_CHECK(activation.dtype() == torch::kFloat32, "activation must be float32");
  TORCH_CHECK(activation.is_contiguous(), "activation must be contiguous");
  TORCH_CHECK(activation.dim() == 2, "activation must be 2D [M, K]");
}

// Return: [row_offsets(int64), col_idx(uint32), values(float32)]
static std::tuple<Tensor, Tensor, Tensor> thr_sparsify_to_csr_sve(const Tensor& activation, double threshold) {
  check_thr_sparsify_to_csr_sve_inputs(activation);

  const int64_t M = activation.size(0);
  const int64_t K = activation.size(1);
  const float* act_ptr = activation.data_ptr<float>();
  const float thr = static_cast<float>(threshold);

  // counts per row (int64)
  Tensor counts_t = torch::empty({M}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* counts = counts_t.data_ptr<int64_t>();

  // ---------------- Pass1: count nnz per row ----------------
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
      const float* row = act_ptr + m * K;
      int64_t nnz = 0;

#if defined(__ARM_FEATURE_SVE)
      int64_t k = 0;
      while (k < K) {
        const svbool_t pg = svwhilelt_b32(k, K);
        const svfloat32_t vx = svld1_f32(pg, row + k);
        const svfloat32_t vabs = svabs_f32_x(pg, vx);
        const svbool_t keep = svcmpge_f32(pg, vabs, svdup_f32(thr)); // ok but dup each iter
        nnz += (int64_t)svcntp_b32(pg, keep);
        k += vl;
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

  // ---------------- Pass2: compact write col_idx + values ----------------
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
      uint32_t* dst_idx = out_idx + offsets[m];
      float* dst_val = out_val + offsets[m];
      int64_t write_pos = 0;

#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_SVE2)
      // Full-keep fast path: nnz == K
      if (nnz == K) {
        int64_t k = 0;
        while (k < K) {
          const svbool_t pg = svwhilelt_b32(k, K);
          const svuint32_t vidx = svindex_u32((uint32_t)k, 1);
          const svfloat32_t vx = svld1_f32(pg, row + k);
          svst1_u32(pg, dst_idx + k, vidx);
          svst1_f32(pg, dst_val + k, vx);
          k += vl;
        }
        continue;
      }

      int64_t k = 0;
      while (k < K) {
        const svbool_t pg = svwhilelt_b32(k, K);

        const svfloat32_t vx = svld1_f32(pg, row + k);
        const svfloat32_t vabs = svabs_f32_x(pg, vx);
        const svbool_t keep = svcmpge_f32(pg, vabs, vthr);

        const uint32_t n_keep = (uint32_t)svcntp_b32(pg, keep);
        if (n_keep) {
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
      for (int64_t k = 0; k < K; ++k) {
        const float x = row[k];
        if (x >= thr || x <= -thr) {
          dst_idx[write_pos] = (uint32_t)k;
          dst_val[write_pos] = x;
          ++write_pos;
        }
      }
#endif
    }
  }

  return {row_offsets, col_idx, values};
}

// 注册到 PyTorch
// 注意：该文件会与其它算子源文件一起编译到同一个扩展中，
// 因此这里必须使用 TORCH_LIBRARY_FRAGMENT，避免与其它 TU 中的 TORCH_LIBRARY 重复定义冲突。
TORCH_LIBRARY_FRAGMENT(sparse_op, m) {
  m.def("thr_sparsify_to_csr_sve(Tensor activation, float threshold) -> (Tensor row_offsets, Tensor nz_col_indices, Tensor values)");
}

TORCH_LIBRARY_IMPL(sparse_op, CPU, m) {
  m.impl("thr_sparsify_to_csr_sve", thr_sparsify_to_csr_sve);
}
