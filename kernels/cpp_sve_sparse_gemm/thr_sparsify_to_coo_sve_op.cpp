// thr_sparsify_to_coo_sve_op.cpp
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

static inline void check_thr_sparsify_to_coo_sve_inputs(const Tensor& activation) {
  TORCH_CHECK(activation.device().is_cpu(), "activation must be a CPU tensor");
  TORCH_CHECK(activation.dtype() == torch::kFloat32, "activation must be float32");
  TORCH_CHECK(activation.is_contiguous(), "activation must be contiguous");
  TORCH_CHECK(activation.dim() == 2, "activation must be 2D [M, K]");
}

/**
 * thr_sparsify_to_coo_sve(activation, threshold) -> (row_indices, col_indices, values)
 * 
 * 将稠密矩阵根据阈值转换为 COO 格式的稀疏矩阵（SVE/SVE2 加速版本）。
 * 
 * Args:
 *   activation: (M, K) float32 稠密矩阵
 *   threshold: float 阈值，绝对值 >= threshold 的元素被保留
 * 
 * Returns:
 *   row_indices: int64 [nnz] 行索引数组（已按行排序）
 *   col_indices: uint32 [nnz] 列索引数组
 *   values: float32 [nnz] 非零元素值数组
 * 
 * 实现策略（SVE 优化）：
 *   Pass 1: 使用 SVE 向量化统计每行的非零元素数量
 *     - svld1_f32: 向量加载
 *     - svabs_f32_x: 向量绝对值
 *     - svcmpge_f32: 向量比较
 *     - svcntp_b32: 计数满足条件的元素
 *   Pass 2: 计算行偏移（前缀和）
 *   Pass 3: 使用 SVE2 compact 指令压缩填充 COO 三元组
 *     - svindex_u32: 生成索引序列
 *     - svcompact_u32/svcompact_f32: 压缩满足条件的元素
 *     - 优化：全保留块跳过 compact 直接写入
 */
static std::tuple<Tensor, Tensor, Tensor> thr_sparsify_to_coo_sve(const Tensor& activation, double threshold) {
  check_thr_sparsify_to_coo_sve_inputs(activation);

  const int64_t M = activation.size(0);
  const int64_t K = activation.size(1);
  const float* act_ptr = activation.data_ptr<float>();
  const float thr = static_cast<float>(threshold);

  // counts per row (int64)
  Tensor counts_t = torch::empty({M}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* counts = counts_t.data_ptr<int64_t>();

  // ---------------- Pass1: count nnz per row (SVE 向量化) ----------------
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
        const svbool_t keep = svcmpge_f32(pg, vabs, vthr);
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
      int64_t* dst_row = out_row + row_offsets[m];
      uint32_t* dst_col = out_col + row_offsets[m];
      float* dst_val = out_val + row_offsets[m];
      int64_t write_pos = 0;

#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_SVE2)
      // Full-keep fast path: nnz == K (所有元素都满足条件)
      if (nnz == K) {
        int64_t k = 0;
        while (k < K) {
          const svbool_t pg = svwhilelt_b32(k, K);
          const svuint32_t vidx = svindex_u32((uint32_t)k, 1);
          const svfloat32_t vx = svld1_f32(pg, row + k);
          
          // 填充行索引（所有元素都是当前行 m）
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

      // General path: 使用 SVE2 compact 指令压缩数据
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
            
            // 填充行索引
            for (int64_t i = 0; i < chunk_len; ++i) {
              dst_row[write_pos + i] = m;
            }
            
            svst1_u32(pg, dst_col + write_pos, vidx);
            svst1_f32(pg, dst_val + write_pos, vx);
            write_pos += chunk_len;
          } else {
            // 使用 compact 指令压缩
            const svuint32_t vidx = svindex_u32((uint32_t)k, 1);
            const svuint32_t packed_col = svcompact_u32(keep, vidx);
            const svfloat32_t packed_val = svcompact_f32(keep, vx);

            const svbool_t pg_out = svwhilelt_b32((uint32_t)0, n_keep);
            
            // 填充行索引
            for (uint32_t i = 0; i < n_keep; ++i) {
              dst_row[write_pos + i] = m;
            }
            
            svst1_u32(pg_out, dst_col + write_pos, packed_col);
            svst1_f32(pg_out, dst_val + write_pos, packed_val);
            write_pos += (int64_t)n_keep;
          }
        }

        k += vl;
      }

#ifndef NDEBUG
      TORCH_CHECK(write_pos == nnz,
                  "thr_sparsify_to_coo_sve: write_pos != nnz at row ", m,
                  " write_pos=", write_pos, " nnz=", nnz);
#endif

#else
      // Scalar fallback
      for (int64_t k = 0; k < K; ++k) {
        const float x = row[k];
        if (x >= thr || x <= -thr) {
          dst_row[write_pos] = m;
          dst_col[write_pos] = (uint32_t)k;
          dst_val[write_pos] = x;
          ++write_pos;
        }
      }
#endif
    }
  }

  return {row_indices, col_indices, values};
}

// 注册到 PyTorch
// 注意：该文件会与其它算子源文件一起编译到同一个扩展中，
// 因此这里必须使用 TORCH_LIBRARY_FRAGMENT，避免与其它 TU 中的 TORCH_LIBRARY 重复定义冲突。
TORCH_LIBRARY_FRAGMENT(sparse_op, m) {
  m.def("thr_sparsify_to_coo_sve(Tensor activation, float threshold) -> (Tensor row_indices, Tensor col_indices, Tensor values)");
}

TORCH_LIBRARY_IMPL(sparse_op, CPU, m) {
  m.impl("thr_sparsify_to_coo_sve", thr_sparsify_to_coo_sve);
}
