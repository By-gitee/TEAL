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
  const float *act_ptr = activation.data_ptr<float>();
  const float thr = static_cast<float>(threshold);

  // 分配每行非零元素计数数组（int64，便于后续前缀和计算）
  Tensor counts_t = torch::empty({M}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t *counts = counts_t.data_ptr<int64_t>();

  // Pass1：统计每行满足阈值的元素个数
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#if defined(__ARM_FEATURE_SVE)
    const svfloat32_t vthr = svdup_f32(thr);      // 将阈值广播至向量
    const size_t vl = svcntw();                   // 每个向量可处理的float数量（动态VL）
#endif

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int64_t m = 0; m < M; ++m) {
      const float *row = act_ptr + m * K;
      int64_t nnz = 0;
#if defined(__ARM_FEATURE_SVE)
      int64_t k = 0;
      // 向量循环展开2倍处理
      while (k + 2 * (int64_t)vl <= K) {
        // 第1个向量块
        svbool_t pg1 = svwhilelt_b32((uint32_t)k, (uint32_t)K);
        svfloat32_t v1 = svld1_f32(pg1, row + k);
        svfloat32_t av1 = svabs_f32_x(pg1, v1);
        svbool_t keep1 = svcmpge_f32(pg1, av1, vthr);
        nnz += (int64_t)svcntp_b32(pg1, keep1);
        // 第2个向量块
        svbool_t pg2 = svwhilelt_b32((uint32_t)(k + vl), (uint32_t)K);
        svfloat32_t v2 = svld1_f32(pg2, row + k + vl);
        svfloat32_t av2 = svabs_f32_x(pg2, v2);
        svbool_t keep2 = svcmpge_f32(pg2, av2, vthr);
        nnz += (int64_t)svcntp_b32(pg2, keep2);
        k += 2 * vl;
      }
      // 处理剩余不足2*vl部分
      while (k < K) {
        svbool_t pg = svwhilelt_b32((uint32_t)k, (uint32_t)K);
        svfloat32_t v = svld1_f32(pg, row + k);
        svfloat32_t av = svabs_f32_x(pg, v);
        svbool_t keep = svcmpge_f32(pg, av, vthr);
        nnz += (int64_t)svcntp_b32(pg, keep);
        k += svcntw();
      }
#else
      // 标量回退路径
      for (int64_t k = 0; k < K; ++k) {
        float x = row[k];
        if (x >= thr || x <= -thr) {
          nnz++;
        }
      }
#endif
      counts[m] = nnz;
    }
  }

  // 计算行前缀和（row_offsets，长度M+1）
  Tensor row_offsets = torch::empty({M + 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t *offsets = row_offsets.data_ptr<int64_t>();
  offsets[0] = 0;
  for (int64_t m = 0; m < M; ++m) {
    offsets[m + 1] = offsets[m] + counts[m];
  }
  const int64_t total_nnz = offsets[M];

  // 分配输出的列索引数组（uint32，一维压平）
  Tensor nz_col_indices = torch::empty({total_nnz}, torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCPU));
  uint32_t *out_idx = (total_nnz > 0 ? nz_col_indices.data_ptr<uint32_t>() : nullptr);

  // Pass2：按照前缀和将各行非零列索引压缩写入输出数组
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#if defined(__ARM_FEATURE_SVE)
    const svfloat32_t vthr = svdup_f32(thr);
    const size_t vl = svcntw();
#endif

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int64_t m = 0; m < M; ++m) {
      const int64_t nnz = counts[m];
      if (nnz == 0) continue;  // 无非零元素则跳过

      const float *row = act_ptr + m * K;
      uint32_t *dst = out_idx + offsets[m];
      int64_t write_pos = 0;
#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_SVE2)
      if (nnz == K) {
        // 整行全满足阈值，直接输出 [0..K-1]
        int64_t kk = 0;
        while (kk < K) {
          svbool_t pg = svwhilelt_b32((uint32_t)kk, (uint32_t)K);
          svuint32_t vidx = svindex_u32((uint32_t)kk, 1);
          svst1_u32(pg, dst + kk, vidx);
          kk += svcntw();
        }
        write_pos = nnz;
      } else {
        int64_t k = 0;
        // 向量输出循环展开2倍
        while (k + 2 * (int64_t)vl <= K) {
          // 第1个向量块筛选并写出
          svbool_t pg1 = svwhilelt_b32((uint32_t)k, (uint32_t)K);
          svfloat32_t v1 = svld1_f32(pg1, row + k);
          svfloat32_t av1 = svabs_f32_x(pg1, v1);
          svbool_t keep1 = svcmpge_f32(pg1, av1, vthr);
          int64_t n1 = (int64_t)svcntp_b32(pg1, keep1);
          if (n1 > 0) {
            svuint32_t vidx1 = svindex_u32((uint32_t)k, 1);
            svuint32_t packed1 = svcompact_u32(keep1, vidx1);
            svbool_t pg_out1 = svwhilelt_b32((uint32_t)0, (uint32_t)n1);
            svst1_u32(pg_out1, dst + write_pos, packed1);
            write_pos += n1;
          }
          // 第2个向量块筛选并写出
          svbool_t pg2 = svwhilelt_b32((uint32_t)(k + vl), (uint32_t)K);
          svfloat32_t v2 = svld1_f32(pg2, row + k + vl);
          svfloat32_t av2 = svabs_f32_x(pg2, v2);
          svbool_t keep2 = svcmpge_f32(pg2, av2, vthr);
          int64_t n2 = (int64_t)svcntp_b32(pg2, keep2);
          if (n2 > 0) {
            svuint32_t vidx2 = svindex_u32((uint32_t)(k + vl), 1);
            svuint32_t packed2 = svcompact_u32(keep2, vidx2);
            svbool_t pg_out2 = svwhilelt_b32((uint32_t)0, (uint32_t)n2);
            svst1_u32(pg_out2, dst + write_pos, packed2);
            write_pos += n2;
          }
          k += 2 * vl;
        }
        // 处理剩余不足2*vl的部分
        while (k < K) {
          svbool_t pg = svwhilelt_b32((uint32_t)k, (uint32_t)K);
          svfloat32_t v = svld1_f32(pg, row + k);
          svfloat32_t av = svabs_f32_x(pg, v);
          svbool_t keep = svcmpge_f32(pg, av, vthr);
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
      // 标量回退路径
      if (nnz == K) {
        // 整行全部输出
        for (uint32_t k = 0; k < (uint32_t)K; ++k) {
          dst[write_pos++] = k;
        }
      } else {
        for (int64_t k = 0; k < K; ++k) {
          float x = row[k];
          if (x >= thr || x <= -thr) {
            dst[write_pos++] = (uint32_t)k;
          }
        }
      }
#endif
      // 校验输出数量是否匹配
      if (write_pos != nnz) {
        // 如果发生不匹配，这里可以记录警告或截断处理（调试用）
      }
    }
  }

  // 构建 nz_counts 数组（仅记录有非零值的行及其nnz），形式：[row_index, nnz, row_index2, nnz2, ...]
  int64_t num_nz_rows = 0;
  for (int64_t m = 0; m < M; ++m) {
    if (counts[m] > 0) num_nz_rows++;
  }
  Tensor nz_counts = torch::empty({2 * num_nz_rows}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t *nzp = nz_counts.data_ptr<int64_t>();
  int64_t p = 0;
  for (int64_t m = 0; m < M; ++m) {
    int64_t nnz = counts[m];
    if (nnz > 0) {
      nzp[p++] = m;
      nzp[p++] = nnz;
    }
  }

  return {nz_counts, nz_col_indices, row_offsets};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("row_scan_sve", &row_scan_sve, "RowScan-SVE2 optimized (index generation with SVE2)");
}
