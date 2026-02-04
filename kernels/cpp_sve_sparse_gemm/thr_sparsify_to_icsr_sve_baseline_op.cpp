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

// // GCC 特有的优化属性，clang 不支持
// #if defined(__GNUC__) && !defined(__clang__)
//   #define NO_VEC __attribute__((optimize("no-tree-vectorize")))
// #else
//   #define NO_VEC
// #endif

// scalar pack — 禁止自动向量化，确保 baseline 不作弊
static inline  void pack_u32(
    const uint32_t* idx,
    const uint32_t* keep,
    int lanes,
    uint32_t* dst,
    int64_t& pos) {
  for (int i = 0; i < lanes; ++i) {
    if (keep[i]) dst[pos++] = idx[i];
  }
}

std::tuple<Tensor, Tensor, Tensor>
thr_sparsify_to_icsr_sve_baseline(const Tensor& activation, double threshold) {
  check_inputs(activation);

  const int64_t M = activation.size(0);
  const int64_t K = activation.size(1);
  const float thr = (float)threshold;
  const float* act = activation.data_ptr<float>();

  Tensor counts_t = torch::empty({M}, torch::kInt64);
  int64_t* counts = counts_t.data_ptr<int64_t>();

// ===================== Pass1: count =====================

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
    int64_t k = 0;

    while (k + 2 * vl <= K) {
      svbool_t pg1 = svptrue_b32();
      svfloat32_t x1 = svld1_f32(pg1, row + k);
      svfloat32_t a1 = svabs_f32_x(pg1, x1);
      svbool_t keep1 = svcmpge_f32(pg1, a1, vthr);
      nnz += svcntp_b32(pg1, keep1);

      svbool_t pg2 = svptrue_b32();
      svfloat32_t x2 = svld1_f32(pg2, row + k + vl);
      svfloat32_t a2 = svabs_f32_x(pg2, x2);
      svbool_t keep2 = svcmpge_f32(pg2, a2, vthr);
      nnz += svcntp_b32(pg2, keep2);

      k += 2 * vl;
    }

    while (k < K) {
      svbool_t pg = svwhilelt_b32(k, K);
      svfloat32_t x = svld1_f32(pg, row + k);
      svfloat32_t a = svabs_f32_x(pg, x);
      svbool_t keep = svcmpge_f32(pg, a, vthr);
      nnz += svcntp_b32(pg, keep);
      k += vl;
    }
#else
    for (int64_t k = 0; k < K; ++k) {
      float v = row[k];
      if (v >= thr || v <= -thr) nnz++;
    }
#endif
    counts[m] = nnz;
  }
}

// ================= row_offsets =================

Tensor row_offsets = torch::empty({M + 1}, torch::kInt64);
int64_t* offs = row_offsets.data_ptr<int64_t>();
offs[0] = 0;
for (int64_t m = 0; m < M; ++m) offs[m + 1] = offs[m] + counts[m];
const int64_t total = offs[M];

Tensor nz_col = torch::empty({total}, torch::kUInt32);
uint32_t* out = (total > 0) ? nz_col.data_ptr<uint32_t>() : nullptr;

// ===================== Pass2: pack =====================

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
    uint32_t* dst = out + offs[m];
    int64_t pos = 0;

#if defined(__ARM_FEATURE_SVE)
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
    while (k < K) {
      svbool_t pg = svwhilelt_b32(k, K);
      svfloat32_t x = svld1_f32(pg, row + k);
      svfloat32_t a = svabs_f32_x(pg, x);
      svbool_t km = svcmpge_f32(pg, a, vthr);

      svuint32_t vidx = svindex_u32(k, 1);
      svst1_u32(pg, idx, vidx);

      svuint32_t ones = svdup_u32_z(km, 1);
      svst1_u32(pg, keep, ones);

      int lanes = std::min<int64_t>(vl, K - k);
      pack_u32(idx, keep, lanes, dst, pos);

      k += vl;
    }
#else
    for (int64_t k = 0; k < K; ++k) {
      float v = row[k];
      if (v >= thr || v <= -thr) dst[pos++] = (uint32_t)k;
    }
#endif
  }
}

// ================= nz_counts =================

int64_t rows = 0;
for (int64_t m = 0; m < M; ++m) if (counts[m] > 0) rows++;

Tensor nz_counts = torch::empty({2 * rows}, torch::kInt64);
int64_t* nzp = nz_counts.data_ptr<int64_t>();

int64_t p = 0;
for (int64_t m = 0; m < M; ++m) {
  if (counts[m] > 0) {
    nzp[p++] = m;
    nzp[p++] = counts[m];
  }
}

return {nz_counts, nz_col, row_offsets};
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("thr_sparsify_to_icsr_sve_baseline",
//         &thr_sparsify_to_icsr_sve_baseline);
// }

// 注册到 PyTorch
// 注意：该文件会与其它算子源文件一起编译到同一个扩展中，
// 因此这里必须使用 TORCH_LIBRARY_FRAGMENT，避免与其它 TU 中的 TORCH_LIBRARY 重复定义冲突。
TORCH_LIBRARY_FRAGMENT(sparse_op, m) {
    m.def("thr_sparsify_to_icsr_sve_baseline(Tensor activation, float threshold) -> (Tensor, Tensor, Tensor)");
  }
  
  TORCH_LIBRARY_IMPL(sparse_op, CPU, m) {
    m.impl("thr_sparsify_to_icsr_sve_baseline", thr_sparsify_to_icsr_sve_baseline);
  }
  