#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#include <omp.h>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

namespace {

inline void check_inputs_gemm(
    const torch::Tensor& A,
    const torch::Tensor& B) {
  TORCH_CHECK(A.device().is_cpu(), "A must be a CPU tensor");
  TORCH_CHECK(B.device().is_cpu(), "B must be a CPU tensor");

  TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
  TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");

  TORCH_CHECK(A.dim() == 2, "A must be 2D");
  TORCH_CHECK(B.dim() == 2, "B must be 2D");

  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

  TORCH_CHECK(A.size(1) == B.size(0),
              "A.size(1) must match B.size(0)");
}

/**
 * Pack B block:
 *   Original B layout: [K, N]
 *   Packed layout:     [kk, nn] contiguous
 *
 * packB[(k - k0) * n_len + (n - n0)] = B[k, n]
 */
inline void pack_B_block(
    const float* B,
    float* packB,
    int64_t N,
    int64_t k0,
    int64_t k_len,
    int64_t n0,
    int64_t n_len) {
  for (int64_t kk = 0; kk < k_len; ++kk) {
    const float* src = B + (k0 + kk) * N + n0;
    float* dst = packB + kk * n_len;
    std::memcpy(dst, src, sizeof(float) * n_len);
  }
}

#if defined(__ARM_FEATURE_SVE)
/**
 * 4xN micro-kernel:
 *   C[i + 0 : i + 3, j : j + n_len] += A_block * packed_B
 *
 * A rows are contiguous in K dimension.
 * packed_B is [k_len, n_len].
 */
inline void gemm_microkernel_4xN_sve(
    const float* A_row0,
    const float* A_row1,
    const float* A_row2,
    const float* A_row3,
    const float* packB,
    float* C_row0,
    float* C_row1,
    float* C_row2,
    float* C_row3,
    int64_t lda,          // actually K, kept for interface symmetry
    int64_t ldc,          // actually N, kept for interface symmetry
    int64_t k_len,
    int64_t n_len) {
  (void)lda;
  (void)ldc;

  int64_t j = 0;
  for (; j < n_len; j += svcntw()) {
    svbool_t pg = svwhilelt_b32(j, n_len);

    svfloat32_t c0 = svld1_f32(pg, C_row0 + j);
    svfloat32_t c1 = svld1_f32(pg, C_row1 + j);
    svfloat32_t c2 = svld1_f32(pg, C_row2 + j);
    svfloat32_t c3 = svld1_f32(pg, C_row3 + j);

    for (int64_t kk = 0; kk < k_len; ++kk) {
      const svfloat32_t b = svld1_f32(pg, packB + kk * n_len + j);

      c0 = svmla_n_f32_m(pg, c0, b, A_row0[kk]);
      c1 = svmla_n_f32_m(pg, c1, b, A_row1[kk]);
      c2 = svmla_n_f32_m(pg, c2, b, A_row2[kk]);
      c3 = svmla_n_f32_m(pg, c3, b, A_row3[kk]);
    }

    svst1_f32(pg, C_row0 + j, c0);
    svst1_f32(pg, C_row1 + j, c1);
    svst1_f32(pg, C_row2 + j, c2);
    svst1_f32(pg, C_row3 + j, c3);
  }
}

/**
 * 1xN micro-kernel for tail rows.
 */
inline void gemm_microkernel_1xN_sve(
    const float* A_row,
    const float* packB,
    float* C_row,
    int64_t k_len,
    int64_t n_len) {
  int64_t j = 0;
  for (; j < n_len; j += svcntw()) {
    svbool_t pg = svwhilelt_b32(j, n_len);
    svfloat32_t c = svld1_f32(pg, C_row + j);

    for (int64_t kk = 0; kk < k_len; ++kk) {
      const svfloat32_t b = svld1_f32(pg, packB + kk * n_len + j);
      c = svmla_n_f32_m(pg, c, b, A_row[kk]);
    }

    svst1_f32(pg, C_row + j, c);
  }
}
#endif

inline void gemm_microkernel_4xN_scalar(
    const float* A_row0,
    const float* A_row1,
    const float* A_row2,
    const float* A_row3,
    const float* packB,
    float* C_row0,
    float* C_row1,
    float* C_row2,
    float* C_row3,
    int64_t k_len,
    int64_t n_len) {
  for (int64_t kk = 0; kk < k_len; ++kk) {
    const float a0 = A_row0[kk];
    const float a1 = A_row1[kk];
    const float a2 = A_row2[kk];
    const float a3 = A_row3[kk];
    const float* b_row = packB + kk * n_len;

    for (int64_t j = 0; j < n_len; ++j) {
      const float bv = b_row[j];
      C_row0[j] += a0 * bv;
      C_row1[j] += a1 * bv;
      C_row2[j] += a2 * bv;
      C_row3[j] += a3 * bv;
    }
  }
}

inline void gemm_microkernel_1xN_scalar(
    const float* A_row,
    const float* packB,
    float* C_row,
    int64_t k_len,
    int64_t n_len) {
  for (int64_t kk = 0; kk < k_len; ++kk) {
    const float a = A_row[kk];
    const float* b_row = packB + kk * n_len;
    for (int64_t j = 0; j < n_len; ++j) {
      C_row[j] += a * b_row[j];
    }
  }
}

} // namespace

torch::Tensor dense_gemm_sve_omp(
    torch::Tensor A,
    torch::Tensor B) {
  check_inputs_gemm(A, B);

  const int64_t M = A.size(0);
  const int64_t K = A.size(1);
  const int64_t N = B.size(1);

  auto C = torch::zeros({M, N}, A.options());
  if (M == 0 || K == 0 || N == 0) {
    return C;
  }

  const float* A_ptr = A.data_ptr<float>();
  const float* B_ptr = B.data_ptr<float>();
  float* C_ptr = C.data_ptr<float>();

  /**
   * Blocking parameters
   *
   * 这些值不是绝对最优，需要按平台调。
   * 对于 ARM server CPU，一般可以从这组起步。
   */
  constexpr int64_t MC = 64;   // row block of A/C
  constexpr int64_t NC = 128;  // col block of B/C
  constexpr int64_t KC = 128;  // reduction block

  const int64_t num_mc = (M + MC - 1) / MC;
  const int64_t num_nc = (N + NC - 1) / NC;

  /**
   * Parallel over (mc, nc) tiles.
   * Each thread owns a distinct C tile, so no write conflict.
   */
#pragma omp parallel
  {
    std::vector<float> packB(KC * NC);

#pragma omp for collapse(2) schedule(static)
    for (int64_t mc_idx = 0; mc_idx < num_mc; ++mc_idx) {
      for (int64_t nc_idx = 0; nc_idx < num_nc; ++nc_idx) {
        const int64_t m0 = mc_idx * MC;
        const int64_t n0 = nc_idx * NC;
        const int64_t m_len = std::min<int64_t>(MC, M - m0);
        const int64_t n_len = std::min<int64_t>(NC, N - n0);

        for (int64_t k0 = 0; k0 < K; k0 += KC) {
          const int64_t k_len = std::min<int64_t>(KC, K - k0);

          // Pack current B panel: [k0:k0+k_len, n0:n0+n_len]
          pack_B_block(B_ptr, packB.data(), N, k0, k_len, n0, n_len);

          int64_t mi = 0;
          for (; mi + 3 < m_len; mi += 4) {
            const float* A_row0 = A_ptr + (m0 + mi + 0) * K + k0;
            const float* A_row1 = A_ptr + (m0 + mi + 1) * K + k0;
            const float* A_row2 = A_ptr + (m0 + mi + 2) * K + k0;
            const float* A_row3 = A_ptr + (m0 + mi + 3) * K + k0;

            float* C_row0 = C_ptr + (m0 + mi + 0) * N + n0;
            float* C_row1 = C_ptr + (m0 + mi + 1) * N + n0;
            float* C_row2 = C_ptr + (m0 + mi + 2) * N + n0;
            float* C_row3 = C_ptr + (m0 + mi + 3) * N + n0;

#if defined(__ARM_FEATURE_SVE)
            gemm_microkernel_4xN_sve(
                A_row0, A_row1, A_row2, A_row3,
                packB.data(),
                C_row0, C_row1, C_row2, C_row3,
                K, N, k_len, n_len);
#else
            gemm_microkernel_4xN_scalar(
                A_row0, A_row1, A_row2, A_row3,
                packB.data(),
                C_row0, C_row1, C_row2, C_row3,
                k_len, n_len);
#endif
          }

          // Tail rows
          for (; mi < m_len; ++mi) {
            const float* A_row = A_ptr + (m0 + mi) * K + k0;
            float* C_row = C_ptr + (m0 + mi) * N + n0;
#if defined(__ARM_FEATURE_SVE)
            gemm_microkernel_1xN_sve(
                A_row, packB.data(), C_row, k_len, n_len);
#else
            gemm_microkernel_1xN_scalar(
                A_row, packB.data(), C_row, k_len, n_len);
#endif
          }
        }
      }
    }
  }

  return C;
}

// Register to torch.ops.dense_op
TORCH_LIBRARY_FRAGMENT(dense_op, m) {
  m.def("dense_gemm_sve_omp(Tensor A, Tensor B) -> Tensor");
}

TORCH_LIBRARY_IMPL(dense_op, CPU, m) {
  m.impl("dense_gemm_sve_omp", dense_gemm_sve_omp);
}