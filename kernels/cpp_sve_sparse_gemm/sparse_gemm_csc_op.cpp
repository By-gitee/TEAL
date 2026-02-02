// #include <torch/extension.h>

// #include <cstdint>
// #include <iostream>
// #include <limits>
// #include <vector>
// #include <algorithm>

// #ifdef _OPENMP
// #include <omp.h>
// #endif

// #if defined(__ARM_FEATURE_SVE)
// #include <arm_sve.h>
// #endif

// /**
//  * CSC sparse GEMM:  Out(M,N) = A_csc(M,K) * W(K,N)
//  * A in CSC: col_ptr (K+1), row_indices (uint32), values (float)
//  *
//  * Parallel strategy (方案A):
//  *   - Parallelize over K blocks (each thread owns a disjoint k-range)
//  *   - Each thread accumulates into a private output buffer (M*N)
//  *   - Reduce private buffers into final output
//  */

// namespace {

// static inline void check_sparse_gemm_csc_inputs(
//     const torch::Tensor& weight,
//     const torch::Tensor& col_ptr,
//     const torch::Tensor& row_indices,
//     const torch::Tensor& values,
//     int64_t M) {

//   TORCH_CHECK(weight.device().is_cpu(), "weight must be a CPU tensor");
//   TORCH_CHECK(col_ptr.device().is_cpu(), "col_ptr must be a CPU tensor");
//   TORCH_CHECK(row_indices.device().is_cpu(), "row_indices must be a CPU tensor");
//   TORCH_CHECK(values.device().is_cpu(), "values must be a CPU tensor");

//   TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
//   TORCH_CHECK(col_ptr.dtype() == torch::kInt64, "col_ptr must be int64");
//   // 为避免错读，这里强制 uint32（你原代码允许 int32 但后面按 uint32 读，会出错）
//   TORCH_CHECK(row_indices.dtype() == torch::kUInt32, "row_indices must be uint32");
//   TORCH_CHECK(values.dtype() == torch::kFloat32, "values must be float32");

//   TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
//   TORCH_CHECK(col_ptr.dim() == 1, "col_ptr must be 1D");
//   TORCH_CHECK(row_indices.dim() == 1, "row_indices must be 1D");
//   TORCH_CHECK(values.dim() == 1, "values must be 1D");

//   TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
//   TORCH_CHECK(col_ptr.is_contiguous(), "col_ptr must be contiguous");
//   TORCH_CHECK(row_indices.is_contiguous(), "row_indices must be contiguous");
//   TORCH_CHECK(values.is_contiguous(), "values must be contiguous");

//   const int64_t K = weight.size(0);
//   TORCH_CHECK(M >= 0, "M must be non-negative");
//   TORCH_CHECK(K > 0, "K must be positive");
//   TORCH_CHECK(weight.size(1) >= 0, "N must be non-negative");
//   TORCH_CHECK(col_ptr.size(0) == K + 1, "col_ptr length must be K+1");

//   const int64_t* col_ptr_ptr = col_ptr.data_ptr<int64_t>();
//   TORCH_CHECK(col_ptr_ptr[0] == 0, "col_ptr[0] must be 0");
//   const int64_t nnz = col_ptr_ptr[K];
//   TORCH_CHECK(nnz == row_indices.size(0), "col_ptr[K] must equal row_indices length");
//   TORCH_CHECK(nnz == values.size(0), "values length must equal col_ptr[K] (nnz)");
// }

// // 每线程计算：处理 k in [k0, k1)，把贡献累加到 out_private (M*N)
// static inline void compute_k_range_private_out(
//     const float* values_ptr,
//     const uint32_t* row_indices_ptr,
//     const int64_t* col_ptr,
//     const float* weight_ptr,
//     float* out_private,     // [M*N], thread-private
//     int64_t M,
//     int64_t K,
//     int64_t N,
//     int64_t k0,
//     int64_t k1) {

// #if defined(__ARM_FEATURE_SVE)
//   const int64_t vl = svcntw();
// #else
//   (void)M;
// #endif

//   for (int64_t k = k0; k < k1; ++k) {
//     const int64_t col_start = col_ptr[k];
//     const int64_t col_end   = col_ptr[k + 1];
//     if (col_start == col_end) continue;

//     const float* w_row = weight_ptr + k * N;  // W[k, :]

//     for (int64_t idx = col_start; idx < col_end; ++idx) {
//       const int64_t m = (int64_t)row_indices_ptr[idx];
//       // 可选：如果你担心 row_indices 越界，可以加个 debug check（release 下建议去掉）
//       // if ((uint64_t)m >= (uint64_t)M) continue;

//       const float a_val = values_ptr[idx];
//       float* out_row = out_private + m * N;

// #if defined(__ARM_FEATURE_SVE)
//     const int64_t n_full = (N / vl) * vl;
//     int64_t n = 0;
//     for (; n < n_full; n += vl) {
//         svbool_t pg = svwhilelt_b32(n, N);
//         const svfloat32_t wv  = svld1_f32(pg, w_row  + n);
//         const svfloat32_t ov  = svld1_f32(pg, out_row + n);
//         const svfloat32_t rv  = svmla_n_f32_x(pg, ov, wv, a_val);
//         svst1_f32(pg, out_row + n, rv);
//     }

// #else
//       // scalar fallback
//       for (int64_t n = 0; n < N; ++n) {
//         out_row[n] += a_val * w_row[n];
//       }
// #endif
//     }
//   }
// }

// } // namespace

// torch::Tensor sparse_gemm_csc(
//     torch::Tensor weight,
//     torch::Tensor col_ptr,
//     torch::Tensor row_indices,
//     torch::Tensor values,
//     int64_t M,
//     int64_t ncore = 0) {

//   check_sparse_gemm_csc_inputs(weight, col_ptr, row_indices, values, M);

//   const int64_t K = weight.size(0);
//   const int64_t N = weight.size(1);

//   auto output = torch::zeros({M, N}, weight.options());
//   if (M == 0 || K == 0 || N == 0) return output;

//   const float* weight_ptr = weight.data_ptr<float>();
//   const int64_t* col_ptr_data = col_ptr.data_ptr<int64_t>();
//   const uint32_t* row_indices_ptr = row_indices.data_ptr<uint32_t>();
//   const float* values_ptr = values.data_ptr<float>();
//   float* out_ptr = output.data_ptr<float>();

//   int64_t nthreads = 1;
// #ifdef _OPENMP
//   nthreads = (ncore <= 0) ? (int64_t)omp_get_max_threads() : ncore;
// #else
//   (void)ncore;
//   nthreads = 1;
// #endif
//   if (nthreads < 1) nthreads = 1;

//   // --- 方案A：每线程私有输出 ---
//   // 注意：内存开销 = nthreads * M * N * 4 bytes
//   // M <= 256 时通常还能接受；若 N 很大、线程很多，建议改成 (M*N_tile) 缓冲版。
//   std::vector<std::vector<float>> priv((size_t)nthreads);

// #ifdef _OPENMP
//   #pragma omp parallel num_threads((int)nthreads)
// #endif
//   {
//     int tid = 0;
// #ifdef _OPENMP
//     tid = omp_get_thread_num();
// #endif
//     // 线程私有缓冲清零
//     priv[(size_t)tid].assign((size_t)M * (size_t)N, 0.0f);
//     float* out_private = priv[(size_t)tid].data();

//     // 静态按 K 切块：线程 tid 负责 [k0,k1)
//     const int64_t k0 = (K * (int64_t)tid) / nthreads;
//     const int64_t k1 = (K * (int64_t)(tid + 1)) / nthreads;

//     compute_k_range_private_out(values_ptr, row_indices_ptr, col_ptr_data,
//                                 weight_ptr, out_private, M, K, N, k0, k1);
//   }

//   // --- 归约：把每个线程的私有输出加到最终 output ---
//   const int64_t MN = M * N;

// #ifdef _OPENMP
//   #pragma omp parallel for schedule(static) num_threads((int)nthreads)
// #endif
//   for (int64_t i = 0; i < MN; ++i) {
//     float acc = 0.0f;
//     for (int t = 0; t < (int)nthreads; ++t) {
//       acc += priv[(size_t)t][(size_t)i];
//     }
//     out_ptr[(size_t)i] += acc;
//   }

//   return output;
// }

// // 注册到 PyTorch
// TORCH_LIBRARY_FRAGMENT(sparse_op, m) {
//   m.def("sparse_gemm_csc(Tensor weight, Tensor col_ptr, Tensor row_indices, Tensor values, int M, int ncore=0) -> Tensor");
// }

// TORCH_LIBRARY_IMPL(sparse_op, CPU, m) {
//   m.impl("sparse_gemm_csc", sparse_gemm_csc);
// }

#include <torch/extension.h>

#include <cstdint>
#include <iostream>
#include <omp.h>
#include <limits>
#include <vector>
#include <algorithm>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

/**
 * CSC (Compressed Sparse Column) 格式的稀疏矩阵乘法算子
 * 
 * 输入 CSC 格式：
 *   - weight: (K, N) 稠密权重矩阵
 *   - col_ptr: 1D int64, length = K + 1, CSC格式的列指针（前缀和）
 *   - row_indices: 1D uint32/int32, CSC格式的行索引
 *   - values: 1D float32, CSC格式的非零元素值
 *   - M: 输出矩阵的行数
 * 
 * 功能：
 *   1. 直接使用输入的 CSC 格式 (values, row_indices, col_ptr)
 *   2. 按 weight 矩阵行进行分块，分配到 ncore 个循环中（负载均衡）
 *   3. 使用稀疏 activation 非零值标量 + SIMD 加载 weight 行数据进行加速
 * 
 * 输出：(M, N) 矩阵
 */

namespace {

// 计算输出列块 [n0, n1)：
// 重要：并行必须按 N 维切分，确保不同线程写的 output 区域不重叠，避免数据竞争。
void compute_n_tile_sve(
    const float* values_ptr,
    const uint32_t* row_indices_ptr,
    const int64_t* col_ptr,
    const float* weight_ptr,
    float* output_ptr,
    int64_t K,
    int64_t n0,
    int64_t n1,
    int64_t N) {
    
#if defined(__ARM_FEATURE_SVE)
    const int64_t vl = svcntw();
    
    if (vl == 4) {
        const int64_t tile = n1 - n0;
        const int64_t tile_full = (tile / 4) * 4;

        // 遍历每一列 k（CSC 的列）
        for (int64_t k = 0; k < K; ++k) {
            const int64_t col_start = col_ptr[k];
            const int64_t col_end   = col_ptr[k + 1];
            if (col_start == col_end) continue;

            const float* w_row_tile = weight_ptr + k * N + n0;  // weight[k, n0:n1]

            // 遍历该列的所有非零元素（m, a_val）
            for (int64_t idx = col_start; idx < col_end; ++idx) {
                const int64_t m = (int64_t)row_indices_ptr[idx];
                const float a_val = values_ptr[idx];
                float* out_row_tile = output_ptr + m * N + n0;  // out[m, n0:n1]

                int64_t t = 0;
                for (; t < tile_full; t += 4) {
                    const svfloat32_t w_vec   = svld1_f32(svptrue_b32(), w_row_tile + t);
                    const svfloat32_t out_vec = svld1_f32(svptrue_b32(), out_row_tile + t);
                    const svfloat32_t res     = svmla_n_f32_x(svptrue_b32(), out_vec, w_vec, a_val);
                    svst1_f32(svptrue_b32(), out_row_tile + t, res);
                }
                for (; t < tile; ++t) {
                    out_row_tile[t] += a_val * w_row_tile[t];
                }
            }
        }
        return;
    }
#endif
    
    // 标量实现（fallback）
    const int64_t tile = n1 - n0;
    for (int64_t k = 0; k < K; ++k) {
        const int64_t col_start = col_ptr[k];
        const int64_t col_end   = col_ptr[k + 1];
        if (col_start == col_end) continue;

        const float* w_row_tile = weight_ptr + k * N + n0;
        for (int64_t idx = col_start; idx < col_end; ++idx) {
            const int64_t m = (int64_t)row_indices_ptr[idx];
            const float a_val = values_ptr[idx];
            float* out_row_tile = output_ptr + m * N + n0;
            for (int64_t t = 0; t < tile; ++t) {
                out_row_tile[t] += a_val * w_row_tile[t];
            }
        }
    }
}

}  // namespace

namespace {

static inline void check_sparse_gemm_csc_inputs(
    const torch::Tensor& weight,
    const torch::Tensor& col_ptr,
    const torch::Tensor& row_indices,
    const torch::Tensor& values,
    int64_t M) {
    // 输入检查
    TORCH_CHECK(weight.device().is_cpu(), "weight must be a CPU tensor");
    TORCH_CHECK(col_ptr.device().is_cpu(), "col_ptr must be a CPU tensor");
    TORCH_CHECK(row_indices.device().is_cpu(), "row_indices must be a CPU tensor");
    TORCH_CHECK(values.device().is_cpu(), "values must be a CPU tensor");

    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(col_ptr.dtype() == torch::kInt64, "col_ptr must be int64");
    TORCH_CHECK(row_indices.dtype() == torch::kUInt32 || row_indices.dtype() == torch::kInt32,
                "row_indices must be uint32 or int32");
    TORCH_CHECK(values.dtype() == torch::kFloat32, "values must be float32");

    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(col_ptr.dim() == 1, "col_ptr must be 1D");
    TORCH_CHECK(row_indices.dim() == 1, "row_indices must be 1D");
    TORCH_CHECK(values.dim() == 1, "values must be 1D");

    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(col_ptr.is_contiguous(), "col_ptr must be contiguous");
    TORCH_CHECK(row_indices.is_contiguous(), "row_indices must be contiguous");
    TORCH_CHECK(values.is_contiguous(), "values must be contiguous");

    const int64_t K = weight.size(0);
    const int64_t N = weight.size(1);

    TORCH_CHECK(M > 0, "M must be positive");
    TORCH_CHECK(K > 0, "K must be positive");
    TORCH_CHECK(col_ptr.size(0) == K + 1, "col_ptr length must be K+1");

    const int64_t* col_ptr_ptr = col_ptr.data_ptr<int64_t>();
    TORCH_CHECK(col_ptr_ptr[0] == 0, "col_ptr[0] must be 0");
    const int64_t nnz = col_ptr_ptr[K];
    TORCH_CHECK(nnz == row_indices.size(0), "col_ptr[K] must equal row_indices length");
    TORCH_CHECK(nnz == values.size(0), "values length must equal col_ptr[K] (nnz)");
}

}  // namespace

// 主函数：CSC 稀疏 GEMM
torch::Tensor sparse_gemm_csc(
    torch::Tensor weight,
    torch::Tensor col_ptr,
    torch::Tensor row_indices,
    torch::Tensor values,
    int64_t M,
    int64_t ncore = 0) {
    
    check_sparse_gemm_csc_inputs(weight, col_ptr, row_indices, values, M);
    const int64_t K = weight.size(0);
    const int64_t N = weight.size(1);
    
    auto output = torch::zeros({M, N}, weight.options());
    
    if (M == 0 || N == 0 || K == 0) {
        return output;
    }
    
    const float* weight_ptr = weight.data_ptr<float>();
    const int64_t* col_ptr_data = col_ptr.data_ptr<int64_t>();
    const uint32_t* row_indices_ptr = row_indices.data_ptr<uint32_t>();
    const float* values_ptr = values.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();
    
    // 如果 ncore <= 0，则使用 OpenMP 的线程数
    const int64_t nthreads = (ncore <= 0) ? (int64_t)omp_get_max_threads() : ncore;
    
    // 并行计算（按输出列 N 切块，避免不同线程写同一 output 元素的数据竞争）
    const int64_t N_TILE = N / nthreads;  // 可调：越大线程数越少，越小调度开销越大
    #pragma omp parallel for schedule(static) num_threads((int)nthreads)
    for (int64_t n0 = 0; n0 < N; n0 += N_TILE) {
        const int64_t n1 = std::min<int64_t>(N, n0 + N_TILE);
        compute_n_tile_sve(values_ptr, row_indices_ptr, col_ptr_data, 
                          weight_ptr, out_ptr, K, n0, n1, N);
    }
    
    return output;
}

// 注册到 PyTorch
// 注意：该文件会与其它算子源文件一起编译到同一个扩展中，
// 因此这里必须使用 TORCH_LIBRARY_FRAGMENT，避免与其它 TU 中的 TORCH_LIBRARY 重复定义冲突。
TORCH_LIBRARY_FRAGMENT(sparse_op, m) {
    m.def("sparse_gemm_csc(Tensor weight, Tensor col_ptr, Tensor row_indices, Tensor values, int M, int ncore=0) -> Tensor");
}

TORCH_LIBRARY_IMPL(sparse_op, CPU, m) {
    m.impl("sparse_gemm_csc", sparse_gemm_csc);
}

// 该扩展的 PYBIND11_MODULE 入口在其它源文件中统一提供，避免多重定义。
