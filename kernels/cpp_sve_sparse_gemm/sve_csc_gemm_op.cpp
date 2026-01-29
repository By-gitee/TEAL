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
 * CSC (Compressed Sparse Column) 格式的稀疏矩阵乘法算子（废弃）
 * 
 * 输入：
 *   - activation: (M, K) 原始稠密 activation 矩阵
 *   - weight: (K, N) 稠密权重矩阵
 *   - nz_counts: (2 * num_nz_rows) 格式为 [row_idx, count, row_idx, count, ...]
 *   - nz_col_indices: 所有非零元素的列索引（扁平化）
 * 
 * 功能：
 *   1. 将稀疏 activation 转换为 CSC 格式
 *   2. 按 weight 矩阵行进行分块，分配到 ncore 个循环中（负载均衡）
 *   3. 使用稀疏 activation 非零值标量 + SIMD 加载 weight 行数据进行加速
 * 
 * 输出：(M, N) 矩阵
 */

namespace {

// CSC 格式数据结构
struct CSCMatrix {
    std::vector<float> values;           // 非零值
    std::vector<uint32_t> row_indices;  // 每个非零值对应的行索引
    std::vector<int64_t> col_ptr;        // 每列的起始位置，长度为 K+1
    int64_t M;                           // 行数
    int64_t K;                           // 列数
    int64_t nnz;                         // 非零元素数量
};

// 将稀疏 activation 转换为 CSC 格式
CSCMatrix convert_to_csc_simple(
    const float* act_ptr,
    const int64_t* nz_counts_ptr,
    const uint32_t* nz_col_indices_ptr,
    const uint32_t* mask_ptr,
    int64_t M,
    int64_t K,
    int64_t num_nz_rows
    ) {
        CSCMatrix csc;
        (void)mask_ptr; // 不再使用 mask 构建 CSC（改为完全依赖 nz_counts/nz_col_indices）
        csc.M = M;
        csc.K = K;
        csc.col_ptr.assign((size_t)K + 1, 0);
      
        if (M <= 0 || K <= 0 || num_nz_rows <= 0) {
          csc.nnz = 0;
          return csc;
        }
      
        // ---------------------------
        // Phase1: count per column (using nz_col_indices; cheapest)
        // ---------------------------
        std::vector<int64_t> col_counts((size_t)K, 0);
      
        int64_t global_offset = 0;
        for (int64_t i = 0; i < num_nz_rows; ++i) {
          const int64_t row_idx = nz_counts_ptr[2 * i];
          const int64_t count   = nz_counts_ptr[2 * i + 1];
      
          for (int64_t j = 0; j < count; ++j) {
            const uint32_t col = nz_col_indices_ptr[global_offset + j];
            col_counts[col] += 1;
          }
          global_offset += count;
        }
      
        // ---------------------------
        // Phase2: prefix-sum col_ptr
        // ---------------------------
        csc.col_ptr[0] = 0;
        for (int64_t k = 0; k < K; ++k) {
          csc.col_ptr[k + 1] = csc.col_ptr[k] + col_counts[k];
        }
        csc.nnz = csc.col_ptr[K];
      
        csc.values.resize((size_t)csc.nnz);
        csc.row_indices.resize((size_t)csc.nnz);
      
        std::vector<int64_t> col_offsets = csc.col_ptr; // write cursors per col
      
        // ---------------------------
        // Phase3: build (values,row_indices) by iterating nz_col_indices per row
        //   直接按稀疏列索引列表写入，避免扫描 mask（O(nnz)）
        // ---------------------------
        global_offset = 0;
        for (int64_t i = 0; i < num_nz_rows; ++i) {
          const int64_t row_idx = nz_counts_ptr[2 * i];
          const int64_t count = nz_counts_ptr[2 * i + 1];

          // 逐个非零列索引写入 CSC
          for (int64_t j = 0; j < count; ++j) {
            const uint32_t col = nz_col_indices_ptr[global_offset + j];
            const float value = act_ptr[row_idx * K + (int64_t)col];
            const int64_t write_pos = col_offsets[(size_t)col]++; // CSC cursor per column
            csc.values[write_pos] = value;
            csc.row_indices[write_pos] = (uint32_t)row_idx;
          }
          global_offset += count;
        }
      
        return csc;
}

// 计算输出列块 [n0, n1)：
// 重要：并行必须按 N 维切分，确保不同线程写的 output 区域不重叠，避免数据竞争。
void compute_n_tile_sve(
    const CSCMatrix& csc,
    const float* weight_ptr,
    float* output_ptr,
    int64_t n0,
    int64_t n1,
    int64_t N) {
    
#if defined(__ARM_FEATURE_SVE)
    const int64_t vl = svcntw();
    
    if (vl == 4) {
        const int64_t tile = n1 - n0;
        const int64_t tile_full = (tile / 4) * 4;

        // 遍历每一列 k（CSC 的列）
        for (int64_t k = 0; k < csc.K; ++k) {
            const int64_t col_start = csc.col_ptr[k];
            const int64_t col_end   = csc.col_ptr[k + 1];
            if (col_start == col_end) continue;

            const float* w_row_tile = weight_ptr + k * N + n0;  // weight[k, n0:n1]

            // 遍历该列的所有非零元素（m, a_val）
            for (int64_t idx = col_start; idx < col_end; ++idx) {
                const int64_t m = (int64_t)csc.row_indices[idx];
                const float a_val = csc.values[idx];
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
    for (int64_t k = 0; k < csc.K; ++k) {
        const int64_t col_start = csc.col_ptr[k];
        const int64_t col_end   = csc.col_ptr[k + 1];
        if (col_start == col_end) continue;

        const float* w_row_tile = weight_ptr + k * N + n0;
        for (int64_t idx = col_start; idx < col_end; ++idx) {
            const int64_t m = (int64_t)csc.row_indices[idx];
            const float a_val = csc.values[idx];
            float* out_row_tile = output_ptr + m * N + n0;
            for (int64_t t = 0; t < tile; ++t) {
                out_row_tile[t] += a_val * w_row_tile[t];
            }
        }
    }
}

}  // namespace

// 主函数：CSC 稀疏 GEMM
torch::Tensor sve_csc_gemm(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor nz_counts,
    torch::Tensor nz_col_indices,
    torch::Tensor mask,
    int64_t ncore = 0) {
    
    // 输入检查
    TORCH_CHECK(activation.device().is_cpu(), "activation must be a CPU tensor");
    TORCH_CHECK(weight.device().is_cpu(), "weight must be a CPU tensor");
    TORCH_CHECK(nz_counts.device().is_cpu(), "nz_counts must be a CPU tensor");
    TORCH_CHECK(nz_col_indices.device().is_cpu(), "nz_col_indices must be a CPU tensor");
    TORCH_CHECK(mask.device().is_cpu(), "mask must be a CPU tensor");
    
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "activation must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(nz_counts.dtype() == torch::kInt64, "nz_counts must be int64");
    TORCH_CHECK(nz_col_indices.dtype() == torch::kUInt32 || nz_col_indices.dtype() == torch::kInt32,
                "nz_col_indices must be uint32 or int32");
    // TORCH_CHECK(mask.dtype() == torch::kBool, "mask must be bool");

    TORCH_CHECK(activation.dim() == 2, "activation must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(activation.is_contiguous(), "activation must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
    
    const int64_t M = activation.size(0);
    const int64_t K = activation.size(1);
    const int64_t N = weight.size(1);
    
    TORCH_CHECK(weight.size(0) == K, "weight K dimension must match activation K");
    TORCH_CHECK(mask.size(0) == M, "mask M dimension must match activation M");
    TORCH_CHECK(mask.size(1) == K, "mask K dimension must match activation K");
    auto output = torch::zeros({M, N}, activation.options());
    
    if (M == 0 || N == 0 || K == 0) {
        return output;
    }
    
    const float* act_ptr = activation.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const int64_t* counts_ptr = nz_counts.data_ptr<int64_t>();
    const uint32_t* indices_ptr = nz_col_indices.data_ptr<uint32_t>();
    const uint32_t* mask_ptr = mask.data_ptr<uint32_t>();
    float* out_ptr = output.data_ptr<float>();
    
    const int64_t num_nz_rows = nz_counts.size(0) / 2;
    
    // 如果 ncore <= 0，则使用 OpenMP 的线程数
    const int64_t nthreads = (ncore <= 0) ? (int64_t)omp_get_max_threads() : ncore;
    
    // 步骤 1：转换为 CSC 格式
    CSCMatrix csc = convert_to_csc_simple(
        act_ptr, counts_ptr, indices_ptr, mask_ptr, M, K, num_nz_rows);
    
    // 步骤 2：并行计算（按输出列 N 切块，避免不同线程写同一 output 元素的数据竞争）
    constexpr int64_t N_TILE = 688;  // 可调：越大线程数越少，越小调度开销越大
    #pragma omp parallel for schedule(static) num_threads((int)nthreads)
    for (int64_t n0 = 0; n0 < N; n0 += N_TILE) {
        const int64_t n1 = std::min<int64_t>(N, n0 + N_TILE);
        compute_n_tile_sve(csc, weight_ptr, out_ptr, n0, n1, N);
    }
    
    return output;
}

// 注册到 PyTorch
TORCH_LIBRARY(teal_csc, m) {
    m.def("sve_csc_gemm(Tensor activation, Tensor weight, Tensor nz_counts, Tensor nz_col_indices, Tensor mask, int ncore=0) -> Tensor");
}

TORCH_LIBRARY_IMPL(teal_csc, CPU, m) {
    m.impl("sve_csc_gemm", sve_csc_gemm);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
