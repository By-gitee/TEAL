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
    std::vector<int64_t> row_indices;    // 每个非零值对应的行索引
    std::vector<int64_t> col_ptr;        // 每列的起始位置，长度为 K+1
    int64_t M;                           // 行数
    int64_t K;                           // 列数
    int64_t nnz;                         // 非零元素数量
};

// 将稀疏 activation 转换为 CSC 格式
CSCMatrix convert_to_csc(
    const float* act_ptr,
    const int64_t* nz_counts_ptr,
    const uint32_t* nz_col_indices_ptr,
    int64_t M,
    int64_t K,
    int64_t num_nz_rows) {
    
    CSCMatrix csc;
    csc.M = M;
    csc.K = K;
    csc.col_ptr.resize(K + 1, 0);
    
    // 第一步：统计每列的非零元素数量
    std::vector<int64_t> col_counts(K, 0);
    
    int64_t global_offset = 0;
    for (int64_t i = 0; i < num_nz_rows; ++i) {
        int64_t row_idx = nz_counts_ptr[2 * i];
        int64_t count = nz_counts_ptr[2 * i + 1];
        
        for (int64_t j = 0; j < count; ++j) {
            uint32_t col = nz_col_indices_ptr[global_offset + j];
            col_counts[col]++;
        }
        global_offset += count;
    }
    
    // 第二步：构建列指针
    csc.col_ptr[0] = 0;
    for (int64_t k = 0; k < K; ++k) {
        csc.col_ptr[k + 1] = csc.col_ptr[k] + col_counts[k];
    }
    csc.nnz = csc.col_ptr[K];
    
    // 第三步：填充值和行索引
    csc.values.resize(csc.nnz);
    csc.row_indices.resize(csc.nnz);
    
    std::vector<int64_t> col_offsets = csc.col_ptr;  // 临时的写入位置
    
    global_offset = 0;
    for (int64_t i = 0; i < num_nz_rows; ++i) {
        int64_t row_idx = nz_counts_ptr[2 * i];
        int64_t count = nz_counts_ptr[2 * i + 1];
        
        const float* row_ptr = act_ptr + row_idx * K;
        
        for (int64_t j = 0; j < count; ++j) {
            uint32_t col = nz_col_indices_ptr[global_offset + j];
            float value = row_ptr[col];
            
            // 使用 scatter store 的思想：将值写入对应列的位置
            int64_t write_pos = col_offsets[col]++;
            csc.values[write_pos] = value;
            csc.row_indices[write_pos] = row_idx;
        }
        global_offset += count;
    }
    
    return csc;
}

// 负载均衡分配任务
struct WorkBlock {
    int64_t start_col;  // 起始列
    int64_t end_col;    // 结束列（不包含）
    int64_t nnz_count;  // 该块的非零元素数量
};

std::vector<WorkBlock> partition_work(
    const CSCMatrix& csc,
    int64_t ncore) {
    
    std::vector<WorkBlock> blocks(ncore);
    int64_t total_nnz = csc.nnz;
    int64_t target_nnz_per_block = (total_nnz + ncore - 1) / ncore;
    
    int64_t current_block = 0;
    int64_t current_nnz = 0;
    int64_t start_col = 0;
    
    for (int64_t k = 0; k < csc.K; ++k) {
        int64_t col_nnz = csc.col_ptr[k + 1] - csc.col_ptr[k];
        
        // 如果加上这一列会超过目标，且当前块已有数据，则开始新块
        if (current_nnz + col_nnz > target_nnz_per_block && 
            current_nnz > 0 && 
            current_block < ncore - 1) {
            
            blocks[current_block].start_col = start_col;
            blocks[current_block].end_col = k;
            blocks[current_block].nnz_count = current_nnz;
            
            current_block++;
            start_col = k;
            current_nnz = 0;
        }
        
        current_nnz += col_nnz;
    }
    
    // 最后一块
    blocks[current_block].start_col = start_col;
    blocks[current_block].end_col = csc.K;
    blocks[current_block].nnz_count = current_nnz;
    
    // 调整块数量（可能实际使用的块少于 ncore）
    blocks.resize(current_block + 1);
    
    return blocks;
}

// 计算一个工作块：使用稀疏列数据 + SIMD 加载 weight 行
void compute_work_block_sve(
    const CSCMatrix& csc,
    const float* weight_ptr,
    float* output_ptr,
    int64_t start_col,
    int64_t end_col,
    int64_t M,
    int64_t N) {
    
#if defined(__ARM_FEATURE_SVE)
    const int64_t vl = svcntw();
    
    if (vl == 4) {
        // 遍历该块内的每一列
        for (int64_t k = start_col; k < end_col; ++k) {
            int64_t col_start = csc.col_ptr[k];
            int64_t col_end = csc.col_ptr[k + 1];
            int64_t col_nnz = col_end - col_start;
            
            if (col_nnz == 0) continue;
            
            const float* w_row = weight_ptr + k * N;  // weight 的第 k 行
            
            // 遍历该列的所有非零元素
            for (int64_t idx = col_start; idx < col_end; ++idx) {
                int64_t m = csc.row_indices[idx];  // 行索引
                float a_val = csc.values[idx];      // 非零值
                float* out_row = output_ptr + m * N;
                
                // 使用 SIMD 加载 weight 行，与标量 a_val 相乘并累加到输出
                int64_t n = 0;
                for (; n + 4 <= N; n += 4) {
                    svfloat32_t w_vec = svld1_f32(svptrue_b32(), w_row + n);
                    svfloat32_t out_vec = svld1_f32(svptrue_b32(), out_row + n);
                    
                    // out[m, n:n+4] += a_val * w[k, n:n+4]
                    out_vec = svmla_n_f32_x(svptrue_b32(), out_vec, w_vec, a_val);
                    
                    svst1_f32(svptrue_b32(), out_row + n, out_vec);
                }
                
                // 处理剩余元素
                for (; n < N; ++n) {
                    out_row[n] += a_val * w_row[n];
                }
            }
        }
        return;
    }
#endif
    
    // 标量实现（fallback）
    for (int64_t k = start_col; k < end_col; ++k) {
        int64_t col_start = csc.col_ptr[k];
        int64_t col_end = csc.col_ptr[k + 1];
        
        const float* w_row = weight_ptr + k * N;
        
        for (int64_t idx = col_start; idx < col_end; ++idx) {
            int64_t m = csc.row_indices[idx];
            float a_val = csc.values[idx];
            float* out_row = output_ptr + m * N;
            
            for (int64_t n = 0; n < N; ++n) {
                out_row[n] += a_val * w_row[n];
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
    int64_t ncore = 0) {
    
    // 输入检查
    TORCH_CHECK(activation.device().is_cpu(), "activation must be a CPU tensor");
    TORCH_CHECK(weight.device().is_cpu(), "weight must be a CPU tensor");
    TORCH_CHECK(nz_counts.device().is_cpu(), "nz_counts must be a CPU tensor");
    TORCH_CHECK(nz_col_indices.device().is_cpu(), "nz_col_indices must be a CPU tensor");
    
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "activation must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(nz_counts.dtype() == torch::kInt64, "nz_counts must be int64");
    TORCH_CHECK(nz_col_indices.dtype() == torch::kUInt32 || nz_col_indices.dtype() == torch::kInt32,
                "nz_col_indices must be uint32 or int32");
    
    TORCH_CHECK(activation.dim() == 2, "activation must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(activation.is_contiguous(), "activation must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    
    const int64_t M = activation.size(0);
    const int64_t K = activation.size(1);
    const int64_t N = weight.size(1);
    
    TORCH_CHECK(weight.size(0) == K, "weight K dimension must match activation K");
    
    auto output = torch::zeros({M, N}, activation.options());
    
    if (M == 0 || N == 0 || K == 0) {
        return output;
    }
    
    const float* act_ptr = activation.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const int64_t* counts_ptr = nz_counts.data_ptr<int64_t>();
    const uint32_t* indices_ptr = nz_col_indices.data_ptr<uint32_t>();
    float* out_ptr = output.data_ptr<float>();
    
    const int64_t num_nz_rows = nz_counts.size(0) / 2;
    
    // 如果 ncore <= 0，则使用 OpenMP 的线程数
    if (ncore <= 0) {
        ncore = omp_get_max_threads();
    }
    
    // 步骤 1：转换为 CSC 格式
    CSCMatrix csc = convert_to_csc(
        act_ptr, counts_ptr, indices_ptr, M, K, num_nz_rows);
    
    // 步骤 2：负载均衡分块
    std::vector<WorkBlock> work_blocks = partition_work(csc, ncore);
    
    // 步骤 3：并行计算每个块
    #pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < static_cast<int64_t>(work_blocks.size()); ++i) {
        const auto& block = work_blocks[i];
        compute_work_block_sve(
            csc,
            weight_ptr,
            out_ptr,
            block.start_col,
            block.end_col,
            M,
            N
        );
    }
    
    return output;
}

// 注册到 PyTorch
TORCH_LIBRARY(teal_csc, m) {
    m.def("sve_csc_gemm(Tensor activation, Tensor weight, Tensor nz_counts, Tensor nz_col_indices, int ncore=0) -> Tensor");
}

TORCH_LIBRARY_IMPL(teal_csc, CPU, m) {
    m.impl("sve_csc_gemm", sve_csc_gemm);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
