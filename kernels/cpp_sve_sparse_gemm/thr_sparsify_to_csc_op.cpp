#include <torch/extension.h>

#include <cstdint>
#include <vector>
#include <cmath>
#include <cstring>   // memcpy
#include <algorithm>
#include <atomic>

#ifdef _OPENMP
#include <omp.h>
#endif

using torch::Tensor;

static inline void check_thr_sparsify_to_csc_inputs(const Tensor& activation) {
  TORCH_CHECK(activation.device().is_cpu(), "activation must be a CPU tensor");
  TORCH_CHECK(activation.dtype() == torch::kFloat32, "activation must be float32");
  TORCH_CHECK(activation.is_contiguous(), "activation must be contiguous");
  TORCH_CHECK(activation.dim() == 2, "activation must be 2D [M, K]");
}

// Fast abs without calling libm (bit trick). Works for IEEE-754 float.
static inline float abs_f32_fast(float x) {
  union { float f; uint32_t u; } v;
  v.f = x;
  v.u &= 0x7FFFFFFFu;
  return v.f;
}

/**
 * thr_sparsify_to_csc(activation, threshold) -> (col_ptr, row_indices, values)
 * 
 * 将稠密矩阵根据阈值转换为 CSC (Compressed Sparse Column) 格式的稀疏矩阵。
 * 
 * Args:
 *   activation: (M, K) float32 稠密矩阵
 *   threshold: float 阈值，绝对值 >= threshold 的元素被保留
 * 
 * Returns:
 *   col_ptr: int64 [K+1] 列指针数组（前缀和）
 *   row_indices: uint32 [nnz] 行索引数组
 *   values: float32 [nnz] 非零元素值数组
 * 
 * 实现策略：
 *   Pass 1: 并行统计每列的非零元素数量（按行遍历，线程局部累加）
 *   Pass 2: 计算列指针（前缀和）
 *   Pass 3: 并行填充 CSC 数据（按行遍历，使用原子操作管理写入位置）
 * 
 * 优化技术：
 *   - OpenMP SIMD 自动向量化
 *   - abs_f32_fast 位运算快速绝对值
 *   - 小缓冲区批量写入减少分支开销
 *   - 原子操作保证线程安全的列写入
 */
static std::tuple<Tensor, Tensor, Tensor>
thr_sparsify_to_csc(torch::Tensor activation, double threshold) {
  check_thr_sparsify_to_csc_inputs(activation);

  const int64_t M = activation.size(0);
  const int64_t K = activation.size(1);
  const float* act_ptr = activation.data_ptr<float>();
  const float thr = static_cast<float>(threshold);

  // ---------------- Pass1: count nnz per column ----------------
  // 为了并行化，每个线程维护局部列计数，最后合并
  
  int num_threads = 1;
#ifdef _OPENMP
  #pragma omp parallel
  {
    #pragma omp single
    num_threads = omp_get_num_threads();
  }
#endif

  // 每个线程的局部列计数 [num_threads][K]
  std::vector<std::vector<int64_t>> local_col_counts(num_threads, std::vector<int64_t>(K, 0));

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif
    int64_t* my_counts = local_col_counts[tid].data();

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int64_t m = 0; m < M; ++m) {
      const float* row = act_ptr + m * K;

      // 使用 OpenMP SIMD 帮助编译器向量化
      // 注意：这里不能用 reduction，因为是数组
      for (int64_t k = 0; k < K; ++k) {
        const float ax = abs_f32_fast(row[k]);
        if (ax >= thr) {
          my_counts[k]++;
        }
      }
    }
  }

  // 合并所有线程的列计数
  std::vector<int64_t> col_counts(K, 0);
  for (int t = 0; t < num_threads; ++t) {
    for (int64_t k = 0; k < K; ++k) {
      col_counts[k] += local_col_counts[t][k];
    }
  }

  // ---------------- Pass2: col_ptr prefix sum (K+1) ----------------
  Tensor col_ptr = torch::empty({K + 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* ptr = col_ptr.data_ptr<int64_t>();
  ptr[0] = 0;
  for (int64_t k = 0; k < K; ++k) {
    ptr[k + 1] = ptr[k] + col_counts[k];
  }
  const int64_t total_nnz = ptr[K];

  // ---------------- Allocate CSC arrays: row_indices + values ----------------
  Tensor row_indices = torch::empty({total_nnz}, torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCPU));
  uint32_t* out_row = (total_nnz > 0) ? row_indices.data_ptr<uint32_t>() : nullptr;

  Tensor values = torch::empty({total_nnz}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  float* out_val = (total_nnz > 0) ? values.data_ptr<float>() : nullptr;

  // ---------------- Pass3: write row_indices + values ----------------
  // 使用原子操作管理每列的当前写入位置
  std::vector<std::atomic<int64_t>> write_positions(K);
  for (int64_t k = 0; k < K; ++k) {
    write_positions[k].store(ptr[k], std::memory_order_relaxed);
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t m = 0; m < M; ++m) {
    const float* row = act_ptr + m * K;

    // 小缓冲区用于批量处理同一列的多个元素
    // 虽然一行中同一列只有一个元素，但缓冲区可以减少原子操作次数
    const int BUFSZ = 64;
    struct Entry {
      int64_t col;
      float val;
    };
    Entry buffer[BUFSZ];
    int buf_n = 0;

    // 遍历当前行的所有列
    int64_t k = 0;
    
    // 轻度循环展开（4路）减少循环开销
    for (; k + 4 <= K; k += 4) {
      const float x0 = row[k + 0];
      const float x1 = row[k + 1];
      const float x2 = row[k + 2];
      const float x3 = row[k + 3];

      const bool keep0 = (abs_f32_fast(x0) >= thr);
      const bool keep1 = (abs_f32_fast(x1) >= thr);
      const bool keep2 = (abs_f32_fast(x2) >= thr);
      const bool keep3 = (abs_f32_fast(x3) >= thr);

      if (keep0) { buffer[buf_n].col = k + 0; buffer[buf_n].val = x0; ++buf_n; }
      if (keep1) { buffer[buf_n].col = k + 1; buffer[buf_n].val = x1; ++buf_n; }
      if (keep2) { buffer[buf_n].col = k + 2; buffer[buf_n].val = x2; ++buf_n; }
      if (keep3) { buffer[buf_n].col = k + 3; buffer[buf_n].val = x3; ++buf_n; }

      // 缓冲区快满时刷新
      if (buf_n >= BUFSZ - 4) {
        for (int i = 0; i < buf_n; ++i) {
          int64_t col = buffer[i].col;
          float val = buffer[i].val;
          int64_t pos = write_positions[col].fetch_add(1, std::memory_order_relaxed);
          out_row[pos] = (uint32_t)m;
          out_val[pos] = val;
        }
        buf_n = 0;
      }
    }

    // 处理尾部
    for (; k < K; ++k) {
      const float x = row[k];
      if (abs_f32_fast(x) >= thr) {
        buffer[buf_n].col = k;
        buffer[buf_n].val = x;
        ++buf_n;
        if (buf_n == BUFSZ) {
          for (int i = 0; i < buf_n; ++i) {
            int64_t col = buffer[i].col;
            float val = buffer[i].val;
            int64_t pos = write_positions[col].fetch_add(1, std::memory_order_relaxed);
            out_row[pos] = (uint32_t)m;
            out_val[pos] = val;
          }
          buf_n = 0;
        }
      }
    }

    // 最终刷新缓冲区
    if (buf_n > 0) {
      for (int i = 0; i < buf_n; ++i) {
        int64_t col = buffer[i].col;
        float val = buffer[i].val;
        int64_t pos = write_positions[col].fetch_add(1, std::memory_order_relaxed);
        out_row[pos] = (uint32_t)m;
        out_val[pos] = val;
      }
    }
  }

#ifndef NDEBUG
  // 验证写入位置是否正确
  for (int64_t k = 0; k < K; ++k) {
    TORCH_CHECK(write_positions[k].load(std::memory_order_relaxed) == ptr[k + 1],
                "thr_sparsify_to_csc: write position mismatch at col ", k);
  }
#endif

  return {col_ptr, row_indices, values};
}

// 注册到 PyTorch
// 注意：该文件会与其它算子源文件一起编译到同一个扩展中，
// 因此这里必须使用 TORCH_LIBRARY_FRAGMENT，避免与其它 TU 中的 TORCH_LIBRARY 重复定义冲突。
TORCH_LIBRARY_FRAGMENT(sparse_op, m) {
  m.def("thr_sparsify_to_csc(Tensor activation, float threshold) -> (Tensor col_ptr, Tensor row_indices, Tensor values)");
}

TORCH_LIBRARY_IMPL(sparse_op, CPU, m) {
  m.impl("thr_sparsify_to_csc", thr_sparsify_to_csc);
}
