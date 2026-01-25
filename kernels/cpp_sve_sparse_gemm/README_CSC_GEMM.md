# SVE CSC 稀疏 GEMM 算子

## 概述

该算子实现了基于 **CSC (Compressed Sparse Column)** 格式的稀疏矩阵乘法，专门针对 ARM SVE 指令集优化。

### 主要特性

1. **CSC 格式转换**：使用 scatter store 思想将稀疏 activation 矩阵转换为 CSC 格式
2. **负载均衡分块**：根据非零元素分布，将 weight 矩阵按行分块，均衡分配到多个线程
3. **SIMD 加速**：使用稀疏 activation 非零值标量与 SIMD 加载的 weight 行向量进行乘加运算
4. **并行优化**：支持自定义线程数（ncore），便于后期并行优化

## 算法流程

### 输入
- `activation`: (M, K) 稀疏激活矩阵（以稠密格式传入，但大部分元素为 0）
- `weight`: (K, N) 稠密权重矩阵
- `nz_counts`: (2 * num_nz_rows) 格式为 `[row_idx, count, row_idx, count, ...]`
  - 存储每个有非零元素的行索引及该行的非零元素数量
- `nz_col_indices`: 扁平化的列索引向量
  - 存储所有非零元素的列索引
- `ncore`: 并行线程数（可选，默认 0 表示自动）

### 输出
- `output`: (M, N) 输出矩阵

### 计算步骤

#### 1. CSC 格式转换
将稀疏 activation 从行格式转换为列格式（CSC）：

```
CSC 格式包含：
- values: 所有非零值（按列优先顺序）
- row_indices: 每个非零值对应的行索引
- col_ptr: 每列的起始位置指针（长度为 K+1）
```

转换过程：
1. **统计阶段**：遍历所有非零元素，统计每列的非零元素数量
2. **指针构建**：根据每列的非零数量构建 `col_ptr` 数组
3. **填充阶段**：使用 scatter store 思想，将每个非零值写入对应列的位置

#### 2. 负载均衡分块
根据每列的非零元素数量，将 K 列分成 ncore 个块：

```
目标：使每个块的非零元素数量尽可能接近 total_nnz / ncore
策略：贪心分配，当前块的非零数量超过目标时开始新块
```

#### 3. 并行计算
每个线程处理一个块，对该块内的每一列：

```cpp
for each column k in block:
    w_row = weight[k, :]  // weight 的第 k 行
    
    for each nonzero (row_idx, value) in column k:
        output[row_idx, :] += value * w_row  // 使用 SIMD 加速
```

**SIMD 优化细节**：
- 使用 ARM SVE 指令 `svld1_f32` 加载 weight 行向量（连续内存）
- 标量非零值与向量相乘：`svmla_n_f32_x`（乘加融合指令）
- 累加到输出行向量：`svst1_f32`

## 与原 `sve_sparse_gemm` 的区别

| 特性 | sve_sparse_gemm | sve_csc_gemm |
|------|-----------------|--------------|
| 数据格式 | 行格式（CSR 风格） | 列格式（CSC） |
| 计算顺序 | 按输出行并行 | 按 weight 行（activation 列）分块 |
| 内存访问 | gather load weight | 连续加载 weight 行 |
| SIMD 策略 | activation 向量 × weight gather | activation 标量 × weight 向量 |
| 并行粒度 | (row, col_block) | (col_block) |
| 适用场景 | 稀疏行较少，列稀疏性高 | 稀疏列较少，行稀疏性高 |

## 性能优势

### 理论分析

1. **内存访问优化**
   - 原算子：对每个非零值，需要 gather 访问 weight 的不连续内存
   - 新算子：连续加载 weight 行，充分利用缓存局部性

2. **SIMD 效率**
   - 原算子：gather load 效率较低，受限于内存带宽
   - 新算子：连续 load，SIMD 单元利用率更高

3. **负载均衡**
   - 原算子：按输出行并行，可能负载不均（某些行非零元素很多）
   - 新算子：按非零元素总数均衡分配，充分利用多核

### 适用场景

**推荐使用 CSC GEMM**：
- activation 矩阵列稀疏性较高（大部分列为全零）
- M >> K（输出行数远大于特征维度）
- 需要充分利用多核并行

**推荐使用原 sparse_gemm**：
- activation 矩阵行稀疏性较高（大部分行为全零）
- K >> N（特征维度远大于输出列数）
- 非零行数量适合并行度

## 使用示例

### Python 接口

```python
from kernels.sve_csc_gemm import sve_csc_gemm, SVECSCGEMMKernel
import torch

# 1. 直接调用函数
M, K, N = 256, 512, 1024
activation = torch.randn(M, K)  # 稀疏矩阵
weight = torch.randn(K, N)

# 构建稀疏元数据
nz_counts = torch.tensor([0, 128, 5, 64, ...], dtype=torch.int64)  # [row, count, ...]
nz_col_indices = torch.tensor([3, 7, 12, ...], dtype=torch.uint32)

output = sve_csc_gemm(activation, weight, nz_counts, nz_col_indices, ncore=4)

# 2. 使用 Kernel 类（支持 torch.compile）
kernel = SVECSCGEMMKernel()
output = kernel(activation, weight, nz_counts, nz_col_indices, ncore=0)  # auto
```

### 测试和性能评估

```bash
# 正确性测试
python scripts/test_sve_csc_gemm.py --test correctness

# 性能测试（不同并行度）
python scripts/test_sve_csc_gemm.py --test performance --M 512 --K 1024 --N 2048 --ncore 1 2 4 8

# 不同稀疏度性能测试
python scripts/test_sve_csc_gemm.py --test sparsity --M 256 --K 512 --N 1024

# 完整测试
python scripts/test_sve_csc_gemm.py --test all --M 256 --K 512 --N 1024 --sparsity 0.9
```

## 编译要求

- **编译器**：GCC 或 Clang，支持 C++17
- **架构**：ARM AArch64 with SVE support
- **依赖**：
  - PyTorch >= 1.13
  - OpenMP (可选，用于并行)

编译标志：
```bash
-std=c++17 -O3 -march=armv8-a+sve -fopenmp
```

## 实现细节

### CSC 转换时间复杂度
- 统计阶段：O(nnz)
- 构建指针：O(K)
- 填充阶段：O(nnz)
- **总计**：O(nnz + K)

### 计算时间复杂度
- 遍历所有非零元素：O(nnz)
- 每个非零值需要处理 N 个输出：O(nnz × N)
- **总计**：O(nnz × N)

### 内存占用
- CSC 数据结构：
  - `values`: nnz × 4 bytes
  - `row_indices`: nnz × 8 bytes
  - `col_ptr`: (K+1) × 8 bytes
- **总计**：约 12 × nnz + 8 × K bytes

## 性能调优建议

1. **选择合适的 ncore**
   - 一般设为 CPU 物理核心数
   - 对于超线程 CPU，可以尝试核心数的 1.5-2 倍
   - 使用 `ncore=0` 让 OpenMP 自动决定

2. **矩阵布局**
   - 确保 activation 和 weight 是连续存储的（contiguous）
   - 考虑使用列优先存储以提高缓存命中率

3. **批量处理**
   - 如果有多个 activation 矩阵，可以考虑批量转换 CSC 格式
   - 重用 CSC 格式数据以分摊转换开销

4. **编译优化**
   - 使用 `-O3` 优化级别
   - 启用 LTO (Link Time Optimization): `-flto`
   - 针对特定 CPU：`-mcpu=native`

## 未来改进方向

1. **缓存优化**
   - 实现分块计算，提高 L1/L2 缓存命中率
   - 考虑列分块 + 行分块的二维分块策略

2. **并行优化**
   - 支持跨节点分布式计算
   - 使用任务队列避免静态负载不均

3. **混合精度**
   - 支持 FP16 计算以提高吞吐量
   - 实现混合精度累加（FP16 计算 + FP32 累加）

4. **自适应策略**
   - 根据稀疏模式自动选择最优算法（CSC vs CSR）
   - 运行时profiling 和自动调优

## 参考资料

- ARM SVE Programming Guide: https://developer.arm.com/architectures/instruction-sets/simd-isas/sve
- Sparse Matrix Storage Formats: https://en.wikipedia.org/wiki/Sparse_matrix
- OpenMP API Specification: https://www.openmp.org/specifications/
