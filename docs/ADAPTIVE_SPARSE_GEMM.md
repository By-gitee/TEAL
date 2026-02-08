# 自适应稀疏 GEMM 实现说明

## 📋 概述

在 gpt-fast + sve_sparse_gemm 基础上实现了 **自适应稀疏 GEMM 路径**，专用于 **prefill 阶段** (seq_len > 1)。

该实现根据 activation 的实际稀疏度，动态调度 **dense GEMM** (torch.matmul) 和 **sparse GEMM** (sve_sparse_gemm) 路径，实现性能优化。

---

## 🎯 核心特性

### 1. 自适应调度策略

```
对每一行 activation:
  计算 sparsity = 1 - (nnz / K)
  
  if sparsity < 0.7 && 连续行数 >= 4:
    → 使用 torch.matmul (dense GEMM)
  else:
    → 使用 sparse_gemm_icsr_sve_gather (sparse GEMM)
```

### 2. 阶段分离

- **Prefill (seq_len > 1)**: 使用自适应稀疏 GEMM
- **Decode (seq_len == 1)**: 保持原有 GEMV 路径 (不改动)

### 3. 总开关

```python
# 环境变量控制
USE_CUSTOM_SPARSE_GEMM = True / False  # 通过 os.getenv("USE_CUSTOM_SPARSE_GEMM")
```

当 `False` 时，完全回退原有 baseline，不影响已有行为。

---

## ⚙️ 配置参数

在 `gpt-fast/model.py` 中定义:

```python
USE_CUSTOM_SPARSE_GEMM = os.getenv("USE_CUSTOM_SPARSE_GEMM", "0") == "1"
DENSE_THRESHOLD = 0.7    # 稀疏度阈值
MIN_DENSE_BLOCK = 4      # 最小 dense block 长度
SPARSE_DEBUG = os.getenv("SPARSE_DEBUG", "0") == "1"
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DENSE_THRESHOLD` | 0.7 | 稀疏度 < 0.7 的行被视为 dense |
| `MIN_DENSE_BLOCK` | 4 | 至少连续 4 行才使用 dense GEMM |
| `SPARSE_DEBUG` | False | 启用调试输出 (显示 dense/sparse 划分) |

---

## 📁 代码结构

### A) `gpt-fast/model.py`

#### 1. `adaptive_sparse_gemm(x, weight, threshold)`

核心自适应 GEMM 函数:

```python
def adaptive_sparse_gemm(x, weight, threshold):
    """
    Args:
        x: (M, K) activation
        weight: (K, N) weight
        threshold: sparsification threshold
    
    Returns:
        output: (M, N) result
    """
    # 1. 稀疏化 activation
    nz_counts, nz_col_indices, row_offsets = thr_sparsify_to_icsr_sve(x, threshold)
    
    # 2. 计算每行 nnz 和稀疏度
    # 3. 划分 dense/sparse blocks
    # 4. 执行计算并拼接结果
    #    - Dense blocks: torch.matmul
    #    - Sparse blocks: sparse_gemm_icsr_sve_gather
```

#### 2. `_new_attn_forward` (Attention 层)

```python
def _new_attn_forward(self, x, ...):
    if USE_CUSTOM_SPARSE_GEMM and seqlen > 1:
        # Prefill: 自适应稀疏 GEMM
        qkv = adaptive_sparse_gemm(...)
    elif seqlen == 1:
        # Decode: 原 GEMV
        qkv = self.gemv1(...)
    else:
        # Baseline
        qkv = self.wqkv(x)
```

#### 3. `_new_ffn_forward` (FeedForward 层)

```python
def _new_ffn_forward(self, x):
    if USE_CUSTOM_SPARSE_GEMM and seqlen > 1:
        # Prefill: 自适应稀疏 GEMM
        gate = adaptive_sparse_gemm(x_flat, self.w1.weight, ...)
        ...
    elif seqlen == 1:
        # Decode: 原 GEMV
        return self.gemv2(...)
    else:
        # Baseline
        return self.w2(...)
```

### B) `kernels/sve_sparse_gemm.py`

使用已有的:
- `thr_sparsify_to_icsr_sve`: 稀疏化函数
- `SparseGEMMiCSRSVEGatherKernel`: sparse GEMM kernel

### C) `scripts/test_adaptive_sparse_gemm.py`

测试脚本，包含:
- 正确性测试
- 调度逻辑测试
- 多配置测试
- 边界情况测试

---

## 🧪 使用方法

### 1. 测试自适应 GEMM 实现

```bash
# 基础测试
python -m scripts.test_adaptive_sparse_gemm

# 启用调试输出
SPARSE_DEBUG=1 python -m scripts.test_adaptive_sparse_gemm
```

### 2. 在 gpt-fast 中使用

#### 启用自适应稀疏 GEMM

```bash
# 启用自适应稀疏 GEMM
export USE_CUSTOM_SPARSE_GEMM=1

# 启用调试输出 (可选)
export SPARSE_DEBUG=1

# 运行推理
cd gpt-fast
python generate.py \
    --checkpoint_path checkpoints/your_model/model.pth \
    --prompt "Hello, world" \
    --max_new_tokens 100 \
    --hist_path path/to/histogram.json \
    --sparsity 0.3
```

#### 禁用自适应稀疏 GEMM (回退 baseline)

```bash
# 方法 1: 不设置环境变量 (默认)
python generate.py ...

# 方法 2: 显式设置为 0
export USE_CUSTOM_SPARSE_GEMM=0
python generate.py ...
```

### 3. 调试输出示例

启用 `SPARSE_DEBUG=1` 后，会输出:

```
[adaptive_sparse_gemm] M=16, K=4096, N=4096
  dense_blocks: [(0, 6), (10, 16)], total_rows: 12
  sparse_blocks: [(6, 10)], total_rows: 4
  nnz_per_row: [3276, 3189, 3245, ..., 410, 398, 425, ...]
```

解释:
- 行 0-5: dense (连续 6 行, 稀疏度低)
- 行 6-9: sparse (连续 4 行, 稀疏度高)
- 行 10-15: dense (连续 6 行, 稀疏度低)

---

## 🔍 调度逻辑详解

### 算法流程

```python
1. 稀疏化: thr_sparsify_to_icsr_sve(activation, threshold)
   → 得到 {nz_counts, nz_col_indices, row_offsets}

2. 计算每行 nnz:
   for m in range(M):
       nnz[m] = row_offsets[m+1] - row_offsets[m]
   
3. 划分 dense/sparse blocks:
   i = 0
   while i < M:
       sparsity = 1 - (nnz[i] / K)
       
       if sparsity < DENSE_THRESHOLD:
           # 尝试扩展 dense block
           j = i + 1
           while j < M and (1 - nnz[j]/K) < DENSE_THRESHOLD:
               j += 1
           
           if j - i >= MIN_DENSE_BLOCK:
               → dense_blocks.append((i, j))
           else:
               → sparse_blocks.append((i, j))
           i = j
       else:
           # 扩展 sparse block
           j = i + 1
           while j < M and (1 - nnz[j]/K) >= DENSE_THRESHOLD:
               j += 1
           → sparse_blocks.append((i, j))
           i = j

4. 执行计算:
   for (start, end) in dense_blocks:
       output[start:end] = torch.matmul(x[start:end], weight)
   
   for (start, end) in sparse_blocks:
       output[start:end] = sparse_gemm_icsr_sve_gather(...)
```

### 示例

假设 `M=10`, `K=1000`, `DENSE_THRESHOLD=0.7`, `MIN_DENSE_BLOCK=4`:

```
行  nnz  稀疏度  判定
0   800  0.20   dense
1   820  0.18   dense
2   790  0.21   dense
3   810  0.19   dense
4   150  0.85   sparse  ← 不满足 MIN_DENSE_BLOCK=4 (只有 4 行)
5   140  0.86   sparse
6   130  0.87   sparse
7   800  0.20   dense
8   810  0.19   dense
9   790  0.21   dense

结果:
  dense_blocks: [(0, 4)]  (前 4 行)
  sparse_blocks: [(4, 7), (7, 10)]  
  注: (7, 10) 虽然是 dense，但只有 3 行 < MIN_DENSE_BLOCK，所以划为 sparse
```

---

## ⚠️ 注意事项

### 1. Decode 阶段不改动

```python
if seqlen == 1:
    # 保持原有 GEMV 路径
    output = self.gemv1(...)
```

确保 decode 性能不受影响。

### 2. Threshold 参数

`adaptive_sparse_gemm` 中的 `threshold` 应该与原有 `sparse_gemv` 使用的 threshold 一致:

```python
# Attention
self.thresh_q, self.thresh_k, self.thresh_v, self.thresh_o

# FeedForward
self.thresh_gate, self.thresh_up, self.thresh_down
```

### 3. 内存开销

- `thr_sparsify_to_icsr_sve` 会分配稀疏格式存储
- Dense blocks 直接复用原 activation
- 总体内存开销 = 稀疏格式 + 输出矩阵

### 4. 性能权衡

- **Dense block 优势**: torch.matmul 高度优化 (BLAS/oneDNN)
- **Sparse block 优势**: 高稀疏度时减少计算量
- **最佳场景**: 混合稀疏度 (部分 dense, 部分 sparse)

---

## 📊 性能评估

### 预期性能表现

| 场景 | 稀疏度 | 预期加速 |
|------|--------|---------|
| 全 dense (< 0.3) | < 30% | ~1.0x (接近 baseline) |
| 混合 (0.3-0.7) | 30-70% | ~1.2-1.5x |
| 高稀疏 (> 0.7) | > 70% | ~1.5-2.5x |

### 测量方法

```bash
# Baseline (无稀疏)
python generate.py \
    --checkpoint_path ... \
    --sparsity 0 \
    --max_new_tokens 100

# 自适应稀疏 GEMM
export USE_CUSTOM_SPARSE_GEMM=1
python generate.py \
    --checkpoint_path ... \
    --hist_path histogram.json \
    --sparsity 0.5 \
    --max_new_tokens 100

# 对比 tokens/sec 和 latency
```

---

## 🐛 调试指南

### 1. 启用 Debug 输出

```bash
export SPARSE_DEBUG=1
python generate.py ...
```

### 2. 检查 Block 划分

查看输出:
```
[adaptive_sparse_gemm] M=16, K=4096, N=4096
  dense_blocks: [(0, 8)]
  sparse_blocks: [(8, 16)]
  nnz_per_row: [3276, 3189, ..., 410, 398]
```

### 3. 验证正确性

```bash
# 对比输出是否与 baseline 一致
python -m scripts.test_adaptive_sparse_gemm
```

### 4. 性能分析

```bash
# 使用 torch profiler
python generate.py --profile output.json ...

# 查看 adaptive_sparse_gemm 占比
# 分析 torch.matmul vs sparse_gemm_icsr_sve_gather 的时间
```

---

## 🔧 参数调优

### 调整 DENSE_THRESHOLD

```python
# model.py
DENSE_THRESHOLD = 0.5  # 更激进的 sparse (更多行走 sparse 路径)
DENSE_THRESHOLD = 0.8  # 更保守的 sparse (更多行走 dense 路径)
```

### 调整 MIN_DENSE_BLOCK

```python
# model.py
MIN_DENSE_BLOCK = 2  # 更小的 dense block (更细粒度划分)
MIN_DENSE_BLOCK = 8  # 更大的 dense block (减少调度开销)
```

### 建议

- **高稀疏度模型 (> 0.7)**: `DENSE_THRESHOLD=0.6`, `MIN_DENSE_BLOCK=4`
- **中等稀疏度 (0.4-0.7)**: `DENSE_THRESHOLD=0.7`, `MIN_DENSE_BLOCK=4`
- **低稀疏度 (< 0.4)**: 建议禁用自适应 GEMM

---

## 📚 API 文档

### `adaptive_sparse_gemm`

```python
def adaptive_sparse_gemm(
    x: torch.Tensor,      # (M, K) activation
    weight: torch.Tensor, # (K, N) weight
    threshold: float,     # sparsification threshold
) -> torch.Tensor:       # (M, N) output
    """
    自适应稀疏 GEMM，根据 activation 稀疏度动态调度 dense/sparse kernel。
    
    调度策略:
      - 连续 >= MIN_DENSE_BLOCK 行且稀疏度 < DENSE_THRESHOLD → dense GEMM
      - 否则 → sparse GEMM
    
    Args:
        x: Activation matrix (M, K)
        weight: Weight matrix (K, N)
        threshold: Sparsification threshold for thr_sparsify_to_icsr_sve
    
    Returns:
        Output matrix (M, N)
    
    Raises:
        AssertionError: 如果 x.shape[1] != weight.shape[0]
    """
```

---

## 🚀 后续优化方向

1. **动态参数调优**: 根据实际 profiling 自动调整 `DENSE_THRESHOLD` 和 `MIN_DENSE_BLOCK`
2. **并行化**: Dense/sparse blocks 可能并行执行 (需要同步)
3. **Kernel fusion**: 将 sparsification + GEMM 融合为单个 kernel
4. **Cache 优化**: 重用 sparsification 结果 (如果 activation 模式稳定)

---

## 📝 许可证

遵循项目原有许可证。

## 🙏 致谢

基于 Meta gpt-fast 和 ARM SVE sparse kernel 实现。
