# Decode 阶段稀疏算子使用指南

## 快速开始

### 1. 仅启用 Decode 阶段稀疏算子

```bash
python gpt-fast/generate.py \
    --checkpoint_path checkpoints/model.pth \
    --hist_path checkpoints/histograms \
    --sparsity 0.5 \
    --max_new_tokens 100
```

**说明**：
- `--hist_path`: 激活分布直方图路径（用于计算稀疏阈值）
- `--sparsity`: 稀疏度级别（0-1 之间，推荐 0.3-0.7）

### 2. 同时启用 Prefill 和 Decode 阶段

```bash
export USE_CUSTOM_SPARSE_GEMM=1

python gpt-fast/generate.py \
    --checkpoint_path checkpoints/model.pth \
    --hist_path checkpoints/histograms \
    --sparsity 0.5 \
    --max_new_tokens 100
```

**说明**：
- `USE_CUSTOM_SPARSE_GEMM=1`: 启用 prefill 阶段的自适应稀疏 GEMM

## 工作原理

### Decode 阶段（seq_len = 1）

当你运行 `generate.py` 时：

1. **模型加载**（`_load_model` 函数）：
   - 读取 `--hist_path` 和 `--sparsity` 参数
   - 为每个 layer 注入 `SparseGEMV` 或 `DenseGEMV` 算子
   - 替换 forward 方法为包含稀疏逻辑的版本

2. **自动使用稀疏算子**：
   - `decode_one_token()` → `model(x, input_pos)`
   - 检测到 `seq_len == 1` → 自动调用稀疏 GEMV
   - 无需在 `generate.py` 中显式调用

3. **关键代码**（在 `model.py` 中）：
   ```python
   # _new_attn_forward 函数
   if seqlen == 1:
       # Decode 阶段：使用稀疏 GEMV
       q, k, v = self.gemv1(x, self.wqkv.weight, self.thresh_q, ...)
   ```

### Prefill 阶段（seq_len > 1）

启用 `USE_CUSTOM_SPARSE_GEMM=1` 后：

```python
# _new_attn_forward 函数
if USE_CUSTOM_SPARSE_GEMM and seqlen > 1:
    # Prefill 阶段：使用自适应稀疏 GEMM
    qkv = adaptive_sparse_gemm(x, self.wqkv.weight, self.thresh_q)
```

## 调试与监控

### 启用调试输出

```bash
export SPARSE_DEBUG=1
export USE_CUSTOM_SPARSE_GEMM=1

python gpt-fast/generate.py ...
```

会输出：
- 每个 block 的 dense/sparse 划分情况
- 每行的 nnz（非零元素数量）
- 调度决策的详细信息

### 查看统计信息

运行结束后会自动打印：
```
SparseGEMV 统计信息:
  Total calls: 1234
  Average sparsity: 0.65
  ...
```

## 测试

### 快速测试

```bash
python scripts/test_decode_sparse_operator.py
```

### 带真实 checkpoint 测试

```bash
python gpt-fast/generate.py \
    --checkpoint_path <your_checkpoint> \
    --hist_path <your_histograms> \
    --sparsity 0.5 \
    --max_new_tokens 10
```

## 关键点

1. **自动化**：稀疏算子的使用完全自动，无需修改 `generate.py` 的调用代码

2. **灵活配置**：
   - Decode 阶段：通过 `--hist_path` + `--sparsity` 启用
   - Prefill 阶段：通过 `USE_CUSTOM_SPARSE_GEMM=1` 启用
   - 可独立控制

3. **向后兼容**：不提供 `--hist_path` 时自动回退到标准 PyTorch Linear

4. **性能建议**：
   - 推荐稀疏度：0.3-0.7
   - CPU: 使用 SparseGEMV（已优化 SVE 指令）
   - 小 batch 推理时效果最好

## 常见问题

**Q: 如何知道稀疏算子是否生效？**

A: 运行时会看到类似输出：
```
Monkeypatching with activation sparsity...
SparseGEMV initialized for layer 0
...
```

**Q: 是否支持推测解码（Speculative Decoding）？**

A: 支持，draft_model 和 target_model 都会自动使用稀疏算子：
```bash
python gpt-fast/generate.py \
    --checkpoint_path checkpoints/target_model.pth \
    --draft_checkpoint_path checkpoints/draft_model.pth \
    --hist_path checkpoints/histograms \
    --sparsity 0.5 \
    --speculate_k 5
```

**Q: 为什么我的 decode 没有加速？**

A: 可能的原因：
1. 未提供 `--hist_path` 参数
2. 稀疏度不够高（激活值不够稀疏）
3. Batch size 太大（稀疏算子针对小 batch 优化）

## 相关文档

- 详细技术文档：`docs/DECODE_SPARSE_OPERATOR_INTEGRATION.md`
- 自适应稀疏 GEMM：`README_ADAPTIVE_SPARSE_GEMM.md`
- 实现报告：`IMPLEMENTATION_SUMMARY.md`
