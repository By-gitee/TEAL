# Decode 阶段稀疏算子集成说明

## 概述

已完成在 `gpt-fast/generate.py` 中集成 decode 阶段自定义稀疏算子的支持。decode 阶段会自动使用在 `model.py` 中配置的稀疏 GEMV 算子。

## 修改内容

### 1. generate.py 文件头部

添加了详细的文档字符串，说明稀疏算子的使用方式：

- **Prefill 阶段** (seq_len > 1): 通过环境变量 `USE_CUSTOM_SPARSE_GEMM=1` 启用自适应稀疏 GEMM
- **Decode 阶段** (seq_len = 1): 通过 `--hist_path` 和 `--sparsity` 参数在 `_load_model` 中配置稀疏 GEMV

### 2. 关键函数注释更新

#### `prefill()` 函数
```python
def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    """
    Prefill 阶段 (seq_len > 1)。
    
    如果启用了 USE_CUSTOM_SPARSE_GEMM，会自动使用 adaptive_sparse_gemm 进行自适应稀疏计算。
    否则使用标准的 PyTorch Linear 或 monkeypatch 的 GEMV 算子。
    """
```

#### `decode_one_token()` 函数
```python
def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs):
    # decode 阶段 (seq_len=1) 会自动使用 model.py 中的 GEMV 稀疏算子
```

#### `decode_n_tokens()` 函数
```python
def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, ...):
    """
    解码多个 token。
    
    在 decode 阶段 (seq_len=1)，会自动使用 model.py 中配置的稀疏 GEMV 算子：
    - 如果 monkeypatch 已启用，则使用 SparseGEMV/DenseGEMV
    - 否则回退到标准的 PyTorch Linear
    """
```

#### `speculative_decode()` 函数
```python
def speculative_decode(model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs):
    """
    推测解码（Speculative Decoding）。
    
    draft_model 和 model 都会自动使用配置的稀疏算子：
    - decode 阶段使用 monkeypatch 的 GEMV 算子
    - 如果启用了 USE_CUSTOM_SPARSE_GEMM，prefill 阶段使用自适应稀疏 GEMM
    """
```

#### `generate()` 函数
```python
def generate(model, prompt, max_new_tokens, ...):
    """
    基于条件序列（prompt）生成指定数量的 token。
    
    稀疏算子说明:
        - Prefill 阶段: 如果 USE_CUSTOM_SPARSE_GEMM=1，使用 adaptive_sparse_gemm
        - Decode 阶段: 如果 model 已经通过 monkeypatch 配置，使用 SparseGEMV/DenseGEMV
        - 所有这些逻辑都在 model.forward 中自动处理，无需在此显式调用
    """
```

## 工作原理

### Decode 阶段稀疏算子使用流程

1. **模型加载时配置** (`_load_model` 函数)：
   - 如果提供了 `--hist_path` 和 `--sparsity` 参数
   - 对每个 layer 调用 `monkeypatch_layer()`
   - 注入 `SparseGEMV` 或 `DenseGEMV` 算子到 layer 的属性中
   - 替换 `forward` 方法为 `_new_attn_forward` 和 `_new_ffn_forward`

2. **Decode 时自动使用** (在 `model.py` 中)：
   - `decode_one_token()` 调用 `model(x, input_pos)`
   - 当 `seqlen == 1` 时，`_new_attn_forward` 和 `_new_ffn_forward` 检测到 decode 阶段
   - 自动调用 `self.gemv1()` 和 `self.gemv2()` 使用稀疏算子
   - 例如：`q, k, v = self.gemv1(x, self.wqkv.weight, self.thresh_q, ...)`

3. **无需显式调用**：
   - `generate.py` 中的所有函数都直接调用 `model.forward()`
   - 不需要在 `generate.py` 中显式调用稀疏算子
   - 所有逻辑都封装在 `model.py` 的 monkeypatch 中

### Prefill 阶段稀疏算子使用流程

1. **设置环境变量**：
   ```bash
   export USE_CUSTOM_SPARSE_GEMM=1
   ```

2. **自动使用 adaptive_sparse_gemm**：
   - 当 `seqlen > 1` 时，`_new_attn_forward` 和 `_new_ffn_forward` 检测到 prefill 阶段
   - 自动调用 `adaptive_sparse_gemm()` 函数
   - 根据激活稀疏度动态选择 dense/sparse kernel

## 使用示例

### 示例 1: 仅启用 Decode 阶段稀疏算子

```bash
python gpt-fast/generate.py \
    --checkpoint_path checkpoints/meta-Transformer/Transformer-2-7b/model.pth \
    --hist_path checkpoints/meta-Transformer/Transformer-2-7b/histograms \
    --sparsity 0.5 \
    --max_new_tokens 100
```

### 示例 2: 同时启用 Prefill 和 Decode 阶段稀疏算子

```bash
export USE_CUSTOM_SPARSE_GEMM=1
export SPARSE_DEBUG=1  # 可选: 启用调试输出

python gpt-fast/generate.py \
    --checkpoint_path checkpoints/meta-Transformer/Transformer-2-7b/model.pth \
    --hist_path checkpoints/meta-Transformer/Transformer-2-7b/histograms \
    --sparsity 0.5 \
    --max_new_tokens 100
```

### 示例 3: 推测解码（Speculative Decoding）

```bash
export USE_CUSTOM_SPARSE_GEMM=1

python gpt-fast/generate.py \
    --checkpoint_path checkpoints/meta-Transformer/Transformer-2-7b/model.pth \
    --draft_checkpoint_path checkpoints/meta-Transformer/Transformer-2-1b/model.pth \
    --hist_path checkpoints/meta-Transformer/Transformer-2-7b/histograms \
    --sparsity 0.5 \
    --speculate_k 5 \
    --max_new_tokens 100
```

## 测试脚本

创建了 `scripts/test_decode_sparse_operator.py` 用于测试 decode 阶段稀疏算子：

```bash
# 基础测试（不需要真实 checkpoint）
python scripts/test_decode_sparse_operator.py

# 带有真实 checkpoint 的测试
python gpt-fast/generate.py \
    --checkpoint_path <your_checkpoint> \
    --hist_path <your_histograms> \
    --sparsity 0.5 \
    --max_new_tokens 10
```

测试脚本会验证：
- Prefill 阶段是否正确处理多 token 输入
- Decode 阶段是否正确处理单 token 输入
- 稀疏算子是否被正确调用
- 统计信息是否正确收集

## 关键设计决策

1. **自动化**：稀疏算子的使用完全自动化，不需要在 `generate.py` 中显式调用

2. **向后兼容**：如果不提供 `--hist_path`，代码会回退到标准 PyTorch Linear

3. **灵活性**：
   - Prefill 和 Decode 阶段可以独立启用
   - 通过环境变量和命令行参数灵活控制

4. **调试友好**：
   - 通过 `SPARSE_DEBUG=1` 可以查看详细的调度信息
   - 通过 `DenseGEMV.print_statistics()` 可以查看统计信息

## 相关文件

- `gpt-fast/generate.py`: 生成主逻辑（已修改）
- `gpt-fast/model.py`: 模型定义和 monkeypatch 逻辑（包含 adaptive_sparse_gemm）
- `kernels/sparse_gemv.py`: GEMV 稀疏算子实现
- `kernels/sve_sparse_gemm.py`: GEMM 稀疏算子实现
- `scripts/test_decode_sparse_operator.py`: 测试脚本（新增）

## 验证清单

- ✅ Decode 阶段自动使用稀疏 GEMV 算子
- ✅ Prefill 阶段可选使用自适应稀疏 GEMM
- ✅ 推测解码支持稀疏算子
- ✅ 向后兼容标准 PyTorch Linear
- ✅ 添加了详细的文档注释
- ✅ 创建了测试脚本
- ✅ 支持 CPU 和 CUDA 设备

## 总结

通过此次修改，`generate.py` 现在完整支持 decode 阶段的自定义稀疏算子。所有的 decode 函数（`decode_one_token`、`decode_n_tokens`、`speculative_decode`）都会自动使用在模型加载时配置的稀疏算子，无需任何额外的代码修改。
