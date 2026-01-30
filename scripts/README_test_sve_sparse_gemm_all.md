# SVE 稀疏 GEMM 综合测试脚本使用说明

## 概述

`test_sve_sparse_gemm_all.py` 是一个综合测试脚本，用于测试不同稀疏矩阵格式的稀疏化算子和GEMM算子的所有组合。

## 测试的算子组合

### 自定义 SVE 算子（13个组合）

### iCSR 格式（4个组合）
1. `thr_sparsify_to_icsr` + `sparse_gemm_icsr_sve_gather`
2. `thr_sparsify_to_icsr_sve` + `sparse_gemm_icsr_sve_gather`
3. `thr_sparsify_to_icsr` + `sparse_gemm_icsr`
4. `thr_sparsify_to_icsr_sve` + `sparse_gemm_icsr`

### CSR 格式（4个组合）
1. `thr_sparsify_to_csr` + `sparse_gemm_csr`
2. `thr_sparsify_to_csr_sve` + `sparse_gemm_csr`
3. `thr_sparsify_to_csr` + `sparse_gemm_csr_sve_gather`
4. `thr_sparsify_to_csr_sve` + `sparse_gemm_csr_sve_gather`

### COO 格式（4个组合）
1. `thr_sparsify_to_coo` + `sparse_gemm_coo`
2. `thr_sparsify_to_coo_sve` + `sparse_gemm_coo`
3. `thr_sparsify_to_coo` + `sparse_gemm_coo_sve_gather`
4. `thr_sparsify_to_coo_sve` + `sparse_gemm_coo_sve_gather`

### CSC 格式（1个组合）
1. `thr_sparsify_to_csc` + `sparse_gemm_csc`

### PyTorch 参考实现（4个）
1. **PyTorch 稠密 matmul**：直接应用阈值后使用 torch.matmul
2. **PyTorch 稀疏 CSR + sparse.mm**：将activation转换为CSR格式后使用 torch.sparse.mm
3. **PyTorch 稀疏 CSC + sparse.mm**：将activation转换为CSC格式后使用 torch.sparse.mm
4. **PyTorch 选择性加载 weight 非零行 + matmul**：
   - 对每行activation，找出非零列索引
   - 只从weight中选择对应的行（通过索引）
   - 进行小规模的matmul：`(1, nnz) @ (nnz, N) -> (1, N)`
   - 优点：减少内存访问，避免加载不需要的weight数据
   - 适用场景：activation稀疏度高时效果更好

**总计：17 个算子组合**

## 运行方式

### 基本运行
```bash
python -m scripts.test_sve_sparse_gemm_all
```

### 带参数运行
```bash
python -m scripts.test_sve_sparse_gemm_all --seed 42 --threshold 0.8 --M 16 --K 512 --N 1024
```

## 命令行参数

- `--seed`: 随机种子（默认：42）
- `--threshold`: 稀疏化阈值，绝对值大于此值的元素被保留（默认：0.8）
- `--M`: activation矩阵的行数（默认：16）
- `--K`: activation矩阵的列数 / weight矩阵的行数（默认：512）
- `--N`: weight矩阵的列数（默认：1024）

## 测试内容

该脚本对所有算子组合执行以下测试：

1. **正确性验证**：将每个组合的结果与参考实现（PyTorch的matmul）进行比较
2. **性能测量**：测量每个组合的平均延迟
3. **性能排名**：按延迟对所有组合进行排名，找出最快的组合
4. **加速比计算**：计算相对于PyTorch稠密实现的加速比
5. **统计分析**：统计有多少自定义算子比PyTorch实现更快

## 输出示例

```
================================================================================
ARM SVE 稀疏 GEMM 算子综合测试
================================================================================
配置参数:
  - 随机种子: 42
  - 阈值: 0.8
  - 矩阵尺寸: activation (16, 512), weight (512, 1024)
  - 稀疏度: 75.2% (2048/8192 非零元素)

================================================================================
测试 iCSR 格式组合
================================================================================

[iCSR-1] thr_sparsify_to_icsr + sparse_gemm_icsr_sve_gather
  延迟: 0.1234 ms

[iCSR-2] thr_sparsify_to_icsr_sve + sparse_gemm_icsr_sve_gather
  延迟: 0.0987 ms

...

================================================================================
测试 PyTorch 参考实现
================================================================================

[PyTorch-1] 稠密 matmul
  延迟: 0.5432 ms

[PyTorch-2] 稀疏 CSR + sparse.mm
  延迟: 0.3210 ms

[PyTorch-3] 稀疏 CSC + sparse.mm
  延迟: 0.3456 ms

[PyTorch-4] 选择性加载 weight 非零行 + matmul
  延迟: 0.2876 ms

================================================================================
正确性验证
================================================================================

iCSR:
  ✅ iCSR-1: thr_sparsify_to_icsr + sparse_gemm_icsr_sve_gather
      最大误差: 1.234e-05, 平均误差: 2.345e-06
  ...

PyTorch:
  ✅ PyTorch-1: 稠密 matmul
      最大误差: 0.000e+00, 平均误差: 0.000e+00
  ...

================================================================================
性能对比总结
================================================================================

延迟排名（从快到慢）：
--------------------------------------------------------------------------------------
排名 算子组合                                                      延迟(ms)     加速比    
--------------------------------------------------------------------------------------
 1. 🚀 iCSR-2: thr_sparsify_to_icsr_sve + sparse_gemm_icsr_sve_gather  0.0987 ms   5.50x
 2. 🚀 CSR-4: thr_sparsify_to_csr_sve + sparse_gemm_csr_sve_gather     0.1023 ms   5.31x
 3. 🚀 COO-4: thr_sparsify_to_coo_sve + sparse_gemm_coo_sve_gather     0.1156 ms   4.70x
 ...
13. 📊 PyTorch-4: 选择性加载 weight 非零行 + matmul                    0.2876 ms   1.89x
14. 📊 PyTorch-2: 稀疏 CSR + sparse.mm                                 0.3210 ms   1.69x
15. 📊 PyTorch-3: 稀疏 CSC + sparse.mm                                 0.3456 ms   1.57x
16. 📊 PyTorch-1: 稠密 matmul                                          0.5432 ms   1.00x

================================================================================
⚡ 最快的组合: iCSR-2: thr_sparsify_to_icsr_sve + sparse_gemm_icsr_sve_gather
   延迟: 0.0987 ms
   相比PyTorch稠密实现加速比: 5.50x
================================================================================

================================================================================
自定义算子性能统计
================================================================================

最快的自定义算子: iCSR-2: thr_sparsify_to_icsr_sve + sparse_gemm_icsr_sve_gather
  延迟: 0.0987 ms
  相比PyTorch稠密实现加速比: 5.50x

比PyTorch稠密实现更快的自定义算子数量: 13/13

PyTorch最快的实现（不含稠密）: PyTorch-4: 选择性加载 weight 非零行 + matmul
  延迟: 0.2876 ms

比PyTorch最快非稠密实现更快的自定义算子数量: 10/13
================================================================================
```

## 注意事项

1. 所有算子组合（包括PyTorch参考实现）使用相同的随机生成的activation、threshold、weight进行测试，确保公平比较
2. 测试会自动编译C++扩展，首次运行可能需要较长时间
3. 建议在ARM平台上运行以充分利用SVE指令集
4. 不同的稀疏度（通过调整threshold）可能会影响不同算子的相对性能
5. 加速比是相对于PyTorch稠密matmul实现计算的，值越大表示性能越好
6. 🚀 标记表示自定义SVE算子实现，📊 标记表示PyTorch参考实现
