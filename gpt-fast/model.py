# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # hack import parent dir
from kernels.sparse_gemv import SparseGEMV

import types

# 自适应稀疏 GEMM 配置
USE_CUSTOM_SPARSE_GEMM = os.getenv("USE_CUSTOM_SPARSE_GEMM", "0") == "1"
DENSE_THRESHOLD = 0.8  # 稀疏度 < 0.7 的行被视为 dense
MIN_DENSE_BLOCK = 4    # 至少连续 4 行才用 dense GEMM
SPARSE_DEBUG = os.getenv("SPARSE_DEBUG", "0") == "1"

def adaptive_sparse_gemm(
    x: Tensor,  # (M, K) activation
    weight: Tensor,  # (N, K) weight in PyTorch Linear (but column-major after monkeypatch)
    threshold: float,
) -> Tensor:
    """
    自适应稀疏 GEMM：根据激活稀疏度自适应调度 dense/sparse kernel。
    
    仅用于 prefill (seq_len > 1)，decode 阶段请使用原 GEMV。
    
    调度策略:
    - 先对 activation 执行 thr_sparsify_to_icsr_sve
    - 根据每行 nnz 计算稀疏度
    - 连续 >= MIN_DENSE_BLOCK 行且稀疏度 < DENSE_THRESHOLD → dense torch.matmul
    - 否则 → sparse_gemm_icsr_sve_gather
    
    Args:
        x: (M, K) activation matrix
        weight: (N, K) weight matrix from PyTorch Linear layer (column-major after monkeypatch)
        threshold: sparsification threshold
    
    Returns:
        output: (M, N) result matrix
    """
    from kernels.sve_sparse_gemm import (
        thr_sparsify_to_icsr_sve,
        SparseGEMMiCSRSVEGatherKernel,
    )
    
    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w, f"Dimension mismatch: x.shape[1]={K}, weight.shape[1]={K_w}"
    
    # 1. 稀疏化 activation
    nz_counts, nz_col_indices, row_offsets = thr_sparsify_to_icsr_sve(x, threshold)
    
    # 2. 计算每行 nnz
    nnz_per_row = torch.zeros(M, dtype=torch.int32)
    for m in range(M):
        start = row_offsets[m].item()
        end = row_offsets[m + 1].item()
        nnz_per_row[m] = end - start
    
    # 边界情况：对于非常小的 M（< MIN_DENSE_BLOCK），根据整体稀疏度决定
    if M < MIN_DENSE_BLOCK:
        total_nnz = sum(nnz_per_row).item()
        avg_sparsity = 1.0 - (total_nnz / (M * K))
        
        if avg_sparsity < DENSE_THRESHOLD:
            # 稀疏度低，用 dense
            if SPARSE_DEBUG:
                print(f"[adaptive_sparse_gemm] M={M} < MIN_DENSE_BLOCK={MIN_DENSE_BLOCK}, avg_sparsity={avg_sparsity:.2f} < {DENSE_THRESHOLD}, using dense matmul")
            return torch.matmul(x, weight.T)
        else:
            # 稀疏度高，用 sparse
            if SPARSE_DEBUG:
                print(f"[adaptive_sparse_gemm] M={M} < MIN_DENSE_BLOCK={MIN_DENSE_BLOCK}, avg_sparsity={avg_sparsity:.2f} >= {DENSE_THRESHOLD}, using sparse GEMM")
            
            from kernels.sve_sparse_gemm import load_sve_sparse_gemm_extension
            load_sve_sparse_gemm_extension()
            
            weight_kn = weight.T.contiguous()
            output = torch.ops.sparse_op.sparse_gemm_icsr_sve_gather(
                x, weight_kn, row_offsets, nz_col_indices
            )
            return output
    
    # 3. 划分 dense/sparse block (M >= MIN_DENSE_BLOCK 的情况)
    # 策略: 扫描每行，连续 >= MIN_DENSE_BLOCK 行且稀疏度 < DENSE_THRESHOLD → dense block
    dense_blocks = []  # [(start_row, end_row), ...]
    sparse_blocks = []  # [(start_row, end_row), ...]
    
    i = 0
    while i < M:
        # 检查当前位置能否形成 dense block
        sparsity = 1.0 - (nnz_per_row[i].item() / K)
        
        if sparsity < DENSE_THRESHOLD:
            # 尝试扩展 dense block
            j = i + 1
            while j < M:
                s_j = 1.0 - (nnz_per_row[j].item() / K)
                if s_j < DENSE_THRESHOLD:
                    j += 1
                else:
                    break
            
            # 检查是否满足最小 block 长度
            if j - i >= MIN_DENSE_BLOCK:
                dense_blocks.append((i, j))
                i = j
            else:
                # 不满足长度要求，划为 sparse
                sparse_blocks.append((i, j))
                i = j
        else:
            # 当前行稀疏度高，划为 sparse
            j = i + 1
            # 扩展连续 sparse 行
            while j < M:
                s_j = 1.0 - (nnz_per_row[j].item() / K)
                if s_j >= DENSE_THRESHOLD:
                    j += 1
                else:
                    break
            sparse_blocks.append((i, j))
            i = j
    
    # 4. Debug 输出
    if SPARSE_DEBUG:
        dense_rows = sum(end - start for start, end in dense_blocks)
        sparse_rows = sum(end - start for start, end in sparse_blocks)
        print(f"[adaptive_sparse_gemm] M={M}, K={K}, N={N} threshold={threshold}")
        print(f"  dense_blocks: {dense_blocks}, total_rows: {dense_rows}")
        print(f"  sparse_blocks: {sparse_blocks}, total_rows: {sparse_rows}")
        print(f"  nnz_per_row: {nnz_per_row.tolist()}")
    
    # 5. 执行计算并拼接结果
    output = torch.zeros(M, N, dtype=x.dtype, device=x.device)
    
    # Dense blocks: torch.matmul
    # weight is (N, K) in column-major layout, so x @ weight.T
    for start, end in dense_blocks:
        x_block = x[start:end, :]  # (block_size, K)
        output[start:end, :] = torch.matmul(x_block, weight.T)
    
    # Sparse blocks: sparse_gemm_icsr_sve_gather
    # sparse_gemm_icsr_sve_gather expects weight as (K, N)
    # Since weight is (N, K) column-major, we need to transpose it
    weight_kn = weight.T.contiguous()  # (K, N)
    
    # 直接使用 torch.ops 调用
    from kernels.sve_sparse_gemm import load_sve_sparse_gemm_extension
    load_sve_sparse_gemm_extension()
    
    for start, end in sparse_blocks:
        block_M = end - start
        # 提取当前 block 的 row_offsets
        # row_offsets[start] 是第 start 行在 nz_col_indices 中的起始位置
        offset_start = row_offsets[start].item()
        offset_end = row_offsets[end].item()
        
        # 重建 block 的 row_offsets (0-indexed)
        block_row_offsets = torch.zeros(block_M + 1, dtype=torch.int64)
        for i in range(block_M + 1):
            block_row_offsets[i] = row_offsets[start + i] - offset_start
        
        # 提取对应的 nz_col_indices
        block_nz_col_indices = nz_col_indices[offset_start:offset_end]
        
        # 提取 activation block
        x_block = x[start:end, :]  # (block_M, K)
        
        # 调用 sparse GEMM
        output[start:end, :] = torch.ops.sparse_op.sparse_gemm_icsr_sve_gather(
            x_block, weight_kn, block_row_offsets, block_nz_col_indices
        )
    
    return output

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5

    sparsify: bool = False
    hist_path: str = None

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [config for config in transformer_configs if config.lower() in str(name).lower()]

        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name # make sure only one 'best' match
            
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(block_size=16384, vocab_size=32000, n_layer=32, dim = 4096, rope_base=1000000),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(n_layer=48, n_head=64, dim=8192, vocab_size=32000, n_local_heads=8, intermediate_size=22016, rope_base=1000000), # CodeLlama-34B-Python-hf
    "70B": dict(n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
    "Mistral-7B": dict(n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=32000),
    "stories15M": dict(n_layer=6, n_head=6, dim=288),
    "stories110M": dict(n_layer=12, n_head=12, dim=768),

    "llama-3-8b": dict(block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000),
    "llama-3-70b": dict(block_size=8192, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000),
}

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.float16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.output.weight.dtype
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, dtype)

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base, dtype)
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

def _new_attn_forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
    bsz, seqlen, _ = x.shape

    kv_size = self.n_local_heads * self.head_dim

    if USE_CUSTOM_SPARSE_GEMM:
        # 使用自适应稀疏 GEMM (适用于 prefill 和 decode 阶段)
        # 处理 wqkv (三个矩阵拼接)
        qkv = adaptive_sparse_gemm(
            x.view(-1, self.dim),  # (bsz*seqlen, dim)
            self.wqkv.weight,
            self.thresh_q,  # 使用 q 的 threshold (或取平均)
        )
        q, k, v = qkv.view(bsz, seqlen, -1).split([self.dim, kv_size, kv_size], dim=-1)
    elif seqlen == 1:
        # Decode 阶段: 使用原 GEMV
        q, k, v = self.gemv1(x, self.wqkv.weight, self.thresh_q, self.thresh_k, self.thresh_v, self.sparsity_bin, kv_size).split([self.dim, kv_size, kv_size], dim=-1)
    else:
        # 回退 baseline
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

    q = q.view(bsz, seqlen, self.n_head, self.head_dim)
    k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

    q = apply_rotary_emb(q, freqs_cis)
    k = apply_rotary_emb(k, freqs_cis)

    q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

    if self.kv_cache is not None:
        k, v = self.kv_cache.update(input_pos, k, v)

    k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
    v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

    y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

    if USE_CUSTOM_SPARSE_GEMM:
        # 使用自适应稀疏 GEMM (适用于 prefill 和 decode 阶段)
        y = adaptive_sparse_gemm(
            y.view(-1, self.dim),
            self.wo.weight,
            self.thresh_o,
        ).view(bsz, seqlen, self.dim)
    elif seqlen == 1:
        # Decode 阶段: 使用原 GEMV
        y = self.gemv2(y, self.wo.weight, self.thresh_o, self.sparsity_bin)
    else:
        # 回退 baseline
        y = self.wo(y)

    return y

class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def qkv_prefill(self, x: Tensor):
        pass

    def qkv_decode(self, x: Tensor):
        pass

    def apply_monkeypatch(self):
        self.old_forward = self.forward
        self.forward = types.MethodType(_new_attn_forward, self)

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim


        # q,k,v = self.gemv1(x, self.wqkv.weight, self.thresh_q, self.thresh_k, self.thresh_v, self.sparsity_bin, kv_size).split([self.dim, kv_size, kv_size], dim=-1) # prefill logic taken care of in gemv
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1) # baseline

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        # y = self.gemv2(y, self.wo.weight, self.thresh_o, self.sparsity_bin) # prefill logic taken care of in gemv
        y = self.wo(y) # baseline

        return y

def _new_ffn_forward(self, x: Tensor) -> Tensor:
    bsz, seqlen, _ = x.shape
    
    if USE_CUSTOM_SPARSE_GEMM:
        # 使用自适应稀疏 GEMM (适用于 prefill 和 decode 阶段)
        x_flat = x.view(-1, x.shape[-1])  # (bsz*seqlen, dim)
        
        gate = adaptive_sparse_gemm(x_flat, self.w1.weight, self.thresh_gate)
        up = adaptive_sparse_gemm(x_flat, self.w3.weight, self.thresh_up)
        hidden = F.silu(gate) * up
        output = adaptive_sparse_gemm(hidden, self.w2.weight, self.thresh_down)
        
        return output.view(bsz, seqlen, -1)
    elif seqlen == 1:
        # Decode 阶段: 使用原 GEMV
        return self.gemv2(
            F.silu(self.gemv1(x, self.w1.weight, self.thresh_gate, self.sparsity_bin)) * 
            self.gemv1(x, self.w3.weight, self.thresh_up, self.sparsity_bin), 
            self.w2.weight, self.thresh_down, self.sparsity_bin
        )
    else:
        # 回退 baseline
        return self.w2(F.silu(self.w1(x)) * self.w3(x)) 

class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

        self.sparsify = False

    def apply_monkeypatch(self):
        self.old_forward = self.forward
        self.forward = types.MethodType(_new_ffn_forward, self)

    def forward(self, x: Tensor) -> Tensor:
        # prefill logic taken care of in gemv
        # return self.gemv2(F.silu(self.gemv1(x, self.w1.weight, self.thresh_gate, self.sparsity_bin)) * self.gemv1(x, self.w3.weight, self.thresh_up, self.sparsity_bin), self.w2.weight, self.thresh_down, self.sparsity_bin) 
        return self.w2(F.silu(self.w1(x)) * self.w3(x)) # baseline


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000,
    dtype: torch.dtype = torch.float16
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
