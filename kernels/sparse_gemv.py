# adapted from deja vu

from typing import Optional

import torch
# import triton
# import triton.language as tl
def init_to_zero(*names):
    def init_func(nargs):
        for name in names:
            nargs[name].zero_()
    return init_func

# NOTE: will need to warm up kernels each time, triton autotune caching isn't a thing right now

# configs=[
#     triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=2, pre_hook=init_to_zero("Y")), 

#     triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, pre_hook=init_to_zero("Y")),
#     triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=2, pre_hook=init_to_zero("Y")),
#     triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
#     triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
#     triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
#     triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
#     triton.Config({"BLOCK_M": 128, "BLOCK_N": 16}, num_warps=4, pre_hook=init_to_zero("Y")),
#     triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, pre_hook=init_to_zero("Y")),
#     triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, pre_hook=init_to_zero("Y")),
#     triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, pre_hook=init_to_zero("Y")),
#     triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),

#     triton.Config({"BLOCK_M": 128, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),
#     triton.Config({"BLOCK_M": 64, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),
#     triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),
#     triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),


#     # Llama 3 variants can use BLOCK_N >= 1024
#     # triton.Config({"BLOCK_M": 128, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
#     # triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
#     # triton.Config({"BLOCK_M": 64, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
#     # triton.Config({"BLOCK_M": 32, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
#     # triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
# ]

# @triton.autotune(
#     configs=configs,
#     key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE", "SPARSITY_BIN"],
# )
# @triton.jit
# def splitk_sparse_gemv_kernel(
#     Y, # Pointers to matrices
#     A, X, threshold,
#     # Matrix dimensions
#     N, M,
#     CACHE_KEY_N, CACHE_KEY_M,
#     # Meta-parameters
#     BATCHSIZE: tl.constexpr, SPARSITY_BIN: tl.constexpr,
#     BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
# ):

# def splitk_sparse_gemv_kernel():
    # start_n = tl.program_id(0)
    # start_m = tl.program_id(1)
    # # now compute the block that each program will go through
    # # rn (resp. rm) denotes a range of indices for rows (resp. col) of A
    
    # rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # A_ptr = A + (rm[:, None] * N + rn[None, :])
    # X_ptr = X + rm
    # Y_ptr = Y + rn
    
    # # eviction policy go brrr
    # if BATCHSIZE == 1:
    #     x0 = tl.load(X_ptr, mask=rm < M, other=0.0, eviction_policy='evict_last') # reuse x across threadblocks
    #     idx = tl.abs(x0) > threshold
    #     # selectively load weight rows
    #     a = tl.load(A_ptr, mask=idx[:, None], other=0.0, eviction_policy='evict_first') # only load weights once per threadblock
    #     acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)

    # # rematerialize rm and rn to save registers
    # rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # tl.atomic_add(Y_ptr, acc0, mask=rn < N)


# NOTE: assumes that weight is column major
def splitk_sparse_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
    sparsity_bin: int
) -> torch.Tensor:
    """
    Compute y = sparse(X) @ weight.
    :param x: input tensor [1, 1, Z]
    :param weight: weight matrix [N, Z]
    :param threshold: threshold for the absolute value of x
    :param sparsity_bin: sparsity level to get tuned kernel
    :return: result tensor y
    """
    pass
    # N, Z = weight.shape
    # beam_width, seq_len, _ = x.shape
    # assert x.shape[2] == Z
    # x = x.contiguous()
    
    # assert weight.stride(1) > 1, "weight should be column major"

    # 1D launch kernel where each block gets its own program.
    # grid = lambda META: (
    #     triton.cdiv(N, META["BLOCK_N"]),
    #     triton.cdiv(Z, META["BLOCK_M"]),
    # )  # noqa

    # output = torch.empty(
    #     beam_width,
    #     seq_len,
    #     N,
    #     device=x.device,
    #     dtype=torch.float16,
    # )


    # kernel = splitk_sparse_gemv_kernel
    # kernel[grid](
    #     output,  # data ptrs
    #     weight,
    #     x,
    #     threshold,
    #     N,  # shapes
    #     Z,
    #     N // 16,  # key for triton cache (limit number of compilations)
    #     Z // 16,
    #     beam_width,  # BATCHSIZE
    #     sparsity_bin, # SPARSITY_BIN
    #     # can't use kwargs because auto-tuner requires args
    # )

    # if x.dtype is not output.dtype:
    #     print(f"Warning: incuring dtype conversion overhead since input dtype is not torch.float16. Detected dtype: {x.dtype}. ")
    #     return output.to(dtype=x.dtype)

    # return output


# fused implementation of qkv with three thresholds
# is unnecessary for uniform but is needed for block-wise greedy
# @triton.autotune(
#     configs=configs, 
#     key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE", "SPARSITY_BIN"],
# )
# @triton.jit
# def qkv_kernel(
#     Y,  # Pointers to output matrices
#     A, 
#     X, 
#     threshold_q, threshold_k, threshold_v,
#     # Matrix dimensions
#     N, N_q, N_kv, M,
#     CACHE_KEY_N, CACHE_KEY_M,
#     # Meta-parameters
#     BATCHSIZE: tl.constexpr, SPARSITY_BIN: tl.constexpr,
#     BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
# ):
def qkv_kernel():
    # start_n = tl.program_id(0)
    # start_m = tl.program_id(1)

    # is_q = start_n * BLOCK_N < N_q
    # is_v = N_q + N_kv <= start_n * BLOCK_N

    
    # rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # rn = start_n*BLOCK_N + tl.arange(0, BLOCK_N)
        
    # A_ptr = A + rm[:, None] * N + rn[None, :]

    # X_ptr = X + rm

    # Y_ptr = Y + rn

    # threshold = tl.where(is_q, threshold_q, tl.where(is_v, threshold_v, threshold_k))
    
    # if BATCHSIZE == 1:
    #     x0 = tl.load(X_ptr, mask=rm < M, other=0.0, eviction_policy='evict_last')
    #     idx = tl.abs(x0) > threshold
    #     a = tl.load(A_ptr, mask=idx[:, None], other=0.0, eviction_policy='evict_first')
    #     acc = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)
    
    # # rematerialize to reduce register pressure

    # rn = start_n*BLOCK_N + tl.arange(0, BLOCK_N)
    # mask_n = rn < N

    # tl.atomic_add(Y_ptr, acc, mask=mask_n)
    pass

def qkv_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    threshold_q: float,
    threshold_k: float,
    threshold_v: float,
    sparsity_bin: int,
    kv_size: int
):
    pass
    # N, Z = weight.shape
    # beam_width, seq_len, _ = x.shape
    # assert x.shape[2] == Z
    # x = x.contiguous()
    
    # assert weight.stride(1) > 1, "weights should be column major"

    # N_q = N - 2*kv_size
    # N_k = kv_size

    # grid = lambda META: (
    #     triton.cdiv(N, META["BLOCK_N"]),
    #     triton.cdiv(Z, META["BLOCK_M"]),
    # )

    # output = torch.empty(beam_width, seq_len, N, device=x.device, dtype=torch.float16)

    # qkv_kernel[grid](
    #     output,
    #     weight,
    #     x,
    #     threshold_q, threshold_k, threshold_v,
    #     N, N_q, N_k, Z,
    #     N // 16, Z // 16,
    #     beam_width,
    #     sparsity_bin,
    # )

    # if x.dtype is not output.dtype:
    #     print(f"Warning: incurring dtype conversion overhead. Input dtype: {x.dtype}")
    #     return output.to(dtype=x.dtype)

    # return output


import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'kernels'))
from compile_wrapper import BaseKernel

# wrappers for compatibility with torch.compile
class SparseGEMV(BaseKernel):

    def meta(
        self,
        # The hidden states -> [ L, seq, D ].
        hidden_states: torch.Tensor,
        # weights [I, D]
        weights: torch.Tensor,
        threshold: float,
        sparsity_bin: int,
    ) -> torch.Tensor:
        pass
        # return hidden_states.new_empty((hidden_states.size(0), hidden_states.size(1), weights.size(0)))

    def forward(
        self,
        # The hidden states -> [ L, seq, D ].
        hidden_states: torch.Tensor,
        # weights [I, D]
        weights: torch.Tensor,
        threshold: float,
        sparsity_bin: int,
    ) -> torch.Tensor:
        pass
        # return splitk_sparse_gemv(hidden_states, weights, threshold, sparsity_bin) if hidden_states.shape[1] == 1 else torch.matmul(hidden_states, weights.T) 
        # this will incur some prefill overhead since weights are column major

from typing import Tuple
class SparseQKVGEMV(BaseKernel):
    def meta(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        threshold_q: float,
        threshold_k: float,
        threshold_v: float,
        sparsity_bin: int,
        kv_size: int
    ) -> torch.Tensor:
        return x.new_empty(x.shape[0], x.shape[1], weight.shape[0])

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        threshold_q: float,
        threshold_k: float,
        threshold_v: float,
        sparsity_bin: int,
        kv_size: int
    ) -> torch.Tensor:
        pass
        # return qkv_gemv(x, weight, threshold_q, threshold_k, threshold_v, sparsity_bin, kv_size) if x.shape[1] == 1 else torch.matmul(x, weight.T)

# for testing purposes, to see if overhead at 0% is really due to strengthening torch.matmul (seems like it is)
class DenseGEMV(BaseKernel):
    # 用于跟踪已打印的形状组合
    _printed_shapes = set()
    # 统计总调用次数
    _total_calls = 0
    # 统计每种形状的调用次数 {(x_shape, W_shape): count}
    _shape_call_counts = {}
    # 分别统计 prefill (seq_len > 1) 和 decode (seq_len == 1)
    _prefill_calls = 0
    _decode_calls = 0
    _prefill_shape_counts = {}  # prefill 阶段每种形状的调用次数
    _decode_shape_counts = {}   # decode 阶段每种形状的调用次数

    def meta(self, x: torch.Tensor, W: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x.new_empty(x.shape[0], x.shape[1], W.shape[0])
    
    def forward(self, x: torch.Tensor, W: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # 更新总调用次数
        DenseGEMV._total_calls += 1
        
        shape_key = (tuple(x.shape), tuple(W.shape))
        seq_len = x.shape[1] if len(x.shape) >= 2 else 1
        
        # 判断是 prefill 还是 decode
        is_prefill = seq_len > 1
        
        # 更新形状调用次数
        if shape_key not in DenseGEMV._shape_call_counts:
            DenseGEMV._shape_call_counts[shape_key] = 0
        DenseGEMV._shape_call_counts[shape_key] += 1
        
        # 分别统计 prefill 和 decode
        if is_prefill:
            DenseGEMV._prefill_calls += 1
            if shape_key not in DenseGEMV._prefill_shape_counts:
                DenseGEMV._prefill_shape_counts[shape_key] = 0
            DenseGEMV._prefill_shape_counts[shape_key] += 1
        else:
            DenseGEMV._decode_calls += 1
            if shape_key not in DenseGEMV._decode_shape_counts:
                DenseGEMV._decode_shape_counts[shape_key] = 0
            DenseGEMV._decode_shape_counts[shape_key] += 1
        
        # 记录遇到的形状（不打印）
        DenseGEMV._printed_shapes.add(shape_key)
        
        return torch.matmul(x, W.T)
    
    @classmethod
    def print_statistics(cls):
        """打印统计信息"""
        print(f"\n{'='*80}")
        print(f"[DenseGEMV Statistics]")
        print(f"  Total calls: {cls._total_calls}")
        print(f"    - Prefill (seq_len > 1): {cls._prefill_calls} ({cls._prefill_calls / cls._total_calls * 100:.1f}%)" if cls._total_calls > 0 else "    - Prefill (seq_len > 1): 0")
        print(f"    - Decode (seq_len == 1): {cls._decode_calls} ({cls._decode_calls / cls._total_calls * 100:.1f}%)" if cls._total_calls > 0 else "    - Decode (seq_len == 1): 0")
        print(f"  Total unique shapes: {len(cls._printed_shapes)}")
        
        if cls._prefill_shape_counts:
            print(f"\n  Prefill (seq_len > 1) - Shape call counts:")
            print(f"  {'No.':<5} {'seq_len':<10} {'x.shape':<30} {'W.shape':<30} {'Calls':<10} {'%':<8}")
            print(f"  {'-'*5} {'-'*10} {'-'*30} {'-'*30} {'-'*10} {'-'*8}")
            sorted_prefill = sorted(cls._prefill_shape_counts.items(), key=lambda x: x[1], reverse=True)
            for i, ((x_shape, w_shape), count) in enumerate(sorted_prefill, 1):
                seq_len = x_shape[1] if len(x_shape) >= 2 else 1
                percentage = count / cls._prefill_calls * 100 if cls._prefill_calls > 0 else 0
                print(f"  {i:<5} {seq_len:<10} {str(x_shape):<30} {str(w_shape):<30} {count:<10} {percentage:>6.1f}%")
            print(f"  {'='*5} {'='*10} {'='*30} {'='*30} {'='*10} {'='*8}")
            print(f"  Total prefill calls: {cls._prefill_calls}")
        
        if cls._decode_shape_counts:
            print(f"\n  Decode (seq_len == 1) - Shape call counts:")
            print(f"  {'No.':<5} {'x.shape':<30} {'W.shape':<30} {'Calls':<10} {'%':<8}")
            print(f"  {'-'*5} {'-'*30} {'-'*30} {'-'*10} {'-'*8}")
            sorted_decode = sorted(cls._decode_shape_counts.items(), key=lambda x: x[1], reverse=True)
            for i, ((x_shape, w_shape), count) in enumerate(sorted_decode, 1):
                percentage = count / cls._decode_calls * 100 if cls._decode_calls > 0 else 0
                print(f"  {i:<5} {str(x_shape):<30} {str(w_shape):<30} {count:<10} {percentage:>6.1f}%")
            print(f"  {'='*5} {'='*30} {'='*30} {'='*10} {'='*8}")
            print(f"  Total decode calls: {cls._decode_calls}")
        
        print(f"{'='*80}\n")
    
    @classmethod
    def reset_statistics(cls):
        """重置统计信息"""
        cls._printed_shapes.clear()
        cls._total_calls = 0
        cls._shape_call_counts.clear()
        cls._prefill_calls = 0
        cls._decode_calls = 0
        cls._prefill_shape_counts.clear()
        cls._decode_shape_counts.clear()
        print("[DenseGEMV] Statistics reset.")