import time
from typing import Callable

import torch
from scipy.stats import gmean

def measure_latency(
    fn: Callable[[], torch.Tensor],
    warmup: int = 20,
    iters: int = 100,
) -> float:
    """
    简单的端到端延迟测量（毫秒）。
    - CPU: 使用 time.perf_counter
    - GPU: 在关键点同步以避免异步干扰
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for _ in range(warmup):
        fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / iters

def benchmark(fn, warmup=100, rep=200, quantiles=None, fast_flush=True):
    # https://github.com/nox-410/tvm.tl/blob/tl/python/tvm/tl/utils.py#L144    
    fn()
    torch.cuda.synchronize()

    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]

    # Warm-up
    for _ in range(n_warmup):
        fn()
    
    # Benchmark
    for i in range(n_repeat):
        cache.zero_()
        start_event[i].record()
        fn()
        end_event[i].record()
    torch.cuda.synchronize()
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float
    )
    times_list = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return gmean(times_list)