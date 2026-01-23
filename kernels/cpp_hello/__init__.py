"""
示例：基于 C++ 自定义算子的开发框架。

功能：
1. 使用 `torch.utils.cpp_extension.load` 编译并注册 C++ 算子 `teal::hello_native`。
2. 提供 Python 端包装类，便于在 `torch.compile` 环境中调用。
3. 附带简单的延迟测量工具。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.cpp_extension import load

from kernels.compile_wrapper import BaseKernel


ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / "_build"
EXT_NAME = "teal_hello_ext"


def _extra_cflags() -> list[str]:
    # Windows/Unix 对应的 C++17 开关
    return ["/std:c++17"] if os.name == "nt" else ["-std=c++17"]


def load_hello_extension(
    rebuild: bool = False,
    verbose: bool = False,
) -> Optional[torch.types.ModuleType]:
    """
    编译并加载 C++ 扩展。若已有同名算子则跳过重复构建。
    """
    # 算子已注册则直接返回
    if not rebuild and hasattr(torch.ops, "teal") and hasattr(torch.ops.teal, "hello_native"):
        return None

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    return load(
        name=EXT_NAME,
        sources=[str(ROOT / "hello_op.cpp")],
        build_directory=str(BUILD_DIR),
        extra_cflags=_extra_cflags(),
        verbose=verbose,
    )


class HelloWorldKernel(BaseKernel):
    """
    Python 端包装，负责：
    - meta 用于推理阶段（torch.compile 时的 shape 推断）
    - forward 调用已注册的 C++ 算子
    """

    def meta(self, input: torch.Tensor) -> torch.Tensor:
        return input.new_empty(input.shape, device="meta")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        load_hello_extension()
        return torch.ops.teal.hello_native(input)


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
