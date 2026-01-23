"""
自定义 C++ 算子示例的正确性与性能测试。

运行方式:
    python -m scripts.test_cpp_hello
"""

from __future__ import annotations

import torch

from kernels.cpp_hello import HelloWorldKernel, load_hello_extension, measure_latency


def check_correctness() -> None:
    load_hello_extension(verbose=False)
    kernel = HelloWorldKernel.initialize(name="hello_cpp", target="CPU")
    op = kernel.operator(compiled=True)

    x = torch.arange(8, dtype=torch.float32)
    out = op(x)
    expected = x + 1

    assert torch.allclose(out, expected), "C++ 算子输出与期望不一致"
    print("✅ 正确性通过：输出等于输入 + 1")


def benchmark_latency() -> None:
    kernel = HelloWorldKernel.initialize(name="hello_cpp", target="CPU")
    op = kernel.operator(compiled=True)

    x = torch.randn(1024, dtype=torch.float32)
    lat_ms = measure_latency(lambda: op(x), warmup=30, iters=200)
    print(f"⏱️  平均延迟: {lat_ms:.4f} ms (输入形状 {tuple(x.shape)})")


def main() -> None:
    check_correctness()
    benchmark_latency()


if __name__ == "__main__":
    main()
