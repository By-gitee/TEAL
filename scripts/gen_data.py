"""
生成 test_csr_icsr_sve_spmm.py 等 benchmark 所需的数据文件。
文件约定（与 test_csr_icsr_sve_spmm.py 一致）：
  - activation_{M}_{K}.bin：M×K 个 float32 row-major
  - weight_{K}_{N}.bin：K×N 个 float32 row-major
"""
import argparse
import sys
from pathlib import Path

import numpy as np

import argparse
import csv
import os
import struct
import subprocess
import sys
from datetime import datetime
from pathlib import Path
# 与 test_csr_icsr_sve_spmm.py 默认配置一致
DEFAULT_M_LIST = [1, 4, 16, 32, 64, 128,256]
DEFAULT_KN_LIST = [
    (512, 512), (1024, 1024), (2048, 2048), (4096, 4096), (5120, 5120),
    (512, 2048), (1024, 4096), (2048, 8192), (4096, 11008), (5120, 13824),
    (2048, 512), (4096, 1024), (8192, 2048), (11008, 4096), (13824, 5120),
]
DEFAULT_DATA_DIR = Path("/dev/xvdb/data")


def weight_path(data_dir: Path, K: int, N: int) -> Path:
    return data_dir / f"weight_{K}_{N}.bin"


def activation_path(data_dir: Path, M: int, K: int) -> Path:
    return data_dir / f"activation_{M}_{K}.bin"


def ensure_weight_file(
    data_dir: Path,
    K: int,
    N: int,
) -> Path:
    """生成或复用 weight_{K}_{N}.bin（K×N row-major float32）。"""
    path = weight_path(data_dir, K, N)
    if path.exists():
        expected = K * N * 4
        if path.stat().st_size == expected:
            return path
    import random
    random.seed(123)
    with open(path, "wb") as f:
        for _ in range(K * N):
            # [-0.1, 0.1) 与各 benchmark 原 gen_weight 一致
            v = 0.2 * random.random() - 0.1
            f.write(struct.pack("<f", v))
    return path


def ensure_activation_file(
    data_dir: Path,
    M: int,
    K: int,
) -> Path:
    """生成或复用 activation_{M}_{K}.bin（M×K row-major float32）。"""
    path = activation_path(data_dir, M, K)
    if path.exists():
        expected = M * K * 4
        if path.stat().st_size == expected:
            return path
    import random
    random.seed(42)
    with open(path, "wb") as f:
        for _ in range(M * K):
            # [-1, 1) 与各 benchmark 原 activation 一致
            v = 2.0 * random.random() - 1.0
            f.write(struct.pack("<f", v))
    return path


def load_bin_matrix(path: Path, shape: tuple) -> np.ndarray:
    """与 test_csr_icsr_sve_spmm.py 中读取方式一致。"""
    arr = np.fromfile(path, dtype=np.float32)
    return arr.reshape(shape)


def run(
    data_dir: Path,
    M_list: list[int],
    kn_list: list[tuple[int, int]],
) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 去重：同一 (K,N) 只生成一次 weight，同一 (M,K) 只生成一次 activation
    for K, N in kn_list:
        p = ensure_weight_file(data_dir, K, N)
        print(f"weight: {p} ({K}x{N})")

    for M in M_list:
        for K, N in kn_list:
            p = ensure_activation_file(
                data_dir, M, K
            )
            print(f"activation: {p} ({M}x{K})")


def main():
    parser = argparse.ArgumentParser(
        description="生成 test_csr_icsr_sve_spmm 等 benchmark 所需 bin 数据"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"数据目录，默认 {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--M",
        type=int,
        nargs="+",
        default=DEFAULT_M_LIST,
        help=f"M 列表，默认 {DEFAULT_M_LIST}",
    )
    parser.add_argument(
        "--KN",
        type=int,
        nargs="+",
        metavar="K N",
        default=None,
        help="K N 对，可多组，如 --KN 5120 5120 4096 4096",
    )
    args = parser.parse_args()

    if args.KN is not None:
        if len(args.KN) % 2 != 0:
            print("--KN 需要成对的 K N", file=sys.stderr)
            sys.exit(1)
        kn_list = [(args.KN[i], args.KN[i + 1]) for i in range(0, len(args.KN), 2)]
    else:
        kn_list = DEFAULT_KN_LIST

    run(
        data_dir=args.data_dir,
        M_list=args.M,
        kn_list=kn_list,
    )


if __name__ == "__main__":
    main()
