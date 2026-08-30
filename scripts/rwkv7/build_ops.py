#!/usr/bin/env python3
"""Build only the RWKV-7 CUDA extension against an installed vLLM runtime."""

from __future__ import annotations

import argparse
import importlib.machinery
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--install-dir", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    source_dir = root / "csrc/libtorch_stable/rwkv7"
    sources = [
        source_dir / "rwkv7_fast_ops_fp16.cpp",
        source_dir / "rwkv7_fast_ops_fp16.cu",
        source_dir / "rwkv7_registration.cpp",
        source_dir / "rwkv7_v3a_ops.cpp",
        source_dir / "rwkv7_v3a_ops.cu",
        source_dir / "rwkv7_wkv_fp16_v2.cpp",
        source_dir / "rwkv7_wkv_fp16_v2.cu",
        source_dir / "rwkv7_wkv_fp32_v2.cpp",
        source_dir / "rwkv7_wkv_fp32_v2.cu",
    ]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing RWKV-7 sources: " + ", ".join(missing))

    build_dir = (args.build_dir or root / "build/rwkv7_ops").resolve()
    install_dir = (args.install_dir or root / "vllm").resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    install_dir.mkdir(parents=True, exist_ok=True)

    from torch.utils.cpp_extension import load

    module = load(
        name="rwkv7_ops",
        sources=[str(path) for path in sources],
        extra_include_paths=[str(root / "csrc")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--extra-device-vectorization",
            "-Xptxas",
            "-O3",
            "-D_IO_FP16_",
        ],
        with_cuda=True,
        is_python_module=True,
        build_directory=str(build_dir),
        verbose=args.verbose,
    )
    source = Path(module.__file__).resolve()
    if not any(source.name.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES):
        raise RuntimeError(f"unexpected extension filename: {source}")
    destination = install_dir / source.name
    shutil.copy2(source, destination)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
