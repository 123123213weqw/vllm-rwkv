#!/usr/bin/env python3
"""Dependency-free CI smoke test for the strict TPS release contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks/rwkv7/benchmark_faster3a.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("rwkv7_tps_contract", BENCHMARK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _check(bench, vllm_tps: float, albatross_tps: float):
    return bench._evaluate_model_only(
        {
            "model_only_steady_decode": {
                "albatross_tokens_per_s": albatross_tps,
                "vllm_tokens_per_s": vllm_tps,
            }
        },
        [],
    )


def main() -> None:
    bench = _load_benchmark()
    threshold = bench.ACCEPTANCE_THRESHOLDS["model_only_steady_decode"]
    assert threshold["min_vllm_to_albatross_ratio"] == 1.0
    assert threshold["max_latency_slowdown_pct"] == 0.0
    assert _check(bench, 100.0, 100.0)["status"] == "passed"
    assert _check(bench, 99.999, 100.0)["status"] == "failed"
    print("strict RWKV-7 TPS contract: passed")


if __name__ == "__main__":
    main()
