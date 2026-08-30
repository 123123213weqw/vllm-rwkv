# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from benchmarks.rwkv7 import benchmark_faster3a as bench


def _report(vllm_tps: float, albatross_tps: float):
    measurements = {
        "model_only_steady_decode": {
            "albatross_tokens_per_s": albatross_tps,
            "vllm_tokens_per_s": vllm_tps,
        }
    }
    return bench._evaluate_model_only(measurements, [])


def test_equal_tps_passes_strict_floor() -> None:
    check = _report(100.0, 100.0)
    assert check["status"] == "passed"
    assert check["metrics"]["vllm_to_albatross_ratio"] == 1.0


def test_any_regression_fails_strict_floor() -> None:
    check = _report(99.999, 100.0)
    assert check["status"] == "failed"


def test_release_threshold_cannot_hide_slowdown() -> None:
    threshold = bench.ACCEPTANCE_THRESHOLDS["model_only_steady_decode"]
    assert threshold["min_vllm_to_albatross_ratio"] == 1.0
    assert threshold["max_latency_slowdown_pct"] == 0.0
    assert Path(bench.__file__).name == "benchmark_faster3a.py"
