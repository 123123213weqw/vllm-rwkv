#!/usr/bin/env python3
"""Evaluate only the strict same-machine Albatross model-core TPS contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_benchmark(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("rwkv7_faster3a_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load benchmark module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bench = _load_benchmark(args.benchmark)
    measurement = json.loads(args.measurement.read_text(encoding="utf-8"))
    check = bench._evaluate_model_only(measurement, [])
    report = {
        "schema_version": 1,
        "benchmark": bench.BENCHMARK_NAME,
        "overall_status": check["status"],
        "source": {
            "albatross_repo": bench.ALBATROSS_REPO,
            "albatross_commit": bench.ALBATROSS_COMMIT,
            "albatross_impl": bench.ALBATROSS_IMPL,
        },
        "acceptance": bench.ACCEPTANCE_THRESHOLDS["model_only_steady_decode"],
        "check": check,
        "measurement": str(args.measurement),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if check["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
