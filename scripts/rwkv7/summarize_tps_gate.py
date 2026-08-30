#!/usr/bin/env python3
"""Aggregate per-case TPS reports and fail when any case is below 1.00x."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()

    cases = []
    for path in args.reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        metrics = report["check"]["metrics"]
        cases.append(
            {
                "report": str(path),
                "status": report["overall_status"],
                "albatross_tokens_per_s": metrics["albatross_tokens_per_s"],
                "vllm_tokens_per_s": metrics["vllm_tokens_per_s"],
                "vllm_to_albatross_ratio": metrics["vllm_to_albatross_ratio"],
            }
        )
    passed = all(case["status"] == "passed" for case in cases)
    summary = {
        "schema_version": 1,
        "overall_status": "passed" if passed else "failed",
        "required_ratio": 1.0,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
