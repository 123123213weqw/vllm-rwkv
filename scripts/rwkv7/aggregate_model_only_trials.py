#!/usr/bin/env python3
"""Aggregate alternating Albatross/vLLM trials with a median estimator."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"measurement must contain an object: {path}")
    return value


def _metrics(measurement: dict[str, Any], path: Path) -> dict[str, Any]:
    value = measurement.get("model_only_steady_decode")
    if not isinstance(value, dict):
        raise ValueError(f"missing model_only_steady_decode: {path}")
    return value


def _median_prefixed(
    measurements: list[dict[str, Any]],
    paths: list[Path],
    prefix: str,
) -> dict[str, Any]:
    rows = [_metrics(item, path) for item, path in zip(measurements, paths)]
    result = copy.deepcopy(rows[0])
    keys = set.intersection(*(set(row) for row in rows))
    for key in sorted(keys):
        if not key.startswith(prefix):
            continue
        values = [row[key] for row in rows]
        if all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            result[key] = statistics.median(values)
    return result


def aggregate(albatross_paths: list[Path], vllm_paths: list[Path]) -> dict[str, Any]:
    if len(albatross_paths) != len(vllm_paths):
        raise ValueError("Albatross and vLLM trial counts must match")
    if not albatross_paths:
        raise ValueError("at least one paired trial is required")

    albatross = [_load(path) for path in albatross_paths]
    vllm = [_load(path) for path in vllm_paths]
    abl_metrics = [
        _metrics(item, path) for item, path in zip(albatross, albatross_paths)
    ]
    vllm_metrics = [_metrics(item, path) for item, path in zip(vllm, vllm_paths)]

    cases = {
        (
            int(metrics["albatross_batch_size"]),
            int(metrics["albatross_seq_len"]),
        )
        for metrics in abl_metrics
    }
    cases.update(
        (
            int(metrics["vllm_batch_size"]),
            int(metrics["vllm_seq_len"]),
        )
        for metrics in vllm_metrics
    )
    if len(cases) != 1:
        raise ValueError(
            f"all paired trials must use one BxT case, got {sorted(cases)}"
        )

    output = copy.deepcopy(albatross[0])
    combined = _median_prefixed(albatross, albatross_paths, "albatross_")
    combined.update(_median_prefixed(vllm, vllm_paths, "vllm_"))
    output["model_only_steady_decode"] = combined
    config = output.setdefault("config", {})
    config.update(vllm[0].get("config", {}))
    config.update(
        {
            "measurement_source": "median_of_paired_alternating_trials",
            "trial_count": len(albatross_paths),
            "trial_order": "alternating_albatross_first_vllm_first",
            "trial_estimator": "median",
            "albatross_trial_files": [str(path) for path in albatross_paths],
            "vllm_trial_files": [str(path) for path in vllm_paths],
        }
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--albatross", type=Path, nargs="+", required=True)
    parser.add_argument("--vllm", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.albatross, args.vllm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
