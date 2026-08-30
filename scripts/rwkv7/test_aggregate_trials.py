#!/usr/bin/env python3
"""Dependency-free smoke test for paired-trial aggregation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from aggregate_model_only_trials import aggregate


def _write(path: Path, side: str, tps: float, latency: float) -> None:
    prefix = "albatross" if side == "albatross" else "vllm"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark": "rwkv7_faster3a",
                "config": {"side": side},
                "model_only_steady_decode": {
                    f"{prefix}_batch_size": 64,
                    f"{prefix}_seq_len": 1,
                    f"{prefix}_tokens_per_s": tps,
                    f"{prefix}_p50_ms": latency,
                    f"{prefix}_label": side,
                },
            }
        ),
        encoding="utf-8",
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        abl = [root / f"a-{index}.json" for index in range(4)]
        vllm = [root / f"v-{index}.json" for index in range(4)]
        for path, tps in zip(abl, (100.0, 102.0, 101.0, 103.0)):
            _write(path, "albatross", tps, 64_000.0 / tps)
        for path, tps in zip(vllm, (103.0, 105.0, 104.0, 106.0)):
            _write(path, "vllm", tps, 64_000.0 / tps)
        result = aggregate(abl, vllm)
        metrics = result["model_only_steady_decode"]
        assert metrics["albatross_tokens_per_s"] == 101.5
        assert metrics["vllm_tokens_per_s"] == 104.5
        assert result["config"]["trial_count"] == 4
    print("paired RWKV-7 trial aggregation: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
