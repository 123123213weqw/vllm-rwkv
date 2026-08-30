# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible continuous-batching churn benchmark for RWKV7.

The workload uses Poisson arrivals, mixed prompt/output lengths, session reuse,
and deterministic early disconnects. It is intended to exercise request-row
allocation/removal rather than report an idealized locked-batch kernel number.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    session_id: str
    prompt_tokens: int
    output_tokens: int
    cancel_after_tokens: int | None
    arrival_delay: float


@dataclass
class RequestResult:
    request_id: int
    session_id: str
    status: str
    route: str = "unknown"
    prompt_tokens: int = 0
    output_tokens: int = 0
    ttft_seconds: float | None = None
    e2e_seconds: float = 0.0
    error: str = ""


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def parse_sse_data(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) else None


def event_token_increment(event: dict[str, Any]) -> int:
    usage = event.get("usage")
    if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
        return 0
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return 0
    choice = choices[0]
    if not isinstance(choice, dict):
        return 0
    delta = choice.get("delta", choice)
    if not isinstance(delta, dict):
        return 0
    content = delta.get("content", delta.get("text"))
    return int(content is not None and content != "")


def event_completion_tokens(event: dict[str, Any]) -> int | None:
    usage = event.get("usage")
    if not isinstance(usage, dict) or usage.get("completion_tokens") is None:
        return None
    return int(usage["completion_tokens"])


def build_workload(
    *,
    requests: int,
    arrival_rate: float,
    session_pool: int,
    prompt_range: tuple[int, int],
    output_range: tuple[int, int],
    cancel_fraction: float,
    seed: int,
) -> list[RequestSpec]:
    if requests < 1 or arrival_rate <= 0 or session_pool < 1:
        raise ValueError("requests, arrival_rate, and session_pool must be positive")
    if not 0 <= cancel_fraction <= 1:
        raise ValueError("cancel_fraction must be between zero and one")
    rng = random.Random(seed)
    workload = []
    for request_id in range(requests):
        output_tokens = rng.randint(*output_range)
        cancel_after = None
        if output_tokens > 1 and rng.random() < cancel_fraction:
            cancel_after = rng.randint(1, output_tokens - 1)
        workload.append(
            RequestSpec(
                request_id=request_id,
                session_id=f"churn-{rng.randrange(session_pool)}",
                prompt_tokens=rng.randint(*prompt_range),
                output_tokens=output_tokens,
                cancel_after_tokens=cancel_after,
                arrival_delay=(
                    0.0 if request_id == 0 else rng.expovariate(arrival_rate)
                ),
            )
        )
    return workload


def synthetic_prompt(tokens: int) -> str:
    return " ".join(f"rwkv{index % 97}" for index in range(tokens))


async def run_request(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    spec: RequestSpec,
    *,
    api_key: str,
) -> RequestResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": synthetic_prompt(spec.prompt_tokens)}],
        "max_tokens": spec.output_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"x-rwkv-session-id": spec.session_id}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    started = time.perf_counter()
    result = RequestResult(
        request_id=spec.request_id,
        session_id=spec.session_id,
        status="error",
        prompt_tokens=spec.prompt_tokens,
    )
    try:
        async with client.stream(
            "POST", endpoint, json=payload, headers=headers
        ) as response:
            result.route = response.headers.get("x-rwkv-route", "direct")
            if response.status_code >= 400:
                detail = (await response.aread()).decode(errors="replace")[:500]
                result.error = f"HTTP {response.status_code}: {detail}"
                return result
            async for line in response.aiter_lines():
                event = parse_sse_data(line)
                if event is None:
                    continue
                increment = event_token_increment(event)
                exact_output_tokens = event_completion_tokens(event)
                if exact_output_tokens is not None:
                    result.output_tokens = exact_output_tokens
                if increment and result.ttft_seconds is None:
                    result.ttft_seconds = time.perf_counter() - started
                if exact_output_tokens is None:
                    result.output_tokens += increment
                if (
                    spec.cancel_after_tokens is not None
                    and result.output_tokens >= spec.cancel_after_tokens
                ):
                    result.status = "cancelled"
                    break
            else:
                result.status = "success"
    except (httpx.HTTPError, asyncio.TimeoutError) as exc:
        result.error = str(exc)
    finally:
        result.e2e_seconds = time.perf_counter() - started
    return result


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    workload = build_workload(
        requests=args.requests,
        arrival_rate=args.arrival_rate,
        session_pool=args.session_pool,
        prompt_range=(args.prompt_min, args.prompt_max),
        output_range=(args.output_min, args.output_max),
        cancel_fraction=args.cancel_fraction,
        seed=args.seed,
    )
    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(
        max_connections=args.max_connections,
        max_keepalive_connections=args.max_connections,
    )
    started = time.perf_counter()
    tasks: list[asyncio.Task[RequestResult]] = []
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for spec in workload:
            await asyncio.sleep(spec.arrival_delay)
            tasks.append(
                asyncio.create_task(
                    run_request(
                        client,
                        args.endpoint,
                        args.model,
                        spec,
                        api_key=args.api_key,
                    )
                )
            )
        results = await asyncio.gather(*tasks)
    duration = time.perf_counter() - started
    ttfts = [item.ttft_seconds for item in results if item.ttft_seconds is not None]
    latencies = [item.e2e_seconds for item in results]
    output_tokens = sum(item.output_tokens for item in results)
    statuses: dict[str, int] = {}
    routes: dict[str, int] = {}
    for result in results:
        statuses[result.status] = statuses.get(result.status, 0) + 1
        routes[result.route] = routes.get(result.route, 0) + 1
    return {
        "schema_version": 1,
        "workload": {
            "requests": args.requests,
            "arrival_rate": args.arrival_rate,
            "session_pool": args.session_pool,
            "prompt_range": [args.prompt_min, args.prompt_max],
            "output_range": [args.output_min, args.output_max],
            "cancel_fraction": args.cancel_fraction,
            "seed": args.seed,
        },
        "summary": {
            "duration_seconds": duration,
            "output_tokens": output_tokens,
            "output_tps": output_tokens / duration,
            "request_throughput": len(results) / duration,
            "statuses": statuses,
            "routes": routes,
            "ttft_seconds": {
                "mean": statistics.fmean(ttfts) if ttfts else None,
                "p50": percentile(ttfts, 0.50),
                "p95": percentile(ttfts, 0.95),
                "p99": percentile(ttfts, 0.99),
            },
            "e2e_seconds": {
                "mean": statistics.fmean(latencies),
                "p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95),
                "p99": percentile(latencies, 0.99),
            },
        },
        "requests": [asdict(item) for item in results],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:8080/v1/chat/completions"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--arrival-rate", type=float, default=20.0)
    parser.add_argument("--session-pool", type=int, default=64)
    parser.add_argument("--prompt-min", type=int, default=8)
    parser.add_argument("--prompt-max", type=int, default=512)
    parser.add_argument("--output-min", type=int, default=8)
    parser.add_argument("--output-max", type=int, default=128)
    parser.add_argument("--cancel-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-connections", type=int, default=512)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(run_benchmark(args))
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
