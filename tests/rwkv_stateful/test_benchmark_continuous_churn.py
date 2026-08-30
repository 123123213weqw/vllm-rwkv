# SPDX-License-Identifier: Apache-2.0
from benchmarks.rwkv7.benchmark_continuous_churn import (
    build_workload,
    event_completion_tokens,
    event_token_increment,
    parse_sse_data,
    percentile,
)


def test_sse_parser_and_token_counter() -> None:
    event = parse_sse_data('data: {"choices":[{"delta":{"content":"x"}}]}')
    assert event is not None
    assert event_token_increment(event) == 1
    assert event_completion_tokens({"usage": {"completion_tokens": 12}}) == 12
    assert parse_sse_data("data: [DONE]") is None
    assert parse_sse_data("event: message") is None


def test_seeded_churn_workload_has_arrivals_and_cancellations() -> None:
    first = build_workload(
        requests=100,
        arrival_rate=20,
        session_pool=8,
        prompt_range=(8, 64),
        output_range=(8, 32),
        cancel_fraction=0.5,
        seed=7,
    )
    second = build_workload(
        requests=100,
        arrival_rate=20,
        session_pool=8,
        prompt_range=(8, 64),
        output_range=(8, 32),
        cancel_fraction=0.5,
        seed=7,
    )
    assert first == second
    assert first[0].arrival_delay == 0
    assert any(item.cancel_after_tokens is not None for item in first)
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
