# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm.rwkv_stateful.policy import (
    EndpointState,
    NoHealthyEndpointError,
    RouteContext,
    StatefulRoutingPolicy,
)
from vllm.rwkv_stateful.registry import SessionRecord


def endpoints(**local_overrides: object) -> tuple[EndpointState, EndpointState]:
    local_values = {
        "name": "local",
        "url": "http://local",
        "healthy": True,
        "queue_depth": 0,
        "capacity": 10,
        "state_transfer_supported": True,
    }
    local_values.update(local_overrides)
    return EndpointState(**local_values), EndpointState(
        "cloud", "http://cloud", True, 0, 100, True
    )


def test_policy_is_local_first_then_spills_to_cloud() -> None:
    policy = StatefulRoutingPolicy(local_utilization_limit=0.8)
    local, cloud = endpoints()
    assert (
        policy.choose(RouteContext(), local=local, cloud=cloud).target.name == "local"
    )
    local, cloud = endpoints(queue_depth=8)
    assert (
        policy.choose(RouteContext(), local=local, cloud=cloud).target.name == "cloud"
    )


def test_policy_is_sticky_and_marks_state_transfer() -> None:
    policy = StatefulRoutingPolicy()
    local, cloud = endpoints()
    session = SessionRecord(
        "s", placement="cloud", endpoint=cloud.url, state_uri="file:///state"
    )
    sticky = policy.choose(RouteContext(session=session), local=local, cloud=cloud)
    assert sticky.target.name == "cloud"
    assert not sticky.needs_state_transfer
    moved = SessionRecord(
        "moved",
        placement="sleeping",
        endpoint="http://retired-edge",
        state_uri="file:///state",
    )
    restored = policy.choose(RouteContext(session=moved), local=local, cloud=cloud)
    assert restored.target.name == "local"
    assert restored.needs_state_transfer


def test_policy_rejects_when_no_endpoint_is_healthy() -> None:
    policy = StatefulRoutingPolicy()
    local, cloud = endpoints(healthy=False)
    cloud = EndpointState("cloud", "http://cloud", healthy=False)
    with pytest.raises(NoHealthyEndpointError):
        policy.choose(RouteContext(), local=local, cloud=cloud)
