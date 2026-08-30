# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm.rwkv_stateful.registry import (
    SessionConflictError,
    SessionRecord,
    SessionRegistry,
)


def test_registry_compare_and_swap_and_listing() -> None:
    with SessionRegistry(":memory:") as registry:
        first = registry.upsert(
            SessionRecord("session-1", placement="local", endpoint="http://local"),
            expected_generation=0,
            now=10,
        )
        assert first.generation == 1
        assert registry.list() == [first]
        second = registry.upsert(
            SessionRecord("session-1", placement="cloud", endpoint="http://cloud"),
            expected_generation=1,
            now=11,
        )
        assert second.generation == 2
        with pytest.raises(SessionConflictError, match="generation changed"):
            registry.upsert(second, expected_generation=1)


def test_registry_lease_expires_and_checks_owner() -> None:
    with SessionRegistry(":memory:") as registry:
        registry.upsert(SessionRecord("session-1"), now=1)
        claimed = registry.claim("session-1", "worker-a", ttl=10, now=2)
        assert claimed.lease_expires_at == 12
        with pytest.raises(SessionConflictError, match="worker-a"):
            registry.claim("session-1", "worker-b", ttl=10, now=3)
        reclaimed = registry.claim("session-1", "worker-b", ttl=10, now=13)
        assert reclaimed.lease_owner == "worker-b"
        released = registry.release("session-1", "worker-b", now=14)
        assert released.lease_owner == ""
