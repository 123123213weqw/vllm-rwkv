# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

from vllm.rwkv_stateful.registry import SessionRecord


class NoHealthyEndpointError(RuntimeError):
    """Raised when neither the local nor cloud inference endpoint is usable."""


@dataclass(frozen=True)
class EndpointState:
    name: str
    url: str
    healthy: bool = True
    queue_depth: int = 0
    capacity: int = 1
    state_transfer_supported: bool = False
    cost_per_million_tokens: float = 0.0

    @property
    def utilization(self) -> float:
        return self.queue_depth / max(self.capacity, 1)


@dataclass(frozen=True)
class RouteContext:
    session: SessionRecord | None = None
    prompt_tokens: int = 0
    force_cloud: bool = False


@dataclass(frozen=True)
class RouteDecision:
    target: EndpointState
    reason: str
    needs_state_transfer: bool = False


class StatefulRoutingPolicy:
    """Sticky local-first routing with cloud spillover and state awareness."""

    def __init__(
        self,
        *,
        local_utilization_limit: float = 0.8,
        sticky_utilization_limit: float = 1.0,
        local_max_prompt_tokens: int = 8192,
    ) -> None:
        if not 0 < local_utilization_limit <= sticky_utilization_limit:
            raise ValueError("invalid routing utilization limits")
        self.local_utilization_limit = local_utilization_limit
        self.sticky_utilization_limit = sticky_utilization_limit
        self.local_max_prompt_tokens = local_max_prompt_tokens

    @staticmethod
    def transfer_needed(session: SessionRecord | None, target: EndpointState) -> bool:
        if (
            session is None
            or not session.state_uri
            or not target.state_transfer_supported
        ):
            return False
        return bool(session.endpoint and session.endpoint != target.url)

    def choose(
        self,
        context: RouteContext,
        *,
        local: EndpointState,
        cloud: EndpointState,
    ) -> RouteDecision:
        session = context.session
        if context.force_cloud and cloud.healthy:
            return RouteDecision(
                cloud,
                "request forced to cloud",
                self.transfer_needed(session, cloud),
            )

        if session is not None and session.endpoint in {local.url, cloud.url}:
            sticky = local if session.endpoint == local.url else cloud
            if sticky.healthy and sticky.utilization < self.sticky_utilization_limit:
                return RouteDecision(
                    sticky, "session placement is still healthy", False
                )

        local_fits = (
            local.healthy
            and local.utilization < self.local_utilization_limit
            and context.prompt_tokens <= self.local_max_prompt_tokens
        )
        if local_fits:
            return RouteDecision(
                local,
                "local capacity available",
                self.transfer_needed(session, local),
            )
        if cloud.healthy:
            reason = (
                "local prompt limit exceeded"
                if context.prompt_tokens > self.local_max_prompt_tokens
                else "local endpoint saturated or unhealthy"
            )
            return RouteDecision(
                cloud,
                reason,
                self.transfer_needed(session, cloud),
            )
        if local.healthy:
            return RouteDecision(
                local,
                "cloud unavailable; using local endpoint",
                self.transfer_needed(session, local),
            )
        raise NoHealthyEndpointError("no healthy RWKV inference endpoint")


__all__ = [
    "EndpointState",
    "NoHealthyEndpointError",
    "RouteContext",
    "RouteDecision",
    "StatefulRoutingPolicy",
]
