# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from vllm.rwkv_stateful.policy import (
    EndpointState,
    NoHealthyEndpointError,
    RouteContext,
    StatefulRoutingPolicy,
)
from vllm.rwkv_stateful.registry import SessionRecord, SessionRegistry

_FORWARDED_RESPONSE_HEADERS = frozenset(
    {"content-type", "cache-control", "x-request-id", "retry-after"}
)
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)
_CONTROL_HEADERS = frozenset(
    {
        "x-rwkv-state-uri",
        "x-rwkv-state-sha256",
        "x-rwkv-state-generation",
        "x-rwkv-route",
        "x-rwkv-route-reason",
    }
)


@dataclass(frozen=True)
class RouterSettings:
    local_url: str = "http://rwkv-local:8000"
    cloud_url: str = "http://rwkv-cloud:8000"
    local_capacity: int = 8
    cloud_capacity: int = 128
    local_max_prompt_tokens: int = 8192
    local_utilization_limit: float = 0.8
    sticky_utilization_limit: float = 1.0
    probe_timeout_seconds: float = 1.0
    request_timeout_seconds: float = 600.0
    registry_path: str = "/var/lib/rwkv-router/sessions.db"
    local_api_key: str = ""
    cloud_api_key: str = ""
    state_transfer_supported: bool = False

    @classmethod
    def from_env(cls) -> RouterSettings:
        def integer(name: str, default: int) -> int:
            return int(os.environ.get(name, default))

        def floating(name: str, default: float) -> float:
            return float(os.environ.get(name, default))

        return cls(
            local_url=os.environ.get("RWKV_LOCAL_URL", cls.local_url),
            cloud_url=os.environ.get("RWKV_CLOUD_URL", cls.cloud_url),
            local_capacity=integer("RWKV_LOCAL_CAPACITY", cls.local_capacity),
            cloud_capacity=integer("RWKV_CLOUD_CAPACITY", cls.cloud_capacity),
            local_max_prompt_tokens=integer(
                "RWKV_LOCAL_MAX_PROMPT_TOKENS", cls.local_max_prompt_tokens
            ),
            local_utilization_limit=floating(
                "RWKV_LOCAL_UTILIZATION_LIMIT", cls.local_utilization_limit
            ),
            sticky_utilization_limit=floating(
                "RWKV_STICKY_UTILIZATION_LIMIT", cls.sticky_utilization_limit
            ),
            probe_timeout_seconds=floating(
                "RWKV_PROBE_TIMEOUT_SECONDS", cls.probe_timeout_seconds
            ),
            request_timeout_seconds=floating(
                "RWKV_REQUEST_TIMEOUT_SECONDS", cls.request_timeout_seconds
            ),
            registry_path=os.environ.get("RWKV_SESSION_REGISTRY", cls.registry_path),
            local_api_key=os.environ.get("RWKV_LOCAL_API_KEY", ""),
            cloud_api_key=os.environ.get("RWKV_CLOUD_API_KEY", ""),
            state_transfer_supported=os.environ.get(
                "RWKV_STATE_TRANSFER_SUPPORTED", "0"
            ).lower()
            not in {"0", "false", "no"},
        )


class RouterMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "rwkv_router_requests_total",
            "Requests routed by target and outcome.",
            ("target", "outcome"),
            registry=self.registry,
        )
        self.fallbacks = Counter(
            "rwkv_router_fallbacks_total",
            "Requests retried on the other endpoint.",
            ("from_target", "to_target"),
            registry=self.registry,
        )
        self.state_transfers = Counter(
            "rwkv_router_state_transfers_total",
            "Requests carrying a state snapshot reference.",
            ("target",),
            registry=self.registry,
        )
        self.latency = Histogram(
            "rwkv_router_request_seconds",
            "Time until the upstream response headers arrive.",
            ("target",),
            registry=self.registry,
        )
        self.inflight = Gauge(
            "rwkv_router_inflight_requests",
            "Requests currently assigned to each endpoint.",
            ("target",),
            registry=self.registry,
        )


class StatefulRouter:
    def __init__(
        self,
        settings: RouterSettings,
        registry: SessionRegistry,
        client: httpx.AsyncClient,
        metrics: RouterMetrics,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.client = client
        self.metrics = metrics
        self.policy = StatefulRoutingPolicy(
            local_utilization_limit=settings.local_utilization_limit,
            sticky_utilization_limit=settings.sticky_utilization_limit,
            local_max_prompt_tokens=settings.local_max_prompt_tokens,
        )
        self._inflight = {"local": 0, "cloud": 0}
        self._inflight_lock = asyncio.Lock()
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._session_locks_guard = asyncio.Lock()

    async def _acquire_session(self, session_id: str) -> asyncio.Lock:
        async with self._session_locks_guard:
            lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        await lock.acquire()
        return lock

    async def _healthy(self, base_url: str) -> bool:
        try:
            response = await self.client.get(
                f"{base_url.rstrip('/')}/health",
                timeout=self.settings.probe_timeout_seconds,
            )
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    async def endpoints(self) -> tuple[EndpointState, EndpointState]:
        local_health, cloud_health = await asyncio.gather(
            self._healthy(self.settings.local_url),
            self._healthy(self.settings.cloud_url),
        )
        async with self._inflight_lock:
            local_depth = self._inflight["local"]
            cloud_depth = self._inflight["cloud"]
        transfer = self.settings.state_transfer_supported
        return (
            EndpointState(
                "local",
                self.settings.local_url.rstrip("/"),
                local_health,
                local_depth,
                self.settings.local_capacity,
                transfer,
            ),
            EndpointState(
                "cloud",
                self.settings.cloud_url.rstrip("/"),
                cloud_health,
                cloud_depth,
                self.settings.cloud_capacity,
                transfer,
            ),
        )

    @staticmethod
    def _session_id(request: Request, body: dict[str, Any]) -> str:
        header = request.headers.get("x-rwkv-session-id")
        metadata = body.get("metadata")
        metadata_id = metadata.get("session_id") if isinstance(metadata, dict) else None
        return str(header or metadata_id or body.get("request_id") or uuid.uuid4())

    @staticmethod
    def _estimate_prompt_tokens(body: dict[str, Any]) -> int:
        prompt = body.get("prompt", "")
        if "messages" in body:
            prompt = json.dumps(body["messages"], ensure_ascii=False)
        if isinstance(prompt, list):
            prompt = " ".join(str(item) for item in prompt)
        return max(1, (len(str(prompt)) + 3) // 4)

    def _request_headers(
        self,
        request: Request,
        target: EndpointState,
        session: SessionRecord | None,
        needs_state_transfer: bool,
    ) -> dict[str, str]:
        headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() not in _HOP_BY_HOP_HEADERS | _CONTROL_HEADERS
        }
        api_key = (
            self.settings.local_api_key
            if target.name == "local"
            else self.settings.cloud_api_key
        )
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        if needs_state_transfer and session is not None:
            headers["x-rwkv-state-uri"] = session.state_uri
            headers["x-rwkv-state-sha256"] = session.state_sha256
            headers["x-rwkv-state-generation"] = str(session.generation)
            self.metrics.state_transfers.labels(target.name).inc()
        return headers

    async def _send(
        self,
        request: Request,
        body: dict[str, Any],
        target: EndpointState,
        session: SessionRecord | None,
        needs_state_transfer: bool,
    ) -> httpx.Response:
        url = f"{target.url}{request.url.path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        headers = self._request_headers(request, target, session, needs_state_transfer)
        upstream_request = self.client.build_request(
            request.method,
            url,
            headers=headers,
            json=body,
            timeout=self.settings.request_timeout_seconds,
        )
        async with self._inflight_lock:
            self._inflight[target.name] += 1
            self.metrics.inflight.labels(target.name).inc()
        started = time.perf_counter()
        try:
            response = await self.client.send(upstream_request, stream=True)
            self.metrics.latency.labels(target.name).observe(
                time.perf_counter() - started
            )
            return response
        except Exception:
            await self._finish(target.name)
            raise

    async def _finish(self, target_name: str) -> None:
        async with self._inflight_lock:
            self._inflight[target_name] = max(0, self._inflight[target_name] - 1)
            self.metrics.inflight.labels(target_name).dec()

    def _record_placement(
        self,
        session_id: str,
        previous: SessionRecord | None,
        target: EndpointState,
        body: dict[str, Any],
        response: httpx.Response,
    ) -> None:
        model_id = str(body.get("model", previous.model_id if previous else ""))
        state_uri = response.headers.get(
            "x-rwkv-state-uri", previous.state_uri if previous else ""
        )
        state_sha256 = response.headers.get(
            "x-rwkv-state-sha256", previous.state_sha256 if previous else ""
        )
        record = SessionRecord(
            session_id=session_id,
            placement=target.name,
            endpoint=target.url,
            model_id=model_id,
            model_revision=previous.model_revision if previous else "",
            state_uri=state_uri,
            state_sha256=state_sha256,
            lease_owner=previous.lease_owner if previous else "",
            lease_expires_at=previous.lease_expires_at if previous else 0.0,
        )
        self.registry.upsert(record)

    async def proxy(self, request: Request) -> Response:
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")
        session_id = self._session_id(request, body)
        session_lock = await self._acquire_session(session_id)
        response: httpx.Response | None = None
        target: EndpointState | None = None
        try:
            session = self.registry.get(session_id)
            local, cloud = await self.endpoints()
            force_cloud = request.headers.get("x-rwkv-force-cloud", "").lower() in {
                "1",
                "true",
                "yes",
            }
            try:
                decision = self.policy.choose(
                    RouteContext(
                        session=session,
                        prompt_tokens=self._estimate_prompt_tokens(body),
                        force_cloud=force_cloud,
                    ),
                    local=local,
                    cloud=cloud,
                )
            except NoHealthyEndpointError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

            target = decision.target
            route_reason = decision.reason
            failed_target = target
            network_error: httpx.HTTPError | None = None
            try:
                response = await self._send(
                    request,
                    body,
                    target,
                    session,
                    decision.needs_state_transfer,
                )
            except httpx.HTTPError as exc:
                network_error = exc

            upstream_error = response is not None and response.status_code >= 500
            if network_error is not None or upstream_error:
                fallback = cloud if failed_target.name == "local" else local
                if not fallback.healthy and network_error is not None:
                    self.metrics.requests.labels(
                        failed_target.name, "network_error"
                    ).inc()
                    raise HTTPException(
                        status_code=502, detail=str(network_error)
                    ) from network_error
                if fallback.healthy:
                    if response is not None:
                        await response.aclose()
                        response = None
                        await self._finish(failed_target.name)
                    error_kind = (
                        "network error"
                        if network_error is not None
                        else "upstream error"
                    )
                    route_reason = (
                        f"{failed_target.name} {error_kind}; "
                        f"fell back to {fallback.name}"
                    )
                    target = fallback
                    self.metrics.fallbacks.labels(
                        failed_target.name, fallback.name
                    ).inc()
                    try:
                        response = await self._send(
                            request,
                            body,
                            target,
                            session,
                            self.policy.transfer_needed(session, target),
                        )
                    except httpx.HTTPError as fallback_exc:
                        self.metrics.requests.labels(target.name, "network_error").inc()
                        raise HTTPException(
                            status_code=502, detail=str(fallback_exc)
                        ) from fallback_exc

            if response is None:
                raise HTTPException(status_code=502, detail="upstream request failed")

            self._record_placement(session_id, session, target, body, response)
            outcome = "success" if response.status_code < 500 else "upstream_error"
            self.metrics.requests.labels(target.name, outcome).inc()
            response_headers = {
                name: value
                for name, value in response.headers.items()
                if name.lower() in _FORWARDED_RESPONSE_HEADERS
            }
            response_headers.update(
                {
                    "x-rwkv-session-id": session_id,
                    "x-rwkv-route": target.name,
                    "x-rwkv-route-reason": route_reason,
                }
            )

            if bool(body.get("stream")):
                stream_response = response
                stream_target = target
                response = None

                async def stream() -> AsyncIterator[bytes]:
                    try:
                        async for chunk in stream_response.aiter_raw():
                            yield chunk
                    finally:
                        await stream_response.aclose()
                        await self._finish(stream_target.name)
                        session_lock.release()

                return StreamingResponse(
                    stream(),
                    status_code=stream_response.status_code,
                    headers=response_headers,
                    media_type=None,
                )

            try:
                content = await response.aread()
                status_code = response.status_code
            finally:
                await response.aclose()
                response = None
                await self._finish(target.name)
            session_lock.release()
            return Response(
                content=content,
                status_code=status_code,
                headers=response_headers,
                media_type=None,
            )
        except BaseException:
            if response is not None and target is not None:
                await response.aclose()
                await self._finish(target.name)
            if session_lock.locked():
                session_lock.release()
            raise


def create_app(
    settings: RouterSettings | None = None,
    *,
    client: httpx.AsyncClient | None = None,
    registry: SessionRegistry | None = None,
) -> FastAPI:
    settings = settings or RouterSettings.from_env()
    if settings.registry_path != ":memory:":
        Path(settings.registry_path).expanduser().parent.mkdir(
            parents=True, exist_ok=True
        )
    owns_registry = registry is None
    registry = registry or SessionRegistry(settings.registry_path)
    owns_client = client is None
    client = client or httpx.AsyncClient()
    metrics = RouterMetrics()
    router = StatefulRouter(settings, registry, client, metrics)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        if owns_client:
            await client.aclose()
        if owns_registry:
            registry.close()

    app = FastAPI(
        title="RWKV state-aware edge-cloud router",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.rwkv_router = router

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, Any]:
        local, cloud = await router.endpoints()
        if not local.healthy and not cloud.healthy:
            raise HTTPException(status_code=503, detail="no inference endpoint ready")
        return {"local": local.healthy, "cloud": cloud.healthy}

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        return Response(
            generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST
        )

    @app.get("/v1/rwkv/sessions/{session_id}")
    async def get_session(session_id: str) -> dict[str, Any]:
        record = registry.get(session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="session not found")
        public = record.__dict__.copy()
        public["has_state"] = bool(public.pop("state_uri"))
        public.pop("state_sha256")
        return public

    for path in ("/v1/chat/completions", "/v1/completions", "/v1/responses"):
        app.add_api_route(path, router.proxy, methods=["POST"])

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RWKV local-first/cloud-fallback router"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()


__all__ = ["RouterSettings", "StatefulRouter", "create_app", "main"]
