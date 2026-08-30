# SPDX-License-Identifier: Apache-2.0
import asyncio
import json

import httpx

from vllm.rwkv_stateful.registry import SessionRegistry
from vllm.rwkv_stateful.router import RouterSettings, create_app


def test_router_prefers_local_and_records_session() -> None:
    async def run() -> None:
        async def upstream(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200)
            assert request.url.host == "local"
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}}]},
                headers={"x-rwkv-state-uri": "file:///snapshot"},
            )

        upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        registry = SessionRegistry(":memory:")
        app = create_app(
            RouterSettings(
                local_url="http://local",
                cloud_url="http://cloud",
                registry_path=":memory:",
            ),
            client=upstream_client,
            registry=registry,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://router"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "rwkv", "messages": [], "stream": False},
                headers={"x-rwkv-session-id": "session-1"},
            )
        assert response.status_code == 200
        assert response.headers["x-rwkv-route"] == "local"
        assert response.headers["x-rwkv-session-id"] == "session-1"
        record = registry.get("session-1")
        assert record is not None
        assert record.placement == "local"
        assert record.state_uri == "file:///snapshot"
        await upstream_client.aclose()

    asyncio.run(run())


def test_router_falls_back_to_cloud_on_local_5xx() -> None:
    async def run() -> None:
        calls: list[str] = []

        async def upstream(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200)
            calls.append(str(request.url.host))
            if request.url.host == "local":
                return httpx.Response(503, json={"error": "busy"})
            return httpx.Response(200, json={"choices": []})

        upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        app = create_app(
            RouterSettings(
                local_url="http://local",
                cloud_url="http://cloud",
                registry_path=":memory:",
            ),
            client=upstream_client,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://router"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                content=json.dumps({"model": "rwkv", "messages": []}),
                headers={"content-type": "application/json"},
            )
        assert response.status_code == 200
        assert response.headers["x-rwkv-route"] == "cloud"
        assert "fell back to cloud" in response.headers["x-rwkv-route-reason"]
        assert calls == ["local", "cloud"]
        await upstream_client.aclose()

    asyncio.run(run())


def test_router_serializes_requests_with_the_same_session() -> None:
    async def run() -> None:
        active = 0
        max_active = 0

        async def upstream(request: httpx.Request) -> httpx.Response:
            nonlocal active, max_active
            if request.url.path == "/health":
                return httpx.Response(200)
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            active -= 1
            return httpx.Response(200, json={"choices": []})

        upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        app = create_app(
            RouterSettings(
                local_url="http://local",
                cloud_url="http://cloud",
                registry_path=":memory:",
            ),
            client=upstream_client,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://router"
        ) as client:
            responses = await asyncio.gather(
                *(
                    client.post(
                        "/v1/chat/completions",
                        json={"model": "rwkv", "messages": []},
                        headers={"x-rwkv-session-id": "shared"},
                    )
                    for _ in range(3)
                )
            )
        assert all(response.status_code == 200 for response in responses)
        assert max_active == 1
        await upstream_client.aclose()

    asyncio.run(run())
