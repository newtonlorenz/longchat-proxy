from __future__ import annotations

import json

import httpx
import pytest

from longchat_proxy.config import Settings
from longchat_proxy.proxy import create_app


class EventStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n'
        yield b"data: [DONE]\n\n"


def upstream(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/models":
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"id": "main-model"}, {"id": "summary-model"}],
            },
        )
    if request.url.path == "/v1/chat/completions":
        payload = json.loads(request.content)
        assert payload["max_tokens"] <= 64
        if payload.get("stream"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=EventStream(),
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})
    return httpx.Response(404, json={"error": "not found"})


@pytest.mark.asyncio
async def test_auth_model_hiding_and_streaming() -> None:
    settings = Settings(
        client_api_key="client-secret",
        summary_model="summary-model",
        max_output_tokens=64,
    )
    app = create_app(settings, transport=httpx.MockTransport(upstream))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy"
        ) as client,
    ):
        denied = await client.get("/v1/models")
        assert denied.status_code == 401
        headers = {"authorization": "Bearer client-secret"}
        models = await client.get("/v1/models", headers=headers)
        assert [item["id"] for item in models.json()["data"]] == ["main-model"]
        response = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "main-model",
                "messages": [{"role": "user", "content": "Reply OK"}],
                "max_tokens": 1000,
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert "[DONE]" in response.text
        assert response.headers["x-longchat-compacted"] == "false"


@pytest.mark.asyncio
async def test_status_never_exposes_secret() -> None:
    settings = Settings(client_api_key="do-not-leak", upstream_api_key="also-secret")
    app = create_app(settings, transport=httpx.MockTransport(upstream))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy"
        ) as client,
    ):
        response = await client.get(
            "/longchat/status", headers={"authorization": "Bearer do-not-leak"}
        )
        assert response.status_code == 200
        assert "do-not-leak" not in response.text
        assert "also-secret" not in response.text
