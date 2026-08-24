from __future__ import annotations

import json
import logging
import secrets
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from .compactor import CompactionError, Compactor, SummaryClient
from .config import Settings, openai_url
from .store import create_store
from .tokens import TokenCounter

LOGGER = logging.getLogger("longchat_proxy")
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


@dataclass(slots=True)
class Metrics:
    started_at: float = field(default_factory=time.time)
    requests: int = 0
    compactions: int = 0
    cache_hits: int = 0
    summary_calls: int = 0
    compaction_errors: int = 0
    upstream_errors: int = 0


class RateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.limit = requests_per_minute
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now = time.monotonic()
        bucket = self._requests[key]
        while bucket and bucket[0] < now - 60:
            bucket.popleft()
        if len(bucket) >= self.limit:
            return False
        bucket.append(now)
        return True


def _forward_headers(request: Request, settings: Settings) -> dict[str, str]:
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP | {"authorization"}
    }
    if settings.upstream_api_key:
        headers["authorization"] = f"Bearer {settings.upstream_api_key}"
    return headers


def _response_headers(response: httpx.Response) -> dict[str, str]:
    return {key: value for key, value in response.headers.items() if key.lower() not in HOP_BY_HOP}


def _error(status: int, message: str, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": error_type}},
    )


def create_app(
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    configured = settings or Settings.from_env()
    configured.validate()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(
            level=getattr(logging, configured.log_level, logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        app.state.settings = configured
        app.state.metrics = Metrics()
        app.state.rate_limiter = RateLimiter(configured.rate_limit_rpm)
        app.state.store = create_store(
            configured.state_mode,
            configured.resolved_state_path,
            configured.state_max_records,
            configured.state_ttl_days,
        )
        app.state.client = httpx.AsyncClient(
            transport=transport,
            timeout=configured.upstream_timeout_seconds,
            follow_redirects=False,
        )
        counter = TokenCounter(
            configured.tokenizer_path,
            configured.estimator_multiplier,
            configured.image_token_reserve,
        )
        app.state.counter = counter
        app.state.compactor = Compactor(
            configured,
            counter,
            app.state.store,
            SummaryClient(configured, app.state.client),
        )
        LOGGER.info(
            "listening on %s:%s; upstream=%s; token_mode=%s; state=%s",
            configured.host,
            configured.port,
            configured.upstream_base_url,
            counter.mode,
            configured.state_mode,
        )
        yield
        await app.state.client.aclose()
        app.state.store.close()

    app = FastAPI(
        title="LongChat Proxy",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    if configured.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(configured.cors_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[
                "X-LongChat-Compacted",
                "X-LongChat-Cache-Hit",
                "X-LongChat-Estimated-Input-Tokens",
                "X-LongChat-Summary-Calls",
            ],
        )

    def authorized(request: Request) -> bool:
        if not configured.client_api_key:
            return True
        provided = request.headers.get("authorization", "")
        expected = f"Bearer {configured.client_api_key}"
        return secrets.compare_digest(provided, expected)

    @app.get("/longchat/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/longchat/ready")
    async def ready(request: Request) -> Response:
        if not authorized(request):
            return _error(401, "Invalid API key", "authentication_error")
        headers: dict[str, str] = {}
        if configured.upstream_api_key:
            headers["authorization"] = f"Bearer {configured.upstream_api_key}"
        try:
            response = await request.app.state.client.get(
                openai_url(configured.upstream_base_url, "/v1/models"),
                headers=headers,
                timeout=min(configured.upstream_timeout_seconds, 10),
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            LOGGER.warning("readiness check failed: %s", error)
            return _error(503, "Upstream unavailable", "upstream_unavailable")
        return JSONResponse({"status": "ready"})

    @app.get("/longchat/status")
    async def status(request: Request) -> Response:
        if not authorized(request):
            return _error(401, "Invalid API key", "authentication_error")
        metrics: Metrics = request.app.state.metrics
        return JSONResponse(
            {
                "status": "ok",
                "uptime_seconds": round(time.time() - metrics.started_at, 1),
                "token_counter": request.app.state.counter.mode,
                "stored_summaries": request.app.state.store.count(),
                "metrics": {
                    key: value for key, value in asdict(metrics).items() if key != "started_at"
                },
                "config": configured.public_dict(),
            }
        )

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def proxy(request: Request, path: str) -> Response:
        if not authorized(request):
            return _error(401, "Invalid API key", "authentication_error")
        client_ip = request.client.host if request.client else "unknown"
        if not request.app.state.rate_limiter.allow(client_ip):
            return _error(429, "Rate limit exceeded", "rate_limit_error")
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > configured.request_max_bytes:
                    return _error(413, "Request body is too large", "request_too_large")
            except ValueError:
                return _error(400, "Invalid Content-Length header", "invalid_request_error")
        body = await request.body()
        if len(body) > configured.request_max_bytes:
            return _error(413, "Request body is too large", "request_too_large")

        metrics: Metrics = request.app.state.metrics
        metrics.requests += 1
        metadata: dict[str, str] = {}
        request_path = "/" + path

        if request.method == "POST" and request_path.rstrip("/") == "/v1/chat/completions":
            try:
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError("JSON body must be an object")
                conversation_key = request.headers.get("x-longchat-conversation-id", "")[:200]
                result = await request.app.state.compactor.compact(payload, conversation_key)
                body = json.dumps(result.body, ensure_ascii=False, separators=(",", ":")).encode()
                metadata = {
                    "x-longchat-compacted": str(result.compacted).lower(),
                    "x-longchat-cache-hit": str(result.cache_hit).lower(),
                    "x-longchat-estimated-input-tokens": str(result.estimated_input_tokens),
                    "x-longchat-summary-calls": str(result.summaries_generated),
                }
                if result.compacted:
                    metrics.compactions += 1
                if result.cache_hit:
                    metrics.cache_hits += 1
                metrics.summary_calls += result.summaries_generated
            except (json.JSONDecodeError, ValueError):
                return _error(400, "Invalid JSON request body", "invalid_request_error")
            except CompactionError as error:
                metrics.compaction_errors += 1
                LOGGER.warning("compaction failed: %s", error)
                if configured.on_compaction_error == "error":
                    return _error(
                        503,
                        "Conversation compaction could not be completed safely",
                        "compaction_error",
                    )
                payload, _ = request.app.state.compactor.cap_output(payload)
                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()

        upstream_url = openai_url(configured.upstream_base_url, request_path)
        if request.url.query:
            upstream_url += f"?{request.url.query}"
        headers = _forward_headers(request, configured)

        if (
            request.method == "GET"
            and request_path.rstrip("/") == "/v1/models"
            and configured.hide_summary_model
            and configured.summary_model
        ):
            try:
                response = await request.app.state.client.request(
                    request.method, upstream_url, headers=headers, content=body
                )
                payload = response.json()
                if isinstance(payload.get("data"), list):
                    payload["data"] = [
                        model
                        for model in payload["data"]
                        if model.get("id") != configured.summary_model
                    ]
                return JSONResponse(
                    payload,
                    status_code=response.status_code,
                    headers={**_response_headers(response), **metadata},
                )
            except (httpx.HTTPError, ValueError) as error:
                metrics.upstream_errors += 1
                LOGGER.warning("model-list request failed: %s", error)
                return _error(502, "Upstream request failed", "upstream_error")

        try:
            upstream_request = request.app.state.client.build_request(
                request.method,
                upstream_url,
                headers=headers,
                content=body if request.method not in {"GET", "HEAD"} else None,
            )
            upstream_response = await request.app.state.client.send(upstream_request, stream=True)
        except httpx.HTTPError as error:
            metrics.upstream_errors += 1
            LOGGER.warning("upstream request failed: %s", error)
            return _error(502, "Upstream request failed", "upstream_error")

        response_headers = {**_response_headers(upstream_response), **metadata}
        return StreamingResponse(
            upstream_response.aiter_raw(),
            status_code=upstream_response.status_code,
            headers=response_headers,
            background=BackgroundTask(upstream_response.aclose),
        )

    return app


def create_default_app() -> FastAPI:
    return create_app(Settings.from_env())
