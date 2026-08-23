from __future__ import annotations

import ipaddress
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """Raised when LongChat would start with an unsafe or impossible configuration."""


def _env(name: str, default: str = "") -> str:
    return os.environ.get(f"LONGCHAT_{name}", default).strip()


def _bool(name: str, default: bool) -> bool:
    value = _env(name)
    if not value:
        return default
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"LONGCHAT_{name} must be true or false")


def _int(name: str, default: int) -> int:
    value = _env(name)
    try:
        return int(value) if value else default
    except ValueError as error:
        raise ConfigurationError(f"LONGCHAT_{name} must be an integer") from error


def _float(name: str, default: float) -> float:
    value = _env(name)
    try:
        return float(value) if value else default
    except ValueError as error:
        raise ConfigurationError(f"LONGCHAT_{name} must be a number") from error


def _json_object(name: str) -> dict[str, Any]:
    value = _env(name)
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ConfigurationError(f"LONGCHAT_{name} must be valid JSON") from error
    if not isinstance(parsed, dict):
        raise ConfigurationError(f"LONGCHAT_{name} must be a JSON object")
    return parsed


def _csv(name: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in _env(name).split(",") if item.strip())


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def openai_url(base_url: str, path: str) -> str:
    """Join a server URL and API path while accepting bases with or without /v1."""
    base = base_url.rstrip("/")
    normalized_path = f"/{path.lstrip('/')}"
    if base.endswith("/v1") and normalized_path.startswith("/v1/"):
        normalized_path = normalized_path[3:]
    return f"{base}{normalized_path}"


@dataclass(frozen=True, slots=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 1235
    upstream_base_url: str = "http://127.0.0.1:1234"
    upstream_api_key: str = ""
    client_api_key: str = ""
    allow_unauthenticated_remote: bool = False

    context_tokens: int = 8192
    max_output_tokens: int = 2048
    recent_tokens: int = 1600
    summary_tokens: int = 400
    safety_tokens: int = 384
    min_messages: int = 6
    image_token_reserve: int = 1200

    summary_model: str = ""
    summary_base_url: str = ""
    summary_api_key: str = ""
    summary_context_tokens: int = 8192
    summary_temperature: float = 0.2
    summary_extra_body: dict[str, Any] | None = None

    tokenizer_path: str = ""
    estimator_multiplier: float = 1.10
    state_mode: str = "memory"
    state_path: str = "~/.local/share/longchat-proxy/state.sqlite3"
    state_max_records: int = 500
    state_ttl_days: int = 30

    request_max_bytes: int = 16 * 1024 * 1024
    upstream_timeout_seconds: float = 600.0
    summary_timeout_seconds: float = 300.0
    rate_limit_rpm: int = 0
    cors_origins: tuple[str, ...] = ()
    hide_summary_model: bool = True
    on_compaction_error: str = "error"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        settings = cls(
            host=_env("HOST", "127.0.0.1"),
            port=_int("PORT", 1235),
            upstream_base_url=_env("UPSTREAM_BASE_URL", "http://127.0.0.1:1234"),
            upstream_api_key=_env("UPSTREAM_API_KEY"),
            client_api_key=_env("CLIENT_API_KEY"),
            allow_unauthenticated_remote=_bool("ALLOW_UNAUTHENTICATED_REMOTE", False),
            context_tokens=_int("CONTEXT_TOKENS", 8192),
            max_output_tokens=_int("MAX_OUTPUT_TOKENS", 2048),
            recent_tokens=_int("RECENT_TOKENS", 1600),
            summary_tokens=_int("SUMMARY_TOKENS", 400),
            safety_tokens=_int("SAFETY_TOKENS", 384),
            min_messages=_int("MIN_MESSAGES", 6),
            image_token_reserve=_int("IMAGE_TOKEN_RESERVE", 1200),
            summary_model=_env("SUMMARY_MODEL"),
            summary_base_url=_env("SUMMARY_BASE_URL"),
            summary_api_key=_env("SUMMARY_API_KEY"),
            summary_context_tokens=_int("SUMMARY_CONTEXT_TOKENS", 8192),
            summary_temperature=_float("SUMMARY_TEMPERATURE", 0.2),
            summary_extra_body=_json_object("SUMMARY_EXTRA_BODY"),
            tokenizer_path=_env("TOKENIZER_PATH"),
            estimator_multiplier=_float("ESTIMATOR_MULTIPLIER", 1.10),
            state_mode=_env("STATE_MODE", "memory").lower(),
            state_path=_env("STATE_PATH", "~/.local/share/longchat-proxy/state.sqlite3"),
            state_max_records=_int("STATE_MAX_RECORDS", 500),
            state_ttl_days=_int("STATE_TTL_DAYS", 30),
            request_max_bytes=_int("REQUEST_MAX_BYTES", 16 * 1024 * 1024),
            upstream_timeout_seconds=_float("UPSTREAM_TIMEOUT_SECONDS", 600.0),
            summary_timeout_seconds=_float("SUMMARY_TIMEOUT_SECONDS", 300.0),
            rate_limit_rpm=_int("RATE_LIMIT_RPM", 0),
            cors_origins=_csv("CORS_ORIGINS"),
            hide_summary_model=_bool("HIDE_SUMMARY_MODEL", True),
            on_compaction_error=_env("ON_COMPACTION_ERROR", "error").lower(),
            log_level=_env("LOG_LEVEL", "INFO").upper(),
        )
        settings.validate()
        return settings

    @property
    def resolved_state_path(self) -> Path:
        return Path(self.state_path).expanduser()

    @property
    def resolved_summary_base_url(self) -> str:
        return (self.summary_base_url or self.upstream_base_url).rstrip("/")

    @property
    def resolved_summary_api_key(self) -> str:
        return self.summary_api_key or self.upstream_api_key

    def validate(self) -> None:
        if not self.upstream_base_url.startswith(("http://", "https://")):
            raise ConfigurationError("LONGCHAT_UPSTREAM_BASE_URL must be HTTP(S)")
        if not 1 <= self.port <= 65535:
            raise ConfigurationError("LONGCHAT_PORT must be between 1 and 65535")
        if not _is_loopback(self.host) and not (
            self.client_api_key or self.allow_unauthenticated_remote
        ):
            raise ConfigurationError(
                "Refusing an unauthenticated non-loopback listener. Set "
                "LONGCHAT_CLIENT_API_KEY or explicitly opt in with "
                "LONGCHAT_ALLOW_UNAUTHENTICATED_REMOTE=true."
            )
        numeric_positive = {
            "CONTEXT_TOKENS": self.context_tokens,
            "MAX_OUTPUT_TOKENS": self.max_output_tokens,
            "RECENT_TOKENS": self.recent_tokens,
            "SUMMARY_TOKENS": self.summary_tokens,
            "SUMMARY_CONTEXT_TOKENS": self.summary_context_tokens,
            "REQUEST_MAX_BYTES": self.request_max_bytes,
        }
        for name, value in numeric_positive.items():
            if value <= 0:
                raise ConfigurationError(f"LONGCHAT_{name} must be positive")
        if self.context_tokens <= self.max_output_tokens + self.safety_tokens + 256:
            raise ConfigurationError("Context is too small for the output and safety reserves")
        if self.summary_context_tokens <= self.summary_tokens + self.safety_tokens + 256:
            raise ConfigurationError("Summary context is too small for its output reserve")
        if self.state_mode not in {"memory", "sqlite", "off"}:
            raise ConfigurationError("LONGCHAT_STATE_MODE must be memory, sqlite, or off")
        if self.on_compaction_error not in {"error", "passthrough"}:
            raise ConfigurationError("LONGCHAT_ON_COMPACTION_ERROR must be error or passthrough")
        if self.estimator_multiplier < 1:
            raise ConfigurationError("LONGCHAT_ESTIMATOR_MULTIPLIER must be at least 1")
        if self.state_max_records <= 0 or self.state_ttl_days <= 0:
            raise ConfigurationError("State record and TTL limits must be positive")
        if self.rate_limit_rpm < 0:
            raise ConfigurationError("LONGCHAT_RATE_LIMIT_RPM cannot be negative")

    def public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("client_api_key", "upstream_api_key", "summary_api_key"):
            result[key] = "<set>" if result[key] else "<unset>"
        result["summary_extra_body"] = self.summary_extra_body or {}
        return result
