from __future__ import annotations

import pytest

from longchat_proxy.config import ConfigurationError, Settings, openai_url


def test_loopback_without_key_is_safe() -> None:
    Settings(host="127.0.0.1").validate()


def test_remote_listener_requires_authentication() -> None:
    with pytest.raises(ConfigurationError, match="unauthenticated non-loopback"):
        Settings(host="0.0.0.0").validate()


def test_remote_listener_accepts_key() -> None:
    Settings(host="0.0.0.0", client_api_key="a-long-random-secret").validate()


def test_public_config_redacts_keys() -> None:
    public = Settings(client_api_key="secret", upstream_api_key="other").public_dict()
    assert public["client_api_key"] == "<set>"
    assert public["upstream_api_key"] == "<set>"
    assert "secret" not in str(public)


def test_openai_url_accepts_base_with_or_without_v1() -> None:
    assert openai_url("http://localhost:1234", "/v1/models") == ("http://localhost:1234/v1/models")
    assert openai_url("http://localhost:1234/v1/", "/v1/models") == (
        "http://localhost:1234/v1/models"
    )
    assert openai_url("http://localhost:1234/v1", "/custom") == ("http://localhost:1234/v1/custom")


def test_compaction_trigger_must_be_zero_or_useful() -> None:
    with pytest.raises(ConfigurationError, match="zero or at least 512"):
        Settings(compaction_trigger_tokens=128).validate()
