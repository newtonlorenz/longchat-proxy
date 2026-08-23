from __future__ import annotations

from typing import Any

import pytest

from longchat_proxy.compactor import Compactor
from longchat_proxy.config import Settings
from longchat_proxy.store import MemoryStore
from longchat_proxy.tokens import TokenCounter


class FakeSummarizer:
    def __init__(self) -> None:
        self.calls = 0

    async def update(
        self, previous: str, messages: list[dict[str, Any]], fallback_model: str
    ) -> str:
        self.calls += 1
        text = str(messages)
        remembered = "- Facts: exact code ORBITAL-LIME" if "ORBITAL-LIME" in text else ""
        return f"{previous}\n{remembered}\n- Open items: continue the conversation".strip()


def long_body(with_system: bool = False) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if with_system:
        messages.append({"role": "system", "content": "Be concise and accurate."})
    messages.extend(
        [
            {"role": "user", "content": "Remember the exact code ORBITAL-LIME."},
            {"role": "assistant", "content": "I will remember it."},
        ]
    )
    filler = "Discuss materials, timing, budget, ventilation and lighting. " * 7
    for index in range(14):
        messages.append({"role": "user", "content": f"Turn {index}. {filler}"})
        messages.append({"role": "assistant", "content": f"Answer {index}. {filler}"})
    messages.append({"role": "user", "content": "What was the exact code?"})
    return {"model": "main-model", "messages": messages, "max_tokens": 512}


@pytest.mark.asyncio
async def test_long_request_compacts_and_preserves_recent_user_boundary() -> None:
    settings = Settings(
        context_tokens=1800,
        max_output_tokens=512,
        recent_tokens=400,
        summary_tokens=120,
        summary_context_tokens=1600,
    )
    summarizer = FakeSummarizer()
    compactor = Compactor(settings, TokenCounter(), MemoryStore(), summarizer)  # type: ignore[arg-type]
    result = await compactor.compact(long_body())
    assert result.compacted is True
    assert result.summaries_generated >= 1
    assert result.estimated_input_tokens <= 1800 - 512 - settings.safety_tokens
    assert result.body["messages"][0]["role"] == "system"
    assert "ORBITAL-LIME" in result.body["messages"][0]["content"]
    assert result.body["messages"][1]["role"] == "user"
    assert result.body["messages"][-1]["content"] == "What was the exact code?"


@pytest.mark.asyncio
async def test_existing_system_prompt_is_merged_not_duplicated() -> None:
    settings = Settings(
        context_tokens=1800,
        max_output_tokens=512,
        recent_tokens=400,
        summary_tokens=120,
        summary_context_tokens=1600,
    )
    compactor = Compactor(settings, TokenCounter(), MemoryStore(), FakeSummarizer())  # type: ignore[arg-type]
    result = await compactor.compact(long_body(with_system=True))
    system_messages = [item for item in result.body["messages"] if item["role"] == "system"]
    assert len(system_messages) == 1
    assert "Be concise and accurate" in system_messages[0]["content"]


@pytest.mark.asyncio
async def test_repeated_prefix_uses_cached_summary() -> None:
    settings = Settings(
        context_tokens=1800,
        max_output_tokens=512,
        recent_tokens=400,
        summary_tokens=120,
        summary_context_tokens=1600,
    )
    summarizer = FakeSummarizer()
    store = MemoryStore()
    compactor = Compactor(settings, TokenCounter(), store, summarizer)  # type: ignore[arg-type]
    first = await compactor.compact(long_body(), "conversation-1")
    first_calls = summarizer.calls
    second = await compactor.compact(long_body(), "conversation-1")
    assert first.compacted and second.compacted
    assert second.cache_hit is True
    assert summarizer.calls == first_calls


@pytest.mark.asyncio
async def test_main_request_controls_and_early_trigger() -> None:
    settings = Settings(
        context_tokens=8192,
        max_output_tokens=1024,
        recent_tokens=400,
        compaction_trigger_tokens=1500,
        summary_tokens=120,
        summary_context_tokens=1600,
        request_extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        drop_tools=True,
    )
    body = long_body()
    body["tools"] = [{"type": "function", "function": {"name": "volatile"}}]
    body["chat_template_kwargs"] = {"reasoning_effort": "low"}
    compactor = Compactor(settings, TokenCounter(), MemoryStore(), FakeSummarizer())  # type: ignore[arg-type]
    result = await compactor.compact(body)
    assert result.compacted is True
    assert "tools" not in result.body
    assert result.body["chat_template_kwargs"] == {
        "reasoning_effort": "low",
        "enable_thinking": False,
    }
