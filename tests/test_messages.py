from __future__ import annotations

from longchat_proxy.messages import (
    merge_instructions_and_memory,
    retained_user_boundary,
    split_instructions,
    summary_document,
)


def test_system_and_developer_messages_are_merged_with_memory() -> None:
    instructions = [
        {"role": "system", "content": "Be precise."},
        {"role": "developer", "content": "Use short answers."},
    ]
    merged = merge_instructions_and_memory(instructions, "- User prefers blue")
    assert merged["role"] == "system"
    assert "Be precise" in merged["content"]
    assert "Use short answers" in merged["content"]
    assert "User prefers blue" in merged["content"]


def test_instruction_split_only_consumes_leading_instructions() -> None:
    messages = [
        {"role": "system", "content": "A"},
        {"role": "user", "content": "B"},
        {"role": "assistant", "content": "C"},
    ]
    instructions, turns = split_instructions(messages)
    assert len(instructions) == 1
    assert [item["role"] for item in turns] == ["user", "assistant"]


def test_boundary_rewinds_assistant_to_user() -> None:
    turns = [
        {"role": "user"},
        {"role": "assistant"},
        {"role": "user"},
        {"role": "assistant"},
    ]
    assert retained_user_boundary(turns, 3) == 2


def test_summary_document_preserves_tool_metadata() -> None:
    document = summary_document(
        {
            "role": "assistant",
            "content": "Calling weather",
            "tool_calls": [{"id": "1", "function": {"name": "weather"}}],
        }
    )
    assert document["tool_calls"][0]["function"]["name"] == "weather"
