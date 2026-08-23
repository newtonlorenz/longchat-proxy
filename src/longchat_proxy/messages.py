from __future__ import annotations

import hashlib
import json
from typing import Any

MEMORY_HEADING = "LongChat rolling memory"


def canonical_hash(messages: list[dict[str, Any]], count: int) -> str:
    payload = json.dumps(
        messages[:count], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_instructions(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = 0
    while index < len(messages) and messages[index].get("role") in {"system", "developer"}:
        index += 1
    return messages[:index], messages[index:]


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, dict) and item.get("type") in {
                "image",
                "image_url",
                "input_image",
            }:
                parts.append("[image attachment]")
        return "\n".join(parts)
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def summary_document(message: dict[str, Any]) -> dict[str, Any]:
    document: dict[str, Any] = {
        "role": str(message.get("role", "unknown")),
        "content": content_text(message.get("content")),
    }
    for key in ("name", "tool_call_id"):
        if message.get(key) is not None:
            document[key] = message[key]
    if message.get("tool_calls") is not None:
        document["tool_calls"] = message["tool_calls"]
    return document


def memory_block(summary: str) -> str:
    return (
        f"## {MEMORY_HEADING}\n"
        "The following is a compact record of earlier turns. Treat it as established context. "
        "If it conflicts with recent verbatim turns, the recent turns win.\n\n"
        f"{summary.strip()}"
    )


def merge_instructions_and_memory(
    instructions: list[dict[str, Any]], summary: str
) -> dict[str, str]:
    sections: list[str] = []
    for message in instructions:
        text = content_text(message.get("content")).strip()
        if text:
            label = str(message.get("role", "system")).capitalize()
            sections.append(f"## Original {label} instructions\n{text}")
    sections.append(memory_block(summary))
    return {"role": "system", "content": "\n\n".join(sections)}


def retained_user_boundary(turns: list[dict[str, Any]], index: int) -> int:
    """Move backwards to a user turn so tool/assistant chains remain well formed."""
    start = min(max(index, 0), len(turns))
    while start > 0 and start < len(turns) and turns[start].get("role") != "user":
        start -= 1
    return start
