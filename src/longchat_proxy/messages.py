from __future__ import annotations

import hashlib
import json
import re
from typing import Any

MEMORY_HEADING = "LongChat rolling memory"
TIMESTAMP_LINE = re.compile(r"(?im)^\s*Current date and time:[^\n]*(?:\n|$)")


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


def sanitize_client_messages(
    messages: list[dict[str, Any]],
    *,
    strip_system_timestamp: bool = False,
    drop_system_sections: tuple[str, ...] = (),
    system_append: str = "",
) -> list[dict[str, Any]]:
    """Remove configured volatile boilerplate from leading instructions."""
    cleaned = [dict(message) for message in messages]
    instruction_count = 0
    while instruction_count < len(cleaned) and cleaned[instruction_count].get("role") in {
        "system",
        "developer",
    }:
        item = cleaned[instruction_count]
        content = item.get("content")
        if isinstance(content, str):
            if strip_system_timestamp:
                content = TIMESTAMP_LINE.sub("", content)
            for heading in drop_system_sections:
                pattern = re.compile(
                    rf"(?ims)(?:^|\n)#[#]*[ \t]+{re.escape(heading)}[ \t]*\n"
                    r".*?(?=(?:\n#[#]*[ \t]+\S)|\Z)"
                )
                content = pattern.sub("\n", content)
            item["content"] = content.strip()
        instruction_count += 1

    appendix = system_append.strip()
    if appendix:
        if instruction_count:
            original = content_text(cleaned[0].get("content")).strip()
            cleaned[0]["content"] = f"{original}\n\n{appendix}" if original else appendix
        else:
            cleaned.insert(0, {"role": "system", "content": appendix})
    return cleaned


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
