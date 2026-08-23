from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


class TokenCounter:
    """Counts with a configured tokenizer, or conservatively estimates otherwise."""

    def __init__(
        self,
        tokenizer_path: str = "",
        multiplier: float = 1.10,
        image_token_reserve: int = 1200,
    ) -> None:
        self.multiplier = multiplier
        self.image_token_reserve = image_token_reserve
        self._tokenizer: Any = None
        if tokenizer_path:
            try:
                from tokenizers import Tokenizer
            except ImportError as error:
                raise RuntimeError(
                    "TOKENIZER_PATH requires: pip install 'longchat-proxy[tokenizers]'"
                ) from error
            self._tokenizer = Tokenizer.from_file(str(Path(tokenizer_path).expanduser()))

    @property
    def mode(self) -> str:
        return "tokenizer" if self._tokenizer else "conservative-estimate"

    def text(self, value: str) -> int:
        if not value:
            return 0
        if self._tokenizer:
            return len(self._tokenizer.encode(value).ids)
        byte_estimate = len(value.encode("utf-8")) / 3.1
        segment_estimate = len(re.findall(r"\w+|[^\w\s]", value, flags=re.UNICODE)) * 1.25
        return max(1, math.ceil(max(byte_estimate, segment_estimate) * self.multiplier))

    def content(self, value: Any) -> int:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            total = 0
            for item in value:
                if not isinstance(item, dict):
                    total += self.text(json.dumps(item, ensure_ascii=False))
                    continue
                item_type = item.get("type")
                if item_type in {"image", "image_url", "input_image", "video", "input_video"}:
                    total += self.image_token_reserve
                elif isinstance(item.get("text"), str):
                    total += self.text(item["text"])
                else:
                    total += self.text(json.dumps(item, ensure_ascii=False, sort_keys=True))
            return total
        if value is None:
            return 0
        return self.text(json.dumps(value, ensure_ascii=False, sort_keys=True))

    def message(self, message: dict[str, Any]) -> int:
        total = 12 + self.text(str(message.get("role", "")))
        total += self.content(message.get("content"))
        for key in ("name", "tool_call_id", "reasoning_content"):
            if message.get(key):
                total += self.text(str(message[key]))
        if message.get("tool_calls"):
            total += self.text(
                json.dumps(message["tool_calls"], ensure_ascii=False, sort_keys=True)
            )
        return total

    def messages(self, messages: list[dict[str, Any]]) -> int:
        return 12 + sum(self.message(message) for message in messages)

    def request_overhead(self, body: dict[str, Any]) -> int:
        total = 0
        for key in ("tools", "tool_choice", "response_format"):
            if body.get(key) is not None:
                total += self.text(json.dumps(body[key], ensure_ascii=False, sort_keys=True))
        return total
