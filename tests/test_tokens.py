from __future__ import annotations

from longchat_proxy.tokens import TokenCounter


def test_estimator_counts_text_and_images() -> None:
    counter = TokenCounter(image_token_reserve=900)
    text = counter.message({"role": "user", "content": "A concise question"})
    image = counter.message(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            ],
        }
    )
    assert image > text + 800


def test_tools_are_included_in_request_overhead() -> None:
    counter = TokenCounter()
    request = {"tools": [{"type": "function", "function": {"name": "x"}}]}
    assert counter.request_overhead(request) > 0
