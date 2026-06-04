"""Temporary experiment: Alibaba Cloud explicit context cache via OpenAI-compatible API.

Enable with LLM_ALIBABA_CACHE_EXPERIMENT=true.

Alibaba requires:
- content as array of blocks: [{"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}]
- cache_control marks the end of the prefix to cache (everything from message[0] through that block)

Strategy for multi-turn chat:
- Put the marker on the **second-to-last** message (stable prefix before the latest user turn).
- Single-message requests: marker on that sole message.

Remove this module once cache behaviour is validated or integrated properly.
"""

from __future__ import annotations

import copy
import os
from typing import Any

_CACHE_CONTROL = {"type": "ephemeral"}


def is_enabled() -> bool:
    return os.getenv("LLM_ALIBABA_CACHE_EXPERIMENT", "").lower() in (
        "1", "true", "yes", "on",
    )


def apply_cache_markers(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of messages with Alibaba explicit-cache content shape."""
    if not messages:
        return messages

    marker_index = len(messages) - 2 if len(messages) >= 2 else 0
    out: list[dict[str, Any]] = []

    for i, msg in enumerate(messages):
        new_msg = copy.deepcopy(msg)
        new_msg["content"] = _to_cacheable_content(
            msg.get("content", ""),
            add_marker=(i == marker_index),
        )
        out.append(new_msg)

    return out


def _to_cacheable_content(content: Any, *, add_marker: bool) -> list[dict[str, Any]]:
    if isinstance(content, str):
        block: dict[str, Any] = {"type": "text", "text": content}
        if add_marker:
            block["cache_control"] = _CACHE_CONTROL.copy()
        return [block]

    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        last_text_idx = None
        for idx, item in enumerate(content):
            if isinstance(item, dict) and item.get("type") == "text":
                last_text_idx = idx
            parts.append(copy.deepcopy(item) if isinstance(item, dict) else item)

        if add_marker:
            if last_text_idx is not None:
                parts[last_text_idx] = dict(parts[last_text_idx])
                parts[last_text_idx]["cache_control"] = _CACHE_CONTROL.copy()
            else:
                parts.append({
                    "type": "text",
                    "text": "",
                    "cache_control": _CACHE_CONTROL.copy(),
                })
        return parts

    block = {"type": "text", "text": str(content)}
    if add_marker:
        block["cache_control"] = _CACHE_CONTROL.copy()
    return [block]
