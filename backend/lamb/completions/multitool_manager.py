"""
Multitool context sources manager.

Extracts tool configurations from assistant metadata and executes RAG
for each tool in parallel. Tool 0 is always derived from top-level fields.
Additional tools come from the 'tools' array when multitools is enabled.

NOTE: _create_tool_assistant uses Pydantic model_copy() to override fields.
This works because all RAG processors read from assistant.metadata (JSON string)
and assistant.RAG_collections. If a future RAG processor reads other Assistant
fields in unexpected ways, this approach may need revisiting.
"""

import asyncio
import json
from typing import Any, Callable, Coroutine, Dict, List

from lamb.lamb_classes import Assistant
from lamb.logging_config import get_logger

logger = get_logger(__name__, component="MULTITOOL")

TOOL_CONFIG_FIELDS = (
    "rag_processor",
    "RAG_collections",
    "RAG_Top_k",
    "rubric_id",
    "rubric_format",
    "file_path",
)


def _parse_metadata(assistant: Assistant) -> Dict[str, Any]:
    metadata_str = assistant.metadata
    if not metadata_str:
        return {}
    try:
        return json.loads(metadata_str)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse metadata for assistant %s", assistant.id)
        return {}


def get_all_tools_config(assistant: Assistant) -> List[Dict[str, Any]]:
    """
    Extract all tool configurations from assistant metadata.

    Returns list of tool config dicts ordered by tool index (0, 1, 2, ...).
    Each dict includes per-tool fields and a 'context_key' placeholder name.
    """
    metadata = _parse_metadata(assistant)
    tools: List[Dict[str, Any]] = []

    tool_0: Dict[str, Any] = {"context_key": "context"}
    for field in TOOL_CONFIG_FIELDS:
        value = metadata.get(field)
        if value is not None:
            tool_0[field] = value

    if assistant.RAG_collections:
        tool_0["RAG_collections"] = assistant.RAG_collections

    if "rag_processor" not in tool_0:
        tool_0["rag_processor"] = ""

    tools.append(tool_0)

    if not metadata.get("multitools", False):
        return tools

    additional_tools = metadata.get("tools", [])
    if not isinstance(additional_tools, list):
        logger.warning("tools field is not a list for assistant %s, ignoring", assistant.id)
        return tools

    for idx, tool_entry in enumerate(additional_tools, start=1):
        if not isinstance(tool_entry, dict):
            logger.warning("Tool entry %d is not a dict for assistant %s, skipping", idx, assistant.id)
            continue

        context_key = f"context{idx + 1}"
        tool_config: Dict[str, Any] = {"context_key": context_key}
        for field in TOOL_CONFIG_FIELDS:
            if field in tool_entry:
                tool_config[field] = tool_entry[field]

        if "rag_processor" not in tool_config:
            tool_config["rag_processor"] = ""

        tools.append(tool_config)

    logger.info("Assistant %s: %d tools configured", assistant.id, len(tools))
    return tools


def _create_tool_assistant(original: Assistant, tool_config: Dict[str, Any]) -> Assistant:
    """
    Create a copy of the assistant with metadata overridden for a specific tool.

    Uses Pydantic model_copy() so that RAG processors see the correct
    RAG_collections / rubric_id / file_path for this tool.
    """
    metadata = _parse_metadata(original)
    for field in TOOL_CONFIG_FIELDS:
        if field in tool_config:
            metadata[field] = tool_config[field]

    overrides: Dict[str, Any] = {"api_callback": json.dumps(metadata)}
    if "RAG_collections" in tool_config:
        overrides["RAG_collections"] = tool_config["RAG_collections"]
    if "RAG_Top_k" in tool_config:
        overrides["RAG_Top_k"] = tool_config["RAG_Top_k"]

    return original.model_copy(update=overrides)


async def get_all_rag_contexts(
    assistant: Assistant,
    request: Dict[str, Any],
    rag_processors: Dict[str, Any],
    get_rag_context_fn: Callable[..., Coroutine],
) -> Dict[str, Any]:
    """
    Execute RAG for each configured tool in parallel and collect all contexts.

    Sync RAG processors are wrapped with asyncio.to_thread() so they can
    run concurrently with async ones.

    Returns dict mapping context_key -> full rag_result dict:
      {"context": {"context": "...", "sources": [...]}, "context2": {...}, ...}

    Tools with no rag_processor get {"context": "", "sources": []}.
    """
    tools = get_all_tools_config(assistant)
    tasks: List[tuple] = []

    for tool in tools:
        context_key = tool["context_key"]
        rag_name = tool.get("rag_processor", "")

        if not rag_name or rag_name not in rag_processors:
            if rag_name and rag_name not in rag_processors:
                logger.warning("RAG processor '%s' not found for %s", rag_name, context_key)
            continue

        tool_assistant = _create_tool_assistant(assistant, tool)
        coro = get_rag_context_fn(
            request=request,
            rag_processors=rag_processors,
            rag_processor=rag_name,
            assistant_details=tool_assistant,
        )
        tasks.append((context_key, coro))

    contexts: Dict[str, Any] = {}

    if tasks:
        keys = [t[0] for t in tasks]
        coros = [t[1] for t in tasks]
        results = await asyncio.gather(*coros, return_exceptions=True)

        for key, result in zip(keys, results):
            if isinstance(result, Exception):
                logger.error("RAG failed for '%s': %s", key, result)
                contexts[key] = {"context": "", "sources": []}
            elif isinstance(result, dict):
                contexts[key] = result
            else:
                contexts[key] = {"context": "", "sources": []}

    for tool in tools:
        ck = tool["context_key"]
        if ck not in contexts:
            contexts[ck] = {"context": "", "sources": []}

    return contexts
