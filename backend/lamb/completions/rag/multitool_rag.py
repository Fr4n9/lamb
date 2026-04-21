"""Multitool RAG processor — orchestrator-driven multi-source context retrieval.

Pipeline: user query → small-fast-model orchestrator → parallel tool execution
→ aggregated rag_context for simple_augment → connector.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple

from lamb.completions.rag.multitool_schema import (
    RawOrchestratorPlan,
    parse_metadata_multitool,
    parse_orchestrator_response,
)
from lamb.completions.rag.multitool_tools.registry import ToolRegistry
from lamb.completions.rag.multitool_tools import kb_query, rubric
from lamb.completions.small_fast_model_helper import invoke_small_fast_model
from lamb.logging_config import get_logger

logger = get_logger(__name__, component="MULTITOOL")

# ---------------------------------------------------------------------------
# Global registry — built once at import time
# ---------------------------------------------------------------------------

_registry = ToolRegistry()
_registry.register("kb_query", kb_query.execute)
_registry.register("rubric", rubric.execute)

DEFAULT_PER_TOOL_TIMEOUT = 12
DEFAULT_TOTAL_TIMEOUT = 25

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def run_tool_with_timeout(
    name: str,
    coro_factory,
    *,
    timeout_sec: float,
) -> Dict[str, Any]:
    """Execute a zero-arg coroutine factory with a per-tool timeout."""
    try:
        return await asyncio.wait_for(coro_factory(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        return {"ok": False, "error": f"timeout after {timeout_sec}s", "tool": name}
    except Exception as e:
        return {"ok": False, "error": str(e), "tool": name}


def _extract_content(result: Any) -> str:
    """Pull the assistant text from a non-streaming connector response."""
    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""


def _build_orchestrator_prompt(allowed_tools: List[str]) -> str:
    descriptions = {
         # "kb_query": "Use ONLY to search for theoretical knowledge when the user asks a question. Arguments: query (str).",
        #"rubric": "MANDATORY to use whenever the user submits an essay, assignment, or asks to be evaluated/graded. No extra arguments needed.",
        "kb_query": "Search knowledge-base collections for relevant context. Arguments: query (str).",
        "rubric": "Fetch a rubric and format it as evaluation context. No extra arguments needed.",
    }
    lines = ["You are a tool-selection orchestrator for an educational AI assistant.",
             "Given the user's query, decide which tools to call.",
             "Respond ONLY with valid JSON matching this schema:",
             '{"tools":[{"name":"<tool>","arguments":{...}}],"rationale":"..."}',
             "",
             "Available tools:"]
    for t in allowed_tools:
        lines.append(f"- {t}: {descriptions.get(t, 'No description.')}")
    lines.append("")
    lines.append("If no tool is useful, return {\"tools\":[]}.")
    return "\n".join(lines)


async def orchestrate_tool_plan(
    *,
    user_query: str,
    assistant_owner: str,
    allowed_tool_names: List[str],
) -> Tuple[RawOrchestratorPlan, List[str]]:
    """Call the small-fast-model to decide which tools to run."""
    system_prompt = _build_orchestrator_prompt(allowed_tool_names)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    result = await invoke_small_fast_model(
        messages=messages,
        assistant_owner=assistant_owner,
        stream=False,
        body=None,
    )

    content = _extract_content(result)
    if not content:
        return RawOrchestratorPlan(tools=[]), []

    return parse_orchestrator_response(content, allowed_names=allowed_tool_names)


def _strip_extra_args(arguments: Dict[str, Any], allowed: frozenset) -> Dict[str, Any]:
    """Keep only whitelisted keys — defense against prompt injection in args."""
    return {k: v for k, v in arguments.items() if k in allowed}


# ---------------------------------------------------------------------------
# Public API — async rag_processor
# ---------------------------------------------------------------------------


async def _rag_processor_internal(
    messages: List[Dict[str, Any]],
    assistant=None,
    request: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Async RAG processor for the multitool pipeline.

    Compatible with ``get_rag_context`` in ``completions/main.py``.
    """
    metadata_str = getattr(assistant, "metadata", None) or getattr(assistant, "api_callback", "")
    mt = parse_metadata_multitool(metadata_str or "")

    if mt is None:
        return {
            "context": "multitool_rag: no 'multitool' block in assistant metadata",
            "sources": [],
            "tool_results": {},
        }

    enabled = mt.get("enabled_tools", [])
    registry_names = set(_registry.allowed_names())
    allowed = [n for n in enabled if n in registry_names]
    if not allowed:
        return {
            "context": "multitool_rag: no valid tools enabled",
            "sources": [],
            "tool_results": {},
        }

    # Extract last user message
    user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                user_query = " ".join(
                    p.get("text", "") for p in content if p.get("type") == "text"
                )
            else:
                user_query = str(content)
            break

    if not user_query:
        return {
            "context": "multitool_rag: no user message found",
            "sources": [],
            "tool_results": {},
        }

    assistant_owner = getattr(assistant, "owner", "")

    # Orchestrate
    try:
        plan, rejected = await orchestrate_tool_plan(
            user_query=user_query,
            assistant_owner=assistant_owner,
            allowed_tool_names=allowed,
        )
    except Exception as e:
        logger.error("Orchestrator failed: %s", e)
        return {
            "context": f"multitool_rag: orchestrator error — {e}",
            "sources": [],
            "tool_results": {},
        }

    if rejected:
        logger.warning("Orchestrator hallucinated tools: %s", rejected)

    if not plan.tools:
        return {
            "context": "multitool_rag: orchestrator selected no tools",
            "sources": [],
            "tool_results": {},
            "orchestrator_raw": plan.model_dump(),
        }

    # Build coroutines
    orch_cfg = mt.get("orchestrator", {})
    per_tool_timeout = orch_cfg.get("per_tool_timeout_sec", DEFAULT_PER_TOOL_TIMEOUT)
    total_timeout = orch_cfg.get("total_timeout_sec", DEFAULT_TOTAL_TIMEOUT)
    per_tool_cfg = mt.get("per_tool", {})

    _tool_modules = {"kb_query": kb_query, "rubric": rubric}

    tasks = []
    task_names = []
    for tc in plan.tools:
        tool_module = _tool_modules.get(tc.name)
        if tool_module is None:
            continue
        tool_allowed_args = getattr(tool_module, "ALLOWED_ARGS", None) or set()

        merged_args = {**per_tool_cfg.get(tc.name, {}), **tc.arguments}
        if tool_allowed_args:
            merged_args = _strip_extra_args(merged_args, tool_allowed_args)
        merged_args["assistant_owner"] = assistant_owner
        if tc.name == "kb_query":
            merged_args.setdefault("query", user_query)

        # Resolve execute at call time so test patches take effect
        def _make_coro(mod=tool_module, a=merged_args):
            return mod.execute(**a)

        tasks.append(
            run_tool_with_timeout(tc.name, _make_coro, timeout_sec=per_tool_timeout)
        )
        task_names.append(tc.name)

    # Execute in parallel with a global timeout
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=False),
            timeout=total_timeout,
        )
    except asyncio.TimeoutError:
        results = [{"ok": False, "error": f"total timeout after {total_timeout}s", "tool": n} for n in task_names]

    # Aggregate
    context_parts: List[str] = []
    all_sources: List[Dict[str, Any]] = []
    tool_results: Dict[str, Any] = {}

    for name, res in zip(task_names, results):
        tool_results[name] = res
        if res.get("ok"):
            context_parts.append(f"=== Tool: {name} (ok) ===\n{res.get('context', '')}")
            all_sources.extend(res.get("sources", []))
        else:
            context_parts.append(f"=== Tool: {name} (error) ===\n{res.get('error', 'unknown error')}")

    final_context = "\n\n".join(context_parts)

    return {
        "context": final_context,
        "sources": all_sources,
        "tool_results": tool_results,
        "orchestrator_raw": plan.model_dump(),
    }


async def rag_processor(
    messages: List[Dict[str, Any]],
    assistant=None,
    request: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Async RAG processor wrapper that intercepts and dumps all returned contexts."""
    result = await _rag_processor_internal(messages, assistant, request)
    
    try:
        import os, time
        dump_dir = "testing/context_dumps"
        os.makedirs(dump_dir, exist_ok=True)
        dump_path = os.path.join(dump_dir, f"context_dump_{int(time.time())}.md")
        with open(dump_path, "w", encoding="utf-8") as f:
            f.write(f"# Final RAG Context Dump ({time.strftime('%Y-%m-%d %H:%M:%S')})\n\n")
            f.write(result.get("context", "NO CONTEXT GENERATED"))
    except Exception as e:
        logger.error("Failed to write context dump: %s", e)
        
    return result
