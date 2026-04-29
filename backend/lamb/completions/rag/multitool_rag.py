"""Multitool RAG processor — orchestrator-driven multi-source context retrieval.

Pipeline: user query → small-fast-model orchestrator → parallel tool execution
→ aggregated rag_context for simple_augment → connector.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
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

_USER_MSG_HEAD = 800
_USER_MSG_TAIL = 800
_MAX_DUMP_TEXT = 20000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mask_owner(owner: str) -> str:
    """Keep domain, mask local part for debug dumps."""
    if not owner or "@" not in owner:
        return owner or ""
    local, domain = owner.rsplit("@", 1)
    if not local:
        return f"***@{domain}"
    return f"{local[0]}***@{domain}"


def _user_query_stats(user_text: str, head: int = _USER_MSG_HEAD, tail: int = _USER_MSG_TAIL) -> Dict[str, Any]:
    n = len(user_text)
    if n <= head + tail:
        return {"len": n, "head": user_text, "tail": ""}
    return {"len": n, "head": user_text[:head], "tail": user_text[-tail:]}


def _redact_key(key: str) -> bool:
    kl = key.lower()
    for frag in ("token", "password", "secret", "api_key", "auth"):
        if frag in kl:
            return True
    return False


def _redact_value_for_debug(obj: Any, max_str: int = 80) -> Any:
    """Redact secrets and truncate long strings in nested structures."""
    if isinstance(obj, dict):
        return {
            k: ("[REDACTED]" if _redact_key(k) else _redact_value_for_debug(v, max_str))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact_value_for_debug(x, max_str) for x in obj]
    if isinstance(obj, str) and len(obj) > max_str:
        return obj[: min(40, max_str)] + "...[truncated]"
    return obj


def _default_multitool_dump_dir() -> Path:
    """`lamb/testing/context_dumps` from package location (avoids CWD)."""
    return Path(__file__).resolve().parents[4] / "testing" / "context_dumps"


def _trunc_for_dump(text: str, max_len: int = _MAX_DUMP_TEXT) -> str:
    if not text or len(text) <= max_len:
        return text
    return text[: max_len - 30] + "\n\n[truncated]\n"


def _write_multitool_context_dump(
    *,
    dump_dir: Path,
    multitool_debug: Dict[str, Any],
    final_context: str,
    assistant: Any,
) -> Optional[str]:
    """Write structured Markdown; returns path on success. Caller gates via env."""
    try:
        dump_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("Failed to create multitool dump dir %s: %s", dump_dir, e)
        return None
    out_path = dump_dir / f"context_dump_{int(time.time())}.md"
    orch = multitool_debug.get("orchestrator") or {}
    raw = _trunc_for_dump(str(orch.get("raw_llm_text", "")))
    parsed = orch.get("parsed")
    try:
        parsed_s = (
            json.dumps(parsed, ensure_ascii=False, indent=2) if parsed is not None else "null"
        )
    except (TypeError, ValueError):
        parsed_s = repr(parsed)
    parsed_s = _trunc_for_dump(parsed_s)

    exec_block = json.dumps(multitool_debug.get("executed", []), ensure_ascii=False, indent=2)
    exec_block = _trunc_for_dump(exec_block)
    full_dbg = _trunc_for_dump(
        json.dumps(multitool_debug, ensure_ascii=False, indent=2, default=str)
    )
    ctx_block = _trunc_for_dump(final_context or "")

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    aid = getattr(assistant, "id", None)
    own = _mask_owner(getattr(assistant, "owner", "") or "")

    timings = multitool_debug.get("timings_ms") or {}
    orch_ms = timings.get("orchestrate_ms")
    tools_ms = timings.get("tools_total_ms")
    orch_s = str(orch_ms) if orch_ms is not None else "—"
    tools_s = str(tools_ms) if tools_ms is not None else "—"

    llm_timings_section = (
        f"## Latencies (Timings)\n\n"
        f"- orchestrate_ms (LLM call): {orch_s}\n"
        f"- tools_total_ms (Parallel execution): {tools_s}\n"
    )

    md = (
        f"# Multitool RAG context dump\n\n"
        f"- timestamp: {ts}\n"
        f"- assistant_id: {aid}\n"
        f"- owner: {own}\n\n"
        f"{llm_timings_section}\n"
        f"## Orchestrator raw\n\n"
        f"```\n{raw}\n```\n\n"
        f"## Parsed plan\n\n"
        f"```json\n{parsed_s}\n```\n\n"
        f"## Tool execution\n\n"
        f"```json\n{exec_block}\n```\n\n"
        f"## Full multitool_debug (JSON)\n\n"
        f"```json\n{full_dbg}\n```\n\n"
        f"## Final injected context\n\n{ctx_block}\n"
    )
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
    except OSError as e:
        logger.error("Failed to write multitool context dump: %s", e)
        return None
    return str(out_path)


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
) -> Tuple[RawOrchestratorPlan, List[str], str]:
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
        return RawOrchestratorPlan(tools=[]), [], ""

    plan, rejected = parse_orchestrator_response(content, allowed_names=allowed_tool_names)
    return plan, rejected, content


def _strip_extra_args(arguments: Dict[str, Any], allowed: frozenset) -> Dict[str, Any]:
    """Keep only whitelisted keys — defense against prompt injection in args."""
    return {k: v for k, v in arguments.items() if k in allowed}


# ---------------------------------------------------------------------------
# Public API — async rag_processor
# ---------------------------------------------------------------------------


def _base_multitool_debug(
    assistant: Any,
    *,
    allowed: List[str],
    user_query: str,
    per_tool_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "assistant_id": getattr(assistant, "id", None),
        "owner_masked": _mask_owner(getattr(assistant, "owner", "") or ""),
        "allowed_tools": list(allowed),
        "rejected_by_registry": [],
        "user_query_stats": _user_query_stats(user_query) if user_query else {"len": 0, "head": "", "tail": ""},
        "per_tool_config_summary": _redact_value_for_debug(dict(per_tool_cfg)),
        "orchestrator": {},
        "executed": [],
        "timings_ms": {},
    }


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
        user_query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    user_query = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
                else:
                    user_query = str(content)
                break
        ustats0 = _user_query_stats(user_query) if user_query else {"len": 0, "head": "", "tail": ""}
        return {
            "context": "multitool_rag: no 'multitool' block in assistant metadata",
            "sources": [],
            "tool_results": {},
            "multitool_debug": {
                "error": "no_multitool_block",
                "user_query_stats": ustats0,
            },
        }

    enabled = mt.get("enabled_tools", [])
    registry_names = set(_registry.allowed_names())
    allowed = [n for n in enabled if n in registry_names]

    user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                user_query = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
            else:
                user_query = str(content)
            break

    per_tool_cfg = mt.get("per_tool", {})
    
    if not allowed:
        return {
            "context": "multitool_rag: no valid tools enabled",
            "sources": [],
            "tool_results": {},
            "multitool_debug": {
                **_base_multitool_debug(assistant, allowed=[], user_query=user_query, per_tool_cfg=per_tool_cfg),
                "error": "no_valid_tools",
            },
        }

    if not user_query:
        return {
            "context": "multitool_rag: no user message found",
            "sources": [],
            "tool_results": {},
            "multitool_debug": {
                **_base_multitool_debug(assistant, allowed=allowed, user_query="", per_tool_cfg=per_tool_cfg),
                "error": "no_user_message"
            },
        }

    assistant_owner = getattr(assistant, "owner", "")

    dbg = _base_multitool_debug(
        assistant,
        allowed=allowed,
        user_query=user_query,
        per_tool_cfg=per_tool_cfg,
    )

    t_orch0 = time.perf_counter()
    try:
        plan, rejected, raw_llm_text = await orchestrate_tool_plan(
            user_query=user_query,
            assistant_owner=assistant_owner,
            allowed_tool_names=allowed,
        )
    except Exception as e:
        logger.error("Orchestrator failed: %s", e)
        dbg["rejected_by_registry"] = []
        dbg["orchestrator"] = {
            "raw_llm_text": "",
            "parsed": None,
            "error": str(e),
        }
        return {
            "context": f"multitool_rag: orchestrator error — {e}",
            "sources": [],
            "tool_results": {},
            "multitool_debug": dbg,
        }

    t_orch1 = time.perf_counter()
    dbg["timings_ms"]["orchestrate_ms"] = round((t_orch1 - t_orch0) * 1000, 3)
    dbg["rejected_by_registry"] = list(rejected)
    dbg["orchestrator"] = {
        "raw_llm_text": raw_llm_text,
        "parsed": plan.model_dump(),
    }

    if rejected:
        logger.warning("Orchestrator hallucinated tools: %s", rejected)

    if not plan.tools:
        return {
            "context": "multitool_rag: orchestrator selected no tools",
            "sources": [],
            "tool_results": {},
            "orchestrator_raw": plan.model_dump(),
            "multitool_debug": dbg,
        }

    # Build coroutines
    orch_cfg = mt.get("orchestrator", {})
    per_tool_timeout = orch_cfg.get("per_tool_timeout_sec", DEFAULT_PER_TOOL_TIMEOUT)
    total_timeout = orch_cfg.get("total_timeout_sec", DEFAULT_TOTAL_TIMEOUT)

    _tool_modules = {"kb_query": kb_query, "rubric": rubric}

    tasks: List[Any] = []
    task_names: List[str] = []
    run_steps: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for tc in plan.tools:
        tool_module = _tool_modules.get(tc.name)
        if tool_module is None:
            skipped.append(tc.name)
            run_steps.append(
                {
                    "name": tc.name,
                    "merged_args": _redact_value_for_debug(dict(tc.arguments)),
                    "ok": False,
                    "error": "no_executor_for_tool",
                }
            )
            continue

        tool_allowed_args = getattr(tool_module, "ALLOWED_ARGS", None) or set()
        merged_args = {**per_tool_cfg.get(tc.name, {}), **tc.arguments}
        if tool_allowed_args:
            merged_args = _strip_extra_args(merged_args, tool_allowed_args)
        merged_args["assistant_owner"] = assistant_owner
        if tc.name == "kb_query":
            merged_args.setdefault("query", user_query)

        run_steps.append(
            {
                "name": tc.name,
                "merged_args": _redact_value_for_debug(dict(merged_args)),
                "task_idx": len(tasks),
            }
        )

        # Resolve execute at call time so test patches take effect
        def _make_coro(mod=tool_module, a=merged_args):
            return mod.execute(**a)

        tasks.append(
            run_tool_with_timeout(tc.name, _make_coro, timeout_sec=per_tool_timeout)
        )
        task_names.append(tc.name)

    dbg["skipped_no_executor"] = skipped
    t_tools0 = time.perf_counter()

    # Execute in parallel with a global timeout
    if tasks:
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=False),
                timeout=total_timeout,
            )
        except asyncio.TimeoutError:
            results = [{"ok": False, "error": f"total timeout after {total_timeout}s", "tool": n} for n in task_names]
    else:
        results = []

    t_tools1 = time.perf_counter()
    dbg["timings_ms"]["tools_total_ms"] = round((t_tools1 - t_tools0) * 1000, 3)

    executed: List[Dict[str, Any]] = []
    for st in run_steps:
        if st.get("error") == "no_executor_for_tool":
            executed.append(
                {k: v for k, v in st.items() if k in ("name", "merged_args", "ok", "error")}
            )
            continue
        t_i = st["task_idx"]
        res = results[t_i]
        ed: Dict[str, Any] = {
            "name": st["name"],
            "merged_args": st["merged_args"],
            "ok": bool(res.get("ok")),
        }
        if not res.get("ok"):
            ed["error"] = res.get("error")
        executed.append(ed)
    dbg["executed"] = executed

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
        "multitool_debug": dbg,
    }


async def rag_processor(
    messages: List[Dict[str, Any]],
    assistant=None,
    request: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Async RAG processor wrapper that intercepts and dumps all returned contexts."""
    result = await _rag_processor_internal(messages, assistant, request)
    
    mdbg = result.get("multitool_debug")
    if mdbg is not None:
        p = _write_multitool_context_dump(
            dump_dir=_default_multitool_dump_dir(),
            multitool_debug=mdbg,
            final_context=result.get("context", ""),
            assistant=assistant,
        )
        if p:
            logger.info("Wrote multitool context dump: %s", p)
    else:
        # Fallback just in case mdbg failed to generate
        try:
            dump_dir = _default_multitool_dump_dir()
            dump_dir.mkdir(parents=True, exist_ok=True)
            dump_path = dump_dir / f"context_dump_{int(time.time())}.md"
            with open(dump_path, "w", encoding="utf-8") as f:
                f.write(f"# Final RAG Context Dump ({time.strftime('%Y-%m-%d %H:%M:%S')})\n\n")
                f.write(result.get("context", "NO CONTEXT GENERATED"))
        except Exception as e:
            logger.error("Failed to write context dump fallback: %s", e)
        
    return result
