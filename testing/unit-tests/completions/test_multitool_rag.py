"""
Unit tests for multitool_rag: orchestrator, schema, timeout, security.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from backend.lamb.completions.rag.multitool_schema import (
    parse_metadata_multitool,
    parse_orchestrator_response,
)
from backend.lamb.completions.rag.multitool_rag import (
    run_tool_with_timeout,
    orchestrate_tool_plan,
    rag_processor,
)


# ---------------------------------------------------------------------------
# Task 1: Schema + metadata parsing
# ---------------------------------------------------------------------------

def test_parse_metadata_multitool_missing_returns_none():
    assert parse_metadata_multitool("{}") is None


def test_parse_metadata_multitool_returns_dict_when_present():
    raw = '{"multitool":{"enabled_tools":["kb_query"]}}'
    mt = parse_metadata_multitool(raw)
    assert mt is not None
    assert mt["enabled_tools"] == ["kb_query"]


# ---------------------------------------------------------------------------
# Task 2: Orchestrator JSON parsing + hallucinated-tool filter
# ---------------------------------------------------------------------------

def test_parse_orchestrator_response_filters_unknown_tool():
    raw = '{"tools":[{"name":"kb_query","arguments":{}},{"name":"ghost_tool","arguments":{}}]}'
    plan, rejected = parse_orchestrator_response(raw, allowed_names=["kb_query"])
    assert [t.name for t in plan.tools] == ["kb_query"]
    assert "ghost_tool" in rejected


# ---------------------------------------------------------------------------
# Task 3: Async execution with timeout
# ---------------------------------------------------------------------------

def test_run_tool_with_timeout_marks_error():
    async def slow():
        await asyncio.sleep(10)
        return {"ok": True}

    out = asyncio.run(run_tool_with_timeout("kb_query", slow, timeout_sec=0.05))
    assert out["ok"] is False
    assert "timeout" in out["error"].lower()


def test_run_tool_with_timeout_happy_path():
    async def fast():
        return {"ok": True, "context": "hello"}

    out = asyncio.run(run_tool_with_timeout("kb_query", fast, timeout_sec=5))
    assert out["ok"] is True
    assert out["context"] == "hello"


def test_run_tool_with_timeout_catches_exception():
    async def broken():
        raise RuntimeError("boom")

    out = asyncio.run(run_tool_with_timeout("kb_query", broken, timeout_sec=5))
    assert out["ok"] is False
    assert "boom" in out["error"]


# ---------------------------------------------------------------------------
# Task 4: Orchestrator wiring (mock small-fast-model)
# ---------------------------------------------------------------------------

def test_orchestrate_tool_plan_calls_small_fast_model():
    fake_response = {
        "choices": [
            {
                "message": {
                    "content": '{"tools":[{"name":"kb_query","arguments":{"query":"x"}}]}'
                }
            }
        ]
    }
    with patch(
        "backend.lamb.completions.rag.multitool_rag.invoke_small_fast_model",
        new=AsyncMock(return_value=fake_response),
    ) as m:
        plan, rejected = asyncio.run(
            orchestrate_tool_plan(
                user_query="hello",
                assistant_owner="a@b.com",
                allowed_tool_names=["kb_query"],
            )
        )
    assert m.await_count == 1
    assert plan.tools[0].name == "kb_query"
    assert rejected == []


# ---------------------------------------------------------------------------
# Task 5: Full rag_processor integration (mocked tools + orchestrator)
# ---------------------------------------------------------------------------

def _make_assistant(metadata_dict):
    return SimpleNamespace(
        metadata=json.dumps(metadata_dict),
        owner="test@example.com",
    )


def _fake_orchestrator_response(tool_names):
    tools = [{"name": n, "arguments": {}} for n in tool_names]
    return {
        "choices": [{"message": {"content": json.dumps({"tools": tools})}}]
    }


def test_rag_processor_happy_path_two_tools():
    assistant = _make_assistant({
        "multitool": {
            "enabled_tools": ["kb_query", "rubric"],
            "per_tool": {
                "kb_query": {"collections": ["c1"], "top_k": 2},
                "rubric": {"rubric_id": "r1"},
            },
            "orchestrator": {"per_tool_timeout_sec": 5, "total_timeout_sec": 10},
        }
    })
    messages = [{"role": "user", "content": "explain Newton"}]

    async def fake_kb(**kw):
        return {"ok": True, "tool": "kb_query", "context": "F=ma", "sources": []}

    async def fake_rubric(**kw):
        return {"ok": True, "tool": "rubric", "context": "Criteria 1", "sources": [{"type": "rubric"}]}

    with patch(
        "backend.lamb.completions.rag.multitool_rag.invoke_small_fast_model",
        new=AsyncMock(return_value=_fake_orchestrator_response(["kb_query", "rubric"])),
    ), patch(
        "backend.lamb.completions.rag.multitool_rag.kb_query.execute",
        side_effect=fake_kb,
    ), patch(
        "backend.lamb.completions.rag.multitool_rag.rubric.execute",
        side_effect=fake_rubric,
    ):
        ctx = asyncio.run(rag_processor(messages, assistant))

    assert "F=ma" in ctx["context"]
    assert "Criteria 1" in ctx["context"]
    assert ctx["tool_results"]["kb_query"]["ok"] is True
    assert ctx["tool_results"]["rubric"]["ok"] is True
    assert len(ctx["sources"]) == 1


def test_rag_processor_missing_multitool_metadata():
    assistant = _make_assistant({"connector": "openai"})
    messages = [{"role": "user", "content": "hi"}]
    ctx = asyncio.run(rag_processor(messages, assistant))
    assert "no 'multitool' block" in ctx["context"]


# ---------------------------------------------------------------------------
# Task 6: Negative tests (security / robustness)
# ---------------------------------------------------------------------------

def test_rag_processor_no_valid_tools_enabled():
    assistant = _make_assistant({
        "multitool": {"enabled_tools": ["nonexistent_tool"]}
    })
    messages = [{"role": "user", "content": "hi"}]
    ctx = asyncio.run(rag_processor(messages, assistant))
    assert "no valid tools enabled" in ctx["context"]


def test_rag_processor_orchestrator_returns_invalid_json():
    assistant = _make_assistant({
        "multitool": {
            "enabled_tools": ["kb_query"],
            "per_tool": {"kb_query": {"collections": ["c1"]}},
        }
    })
    messages = [{"role": "user", "content": "hi"}]

    bad_response = {"choices": [{"message": {"content": "NOT JSON AT ALL"}}]}
    with patch(
        "backend.lamb.completions.rag.multitool_rag.invoke_small_fast_model",
        new=AsyncMock(return_value=bad_response),
    ):
        ctx = asyncio.run(rag_processor(messages, assistant))

    assert "orchestrator error" in ctx["context"].lower()


def test_rag_processor_orchestrator_returns_empty_tool_list():
    assistant = _make_assistant({
        "multitool": {
            "enabled_tools": ["kb_query"],
            "per_tool": {"kb_query": {"collections": ["c1"]}},
        }
    })
    messages = [{"role": "user", "content": "hi"}]

    empty_response = {"choices": [{"message": {"content": '{"tools":[]}'}}]}
    with patch(
        "backend.lamb.completions.rag.multitool_rag.invoke_small_fast_model",
        new=AsyncMock(return_value=empty_response),
    ):
        ctx = asyncio.run(rag_processor(messages, assistant))

    assert "selected no tools" in ctx["context"]


def test_rag_processor_tool_timeout_surfaces_error():
    assistant = _make_assistant({
        "multitool": {
            "enabled_tools": ["kb_query"],
            "per_tool": {"kb_query": {"collections": ["c1"]}},
            "orchestrator": {"per_tool_timeout_sec": 0.05, "total_timeout_sec": 1},
        }
    })
    messages = [{"role": "user", "content": "hi"}]

    async def slow_kb(**kw):
        await asyncio.sleep(10)
        return {"ok": True}

    with patch(
        "backend.lamb.completions.rag.multitool_rag.invoke_small_fast_model",
        new=AsyncMock(return_value=_fake_orchestrator_response(["kb_query"])),
    ), patch(
        "backend.lamb.completions.rag.multitool_rag.kb_query.execute",
        side_effect=slow_kb,
    ):
        ctx = asyncio.run(rag_processor(messages, assistant))

    assert ctx["tool_results"]["kb_query"]["ok"] is False
    assert "timeout" in ctx["tool_results"]["kb_query"]["error"].lower()
    assert "(error)" in ctx["context"]


def test_parse_orchestrator_response_invalid_json_raises():
    import pytest
    with pytest.raises(json.JSONDecodeError):
        parse_orchestrator_response("NOT JSON", allowed_names=["kb_query"])


def test_rag_processor_rubric_missing_required_key():
    """rubric tool called without rubric_id in per_tool config → error block."""
    assistant = _make_assistant({
        "multitool": {
            "enabled_tools": ["rubric"],
            "per_tool": {"rubric": {}},
            "orchestrator": {"per_tool_timeout_sec": 5, "total_timeout_sec": 10},
        }
    })
    messages = [{"role": "user", "content": "evaluate me"}]

    async def rubric_needs_id(**kw):
        if not kw.get("rubric_id"):
            return {"ok": False, "error": "rubric_id is required", "tool": "rubric"}
        return {"ok": True}

    with patch(
        "backend.lamb.completions.rag.multitool_rag.invoke_small_fast_model",
        new=AsyncMock(return_value=_fake_orchestrator_response(["rubric"])),
    ), patch(
        "backend.lamb.completions.rag.multitool_rag.rubric.execute",
        side_effect=rubric_needs_id,
    ):
        ctx = asyncio.run(rag_processor(messages, assistant))

    assert ctx["tool_results"]["rubric"]["ok"] is False
    assert "rubric_id" in ctx["tool_results"]["rubric"]["error"]
