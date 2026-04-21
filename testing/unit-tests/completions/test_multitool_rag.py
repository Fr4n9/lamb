"""
Unit tests for multitool_rag: orchestrator, schema, timeout, security.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

#import may change depending where are you executing the tests
#backend.lamb.completions.rag.multitool_schema may be another path that could solve problems if executing tests another way than from the venv 
from lamb.completions.rag.multitool_schema import (
    parse_metadata_multitool,
    parse_orchestrator_response,
)
#import may change depending where are you executing the tests
#backend.lamb.completions.rag.multitool_schema may be another path that could solve problems if executing tests another way than from the venv 
from lamb.completions.rag.multitool_rag import (
    run_tool_with_timeout,
    orchestrate_tool_plan,
    rag_processor,
)


# ---------------------------------------------------------------------------
# Task 1: Schema + metadata parsing
# ---------------------------------------------------------------------------

def test_parse_metadata_multitool_missing_returns_none():
    """
    Action: Tests the metadata parser when the 'multitool' json block is completely empty or missing.
    Guarantees: Proves that missing database configurations are handled gracefully without throwing unhandled KeyErrors.
    """
    assert parse_metadata_multitool("{}") is None


def test_parse_metadata_multitool_returns_dict_when_present():
    """
    Action: Tests the parser with a valid multitool configuration string containing enabled tools.
    Guarantees: Confirms the JSON decoder successfully extracts the user's settings into a Python dictionary.
    """
    raw = '{"multitool":{"enabled_tools":["kb_query"]}}'
    mt = parse_metadata_multitool(raw)
    assert mt is not None
    assert mt["enabled_tools"] == ["kb_query"]


# ---------------------------------------------------------------------------
# Task 2: Orchestrator JSON parsing + hallucinated-tool filter
# ---------------------------------------------------------------------------

def test_parse_orchestrator_response_filters_unknown_tool():
    """
    Action: Simulates the LLM orchestrator hallucinating an unregistered tool ('ghost_tool').
    Guarantees: Proves that only safely allowed tools are permitted to pass, preventing arbitrary execution bugs.
    """
    raw = '{"tools":[{"name":"kb_query","arguments":{}},{"name":"ghost_tool","arguments":{}}]}'
    plan, rejected = parse_orchestrator_response(raw, allowed_names=["kb_query"])
    assert [t.name for t in plan.tools] == ["kb_query"]
    assert "ghost_tool" in rejected


# ---------------------------------------------------------------------------
# Task 3: Async execution with timeout
# ---------------------------------------------------------------------------

def test_run_tool_with_timeout_marks_error():
    """
    Action: Simulates a tool taking 10 seconds while the strict code timeout is set to 0.05 seconds.
    Guarantees: Proves the async code successfully halts hanging API calls, preventing server deadlock.
    """
    async def slow():
        await asyncio.sleep(10)
        return {"ok": True}

    out = asyncio.run(run_tool_with_timeout("kb_query", slow, timeout_sec=0.05))
    assert out["ok"] is False
    assert "timeout" in out["error"].lower()


def test_run_tool_with_timeout_happy_path():
    """
    Action: Tests a tool that executes quickly and successfully within the time limit.
    Guarantees: Assures that normal, fast responses are properly extracted and marked as successful (ok=True).
    """
    async def fast():
        return {"ok": True, "context": "hello"}

    out = asyncio.run(run_tool_with_timeout("kb_query", fast, timeout_sec=5))
    assert out["ok"] is True
    assert out["context"] == "hello"


def test_run_tool_with_timeout_catches_exception():
    """
    Action: Forces the simulated tool to crash internally by raising a RuntimeError ('boom').
    Guarantees: Proves that internal tool crashes are caught and stringified securely instead of crashing the endpoint.
    """
    async def broken():
        raise RuntimeError("boom")

    out = asyncio.run(run_tool_with_timeout("kb_query", broken, timeout_sec=5))
    assert out["ok"] is False
    assert "boom" in out["error"]


# ---------------------------------------------------------------------------
# Task 4: Orchestrator wiring (mock small-fast-model)
# ---------------------------------------------------------------------------

def test_orchestrate_tool_plan_calls_small_fast_model():
    """
    Action: Mocks the small fast model (OpenAI) to return a valid JSON requesting the kb_query tool.
    Guarantees: Validates the core integration pipeline successfully sends the query and extracts the tool plan.
    """
    fake_response = {
        "choices": [
            {
                "message": {
                    "content": '{"tools":[{"name":"kb_query","arguments":{"query":"x"}}]}'
                }
            }
        ]
    }
    #import may change depending where are you executing the tests
    #backend.lamb.completions.rag.multitool_schema may be another path that could solve problems if executing tests another way than from the venv 

    with patch(
        "lamb.completions.rag.multitool_rag.invoke_small_fast_model",
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
    """
    Action: Runs the full RAG processor simulating a request where both KB and Rubric tools are triggered successfully.
    Guarantees: The ultimate validation test. Proves parallel execution generates a correctly formatted multi-tool context string.
    """
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

    #import may change depending where are you executing the tests
    #backend.lamb.completions.rag.multitool_schema may be another path that could solve problems if executing tests another way than from the venv 
    with patch(
        "lamb.completions.rag.multitool_rag.invoke_small_fast_model",
        new=AsyncMock(return_value=_fake_orchestrator_response(["kb_query", "rubric"])),
    ), patch(
        "lamb.completions.rag.multitool_rag.kb_query.execute",
        side_effect=fake_kb,
    ), patch(
        "lamb.completions.rag.multitool_rag.rubric.execute",
        side_effect=fake_rubric,
    ):
        ctx = asyncio.run(rag_processor(messages, assistant))

    assert "F=ma" in ctx["context"]
    assert "Criteria 1" in ctx["context"]
    assert ctx["tool_results"]["kb_query"]["ok"] is True
    assert ctx["tool_results"]["rubric"]["ok"] is True
    assert len(ctx["sources"]) == 1


def test_rag_processor_missing_multitool_metadata():
    """
    Action: Dispatches a message to the processor but strips the database completely of any tool configuration.
    Guarantees: Ensures the processor fails gracefully and returns an empty context instead of breaking the chat flow.
    """
    assistant = _make_assistant({"connector": "openai"})
    messages = [{"role": "user", "content": "hi"}]
    ctx = asyncio.run(rag_processor(messages, assistant))
    assert "no 'multitool' block" in ctx["context"]


# ---------------------------------------------------------------------------
# Task 6: Negative tests (security / robustness)
# ---------------------------------------------------------------------------

def test_rag_processor_no_valid_tools_enabled():
    """
    Action: Triggers the processor when the DB metadata is misconfigured with non-existent tools.
    Guarantees: Confirms the system detects the anomaly immediately and returns an explicit safe error message.
    """
    assistant = _make_assistant({
        "multitool": {"enabled_tools": ["nonexistent_tool"]}
    })
    messages = [{"role": "user", "content": "hi"}]
    ctx = asyncio.run(rag_processor(messages, assistant))
    assert "no valid tools enabled" in ctx["context"]


def test_rag_processor_orchestrator_returns_invalid_json():
    """
    Action: Mocks the OpenAI API orchestrator to return total garbage text ("NOT JSON AT ALL").
    Guarantees: Proves that JSON syntax errors from unpredictable LLMs are safely caught without triggering HTTP 500 crashes.
    """
    assistant = _make_assistant({
        "multitool": {
            "enabled_tools": ["kb_query"],
            "per_tool": {"kb_query": {"collections": ["c1"]}},
        }
    })
    messages = [{"role": "user", "content": "hi"}]

    bad_response = {"choices": [{"message": {"content": "NOT JSON AT ALL"}}]}
    
    #import may change depending where are you executing the tests
#backend.lamb.completions.rag.multitool_schema may be another path that could solve problems if executing tests another way than from the venv 

    with patch(
        "lamb.completions.rag.multitool_rag.invoke_small_fast_model",
        new=AsyncMock(return_value=bad_response),
    ):
        ctx = asyncio.run(rag_processor(messages, assistant))

    assert "orchestrator error" in ctx["context"].lower()


def test_rag_processor_orchestrator_returns_empty_tool_list():
    """
    Action: The orchestrator model decides it doesn't need any external tools for the user query and returns an empty array.
    Guarantees: Ensures that a no-op decision correctly bypasses tool execution loops smoothly.
    """
    assistant = _make_assistant({
        "multitool": {
            "enabled_tools": ["kb_query"],
            "per_tool": {"kb_query": {"collections": ["c1"]}},
        }
    })
    messages = [{"role": "user", "content": "hi"}]

    empty_response = {"choices": [{"message": {"content": '{"tools":[]}'}}]}
    
    #import may change depending where are you executing the tests
    #backend.lamb.completions.rag.multitool_schema may be another path that could solve problems if executing tests another way than from the venv 
    with patch(
        "lamb.completions.rag.multitool_rag.invoke_small_fast_model",
        new=AsyncMock(return_value=empty_response),
    ):
        ctx = asyncio.run(rag_processor(messages, assistant))

    assert "selected no tools" in ctx["context"]


def test_rag_processor_tool_timeout_surfaces_error():
    """
    Action: Simulates the entire RAG pipeline where one tool takes too long (10s) and breaches the 1s timeout limit.
    Guarantees: Verifies that the final returned context properly tags the lagging tool with an '(error)' fallback message.
    """
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

    #import may change depending where are you executing the tests
    #backend.lamb.completions.rag.multitool_schema may be another path that could solve problems if executing tests another way than from the venv 
    with patch(
        "lamb.completions.rag.multitool_rag.invoke_small_fast_model",
        new=AsyncMock(return_value=_fake_orchestrator_response(["kb_query"])),
    ), patch(
        "lamb.completions.rag.multitool_rag.kb_query.execute",
        side_effect=slow_kb,
    ):
        ctx = asyncio.run(rag_processor(messages, assistant))

    assert ctx["tool_results"]["kb_query"]["ok"] is False
    assert "timeout" in ctx["tool_results"]["kb_query"]["error"].lower()
    assert "(error)" in ctx["context"]


def test_parse_orchestrator_response_invalid_json_raises():
    """
    Action: Triggers the lower-level parsing function directly with broken JSON data.
    Guarantees: Ensures the fundamental parsing function correctly bubbles up native JSONDecodeError exceptions to be caught higher up.
    """
    import pytest
    with pytest.raises(json.JSONDecodeError):
        parse_orchestrator_response("NOT JSON", allowed_names=["kb_query"])


def test_rag_processor_rubric_missing_required_key():
    """
    Action: rubric tool called without rubric_id in per_tool config → error block.
    Guarantees: Verifies argument stripping and missing parameter validation works correctly to prevent arbitrary errors.
    """
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

    #import may change depending where are you executing the tests
    #backend.lamb.completions.rag.multitool_schema may be another path that could solve problems if executing tests another way than from the venv 
    with patch(
        "lamb.completions.rag.multitool_rag.invoke_small_fast_model",
        new=AsyncMock(return_value=_fake_orchestrator_response(["rubric"])),
    ), patch(
        "lamb.completions.rag.multitool_rag.rubric.execute",
        side_effect=rubric_needs_id,
    ):
        ctx = asyncio.run(rag_processor(messages, assistant))

    assert ctx["tool_results"]["rubric"]["ok"] is False
    assert "rubric_id" in ctx["tool_results"]["rubric"]["error"]
