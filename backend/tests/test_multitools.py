"""Tests for multitool context sources."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from lamb.lamb_classes import Assistant
from lamb.completions.multitool_manager import (
    _create_tool_assistant,
    get_all_tools_config,
    get_all_rag_contexts,
)
from lamb.completions.pps.simple_augment import prompt_processor
from creator_interface.assistant_router import validate_multitools_config


def _make_assistant(
    metadata_dict: dict | None = None,
    rag_collections: str = "",
    rag_top_k: int = 3,
) -> Assistant:
    """Create a real Assistant with the given metadata."""
    metadata_str = json.dumps(metadata_dict or {})
    return Assistant(
        id=1,
        organization_id=1,
        name="test-assistant",
        description="test",
        owner="test@example.com",
        api_callback=metadata_str,
        system_prompt="You are helpful.",
        prompt_template="Context: {context}\nQuestion: {user_input}",
        pre_retrieval_endpoint="",
        post_retrieval_endpoint="",
        RAG_endpoint="",
        RAG_Top_k=rag_top_k,
        RAG_collections=rag_collections,
    )


class TestGetAllToolsConfig:
    def test_single_tool_from_top_level_fields(self):
        assistant = _make_assistant(
            metadata_dict={
                "rag_processor": "simple_rag",
                "connector": "openai",
                "llm": "gpt-4",
                "prompt_processor": "simple_augment",
            },
            rag_collections="col1,col2",
        )
        tools = get_all_tools_config(assistant)
        assert len(tools) == 1
        assert tools[0]["rag_processor"] == "simple_rag"
        assert tools[0]["RAG_collections"] == "col1,col2"
        assert tools[0]["context_key"] == "context"

    def test_multitools_disabled_returns_only_tool_0(self):
        assistant = _make_assistant(
            metadata_dict={
                "rag_processor": "simple_rag",
                "connector": "openai",
                "llm": "gpt-4",
                "prompt_processor": "simple_augment",
                "multitools": False,
                "tools": [{"rag_processor": "no_rag", "RAG_collections": "colX"}],
            },
            rag_collections="col1",
        )
        tools = get_all_tools_config(assistant)
        assert len(tools) == 1
        assert tools[0]["rag_processor"] == "simple_rag"

    def test_multitools_enabled_with_additional_tools(self):
        assistant = _make_assistant(
            metadata_dict={
                "rag_processor": "simple_rag",
                "connector": "openai",
                "llm": "gpt-4",
                "prompt_processor": "simple_augment",
                "multitools": True,
                "tools": [
                    {"rag_processor": "context_aware_rag", "RAG_collections": "col3,col4"},
                    {"rag_processor": "rubric_rag", "rubric_id": "rubric-123"},
                ],
            },
            rag_collections="col1,col2",
        )
        tools = get_all_tools_config(assistant)
        assert len(tools) == 3
        assert tools[0]["context_key"] == "context"
        assert tools[0]["RAG_collections"] == "col1,col2"
        assert tools[1]["context_key"] == "context2"
        assert tools[1]["RAG_collections"] == "col3,col4"
        assert tools[2]["context_key"] == "context3"
        assert tools[2]["rubric_id"] == "rubric-123"

    def test_tool_0_with_no_rag_processor(self):
        assistant = _make_assistant(
            metadata_dict={
                "rag_processor": "",
                "connector": "openai",
                "llm": "gpt-4",
                "prompt_processor": "simple_augment",
            },
        )
        tools = get_all_tools_config(assistant)
        assert len(tools) == 1
        assert tools[0]["rag_processor"] == ""

    def test_empty_tools_array(self):
        assistant = _make_assistant(
            metadata_dict={
                "rag_processor": "simple_rag",
                "connector": "openai",
                "llm": "gpt-4",
                "prompt_processor": "simple_augment",
                "multitools": True,
                "tools": [],
            },
            rag_collections="col1",
        )
        tools = get_all_tools_config(assistant)
        assert len(tools) == 1

    def test_rubric_fields_in_tool_0(self):
        assistant = _make_assistant(
            metadata_dict={
                "rag_processor": "rubric_rag",
                "connector": "openai",
                "llm": "gpt-4",
                "prompt_processor": "simple_augment",
                "rubric_id": "rubric-456",
                "rubric_format": "json",
            },
        )
        tools = get_all_tools_config(assistant)
        assert tools[0]["rubric_id"] == "rubric-456"
        assert tools[0]["rubric_format"] == "json"


class TestGetAllRagContexts:
    def test_single_tool_calls_rag_once(self):
        assistant = _make_assistant(
            metadata_dict={
                "rag_processor": "simple_rag",
                "connector": "openai",
                "llm": "gpt-4",
                "prompt_processor": "simple_augment",
            },
            rag_collections="col1",
        )
        mock_rag_fn = AsyncMock(return_value={"context": "KB result", "sources": []})
        rag_processors = {"simple_rag": lambda **kw: None}

        result = asyncio.get_event_loop().run_until_complete(
            get_all_rag_contexts(
                assistant=assistant,
                request={"messages": [{"role": "user", "content": "hi"}]},
                rag_processors=rag_processors,
                get_rag_context_fn=mock_rag_fn,
            )
        )

        assert "context" in result
        assert result["context"]["context"] == "KB result"
        mock_rag_fn.assert_called_once()

    def test_multitools_calls_rag_for_each_tool(self):
        assistant = _make_assistant(
            metadata_dict={
                "rag_processor": "simple_rag",
                "connector": "openai",
                "llm": "gpt-4",
                "prompt_processor": "simple_augment",
                "multitools": True,
                "tools": [
                    {"rag_processor": "no_rag", "RAG_collections": "col2"},
                ],
            },
            rag_collections="col1",
        )

        call_count = 0

        async def fake_rag_fn(**kwargs):
            nonlocal call_count
            call_count += 1
            name = kwargs["rag_processor"]
            return {"context": f"result-{name}", "sources": []}

        rag_processors = {
            "simple_rag": lambda **kw: None,
            "no_rag": lambda **kw: None,
        }

        result = asyncio.get_event_loop().run_until_complete(
            get_all_rag_contexts(
                assistant=assistant,
                request={"messages": [{"role": "user", "content": "hi"}]},
                rag_processors=rag_processors,
                get_rag_context_fn=fake_rag_fn,
            )
        )

        assert call_count == 2
        assert result["context"]["context"] == "result-simple_rag"
        assert result["context2"]["context"] == "result-no_rag"

    def test_rag_failure_returns_empty_context(self):
        assistant = _make_assistant(
            metadata_dict={
                "rag_processor": "simple_rag",
                "connector": "openai",
                "llm": "gpt-4",
                "prompt_processor": "simple_augment",
            },
            rag_collections="col1",
        )

        async def failing_rag(**kwargs):
            raise RuntimeError("KB server down")

        rag_processors = {"simple_rag": lambda **kw: None}

        result = asyncio.get_event_loop().run_until_complete(
            get_all_rag_contexts(
                assistant=assistant,
                request={"messages": [{"role": "user", "content": "hi"}]},
                rag_processors=rag_processors,
                get_rag_context_fn=failing_rag,
            )
        )

        assert result["context"]["context"] == ""
        assert result["context"]["sources"] == []


class TestMultiContextPrompt:
    def test_single_context_replacement_legacy(self):
        """Legacy dict with 'context' key works unchanged."""
        assistant = _make_assistant(
            metadata_dict={
                "prompt_processor": "simple_augment",
                "connector": "openai",
                "llm": "gpt-4",
                "rag_processor": "simple_rag",
            },
        )
        assistant = assistant.model_copy(update={
            "system_prompt": "You are helpful.",
            "prompt_template": "Context: {context}\nQuestion: {user_input}",
        })

        request = {"messages": [{"role": "user", "content": "What is X?"}]}
        rag_context = {"context": "KB content", "sources": []}

        messages = prompt_processor(request, assistant, rag_context)
        assert "KB content" in messages[1]["content"]
        assert "{context}" not in messages[1]["content"]

    def test_multi_context_dict_of_dicts(self):
        """When rag_context has nested dicts (multitool format), all placeholders are replaced."""
        assistant = _make_assistant(
            metadata_dict={
                "prompt_processor": "simple_augment",
                "connector": "openai",
                "llm": "gpt-4",
                "rag_processor": "simple_rag",
            },
        )
        assistant = assistant.model_copy(update={
            "system_prompt": "You are helpful.",
            "prompt_template": (
                "Primary: {context}\n"
                "Secondary: {context2}\n"
                "Tertiary: {context3}\n"
                "Q: {user_input}"
            ),
        })

        request = {"messages": [{"role": "user", "content": "Compare."}]}
        rag_context = {
            "context": {"context": "Primary data", "sources": []},
            "context2": {"context": "Secondary data", "sources": []},
            "context3": {"context": "Rubric text", "sources": []},
        }

        messages = prompt_processor(request, assistant, rag_context)
        content = messages[1]["content"]
        assert "Primary data" in content
        assert "Secondary data" in content
        assert "Rubric text" in content
        assert "{context}" not in content
        assert "{context2}" not in content
        assert "{context3}" not in content

    def test_missing_context_key_replaced_with_empty(self):
        assistant = _make_assistant(
            metadata_dict={
                "prompt_processor": "simple_augment",
                "connector": "openai",
                "llm": "gpt-4",
                "rag_processor": "simple_rag",
            },
        )
        assistant = assistant.model_copy(update={
            "system_prompt": "",
            "prompt_template": "Data: {context}\nExtra: {context2}\nQ: {user_input}",
        })

        request = {"messages": [{"role": "user", "content": "Hello"}]}
        rag_context = {"context": {"context": "Some data", "sources": []}}

        messages = prompt_processor(request, assistant, rag_context)
        content = messages[-1]["content"]
        assert "Some data" in content
        assert "{context2}" not in content

    def test_none_rag_context_cleans_all_placeholders(self):
        assistant = _make_assistant(
            metadata_dict={
                "prompt_processor": "simple_augment",
                "connector": "openai",
                "llm": "gpt-4",
                "rag_processor": "",
            },
        )
        assistant = assistant.model_copy(update={
            "system_prompt": "",
            "prompt_template": "{context} {context2} {context3} Q: {user_input}",
        })

        request = {"messages": [{"role": "user", "content": "Hi"}]}
        messages = prompt_processor(request, assistant, None)
        content = messages[-1]["content"]
        assert "{context}" not in content
        assert "{context2}" not in content


class TestPerToolTopK:
    def test_tool_0_uses_assistant_level_top_k(self):
        assistant = _make_assistant(
            metadata_dict={
                "rag_processor": "simple_rag",
                "connector": "openai",
                "llm": "gpt-4",
                "prompt_processor": "simple_augment",
            },
            rag_collections="col1",
            rag_top_k=5,
        )
        tools = get_all_tools_config(assistant)
        assert len(tools) == 1
        assert "RAG_Top_k" not in tools[0]

    def test_additional_tool_has_own_top_k(self):
        assistant = _make_assistant(
            metadata_dict={
                "rag_processor": "simple_rag",
                "connector": "openai",
                "llm": "gpt-4",
                "prompt_processor": "simple_augment",
                "multitools": True,
                "tools": [
                    {"rag_processor": "context_aware_rag", "RAG_collections": "col2", "RAG_Top_k": 7},
                ],
            },
            rag_collections="col1",
            rag_top_k=3,
        )
        tools = get_all_tools_config(assistant)
        assert len(tools) == 2
        assert tools[1]["RAG_Top_k"] == 7

    def test_additional_tool_defaults_top_k_when_missing(self):
        assistant = _make_assistant(
            metadata_dict={
                "rag_processor": "simple_rag",
                "connector": "openai",
                "llm": "gpt-4",
                "prompt_processor": "simple_augment",
                "multitools": True,
                "tools": [
                    {"rag_processor": "context_aware_rag", "RAG_collections": "col2"},
                ],
            },
            rag_collections="col1",
            rag_top_k=3,
        )
        tools = get_all_tools_config(assistant)
        assert "RAG_Top_k" not in tools[1]

    def test_create_tool_assistant_overrides_top_k(self):
        assistant = _make_assistant(
            metadata_dict={
                "rag_processor": "simple_rag",
                "connector": "openai",
                "llm": "gpt-4",
                "prompt_processor": "simple_augment",
            },
            rag_collections="col1",
            rag_top_k=3,
        )
        tool_config = {
            "context_key": "context2",
            "rag_processor": "context_aware_rag",
            "RAG_collections": "col2",
            "RAG_Top_k": 7,
        }
        tool_assistant = _create_tool_assistant(assistant, tool_config)
        assert tool_assistant.RAG_Top_k == 7


class TestValidateMultitoolsConfig:
    def test_valid_tools_array(self):
        metadata = {
            "multitools": True,
            "tools": [
                {"rag_processor": "context_aware_rag", "RAG_collections": "col1"},
                {"rag_processor": "rubric_rag", "rubric_id": "rubric-123"},
            ],
        }
        assert validate_multitools_config(metadata) is None

    def test_tool_missing_rag_processor(self):
        metadata = {
            "multitools": True,
            "tools": [{"RAG_collections": "col1"}],
        }
        error = validate_multitools_config(metadata)
        assert error is not None
        assert "rag_processor" in error

    def test_multitools_false_skips_validation(self):
        metadata = {
            "multitools": False,
            "tools": [{"RAG_collections": "col1"}],
        }
        assert validate_multitools_config(metadata) is None

    def test_empty_tools_array_is_valid(self):
        metadata = {"multitools": True, "tools": []}
        assert validate_multitools_config(metadata) is None

    def test_tools_not_array_is_invalid(self):
        metadata = {"multitools": True, "tools": "not-an-array"}
        error = validate_multitools_config(metadata)
        assert error is not None
        assert "array" in error.lower()

    def test_no_multitools_key_skips(self):
        metadata = {"rag_processor": "simple_rag"}
        assert validate_multitools_config(metadata) is None


class TestValidateMultitoolsConfigExtended:
    def test_max_4_tools_in_array(self):
        metadata = {
            "multitools": True,
            "tools": [
                {"rag_processor": "simple_rag", "RAG_collections": f"col{i}"}
                for i in range(5)
            ],
        }
        error = validate_multitools_config(metadata)
        assert error is not None
        assert "maximum" in error.lower() or "4" in error

    def test_exactly_4_tools_is_valid(self):
        metadata = {
            "multitools": True,
            "tools": [
                {"rag_processor": "simple_rag", "RAG_collections": f"col{i}"}
                for i in range(4)
            ],
        }
        error = validate_multitools_config(metadata)
        assert error is None
