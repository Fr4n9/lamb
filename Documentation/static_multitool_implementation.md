# Static Multitool Implementation Log

> This document tracks the implementation of the static multitool backend support.
> It is updated task-by-task as implementation progresses.
> This is the first phase (backend-only); the frontend multitool UI will be built later.

## Overview

Adds backend support for multiple context sources ("tools") per assistant. Each tool
has its own RAG processor and KB collections (or rubric), sharing a single connector.
Context is injected into numbered placeholders ({context}, {context2}, {context3}, ...)
in the prompt template.

## Architecture Summary

- New metadata fields: `multitools` (bool) + `tools` (array)
- Tool 0 = existing top-level fields (backward compatible)
- Additional tools in `tools[]` array, each with its own `rag_processor`
- Parallel RAG execution via `asyncio.gather` + `asyncio.to_thread` for sync processors
- Prompt processor replaces `{context}`, `{context2}` ... `{context5}` placeholders

## Known TODOs and Future Improvements

- [ ] **Vision path multi-context**: `simple_augment.py` vision code path only replaces `{context}`. Multi-context placeholders (`{context2}`-`{context5}`) are NOT supported in vision mode yet. Needs a refactor to extract shared context-replacement logic into a helper used by both vision and text-only paths.
- [x] **Per-tool RAG_Top_k**: Each tool in `tools[]` can override `RAG_Top_k` independently via `TOOL_CONFIG_FIELDS` and `_create_tool_assistant`. Tool 0 uses top-level `assistant.RAG_Top_k`.
- [ ] **Sources formatting for additional tools**: Only tool 0 (primary context) gets source citation formatting in the prompt. Additional tools inject raw context text without source links. This should be revisited when the frontend displays per-tool sources.
- [ ] **`_create_tool_assistant` uses `model_copy()`**: This works because all current RAG processors only read `assistant.metadata` (JSON string) and `assistant.RAG_collections`. If a future RAG processor reads other Assistant fields in unexpected ways (e.g., `owner` for different orgs), this approach may need revisiting.
- [ ] **Dynamic placeholder count**: Currently fixed at 5 placeholders (`{context}` through `{context5}`). If educators need more, switch to scanning the `prompt_template` for `{contextN}` patterns dynamically.
- [ ] **Frontend static multitool UI**: The entire frontend for configuring multiple tools per assistant (adding/removing tools, selecting RAG processor per tool, assigning KBs per tool, inserting `{contextN}` placeholder buttons) is not yet built. This backend work prepares the data model and pipeline for it.
- [x] **Validation on create path**: `validate_multitools_config` is integrated into both `validate_update_plugin_metadata` (update) and `prepare_assistant_body` (create). Max 4 additional tools enforced.

## Implementation Progress

### Task 1: Tool config extraction utility
**Status:** Complete
**Commit:** `5b0a144b`
**Files:** `backend/lamb/completions/multitool_manager.py`, `backend/tests/test_multitools.py`

**What was done:**
- Created `multitool_manager.py` with `get_all_tools_config()`, `_create_tool_assistant()`, `get_all_rag_contexts()`
- Created `test_multitools.py` with tests for all tasks (19 total)
- All 6 `TestGetAllToolsConfig` tests pass

**Deviations:** Test file includes all task tests upfront for simpler maintenance.

**New TODOs discovered:** None

### Task 2: Multi-context RAG execution in completion pipeline
**Status:** Complete
**Commit:** `10db5821`
**Files:** `backend/lamb/completions/main.py`, `backend/tests/test_multitools.py`

**What was done:**
- Extended `load_and_validate_plugins` to validate all tools' RAG processors
- Integrated `get_all_rag_contexts` in `create_completion` and `run_lamb_assistant`
- Wrapped sync RAG processors with `asyncio.to_thread()` in `get_rag_context`
- 3 `TestGetAllRagContexts` tests pass

**Deviations:** None

**New TODOs discovered:** None

### Task 3: Prompt processor handles multiple context placeholders
**Status:** Complete
**Commit:** `875bc072`
**Files:** `backend/lamb/completions/pps/simple_augment.py`, `backend/tests/test_multitools.py`

**What was done:**
- Text-only path in `simple_augment.py` supports multi-context dict-of-dicts format
- Replaces `{context}` through `{context5}` placeholders
- Sources formatting for primary context (tool 0) only
- Added TODO comment for vision path multi-context support
- 4 `TestMultiContextPrompt` tests pass (2 tests use `messages[-1]` when system_prompt is empty)

**Deviations:** Tests use `messages[-1]` instead of `messages[1]` when system_prompt is empty (plan had index bug).

**New TODOs discovered:** None

### Task 4: Validation for tools array in assistant router
**Status:** Complete
**Commit:** `51a45019`
**Files:** `backend/creator_interface/assistant_router.py`, `backend/tests/test_multitools.py`

**What was done:**
- Added `validate_multitools_config()` function
- Integrated into `validate_update_plugin_metadata` for assistant updates
- 6 `TestValidateMultitoolsConfig` tests pass
- Full backend suite: 58 tests pass

**Deviations:** None

**New TODOs discovered:** None

### Task 5: Per-tool RAG_Top_k and create-path validation
**Status:** Complete
**Files:** `backend/lamb/completions/multitool_manager.py`, `backend/creator_interface/assistant_router.py`, `backend/tests/test_multitools.py`

**What was done:**
- Added `RAG_Top_k` to `TOOL_CONFIG_FIELDS`; `_create_tool_assistant` overrides assistant-level `RAG_Top_k` per tool
- Added `MAX_ADDITIONAL_TOOLS = 4` validation in `validate_multitools_config`
- Integrated `validate_multitools_config` into `prepare_assistant_body` (create path)
- 6 new tests: 4 for per-tool top_k, 2 for max tools limit
- All 25 multitools tests pass

**Deviations:** None

**New TODOs discovered:** None
