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
- [ ] **Per-tool RAG_Top_k**: Currently all tools share the top-level `RAG_Top_k`. Each tool in `tools[]` should eventually be able to override `RAG_Top_k` independently. Deferred to frontend phase.
- [ ] **Sources formatting for additional tools**: Only tool 0 (primary context) gets source citation formatting in the prompt. Additional tools inject raw context text without source links. This should be revisited when the frontend displays per-tool sources.
- [ ] **`_create_tool_assistant` uses `model_copy()`**: This works because all current RAG processors only read `assistant.metadata` (JSON string) and `assistant.RAG_collections`. If a future RAG processor reads other Assistant fields in unexpected ways (e.g., `owner` for different orgs), this approach may need revisiting.
- [ ] **Dynamic placeholder count**: Currently fixed at 5 placeholders (`{context}` through `{context5}`). If educators need more, switch to scanning the `prompt_template` for `{contextN}` patterns dynamically.
- [ ] **Frontend static multitool UI**: The entire frontend for configuring multiple tools per assistant (adding/removing tools, selecting RAG processor per tool, assigning KBs per tool, inserting `{contextN}` placeholder buttons) is not yet built. This backend work prepares the data model and pipeline for it.
- [ ] **Validation on create path**: `validate_multitools_config` is integrated into `validate_update_plugin_metadata` (update only). The create path (`prepare_assistant_body`) only applies `_ensure_metadata_defaults`. Since multitools will initially only be set via API/CLI, this is acceptable, but should be added when the frontend UI is built.

## Implementation Progress

### Task 1: Tool config extraction utility
**Status:** Not started
**Files:** `backend/lamb/completions/multitool_manager.py`, `backend/tests/test_multitools.py`

### Task 2: Multi-context RAG execution in completion pipeline
**Status:** Not started
**Files:** `backend/lamb/completions/main.py`, `backend/tests/test_multitools.py`

### Task 3: Prompt processor handles multiple context placeholders
**Status:** Not started
**Files:** `backend/lamb/completions/pps/simple_augment.py`, `backend/tests/test_multitools.py`

### Task 4: Validation for tools array in assistant router
**Status:** Not started
**Files:** `backend/creator_interface/assistant_router.py`, `backend/tests/test_multitools.py`
