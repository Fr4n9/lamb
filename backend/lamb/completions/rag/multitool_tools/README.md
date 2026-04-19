# Multitool RAG Processor

Async RAG processor that uses a lightweight LLM orchestrator to select and execute multiple tools in parallel, aggregating results into a unified `rag_context` for downstream prompt processing.

## Architecture

```
User message
    │
    ▼
┌──────────────────────────────────┐
│  multitool_rag.rag_processor()   │
│                                  │
│  1. Parse metadata.multitool     │
│  2. Intersect enabled_tools      │
│     with ToolRegistry            │
│  3. Call small-fast-model        │
│     (orchestrator)               │
│  4. Filter hallucinated tools    │
│  5. Execute tools in parallel    │
│     (asyncio.gather + timeouts)  │
│  6. Aggregate into rag_context   │
└──────────────┬───────────────────┘
               │
               ▼
       simple_augment (PPS)
       injects context into
       prompt template
               │
               ▼
         Connector (LLM)
```

The multitool RAG processor plugs into the existing pipeline at the RAG slot. No changes to `main.py`, `simple_augment`, or any connector are required.

## Files

| File | Purpose |
|------|---------|
| `multitool_rag.py` | Main RAG processor: orchestrator call, parallel tool execution, context aggregation |
| `multitool_schema.py` | Pydantic models (`RawToolCall`, `RawOrchestratorPlan`) + parsing functions for metadata and orchestrator JSON |
| `multitool_tools/registry.py` | `ToolRegistry` class mapping tool names to async callables |
| `multitool_tools/kb_query.py` | KB query tool — queries one or more KB Server collections via HTTP |
| `multitool_tools/rubric.py` | Rubric tool — fetches a rubric from the DB and formats it for evaluation |
| `multitool_tools/__init__.py` | Package marker |

## Metadata contract

Configuration lives in the assistant's `api_callback` JSON column (accessed as `assistant.metadata`). Add a `multitool` block:

```json
{
  "prompt_processor": "simple_augment",
  "connector": "openai",
  "llm": "gpt-4o-mini",
  "rag_processor": "multitool_rag",
  "multitool": {
    "enabled_tools": ["kb_query", "rubric"],
    "per_tool": {
      "kb_query": { "collections": ["collection-id-1"], "top_k": 3 },
      "rubric": { "rubric_id": "uuid-here" }
    },
    "orchestrator": {
      "max_tools": 4,
      "per_tool_timeout_sec": 12,
      "total_timeout_sec": 25
    }
  }
}
```

### Fields

- **`enabled_tools`**: allowlist of tool names the orchestrator may select. Must exist in the `ToolRegistry`. Tools not in this list are never executed, even if the orchestrator hallucinates them.
- **`per_tool`**: per-tool configuration merged with orchestrator-provided arguments. Server-side values take precedence for security.
- **`orchestrator.per_tool_timeout_sec`**: maximum seconds per individual tool (default: 12).
- **`orchestrator.total_timeout_sec`**: maximum wall-clock seconds for all tools combined (default: 25).

## rag_context output shape

The processor returns a dict compatible with `simple_augment` (which reads `rag_context["context"]`):

```python
{
    "context": "=== Tool: kb_query (ok) ===\n...\n\n=== Tool: rubric (error) ===\ntimeout after 12s",
    "sources": [{"title": "...", "url": "..."}],
    "tool_results": {
        "kb_query": {"ok": True, "context": "...", "sources": [...]},
        "rubric": {"ok": False, "error": "timeout after 12s", "tool": "rubric"}
    },
    "orchestrator_raw": {"tools": [...], "rationale": "..."}
}
```

## Orchestrator

The orchestrator is a single call to the organization's small-fast-model (`invoke_small_fast_model`). It receives:

- A system prompt listing only the allowed tools with descriptions
- The user's query

It returns strict JSON: `{"tools": [{"name": "...", "arguments": {...}}], "rationale": "..."}`.

Unknown tool names in the response are silently filtered and logged. Invalid JSON causes the processor to return an error in `context` (no tools are executed).

## Concurrency and timeouts

- Tools run in parallel via `asyncio.gather`.
- Each tool is wrapped in `asyncio.wait_for` with `per_tool_timeout_sec`.
- The entire gather is wrapped in a global `asyncio.wait_for` with `total_timeout_sec`.
- On timeout, the tool result is `{"ok": False, "error": "timeout after Xs"}`.
- The downstream LLM sees `(error)` sections in the context and can decide how to respond.

## Security

- **Hallucination filter**: tool names from the orchestrator are intersected with `enabled_tools` AND the registry. Unknown names are rejected.
- **Argument stripping**: each tool module defines `ALLOWED_ARGS` (frozenset). Orchestrator-supplied arguments not in this set are silently dropped.
- **Server-side defaults**: `per_tool` config from metadata is merged under orchestrator args, so the assistant owner's configuration takes precedence.

## Adding a new tool

1. Create `multitool_tools/your_tool.py` with:
   - `ALLOWED_ARGS = frozenset({"arg1", "arg2"})` 
   - `async def execute(*, arg1, assistant_owner, **_extra) -> Dict[str, Any]`
   - Return `{"ok": True, "tool": "your_tool", "context": "...", "sources": [...]}`
2. Register it in `multitool_rag.py`:
   - `from lamb.completions.rag.multitool_tools import your_tool`
   - `_registry.register("your_tool", your_tool.execute)`
   - Add to `_tool_modules` dict inside `rag_processor`
3. Add a description in `_build_orchestrator_prompt`
4. Configure in assistant metadata: add to `enabled_tools` and `per_tool`

## Tests

```bash
cd /home/franpv2004/proyecto/lamb
PYTHONPATH=backend:$PYTHONPATH backend/.venv/bin/python -m pytest testing/unit-tests/completions/test_multitool_rag.py -v
```

15 tests covering:

| Category | Tests |
|----------|-------|
| Schema parsing | `test_parse_metadata_multitool_*` (2) |
| Hallucination filter | `test_parse_orchestrator_response_filters_unknown_tool` |
| Timeout handling | `test_run_tool_with_timeout_*` (3) |
| Orchestrator wiring | `test_orchestrate_tool_plan_calls_small_fast_model` |
| Happy path integration | `test_rag_processor_happy_path_two_tools` |
| Missing metadata | `test_rag_processor_missing_multitool_metadata` |
| Invalid tools | `test_rag_processor_no_valid_tools_enabled` |
| Invalid orchestrator JSON | `test_rag_processor_orchestrator_returns_invalid_json` |
| Empty tool selection | `test_rag_processor_orchestrator_returns_empty_tool_list` |
| Tool timeout propagation | `test_rag_processor_tool_timeout_surfaces_error` |
| JSON decode error | `test_parse_orchestrator_response_invalid_json_raises` |
| Missing required tool args | `test_rag_processor_rubric_missing_required_key` |

## Manual testing (no frontend)

Configure an assistant via API:

```bash
curl -X PUT "http://localhost:9099/creator/assistant/update_assistant/<ID>" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"metadata": "{\"prompt_processor\":\"simple_augment\",\"connector\":\"openai\",\"llm\":\"gpt-4o-mini\",\"rag_processor\":\"multitool_rag\",\"multitool\":{\"enabled_tools\":[\"kb_query\"],\"per_tool\":{\"kb_query\":{\"collections\":[\"<COLLECTION_ID>\"],\"top_k\":3}},\"orchestrator\":{\"per_tool_timeout_sec\":12,\"total_timeout_sec\":25}}}"}'
```

Send a completion:

```bash
curl -X POST "http://localhost:9099/lamb/v1/completions/?assistant=<ID>" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Explain the second law of Newton"}], "stream": false}'
```
