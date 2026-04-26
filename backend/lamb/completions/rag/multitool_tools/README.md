# Multitool RAG Processor

Async RAG processor that uses a lightweight LLM orchestrator to select and execute multiple tools in parallel, aggregating results into a unified `rag_context` for downstream prompt processing.

## Architecture

```
User message + Conversation history
    │
    ▼
┌──────────────────────────────────────┐
│  multitool_rag.rag_processor()       │
│                                      │
│  1. Parse metadata.multitool         │
│  2. Intersect enabled_tools          │
│     with ToolRegistry                │
│  3. Extract conversation memory      │
│     (last N turns, default 2)        │
│  4. Query rewriting                  │
│     (memory + current prompt         │
│      → small-fast-model call 1)      │
│  5. Fetch dynamic descriptions:      │
│     - KB: GET /collections/{id}      │
│     - Rubric: DB title + desc        │
│  6. Call small-fast-model            │
│     (CoT orchestrator, call 2)       │
│     → uses REWRITTEN query           │
│     → STEP 1: Classify intent        │
│     → STEP 2: Select tools           │
│  7. Filter hallucinated tools        │
│  8. Execute tools in parallel        │
│     (asyncio.gather + timeouts)      │
│  9. Aggregate into rag_context       │
└──────────────┬───────────────────────┘
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


| File                          | Purpose                                                                                                       |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `multitool_rag.py`            | Main RAG processor: memory extraction, query rewriting, orchestrator call, parallel tool execution, context aggregation |
| `multitool_schema.py`         | Pydantic models (`RawToolCall`, `RawOrchestratorPlan`) + parsing functions for metadata and orchestrator JSON |
| `multitool_tools/registry.py` | `ToolRegistry` class mapping tool names to async callables                                                    |
| `multitool_tools/kb_query.py` | KB query tool — queries one or more KB Server collections via HTTP                                            |
| `multitool_tools/rubric.py`   | Rubric tool — fetches a rubric from the DB and formats it for evaluation                                      |
| `multitool_tools/__init__.py` | Package marker                                                                                                |


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

- `**enabled_tools**`: allowlist of tool names the orchestrator may select. Must exist in the `ToolRegistry`. Tools not in this list are never executed, even if the orchestrator hallucinates them.
- `**per_tool**`: per-tool configuration merged with orchestrator-provided arguments. Server-side values take precedence for security.
- `**orchestrator.per_tool_timeout_sec**`: maximum seconds per individual tool (default: 12).
- `**orchestrator.total_timeout_sec**`: maximum wall-clock seconds for all tools combined (default: 25).

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
    "orchestrator_raw": {"tools": [...], "rationale": "...", "intent": "SEARCH|EVALUATE|BOTH|NONE"},
    "multitool_debug": { ... }  # diagnostics; not injected into the main prompt (see below)
}
```

The main completion prompt still receives only the usual `context` and `sources` from downstream wiring; the extra `multitool_debug` (and the dump file) exist for observability and local debugging.

### `multitool_debug` (JSON-serializable)

When present, this dict may include: `assistant_id`, `owner_masked` (envelope-style masking), `allowed_tools`, `rejected_by_registry` (orchestrator hallucinations), `user_query_stats` (length, head, tail of the last user message, truncated to reduce PII exposure), `kb_desc_keys` / `rubric_desc_keys`, `per_tool_config_summary` (per-tool metadata with secrets and long values redacted), `memory` (`num_turns_requested`, `messages_count`), `query_rewriting` (`original_query`, `rewritten_query`, `was_rewritten`), `orchestrator` (`raw_llm_text` from the small-fast model, plus `parsed` from the filtered plan, or an `error` on parse failure), `executed` (per tool: redacted `merged_args`, `ok`, `error`), `skipped_no_executor` if a tool has no in-process executor, and `timings_ms` (`orchestrate_ms`, `tools_total_ms`, and `query_rewrite_ms` when rewriting occurred).

## Debug: Markdown context dump (always on)

Every call to `rag_processor` writes a structured Markdown file to **`lamb/testing/context_dumps/`** (resolved from the package path via `Path(__file__).resolve().parents[4]`, not dependent on CWD or environment variables). Each file includes:

1. **Header:** timestamp, assistant id, masked owner.
2. **Memory & Query Rewriting:** memory message count, original query, rewritten query.
3. **LLM timings (small-fast-model):** `query_rewrite_ms` (call 1, skipped when there is no memory), `orchestrate_ms` (call 2), and their sum for quick comparison (same values also appear inside `timings_ms` in the full JSON below).
4. **Orchestrator raw:** the full LLM response text before JSON parsing.
5. **Parsed plan:** pretty-printed JSON of the filtered plan (intent, rationale, tools, rejected).
6. **Tool execution:** per-tool merged args (redacted), ok/error status.
7. **Full `multitool_debug` JSON:** all diagnostics (allowed tools, user query stats, timings, etc.).
8. **Final injected context:** the exact text the main LLM sees.

Long fields are truncated at ~20k chars with a `[truncated]` marker.

**PII note:** the dump and `user_query_stats` can contain fragments of the user's message (head 800 / tail 800 chars). In production, consider disabling the dump write or routing the directory to a secured location.

## Orchestrator

The orchestrator is a call to the organization's small-fast-model (`invoke_small_fast_model`) after optional query rewriting (see [Memory & Query Rewriting](#memory--query-rewriting-prototype-3)). It receives:

- A Chain-of-Thought system prompt with intent classification and tool descriptions
- The user's query (rewritten when conversation memory is present)

It returns strict JSON: `{"intent": "SEARCH|EVALUATE|BOTH|NONE", "tools": [{"name": "...", "arguments": {...}}], "rationale": "..."}`.

Unknown tool names in the response are silently filtered and logged. Invalid JSON causes the processor to return an error in `context` (no tools are executed).

### Chain-of-Thought (CoT) Prompt Structure

The orchestrator prompt uses a two-step reasoning structure to improve tool selection accuracy and scalability:

```
## STEP 1: Classify the user's intent
- SEARCH: user wants information → consider kb_query
- EVALUATE: user submits work for correction/grading → consider rubric
- BOTH: user needs information AND evaluation → consider both
- NONE: query doesn't match any tool

## STEP 2: Select tools based on classification
[Dynamic tool descriptions with available KBs/rubrics]

## IMPORTANT RULES
- Must consider ALL tools before deciding
- Correction/evaluation requests → MUST use rubric
- Use kb_query when retrieval from course materials is needed
```

**Why CoT?** Without intent classification, the orchestrator exhibited a bias toward better-documented tools (e.g., `kb_query` with rich KB descriptions) and consistently ignored less-documented tools (e.g., `rubric`). The CoT structure forces the model to reason about intent first, then select — this scales to N tools without bias.

## Smart Routing (Dynamic Descriptions)

Both `kb_query` and `rubric` tools use **dynamic descriptions** — semantic information fetched at runtime and injected into the orchestrator prompt. This gives the model rich context about what each resource contains, enabling precise routing decisions.

### KB Collection Routing

When `kb_query` is enabled, the processor fetches semantic descriptions for each configured collection from the KB server (`GET /collections/{id}`) before calling the orchestrator. These descriptions are injected into the orchestrator's system prompt, enabling it to select only the relevant collections.

The orchestrator can return a `target_collections` argument:

```json
{"tools": [{"name": "kb_query", "arguments": {"query": "Newton's laws", "target_collections": ["95"]}}]}
```

If `target_collections` is provided, `kb_query` only queries those specific collection IDs. If omitted or empty, it falls back to querying all configured collections (backward compatible with Prototype 1).

The `per_tool.kb_query` block remains the same. The `target_collections` argument is provided by the orchestrator at runtime; it is not part of the static metadata configuration.

### Rubric Routing

When `rubric` is enabled, the processor fetches the rubric's title and description from the database (`RubricDatabaseManager.get_rubric_by_id()`) before calling the orchestrator. This is injected into the prompt:

```
- rubric: Retrieve and apply an evaluation rubric to assess, grade, or correct student work.
  Available rubrics:
    - ID abc-123: "Rubric for Roman Empire essays — Evaluates historical accuracy and argumentation"
  Arguments: rubric_id (str), rubric_format ('markdown' or 'json')
```

The orchestrator can then select the rubric by ID:

```json
{"tools": [{"name": "rubric", "arguments": {"rubric_id": "abc-123"}}]}
```

The `rubric_id` from `per_tool.rubric` metadata is used for the description fetch. At execution time, the server-side `per_tool` config takes precedence over orchestrator-supplied arguments (the merge order is `{**orchestrator_args, **per_tool_cfg}`), ensuring the orchestrator cannot override which rubric is used.

### Requirements for administrators

Descriptions should be meaningful and precise for both resource types:

- **KB collections**: Write clear descriptions in the KB Server admin (e.g., "Physics course materials — Newtonian mechanics and thermodynamics"). The orchestrator relies on these to route queries to the correct collection.
- **Rubrics**: Write descriptive titles and descriptions when creating rubrics (e.g., title: "Essay Evaluation — Roman Empire", description: "Evaluates historical accuracy, argumentation, and writing quality"). Vague titles like "Untitled Rubric" will reduce routing accuracy.

## Memory & Query Rewriting (Prototype 3)

The multitool RAG processor uses conversation memory to rewrite ambiguous queries before tool selection. This improves routing accuracy for multi-turn conversations where the user's latest message may contain pronouns or implicit references.

### Flow

1. **Memory extraction**: The last N turns (default 2) of user+assistant messages are extracted from the conversation history. System messages are excluded. The current user message (the prompt being answered) is treated separately.
2. **Query rewriting**: If memory is present, a small-fast-model call rewrites the current user message into a self-contained query by resolving references and incorporating conversation context.
3. **Rewritten query usage**: The rewritten query replaces the raw user message for both the orchestrator (tool selection) and tool execution (e.g., kb_query search). If rewriting fails, the original query is used as fallback.

### LLM calls

Prototype 3 adds **one extra small-fast-model call** per request when conversation memory exists:

| Call | Purpose | Input | Output |
|------|---------|-------|--------|
| 1 (new) | Query rewriting | Memory + current user message | Self-contained rewritten query |
| 2 (existing) | Orchestrator tool selection | Rewritten query + tool descriptions | Tool plan (intent + tools + rationale) |

When there is no conversation history (first message), call 1 is skipped entirely.

### Memory configuration

The number of turns can be configured in the `multitool` metadata block:

```json
{
  "multitool": {
    "memory": {
      "num_turns": 2
    }
  }
}
```

Default: `2` turns (2 user + 2 assistant messages). No token-length filtering is applied in this prototype.

### Example

```
Conversation:
  USER: What is photosynthesis?
  ASSISTANT: Photosynthesis is the process by which plants convert light energy...
  USER: Tell me more about it            ← current message (ambiguous)

Memory extracted: [user: "What is photosynthesis?", assistant: "Photosynthesis is..."]
Rewritten query: "detailed explanation of photosynthesis process and mechanism"
Orchestrator receives: "detailed explanation of photosynthesis process and mechanism"
```

## Concurrency and timeouts

- Tools run in parallel via `asyncio.gather`.
- Each tool is wrapped in `asyncio.wait_for` with `per_tool_timeout_sec`.
- The entire gather is wrapped in a global `asyncio.wait_for` with `total_timeout_sec`.
- On timeout, the tool result is `{"ok": False, "error": "timeout after Xs"}`.
- The downstream LLM sees `(error)` sections in the context and can decide how to respond.

## Security

- **Hallucination filter**: tool names from the orchestrator are intersected with `enabled_tools` AND the registry. Unknown names are rejected.
- **Argument stripping**: each tool module defines `ALLOWED_ARGS` (frozenset). Orchestrator-supplied arguments not in this set are silently dropped.
- **Server-side overrides**: `per_tool` config from metadata is merged *on top of* orchestrator args (`{**orchestrator_args, **per_tool_cfg}`), so the assistant owner's configuration always takes precedence. The orchestrator cannot override `rubric_id`, `collections`, or any other server-configured value.

## Adding a new tool

1. Create `multitool_tools/your_tool.py` with:
   - `ALLOWED_ARGS = frozenset({"arg1", "arg2"})` 
   - `async def execute(*, arg1, assistant_owner, **_extra) -> Dict[str, Any]`
   - Return `{"ok": True, "tool": "your_tool", "context": "...", "sources": [...]}`
   - (Optional) `async def fetch_your_tool_descriptions(...)` for dynamic descriptions
2. Register it in `multitool_rag.py`:
   - `from lamb.completions.rag.multitool_tools import your_tool`
   - `_registry.register("your_tool", your_tool.execute)`
   - Add to `_tool_modules` dict inside `rag_processor`
3. Add a description block in `_build_orchestrator_prompt` (both static fallback and dynamic enriched version)
4. If the tool has dynamic descriptions, add the fetch call in `_rag_processor_internal` (before the orchestrator call)
5. Add an intent category in the CoT prompt STEP 1 if the tool represents a new user intent type
6. Configure in assistant metadata: add to `enabled_tools` and `per_tool`

## Tests

```bash
cd /home/franpv2004/proyecto/lamb
PYTHONPATH=backend:$PYTHONPATH backend/.venv/bin/python -m pytest testing/unit-tests/completions/test_multitool_rag.py -v
```

46 tests covering:

| Category                     | Tests                                                     |
| ---------------------------- | --------------------------------------------------------- |
| Memory extraction            | `test_extract_conversation_memory_*` (5)                  |
| Query rewriting              | `test_rewrite_query_with_memory_*` (4)                    |
| Memory integration (e2e)     | `test_rag_processor_with_memory_*`, `_no_memory_*`, `_rewrite_failure_*` (3) |
| Context dump (memory)        | `test_debug_dump_contains_memory_and_rewriting_sections`  |
| KB description fetch         | `test_fetch_collection_descriptions_*` (3)                |
| KB target filter             | `test_kb_query_execute_*` (2)                             |
| Orchestrator prompt (KB)     | `test_build_orchestrator_prompt_includes/without_kb_descriptions` (2) |
| Orchestrator prompt (CoT)    | `test_build_orchestrator_prompt_has_intent_classification` |
| Orchestrator prompt (rubric) | `test_build_orchestrator_prompt_includes_rubric_descriptions`, `test_build_orchestrator_prompt_rubric_without_descriptions` |
| Orchestrator prompt (rules)  | `test_build_orchestrator_prompt_important_rules`          |
| Rubric description fetch     | `test_fetch_rubric_descriptions_*` (3)                    |
| Schema parsing               | `test_parse_metadata_multitool_*` (2)                     |
| Hallucination filter         | `test_parse_orchestrator_response_filters_unknown_tool`   |
| Intent after filter          | `test_parse_orchestrator_response_preserves_intent_after_filtering` |
| Timeout handling             | `test_run_tool_with_timeout_*` (3)                        |
| Orchestrator wiring          | `test_orchestrate_tool_plan_calls_small_fast_model`       |
| Smart KB routing (e2e)       | `test_rag_processor_smart_routing_queries_only_targeted_collections` |
| Happy path integration       | `test_rag_processor_happy_path_two_tools`                 |
| Missing metadata             | `test_rag_processor_missing_multitool_metadata`           |
| Invalid tools                | `test_rag_processor_no_valid_tools_enabled`               |
| Invalid orchestrator JSON    | `test_rag_processor_orchestrator_returns_invalid_json`    |
| Empty tool selection         | `test_rag_processor_orchestrator_returns_empty_tool_list` |
| Tool timeout propagation     | `test_rag_processor_tool_timeout_surfaces_error`          |
| JSON decode error            | `test_parse_orchestrator_response_invalid_json_raises`    |
| Missing required tool args   | `test_rag_processor_rubric_missing_required_key`          |
| `multitool_debug`            | `test_multitool_debug_includes_orchestrator_raw_and_timings` |
| Context dump (always on)     | `test_debug_dump_always_writes_and_contains_enriched_sections` |


## Manual testing (no frontend)

Configure an assistant with **kb_query only**:

```bash
curl -X PUT "http://localhost:9099/creator/assistant/update_assistant/<ID>" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"metadata": "{\"prompt_processor\":\"simple_augment\",\"connector\":\"openai\",\"llm\":\"gpt-4o-mini\",\"rag_processor\":\"multitool_rag\",\"multitool\":{\"enabled_tools\":[\"kb_query\"],\"per_tool\":{\"kb_query\":{\"collections\":[\"<COLLECTION_ID>\"],\"top_k\":3}},\"orchestrator\":{\"per_tool_timeout_sec\":12,\"total_timeout_sec\":25}}}"}'
```

Configure an assistant with **kb_query + rubric** (CoT routing):

```bash
curl -X PUT "http://localhost:9099/creator/assistant/update_assistant/<ID>" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"metadata": "{\"prompt_processor\":\"simple_augment\",\"connector\":\"openai\",\"llm\":\"gpt-4o-mini\",\"rag_processor\":\"multitool_rag\",\"multitool\":{\"enabled_tools\":[\"kb_query\",\"rubric\"],\"per_tool\":{\"kb_query\":{\"collections\":[\"<COLLECTION_ID>\"],\"top_k\":3},\"rubric\":{\"rubric_id\":\"<RUBRIC_UUID>\"}},\"orchestrator\":{\"per_tool_timeout_sec\":12,\"total_timeout_sec\":25}}}"}'
```

Send a completion:

```bash
curl -X POST "http://localhost:9099/lamb/v1/completions/?assistant=<ID>" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Explain the second law of Newton"}], "stream": false}'
```
