# KV Cache Classroom Test

Simulates **20 students × 15 minutes** against a LAMB assistant (Dual Tool: `library_file_rag` + `query_rewriting_ks_rag` + `kvcache_augment`). Records tokens/costs and generates graphs of **accumulated cost with KV cache vs counterfactual**.

Used for TFG chapter 10 evaluation. See **Recorded runs** below for the canonical result folders committed in this repo.

## Prerequisites

### Backend configuration (in `backend/.env`)
| Variable | Purpose |
|----------|---------|
| `LAMB_LIBRARY_SERVER` | Library Manager URL (port 9091) |
| `LAMB_LIBRARY_TOKEN` | Auth token for Library Manager |
| `LAMB_KVCACHE_DOCUMENT_PLACEMENT` | `system` (Dual Tool, default) or `user_template` (baseline load test only) |

### UI setup
1. **Libraries:** Import the reference document (e.g. `codi-deontologic-coeinf-12-cat.pdf`).
2. **Knowledge Store:** Link variable context (e.g. `Ch7_2526_Compresio_imatges_4.pdf`).
3. **Assistant:** `kvcache_augment`, `library_file_rag`, `query_rewriting_ks_rag`, model Qwen 3.6 Flash or GPT-5 nano, **no restrictive cost quota**.
4. **Cost Management → Model pricing:** Tariffs must match `.env` pricing block for the model under test. Qwen requires `requires_explicit_cache=1`.
5. Copy the **Assistant ID** to `.env`.

### Python dependencies
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Setup

1. Copy `.env.sample` to `.env` and fill in credentials:
```bash
cp .env.sample .env
```

2. Set pricing for the model you are testing (Qwen vs GPT-5 nano — see `.env.sample` comments).

3. Edit `questions/kv_cache_pool.txt` if needed (default pool covers COEINF ethics + JPEG compression topics).

## Run

### Smoke test
```bash
python classroom_simulation.py --smoke --stateless
```
Dual Tool (doc in system): expect high `cached_tokens` on request 2 with Qwen.  
Baseline (`LAMB_KVCACHE_DOCUMENT_PLACEMENT=user_template`): expect **low** cache with Qwen; OpenAI may still cache user-prefix on nano.

### Full simulation
```bash
python classroom_simulation.py --stateless   # one message per request
python classroom_simulation.py --multi-turn  # accumulated history per student
```

Results saved to `results/run_{timestamp}/`.

### Analyze results
```bash
python analyze_kv_cache_run.py results/run_{timestamp}/
```
Generates:
- `cost_comparison_{run_id}.png` — accumulated cost graph
- `cache_by_turn_{run_id}.png` — avg prompt / cache read / cache write by turn
- `turn_breakdown_{run_id}.json` — per-turn averages
- `cost_summary_{run_id}.json` — summary statistics

### Optional: Validate against LAMB DB
```bash
python analyze_kv_cache_run.py results/run_{timestamp}/ --sqlite /path/to/lamb_v4.db
```

## Request flags

Both modes always send:
```json
{
  "stream": false,
  "persist_chat": false
}
```

| Flag | Why |
|------|-----|
| `stream: false` | Full `usage` with cache buckets only in non-streaming responses |
| `persist_chat: false` | History managed by harness (multi-turn), not LAMB DB |

## Recorded runs

Committed result folders for TFG reproducibility. **Canonical runs** (chapter 10) are marked with ★.

### ★ Dual Tool — document in system prompt (`LAMB_KVCACHE_DOCUMENT_PLACEMENT=system`)

| Folder | Model | Mode | Requests | Savings | Cost | Role |
|--------|-------|------|----------|---------|------|------|
| `run_20260619_200426` | Qwen 3.6 Flash | stateless | 252 | 43.5% | $0.89 | ★ Primary Qwen stateless |
| `run_20260619_202643` | Qwen 3.6 Flash | multi-turn | 243 | 47.5% | $0.90 | ★ Primary Qwen multi-turn |
| `run_20260619_222912` | GPT-5 nano | stateless | 215 | 31.2% | $0.32 | ★ Primary Nano stateless |
| `run_20260619_225433` | GPT-5 nano | multi-turn | 226 | 35.4% | $0.33 | ★ Primary Nano multi-turn |

### ★ Baseline — document in user `{context}` (`LAMB_KVCACHE_DOCUMENT_PLACEMENT=user_template`)

Same pipeline and assistants; only document placement differs (validates Dual Tool architecture).

| Folder | Model | Mode | Requests | Savings | Cost | Role |
|--------|-------|------|----------|---------|------|------|
| `run_20260620_002750` | Qwen 3.6 Flash | stateless | 251 | 0.1% | $1.59 | ★ Baseline Qwen stateless |
| `run_20260620_005026` | Qwen 3.6 Flash | multi-turn | 250 | 9.5% | $1.73 | ★ Baseline Qwen multi-turn |
| `run_20260620_011614` | GPT-5 nano | stateless | 238 | 32.9% | $0.34 | ★ Baseline Nano stateless |
| `run_20260620_015526` | GPT-5 nano | multi-turn | 232 | 8.9% | $0.48 | ★ Baseline Nano multi-turn |

### Exploratory runs (June 5–8 — superseded pricing/model, annex only)

Early iterations before canonical Qwen Flash / Nano matrix. Not used as primary TFG results.

| Folder | Requests | Savings | Cost | Notes |
|--------|----------|---------|------|-------|
| `run_20260605_143250` | 196 | 48.8% | $0.99 | First 15 min run |
| `run_20260605_144400` | 73 | 31.6% | $0.49 | Short run |
| `run_20260606_182118` | 185 | 49.7% | $0.89 | Original draft primary run |
| `run_20260606_190505` | 168 | 43.1% | $1.05 | Variant |
| `run_20260608_193639` | 152 | 42.3% | $0.98 | Variant |

### Other

| Folder | Notes |
|--------|-------|
| `test_run` | Early harness smoke (~few requests), no `summary.json` |

Each run folder contains: `requests.jsonl`, `summary.json`, `cost_comparison_*.png`, `cache_by_turn_*.png`, `turn_breakdown_*.json`, `cost_summary_*.json`.

## What this test does NOT include

- LAMB 5-min cache for GET to Library Manager
- Script-based assistant/document creation
- Committing `.env` or `.venv/` (see local `.gitignore`)
