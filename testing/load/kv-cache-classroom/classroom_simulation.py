#!/usr/bin/env python3
"""Classroom simulation: N students × duration against a LAMB assistant.

Sends non-streaming, non-persisted chat requests to the proxy endpoint
and logs usage data (tokens, cache buckets, costs) as JSONL.

Multi-turn mode (default): each simulated student accumulates conversation
history between requests, matching Creator app / Open WebUI behaviour.
LAMB does not truncate history — the client sends the full messages array.

Stateless mode (--stateless): one user message per request (legacy harness).
"""

import argparse
import asyncio
import copy
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

# Load .env from script directory
SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR / ".env")

# Config from environment
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:9099").rstrip("/")
CREATOR_EMAIL = os.getenv("CREATOR_EMAIL")
CREATOR_PASSWORD = os.getenv("CREATOR_PASSWORD")
ASSISTANT_ID = os.getenv("ASSISTANT_ID")

NUM_STUDENTS = int(os.getenv("SIM_NUM_STUDENTS", "20"))
DURATION_MINUTES = int(os.getenv("SIM_DURATION_MINUTES", "15"))
INTERVAL_MIN_S = float(os.getenv("SIM_INTERVAL_MIN_S", "25"))
INTERVAL_MAX_S = float(os.getenv("SIM_INTERVAL_MAX_S", "75"))

# Multi-turn: default true (matches LAMB client behaviour). 0 = unlimited history.
SIM_MULTI_TURN = os.getenv("SIM_MULTI_TURN", "true").lower() in ("1", "true", "yes")
SIM_MAX_HISTORY_MESSAGES = int(os.getenv("SIM_MAX_HISTORY_MESSAGES", "0"))

# Pricing for cost calculation
PRICING_INPUT_PER_1M = float(os.getenv("PRICING_INPUT_PER_1M", "0.80"))
PRICING_CACHE_READ_PER_1M = float(os.getenv("PRICING_CACHE_READ_PER_1M", "0.16"))
PRICING_CACHE_WRITE_PER_1M = float(os.getenv("PRICING_CACHE_WRITE_PER_1M", "1.00"))
PRICING_OUTPUT_PER_1M = float(os.getenv("PRICING_OUTPUT_PER_1M", "2.00"))

# Questions pool
QUESTIONS_FILE = SCRIPT_DIR / "questions" / "kv_cache_pool.txt"

# Set by CLI in main(); used by run_simulation / smoke_test
MULTI_TURN_MODE = SIM_MULTI_TURN


def sim_mode_label(multi_turn: bool | None = None) -> str:
    use_multi = MULTI_TURN_MODE if multi_turn is None else multi_turn
    return "multi_turn" if use_multi else "stateless"


def load_questions():
    if not QUESTIONS_FILE.exists():
        print(f"ERROR: Questions file not found: {QUESTIONS_FILE}")
        sys.exit(1)
    questions = [
        line.strip()
        for line in QUESTIONS_FILE.read_text().splitlines()
        if line.strip()
    ]
    if not questions:
        print("ERROR: No questions in pool")
        sys.exit(1)
    return questions


def compute_costs(usage: dict) -> dict:
    """Compute actual cost (with cache) and counterfactual cost (no cache)."""
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    details = usage.get("prompt_tokens_details", {})
    cached_tokens = details.get("cached_tokens", 0)
    cache_creation = details.get("cache_creation_input_tokens", 0)
    non_cached_prompt = prompt_tokens - cached_tokens - cache_creation

    cost_actual = (
        (non_cached_prompt / 1_000_000) * PRICING_INPUT_PER_1M
        + (cached_tokens / 1_000_000) * PRICING_CACHE_READ_PER_1M
        + (cache_creation / 1_000_000) * PRICING_CACHE_WRITE_PER_1M
        + (completion_tokens / 1_000_000) * PRICING_OUTPUT_PER_1M
    )

    cost_no_cache = (
        (prompt_tokens / 1_000_000) * PRICING_INPUT_PER_1M
        + (completion_tokens / 1_000_000) * PRICING_OUTPUT_PER_1M
    )

    return {
        "cost_usd_actual": round(cost_actual, 6),
        "cost_usd_no_cache": round(cost_no_cache, 6),
        "non_cached_prompt_tokens": non_cached_prompt,
        "cached_tokens": cached_tokens,
        "cache_creation_tokens": cache_creation,
    }


def _build_messages(question: str, history: list | None, multi_turn: bool) -> list[dict]:
    if multi_turn and history:
        return history + [{"role": "user", "content": question}]
    return [{"role": "user", "content": question}]


def _metadata_fields(messages: list, history: list | None, multi_turn: bool) -> dict:
    prior = history or []
    return {
        "sim_mode": sim_mode_label(multi_turn),
        "messages_sent": len(messages),
        "history_turns": len(prior) // 2 if multi_turn else 0,
    }


def _error_result(
    student_id: int,
    request_num: int,
    question: str,
    messages: list,
    history: list | None,
    multi_turn: bool,
    status: int,
    elapsed: float,
    error: str,
) -> dict:
    return {
        "student_id": student_id,
        "request_num": request_num,
        "question": question,
        "status": status,
        "elapsed_s": round(elapsed, 3),
        "timestamp": datetime.now().isoformat(),
        "error": error,
        "cost_usd_actual": 0,
        "cost_usd_no_cache": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        **_metadata_fields(messages, history, multi_turn),
    }


async def login(session: aiohttp.ClientSession) -> str:
    """Login via POST /creator/login and return bearer token."""
    form_data = aiohttp.FormData()
    form_data.add_field("email", CREATOR_EMAIL)
    form_data.add_field("password", CREATOR_PASSWORD)

    async with session.post(
        f"{API_BASE_URL}/creator/login", data=form_data
    ) as resp:
        if resp.status != 200:
            text = await resp.text()
            print(f"Login failed: {resp.status} - {text}")
            sys.exit(1)
        result = await resp.json()
        if not result.get("success"):
            print(f"Login failed: {result.get('error')}")
            sys.exit(1)
        return result["data"]["token"]


async def send_chat(
    session: aiohttp.ClientSession,
    token: str,
    question: str,
    student_id: int,
    request_num: int,
    history: list | None = None,
    multi_turn: bool | None = None,
    max_retries: int = 3,
) -> dict:
    """Send a chat request. Returns usage data; includes assistant_content on 200."""
    use_multi = MULTI_TURN_MODE if multi_turn is None else multi_turn
    messages = _build_messages(question, history, use_multi)

    payload = {
        "messages": messages,
        "stream": False,
        "persist_chat": False,
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries + 1):
        start_time = time.time()
        async with session.post(
            f"{API_BASE_URL}/creator/assistant/{ASSISTANT_ID}/chat/completions",
            json=payload,
            headers=headers,
        ) as resp:
            elapsed = time.time() - start_time
            status = resp.status

            if status == 429 and attempt < max_retries:
                backoff = min(2 ** attempt * 5, 60)
                print(
                    f"  [student {student_id}] 429 rate limit — "
                    f"retry {attempt + 1}/{max_retries} in {backoff}s"
                )
                await asyncio.sleep(backoff)
                continue

            if status == 200:
                data = await resp.json()
                usage = data.get("usage", {})
                costs = compute_costs(usage)
                assistant_content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                return {
                    "student_id": student_id,
                    "request_num": request_num,
                    "question": question,
                    "status": status,
                    "elapsed_s": round(elapsed, 3),
                    "timestamp": datetime.now().isoformat(),
                    "usage": usage,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "assistant_content": assistant_content,
                    **_metadata_fields(messages, history, use_multi),
                    **costs,
                }

            text = await resp.text()
            return _error_result(
                student_id,
                request_num,
                question,
                messages,
                history,
                use_multi,
                status,
                elapsed,
                text,
            )

    return _error_result(
        student_id,
        request_num,
        question,
        messages,
        history,
        use_multi,
        429,
        0,
        "Rate limit exceeded after retries",
    )


def append_to_history(history: list, question: str, assistant_content: str) -> None:
    """Append user + assistant turn to in-memory history (multi-turn mode)."""
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": assistant_content or ""})
    if SIM_MAX_HISTORY_MESSAGES > 0 and len(history) > SIM_MAX_HISTORY_MESSAGES:
        del history[: len(history) - SIM_MAX_HISTORY_MESSAGES]


def result_for_jsonl(result: dict) -> dict:
    """Strip internal fields before writing JSONL."""
    out = copy.deepcopy(result)
    out.pop("assistant_content", None)
    return out


async def smoke_test(token: str):
    """Two-turn smoke: verify cache_read and growing prompt in multi-turn mode."""
    print("\n=== SMOKE TEST ===")
    print(f"Mode: {sim_mode_label()}")

    async with aiohttp.ClientSession() as session:
        questions = load_questions()
        history: list = []

        print("Sending request 1 (no prior history)...")
        r1 = await send_chat(session, token, questions[0], 0, 1, history=None)
        usage1 = r1.get("usage", {})
        details1 = usage1.get("prompt_tokens_details", {})
        cache_read1 = details1.get("cached_tokens", 0)
        cache_write1 = details1.get("cache_creation_input_tokens", 0)
        prompt1 = r1.get("prompt_tokens", 0)
        print(f"  messages_sent={r1.get('messages_sent')}, prompt_tokens={prompt1}")
        print(f"  cache_read={cache_read1}, cache_write={cache_write1}")
        print(f"  Raw usage (turn 1): {json.dumps(usage1, indent=2)}")
        if r1.get("status") != 200:
            print(f"  Status turn 1: {r1.get('status')}")
            if r1.get("error"):
                print(f"  Error turn 1: {r1.get('error')}")

        if MULTI_TURN_MODE and r1.get("status") == 200:
            append_to_history(history, questions[0], r1.get("assistant_content", ""))

        q2 = questions[1] if len(questions) > 1 else questions[0]
        print("Sending request 2...")
        r2 = await send_chat(
            session,
            token,
            q2,
            0,
            2,
            history=history if MULTI_TURN_MODE else None,
        )
        usage2 = r2.get("usage", {})
        details2 = usage2.get("prompt_tokens_details", {})
        cache_read2 = details2.get("cached_tokens", 0)
        prompt2 = r2.get("prompt_tokens", 0)
        print(f"  messages_sent={r2.get('messages_sent')}, prompt_tokens={prompt2}")
        print(f"  cache_read={cache_read2}")
        print(f"  Raw usage (turn 2): {json.dumps(usage2, indent=2)}")

        failed = False
        if cache_read2 <= 0:
            print("❌ SMOKE TEST FAILED: cache_read == 0 on request 2")
            failed = True
        if MULTI_TURN_MODE and prompt2 <= prompt1:
            print(
                f"❌ SMOKE TEST FAILED: prompt_tokens did not grow "
                f"({prompt1} -> {prompt2}) in multi-turn mode"
            )
            failed = True

        if failed:
            print("   Check requires_explicit_cache=1 in Cost Management → Model pricing.")
            sys.exit(1)

        print("✅ SMOKE TEST PASSED")


async def student_loop(
    student_id: int,
    session: aiohttp.ClientSession,
    token: str,
    questions: list,
    duration_s: float,
    results: list,
    results_lock: asyncio.Lock,
):
    """Single student: requests at random intervals until duration expires."""
    start = time.time()
    request_num = 0
    history: list = []

    while time.time() - start < duration_s:
        question = random.choice(questions)
        request_num += 1

        result = await send_chat(
            session,
            token,
            question,
            student_id,
            request_num,
            history=history if MULTI_TURN_MODE else None,
        )

        if MULTI_TURN_MODE and result.get("status") == 200:
            append_to_history(
                history,
                question,
                result.get("assistant_content", ""),
            )

        async with results_lock:
            results.append(result_for_jsonl(result))

        interval = random.uniform(INTERVAL_MIN_S, INTERVAL_MAX_S)
        await asyncio.sleep(interval)


async def run_simulation():
    """Main simulation: NUM_STUDENTS students for DURATION_MINUTES."""
    mode = sim_mode_label()
    print("\n=== CLASSROOM SIMULATION ===")
    print(f"Mode: {mode}")
    print(f"Students: {NUM_STUDENTS}")
    print(f"Duration: {DURATION_MINUTES} minutes")
    print(f"Interval: {INTERVAL_MIN_S}-{INTERVAL_MAX_S}s")
    if SIM_MAX_HISTORY_MESSAGES > 0:
        print(f"History cap: {SIM_MAX_HISTORY_MESSAGES} messages (harness only, not LAMB)")
    print()

    async with aiohttp.ClientSession() as session:
        token = await login(session)
        print(f"Logged in as {CREATOR_EMAIL}")

        questions = load_questions()
        print(f"Loaded {len(questions)} questions from pool")

        results = []
        results_lock = asyncio.Lock()
        duration_s = DURATION_MINUTES * 60

        tasks = [
            student_loop(sid, session, token, questions, duration_s, results, results_lock)
            for sid in range(NUM_STUDENTS)
        ]

        print(f"Starting {NUM_STUDENTS} students for {DURATION_MINUTES} minutes...")
        start_time = time.time()

        async def report_progress():
            while time.time() - start_time < duration_s:
                await asyncio.sleep(30)
                elapsed = time.time() - start_time
                print(f"  [{elapsed/60:.1f}/{DURATION_MINUTES} min] {len(results)} requests so far")

        await asyncio.gather(*tasks, report_progress())

        elapsed_total = time.time() - start_time
        print(f"\nSimulation complete in {elapsed_total/60:.1f} minutes")
        print(f"Total requests: {len(results)}")

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = SCRIPT_DIR / "results" / f"run_{run_id}"
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / "requests.jsonl"
        with open(results_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        successful = [r for r in results if r.get("status") == 200]
        failed = [r for r in results if r.get("status") != 200]
        total_cost = sum(r.get("cost_usd_actual", 0) for r in results)
        total_cost_no_cache = sum(r.get("cost_usd_no_cache", 0) for r in results)
        total_cache_read = sum(
            r.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
            for r in results
        )
        total_cache_write = sum(
            r.get("usage", {}).get("prompt_tokens_details", {}).get("cache_creation_input_tokens", 0)
            for r in results
        )
        requests_with_cache_write = sum(
            1
            for r in successful
            if r.get("usage", {}).get("prompt_tokens_details", {}).get("cache_creation_input_tokens", 0) > 0
        )

        summary = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "sim_mode": mode,
            "num_students": NUM_STUDENTS,
            "duration_minutes": DURATION_MINUTES,
            "total_requests": len(results),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "error_rate": round(len(failed) / len(results) * 100, 2) if results else 0,
            "total_cost_usd_actual": round(total_cost, 4),
            "total_cost_usd_no_cache": round(total_cost_no_cache, 4),
            "savings_usd": round(total_cost_no_cache - total_cost, 4),
            "savings_pct": round(
                (1 - total_cost / total_cost_no_cache) * 100, 2
            ) if total_cost_no_cache > 0 else 0,
            "total_cache_read_tokens": total_cache_read,
            "total_cache_write_tokens": total_cache_write,
            "requests_with_cache_write": requests_with_cache_write,
        }

        summary_file = results_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n=== SUMMARY ===")
        print(f"Mode: {mode}")
        print(f"Requests: {len(successful)} OK, {len(failed)} failed ({summary['error_rate']}%)")
        print(f"Cost with cache:    ${total_cost:.4f}")
        print(f"Cost without cache: ${total_cost_no_cache:.4f}")
        print(f"Savings: ${summary['savings_usd']:.4f} ({summary['savings_pct']}%)")
        print(f"Cache read tokens:  {total_cache_read}")
        print(f"Cache write tokens: {total_cache_write}")
        print(f"Requests with cache_write: {requests_with_cache_write}")
        print(f"\nResults saved to: {results_dir}")
        print(f"Run analysis: python analyze_kv_cache_run.py {results_file}")


def main():
    global MULTI_TURN_MODE

    parser = argparse.ArgumentParser(description="KV Cache Classroom Simulation")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test only")
    parser.add_argument(
        "--stateless",
        action="store_true",
        help="One user message per request (legacy harness, no conversation history)",
    )
    parser.add_argument(
        "--multi-turn",
        action="store_true",
        help="Accumulate conversation history per student (default)",
    )
    args = parser.parse_args()

    if args.stateless and args.multi_turn:
        print("ERROR: Use only one of --stateless or --multi-turn")
        sys.exit(1)

    if args.stateless:
        MULTI_TURN_MODE = False
    elif args.multi_turn:
        MULTI_TURN_MODE = True
    else:
        MULTI_TURN_MODE = SIM_MULTI_TURN

    if not all([CREATOR_EMAIL, CREATOR_PASSWORD, ASSISTANT_ID]):
        print("ERROR: Missing required env vars: CREATOR_EMAIL, CREATOR_PASSWORD, ASSISTANT_ID")
        print("Copy .env.sample to .env and fill in values")
        sys.exit(1)

    async def run():
        async with aiohttp.ClientSession() as session:
            token = await login(session)
            if args.smoke:
                await smoke_test(token)
            else:
                await run_simulation()

    asyncio.run(run())


if __name__ == "__main__":
    main()
