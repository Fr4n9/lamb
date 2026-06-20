#!/usr/bin/env python3
"""Analyze KV cache simulation results and generate cost comparison graph.

Reads JSONL from classroom_simulation.py, computes accumulated costs,
and generates a matplotlib graph showing real cost vs counterfactual.
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime as dt
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# Load .env from script directory
SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR / ".env")

# Pricing (same as simulation)
PRICING_INPUT_PER_1M = float(os.getenv("PRICING_INPUT_PER_1M", "0.80"))
PRICING_CACHE_READ_PER_1M = float(os.getenv("PRICING_CACHE_READ_PER_1M", "0.16"))
PRICING_CACHE_WRITE_PER_1M = float(os.getenv("PRICING_CACHE_WRITE_PER_1M", "1.00"))
PRICING_OUTPUT_PER_1M = float(os.getenv("PRICING_OUTPUT_PER_1M", "2.00"))

# DB table prefix (default LAMB_ → table is LAMB_usage_logs)
LAMB_DB_PREFIX = os.getenv("LAMB_DB_PREFIX", "LAMB_")

# Assistant ID (needed for --sqlite filtering)
ASSISTANT_ID = os.getenv("ASSISTANT_ID", "")


def load_jsonl(filepath: Path) -> list[dict]:
    if not filepath.exists():
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)
    results = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def compute_accumulated_costs(results: list[dict]) -> list[dict]:
    """Compute running totals for actual and counterfactual costs."""
    # Sort by timestamp to ensure chronological order
    results.sort(key=lambda r: r.get("timestamp", ""))

    accumulated = []
    cum_actual = 0.0
    cum_no_cache = 0.0
    cum_cache_read = 0
    cum_cache_write = 0

    for i, r in enumerate(results):
        cum_actual += r.get("cost_usd_actual", 0)
        cum_no_cache += r.get("cost_usd_no_cache", 0)

        usage = r.get("usage", {})
        details = usage.get("prompt_tokens_details", {})
        cum_cache_read += details.get("cached_tokens", 0)
        cum_cache_write += details.get("cache_creation_input_tokens", 0)

        accumulated.append({
            "request_num": i + 1,
            "cum_actual": round(cum_actual, 6),
            "cum_no_cache": round(cum_no_cache, 6),
            "cum_cache_read": cum_cache_read,
            "cum_cache_write": cum_cache_write,
            "delta": round(cum_no_cache - cum_actual, 6),
        })

    return accumulated


def _turn_index(record: dict) -> int:
    """Conversation turn number for this request (1 = first message in thread)."""
    if "history_turns" in record:
        return int(record["history_turns"]) + 1
    return int(record.get("request_num", 1))


def _prompt_tokens(record: dict) -> int:
    if "prompt_tokens" in record:
        return int(record["prompt_tokens"])
    return int(record.get("usage", {}).get("prompt_tokens", 0))


def _cached_tokens(record: dict) -> int:
    if "cached_tokens" in record:
        return int(record["cached_tokens"])
    return int(
        record.get("usage", {})
        .get("prompt_tokens_details", {})
        .get("cached_tokens", 0)
    )


def _cache_write_tokens(record: dict) -> int:
    if "cache_creation_tokens" in record:
        return int(record["cache_creation_tokens"])
    return int(
        record.get("usage", {})
        .get("prompt_tokens_details", {})
        .get("cache_creation_input_tokens", 0)
    )


def breakdown_by_turn(results: list[dict]) -> dict:
    """Aggregate token and cost metrics by conversation turn index."""
    by_turn: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        if r.get("status") != 200:
            continue
        by_turn[_turn_index(r)].append(r)

    stats = {}
    for turn in sorted(by_turn):
        rows = by_turn[turn]
        stats[str(turn)] = {
            "count": len(rows),
            "avg_prompt_tokens": round(mean(_prompt_tokens(r) for r in rows), 1),
            "avg_cached_tokens": round(mean(_cached_tokens(r) for r in rows), 1),
            "avg_cache_write_tokens": round(mean(_cache_write_tokens(r) for r in rows), 1),
            "avg_non_cached_prompt_tokens": round(
                mean(r.get("non_cached_prompt_tokens", 0) for r in rows), 1
            ),
            "avg_cost_usd_actual": round(mean(r.get("cost_usd_actual", 0) for r in rows), 6),
            "avg_messages_sent": round(mean(r.get("messages_sent", 1) for r in rows), 1),
        }
    return stats


def print_turn_breakdown(turn_stats: dict) -> None:
    if not turn_stats:
        print("No successful requests for turn breakdown")
        return

    print("\n=== TURN BREAKDOWN (successful requests) ===")
    print(
        f"{'Turn':>4}  {'Count':>5}  {'Avg prompt':>10}  {'Avg cached':>10}  "
        f"{'Avg write':>9}  {'Avg msgs':>8}"
    )
    for turn in sorted(turn_stats, key=lambda t: int(t)):
        s = turn_stats[turn]
        print(
            f"{turn:>4}  {s['count']:>5}  {s['avg_prompt_tokens']:>10.0f}  "
            f"{s['avg_cached_tokens']:>10.0f}  {s['avg_cache_write_tokens']:>9.0f}  "
            f"{s['avg_messages_sent']:>8.1f}"
        )


def generate_turn_graph(turn_stats: dict, run_id: str, output_dir: Path) -> None:
    """Plot average token buckets by conversation turn."""
    if not turn_stats:
        return

    turns = sorted(turn_stats, key=lambda t: int(t))
    x = [int(t) for t in turns]
    avg_prompt = [turn_stats[t]["avg_prompt_tokens"] for t in turns]
    avg_cached = [turn_stats[t]["avg_cached_tokens"] for t in turns]
    avg_write = [turn_stats[t]["avg_cache_write_tokens"] for t in turns]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, avg_prompt, marker="o", label="Avg prompt tokens", color="#2271b3", linewidth=2)
    ax.plot(x, avg_cached, marker="s", label="Avg cache read tokens", color="#16a34a", linewidth=2)
    ax.plot(x, avg_write, marker="^", label="Avg cache write tokens", color="#ea580c", linewidth=2)

    ax.set_xlabel("Conversation turn (1 = first question)", fontsize=12)
    ax.set_ylabel("Tokens (average per request)", fontsize=12)
    ax.set_title(f"KV Cache tokens by turn — Run {run_id}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / f"cache_by_turn_{run_id}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Turn graph saved: {output_file}")


def generate_graph(accumulated: list[dict], run_id: str, output_dir: Path):
    """Generate cost comparison matplotlib graph."""
    if not accumulated:
        print("No data to graph")
        return

    x = [a["request_num"] for a in accumulated]
    y_actual = [a["cum_actual"] for a in accumulated]
    y_no_cache = [a["cum_no_cache"] for a in accumulated]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(x, y_no_cache, label="Counterfactual (no cache)", color="#94a3b8", linestyle="--", linewidth=2)
    ax.plot(x, y_actual, label="With KV cache", color="#2271b3", linewidth=2)

    # Fill area between lines
    ax.fill_between(x, y_actual, y_no_cache, alpha=0.15, color="#2271b3")

    # Find crossover point (where actual first drops below counterfactual)
    crossover = None
    for a in accumulated:
        if a["cum_actual"] < a["cum_no_cache"]:
            crossover = a["request_num"]
            break

    if crossover:
        ax.axvline(x=crossover, color="#ef4444", linestyle=":", alpha=0.7)
        ax.text(
            crossover, max(y_actual) * 0.5,
            f"Crossover @ request {crossover}",
            rotation=90, color="#ef4444", fontsize=9,
            verticalalignment="center",
        )

    ax.set_xlabel("Request number", fontsize=12)
    ax.set_ylabel("Accumulated cost (USD)", fontsize=12)

    last = accumulated[-1]
    if last["cum_no_cache"] > 0:
        savings_pct = last["delta"] / last["cum_no_cache"] * 100
        subtitle = f"{len(accumulated)} requests | Final savings: ${last['delta']:.4f} ({savings_pct:.1f}%)"
    else:
        subtitle = f"{len(accumulated)} requests | No cost data"

    ax.set_title(
        f"KV Cache Cost Comparison — Run {run_id}\n{subtitle}",
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Narrative note
    note = (
        "Note: Initial requests may show real cost > counterfactual because "
        "cache_write (~125% of input) is more expensive than plain input. "
        "After crossover, cache_read (cheap) dominates."
    )
    fig.text(0.5, 0.01, note, ha="center", fontsize=8, style="italic", color="#64748b")

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    output_file = output_dir / f"cost_comparison_{run_id}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Graph saved: {output_file}")


def validate_against_sqlite(jsonl_path: Path, db_path: Path, assistant_id: str, db_prefix: str) -> dict:
    """Compare script-computed costs vs frozen cost_usd in usage_logs.

    Filters by assistant_id and by the temporal window of the run
    (min/max timestamps from the JSONL) to avoid comparing against
    the entire DB history.
    """
    if not db_path.exists():
        print(f"WARNING: DB not found: {db_path}")
        return {}

    # Load JSONL to get temporal window
    results = load_jsonl(jsonl_path)
    if not results:
        print("WARNING: No results in JSONL to compare")
        return {}

    timestamps = [r.get("timestamp", "") for r in results if r.get("timestamp")]
    if not timestamps:
        print("WARNING: No timestamps in JSONL")
        return {}

    ts_min = min(timestamps)
    ts_max = max(timestamps)

    table_name = f"{db_prefix}usage_logs"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Filter by assistant_id and created_at window
    # created_at is stored as Unix timestamp (integer seconds)
    ts_min_unix = int(dt.fromisoformat(ts_min).timestamp())
    ts_max_unix = int(dt.fromisoformat(ts_max).timestamp())

    cursor.execute(
        f"SELECT SUM(cost_usd) FROM {table_name} "
        f"WHERE assistant_id = ? AND created_at >= ? AND created_at <= ?",
        (assistant_id, ts_min_unix, ts_max_unix),
    )
    db_total = cursor.fetchone()[0] or 0

    conn.close()

    # Sum from JSONL
    jsonl_total = sum(r.get("cost_usd_actual", 0) for r in results)

    delta = abs(db_total - jsonl_total)
    return {
        "db_cost_usd": round(db_total, 4),
        "jsonl_cost_usd": round(jsonl_total, 4),
        "delta": round(delta, 4),
        "delta_pct": round(delta / db_total * 100, 2) if db_total > 0 else 0,
        "db_rows_window": f"assistant_id={assistant_id}, created_at=[{ts_min_unix}, {ts_max_unix}]",
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze KV cache simulation results")
    parser.add_argument("jsonl_file", help="Path to requests.jsonl")
    parser.add_argument("--sqlite", help="Path to LAMB DB for validation")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_file)
    output_dir = jsonl_path.parent

    # Extract run_id from directory name
    run_id = output_dir.name.replace("run_", "")

    print(f"=== ANALYZING RUN: {run_id} ===")
    print(f"Reading: {jsonl_path}")

    # Load and compute
    results = load_jsonl(jsonl_path)
    print(f"Loaded {len(results)} requests")

    accumulated = compute_accumulated_costs(results)
    turn_stats = breakdown_by_turn(results)

    sim_mode = "unknown"
    if results:
        sim_mode = results[0].get("sim_mode", "unknown")
        for r in results:
            if r.get("sim_mode"):
                sim_mode = r["sim_mode"]
                break

    requests_with_cache_write = sum(1 for r in results if _cache_write_tokens(r) > 0)

    generate_graph(accumulated, run_id, output_dir)
    generate_turn_graph(turn_stats, run_id, output_dir)
    print_turn_breakdown(turn_stats)

    turn_breakdown_file = output_dir / f"turn_breakdown_{run_id}.json"
    with open(turn_breakdown_file, "w") as f:
        json.dump(turn_stats, f, indent=2)
    print(f"\nTurn breakdown saved: {turn_breakdown_file}")

    final = accumulated[-1]
    print(f"\n=== SUMMARY ===")
    print(f"Sim mode: {sim_mode}")
    print(f"Total requests: {len(accumulated)}")
    print(f"Cost with cache:    ${final['cum_actual']:.4f}")
    print(f"Cost without cache: ${final['cum_no_cache']:.4f}")
    if final["cum_no_cache"] > 0:
        print(f"Savings: ${final['delta']:.4f} ({final['delta']/final['cum_no_cache']*100:.1f}%)")
    print(f"Total cache read tokens:  {final['cum_cache_read']}")
    print(f"Total cache write tokens: {final['cum_cache_write']}")
    print(f"Requests with cache_write: {requests_with_cache_write}")

    summary = {
        "run_id": run_id,
        "sim_mode": sim_mode,
        "total_requests": len(accumulated),
        "cost_usd_actual": final["cum_actual"],
        "cost_usd_no_cache": final["cum_no_cache"],
        "savings_usd": final["delta"],
        "savings_pct": round(final["delta"] / final["cum_no_cache"] * 100, 2) if final["cum_no_cache"] > 0 else 0,
        "total_cache_read_tokens": final["cum_cache_read"],
        "total_cache_write_tokens": final["cum_cache_write"],
        "requests_with_cache_write": requests_with_cache_write,
        "avg_prompt_tokens_by_turn": {
            turn: stats["avg_prompt_tokens"] for turn, stats in turn_stats.items()
        },
    }
    summary_file = output_dir / f"cost_summary_{run_id}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_file}")

    # Optional SQLite validation
    if args.sqlite:
        print(f"\n=== SQLITE VALIDATION ===")
        if not ASSISTANT_ID:
            print("WARNING: ASSISTANT_ID not set in .env — cannot filter by assistant_id")
            print("         Set ASSISTANT_ID in .env for accurate comparison")
        validation = validate_against_sqlite(
            jsonl_path, Path(args.sqlite), ASSISTANT_ID, LAMB_DB_PREFIX
        )
        if validation:
            print(f"Filter: {validation.get('db_rows_window', 'N/A')}")
            print(f"DB cost:      ${validation['db_cost_usd']:.4f}")
            print(f"JSONL cost:   ${validation['jsonl_cost_usd']:.4f}")
            print(f"Delta:        ${validation['delta']:.4f} ({validation['delta_pct']}%)")
            if validation["delta_pct"] < 5:
                print("✅ Costs match within 5% tolerance")
            else:
                print("⚠️  Costs differ by more than 5% — check pricing alignment")


if __name__ == "__main__":
    main()
