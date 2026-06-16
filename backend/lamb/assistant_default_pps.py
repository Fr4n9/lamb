"""Default prompt_processor resolution (with optional legacy override for E2E/dev)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import config

DEFAULTS_JSON_PATHS = (
    Path(__file__).parent.parent / "static" / "json" / "defaults.json",
    Path("/opt/lamb_v4/backend/static/json/defaults.json"),
    Path("static/json/defaults.json"),
    Path("backend/static/json/defaults.json"),
)


def default_prompt_processor() -> str:
    """Return the default PPS for new assistants."""
    if config.LAMB_LEGACY_PPS_DEFAULT:
        return "simple_augment"
    return "kvcache_augment"


def apply_defaults_pps_policy(mapping: dict[str, Any] | None) -> dict[str, Any] | None:
    """Apply env-driven prompt_processor when serving assistant defaults.

    When LAMB_LEGACY_PPS_DEFAULT is false, always serves kvcache_augment (overrides
    stale org DB values). When true, serves simple_augment (Playwright / legacy E2E).
    """
    if not mapping:
        return mapping
    result = dict(mapping)
    result["prompt_processor"] = default_prompt_processor()
    return result


def load_defaults_json_document() -> dict[str, Any]:
    """Load defaults.json from disk, applying served-defaults PPS policy."""
    for path in DEFAULTS_JSON_PATHS:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data.get("config"), dict):
            data = dict(data)
            data["config"] = apply_defaults_pps_policy(dict(data["config"]))
            return data
        if isinstance(data, dict):
            return {"config": apply_defaults_pps_policy(dict(data))}

    minimal = {
        "connector": "openai",
        "llm": "gpt-4o-mini",
        "prompt_processor": default_prompt_processor(),
        "rag_processor": "No RAG",
    }
    return {"config": minimal}
