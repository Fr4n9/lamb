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


def apply_legacy_pps_override(mapping: dict[str, Any] | None) -> dict[str, Any] | None:
    """When LAMB_LEGACY_PPS_DEFAULT is set, force prompt_processor to simple_augment."""
    if not config.LAMB_LEGACY_PPS_DEFAULT or not mapping:
        return mapping
    result = dict(mapping)
    result["prompt_processor"] = "simple_augment"
    return result


def load_defaults_json_document() -> dict[str, Any]:
    """Load defaults.json from disk, applying legacy PPS override when configured."""
    for path in DEFAULTS_JSON_PATHS:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data.get("config"), dict):
            data = dict(data)
            data["config"] = apply_legacy_pps_override(dict(data["config"]))
            return data
        if isinstance(data, dict):
            return {"config": apply_legacy_pps_override(dict(data))}

    minimal = {
        "connector": "openai",
        "llm": "gpt-4o-mini",
        "prompt_processor": default_prompt_processor(),
        "rag_processor": "No RAG",
    }
    return {"config": minimal}
