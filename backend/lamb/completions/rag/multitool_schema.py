from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class RawToolCall(BaseModel):
    """Lenient: orchestrator may hallucinate names; filter later."""
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class RawOrchestratorPlan(BaseModel):
    tools: List[RawToolCall] = Field(default_factory=list)
    rationale: Optional[str] = None


def parse_metadata_multitool(metadata_json: str) -> Optional[Dict[str, Any]]:
    """Extract the ``multitool`` block from assistant metadata JSON.

    Returns None when the key is absent or not a dict.
    """
    if not metadata_json or not metadata_json.strip():
        return None
    data = json.loads(metadata_json)
    mt = data.get("multitool")
    if not isinstance(mt, dict):
        return None
    return mt


def parse_orchestrator_response(
    raw_json: str,
    *,
    allowed_names: List[str],
) -> Tuple[RawOrchestratorPlan, List[str]]:
    """Parse the orchestrator LLM output and filter unknown tools.

    Returns ``(filtered_plan, rejected_names)``.
    """
    data = json.loads(raw_json)
    full_plan = RawOrchestratorPlan.model_validate(data)

    allowed_set = set(allowed_names)
    kept: List[RawToolCall] = []
    rejected: List[str] = []
    for tc in full_plan.tools:
        if tc.name in allowed_set:
            kept.append(tc)
        else:
            rejected.append(tc.name)

    filtered = RawOrchestratorPlan(tools=kept, rationale=full_plan.rationale)
    return filtered, rejected
