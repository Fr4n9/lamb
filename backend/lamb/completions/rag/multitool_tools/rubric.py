"""Rubric tool — async wrapper around rubric fetch + format.

Reuses patterns from rubric_rag.py as a standalone async tool.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from lamb.logging_config import get_logger

logger = get_logger(__name__, component="MULTITOOL")

ALLOWED_ARGS = frozenset({"rubric_id", "rubric_format"})


async def execute(
    *,
    rubric_id: str,
    rubric_format: str = "markdown",
    assistant_owner: str,
    **_extra: Any,
) -> Dict[str, Any]:
    """Fetch a rubric by ID and return formatted evaluation context."""
    if not rubric_id:
        return {"ok": False, "error": "rubric_id is required", "tool": "rubric"}

    if rubric_format not in ("markdown", "json"):
        rubric_format = "markdown"

    try:
        from lamb.evaluaitor.rubric_database import RubricDatabaseManager
        from lamb.evaluaitor.rubric_service import (
            generate_rubric_evaluation_markdown,
            generate_rubric_evaluation_json,
        )

        db = RubricDatabaseManager()
        rubric = db.get_rubric_by_id(rubric_id, assistant_owner)
        if not rubric:
            return {"ok": False, "error": f"rubric {rubric_id} not found", "tool": "rubric"}

        rubric_data = rubric.get("rubric_data")
        if isinstance(rubric_data, str):
            rubric_data = json.loads(rubric_data)

        if rubric_format == "json":
            context = generate_rubric_evaluation_json(rubric_data)
        else:
            context = generate_rubric_evaluation_markdown(rubric_data)

        sources = [{
            "type": "rubric",
            "rubric_id": rubric_id,
            "title": rubric.get("title", "Unknown Rubric"),
        }]

        return {"ok": True, "tool": "rubric", "context": context, "sources": sources}

    except Exception as e:
        logger.error("Rubric tool error: %s", e, exc_info=True)
        return {"ok": False, "error": str(e), "tool": "rubric"}
