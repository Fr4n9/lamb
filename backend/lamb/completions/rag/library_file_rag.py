import os
import time
from typing import Dict, Any, List, Optional
from lamb.lamb_classes import Assistant
from lamb.completions import document_cache
import json
import logging
import httpx

logger = logging.getLogger('lamb.completions.rag.library_file_rag')
logger.setLevel(logging.WARNING)


def rag_processor(
    messages: List[Dict[str, Any]],
    assistant: Assistant = None,
    request: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    empty_result = {"context": "", "sources": [], "_timing": {"fetch_ms": 0, "cache": "skip"}}

    if assistant is None:
        logger.warning("No assistant provided")
        return empty_result

    try:
        metadata = json.loads(assistant.metadata) if assistant.metadata else {}
    except (json.JSONDecodeError, TypeError):
        logger.warning("Invalid metadata JSON")
        return empty_result

    library_id = metadata.get("library_id")
    item_id = metadata.get("item_id")

    if not library_id or not item_id:
        logger.warning("library_id and item_id are required for library_file_rag")
        return empty_result

    org_id = getattr(assistant, "organization_id", None) or "global"
    return _fetch_with_cache(org_id, library_id, item_id)


def _fetch_with_cache(org_id: str, library_id: str, item_id: str) -> Dict[str, Any]:
    """Fetch document from cache or Library Manager, with timing."""
    cache_key = f"{org_id}:{library_id}:{item_id}"
    start = time.time()

    cached = document_cache.get(cache_key)
    if cached is not None:
        elapsed_ms = (time.time() - start) * 1000
        cached["_timing"] = {"fetch_ms": round(elapsed_ms, 2), "cache": "hit"}
        return cached

    result = _fetch_from_library_manager(library_id, item_id)
    elapsed_ms = (time.time() - start) * 1000

    if result.get("context"):
        document_cache.set(cache_key, {"context": result["context"], "sources": result["sources"]})

    result["_timing"] = {"fetch_ms": round(elapsed_ms, 2), "cache": "miss"}
    return result


def _fetch_from_library_manager(library_id: str, item_id: str) -> Dict[str, Any]:
    empty_result = {"context": "", "sources": [], "_timing": {"fetch_ms": 0, "cache": "skip"}}

    lm_url = os.environ.get("LAMB_LIBRARY_SERVER", "").rstrip("/")
    lm_token = os.environ.get("LAMB_LIBRARY_TOKEN", "")

    if not lm_url or not lm_token:
        logger.warning("LAMB_LIBRARY_SERVER or LAMB_LIBRARY_TOKEN not configured")
        return empty_result

    url = f"{lm_url}/libraries/{library_id}/items/{item_id}/content"
    headers = {"Authorization": f"Bearer {lm_token}"}

    try:
        response = httpx.get(url, params={"format": "markdown"}, headers=headers, timeout=30.0)
        if response.status_code == 200:
            content = response.text
            return {
                "context": content,
                "sources": [{
                    "title": "Library Document",
                    "url": f"/docs/{library_id}/{item_id}",
                    "similarity": 1.0,
                }],
            }
        else:
            logger.warning(
                f"Library Manager returned {response.status_code} for "
                f"library={library_id} item={item_id}"
            )
            return empty_result
    except httpx.HTTPError as e:
        logger.warning(f"Failed to fetch from Library Manager: {e}")
        return empty_result
