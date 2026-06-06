import os
import time
from typing import Dict, Any, List, Optional
from lamb.lamb_classes import Assistant
from lamb.completions import document_cache
import json
import logging
import httpx

logger = logging.getLogger('lamb.completions.rag.single_file_rag')
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
    file_path = metadata.get("file_path")

    if library_id and item_id:
        org_id = getattr(assistant, "organization_id", None) or "global"
        return _fetch_with_cache(org_id, library_id, item_id)
    elif file_path:
        return _read_from_static_file(file_path)
    else:
        logger.warning("No library_id+item_id or file_path in metadata")
        return empty_result


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
    empty_result = {"context": "", "sources": []}

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


def _read_from_static_file(file_path: str) -> Dict[str, Any]:
    base_path = os.path.join('static', 'public')
    full_path = os.path.join(base_path, file_path)

    if '..' in file_path or not os.path.abspath(full_path).startswith(os.path.abspath(base_path)):
        error_msg = f"Error: Invalid file path: {file_path}"
        logger.warning(f"Path traversal attempt detected: {file_path}")
        return {"context": error_msg, "sources": [], "_timing": {"fetch_ms": 0, "cache": "skip"}}

    if not os.path.exists(full_path):
        error_msg = f"Error: File not found: {file_path}"
        logger.warning(f"File not found: {full_path}")
        return {"context": error_msg, "sources": [], "_timing": {"fetch_ms": 0, "cache": "skip"}}

    try:
        start = time.time()
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        elapsed_ms = (time.time() - start) * 1000
        return {
            "context": content,
            "sources": [{
                "title": os.path.basename(file_path),
                "url": f"/static/public/{file_path}",
                "similarity": 1.0,
            }],
            "_timing": {"fetch_ms": round(elapsed_ms, 2), "cache": "skip"},
        }
    except Exception as e:
        error_msg = f"Error reading file {file_path}: {str(e)}"
        logger.warning(f"Error reading file {full_path}: {e}")
        return {"context": error_msg, "sources": [], "_timing": {"fetch_ms": 0, "cache": "skip"}}
