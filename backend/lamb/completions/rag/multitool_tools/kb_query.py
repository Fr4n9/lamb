"""KB query tool — async wrapper around KB Server collection query.

Reuses the same HTTP pattern as simple_rag.py but as a standalone async tool.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import requests

from lamb.completions.org_config_resolver import OrganizationConfigResolver
from lamb.logging_config import get_logger

logger = get_logger(__name__, component="MULTITOOL")

ALLOWED_ARGS = frozenset({"query", "collections", "top_k", "target_collections"})


async def fetch_collection_descriptions(
    collection_ids: List[str],
    kb_url: str,
    kb_token: str,
) -> Dict[str, str]:
    """Fetch semantic descriptions for a list of collection IDs from the KB server.

    Returns a dict mapping collection_id (str) -> description (str).
    Collections that fail to load get an empty description.
    """
    headers = {
        "Authorization": f"Bearer {kb_token}",
        "Content-Type": "application/json",
    }
    descriptions: Dict[str, str] = {}
    for cid in collection_ids:
        try:
            resp = requests.get(f"{kb_url.rstrip('/')}/collections/{cid}", headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                descriptions[str(cid)] = data.get("description") or ""
            else:
                logger.warning("KB description fetch for %s returned %s", cid, resp.status_code)
                descriptions[str(cid)] = ""
        except Exception as e:
            logger.error("KB description fetch error for %s: %s", cid, e)
            descriptions[str(cid)] = ""
    return descriptions


async def execute(
    *,
    query: str,
    collections: List[str],
    top_k: int = 3,
    target_collections: Optional[List[str]] = None,
    assistant_owner: str,
    **_extra: Any,
) -> Dict[str, Any]:
    """Query one or more KB collections and return merged context + sources.

    If target_collections is provided (from the orchestrator), only those
    collection IDs are queried. Otherwise all collections are queried
    (backward-compatible with Prototype 1).
    """
    if not collections:
        return {"ok": False, "error": "no collections configured", "tool": "kb_query"}

    effective_collections = list(collections)
    if target_collections:
        allowed_set = {str(c) for c in collections}
        effective_collections = [c for c in target_collections if str(c) in allowed_set]
        if not effective_collections:
            effective_collections = list(collections)
            logger.warning(
                "target_collections %s had no overlap with configured collections %s, falling back to all",
                target_collections,
                collections,
            )

    config_resolver = OrganizationConfigResolver(assistant_owner)
    kb_config = config_resolver.get_knowledge_base_config() or {}

    kb_url = kb_config.get("server_url")
    kb_token = kb_config.get("api_token")

    if not kb_url:
        import os
        kb_url = os.getenv("LAMB_KB_SERVER")
        kb_token = os.getenv("LAMB_KB_SERVER_TOKEN")

    if not kb_url:
        return {"ok": False, "error": "KB server not configured", "tool": "kb_query"}

    headers = {
        "Authorization": f"Bearer {kb_token}",
        "Content-Type": "application/json",
    }
    payload = {"query_text": query, "top_k": top_k, "threshold": 0.0, "plugin_params": {}}

    contexts: List[str] = []
    sources: List[Dict[str, Any]] = []

    for cid in effective_collections:
        try:
            resp = requests.post(f"{kb_url}/collections/{cid}/query", headers=headers, json=payload)
            if resp.status_code != 200:
                logger.warning("KB query %s returned %s", cid, resp.status_code)
                continue
            data = resp.json()
            docs = data.get("results", data.get("documents", []))
            for doc in docs:
                if "data" in doc:
                    contexts.append(doc["data"])
                meta = doc.get("metadata", {})
                url = meta.get("source_url") or meta.get("original_file_url") or meta.get("file_url")
                if url and not url.startswith("http"):
                    url = f"{kb_url}{url}"
                if url:
                    sources.append({
                        "title": meta.get("filename", meta.get("original_filename", "Unknown")),
                        "url": url,
                        "similarity": doc.get("similarity", 0),
                    })
        except Exception as e:
            logger.error("KB query error for %s: %s", cid, e)

    return {
        "ok": True,
        "tool": "kb_query",
        "context": "\n\n".join(contexts),
        "sources": sources,
    }
