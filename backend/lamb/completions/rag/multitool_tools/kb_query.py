"""KB query tool — async wrapper around KB Server collection query.

Reuses the same HTTP pattern as simple_rag.py but as a standalone async tool.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import requests

from lamb.completions.org_config_resolver import OrganizationConfigResolver
from lamb.logging_config import get_logger

logger = get_logger(__name__, component="MULTITOOL")

ALLOWED_ARGS = frozenset({"query", "collections", "top_k"})


async def execute(
    *,
    query: str,
    collections: List[str],
    top_k: int = 3,
    assistant_owner: str,
    **_extra: Any,
) -> Dict[str, Any]:
    """Query one or more KB collections and return merged context + sources."""
    if not collections:
        return {"ok": False, "error": "no collections configured", "tool": "kb_query"}

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

    for cid in collections:
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
