"""Search trigger layer — centralized search for graph density.

All graph expansion paths call search_triggered() to fill L4-L5 evidence
with real web sources before falling back to mother-only generation.
Search fills evidence, not concepts (the hierarchy handles those).

Rate-limited and deduped per query hash.
"""

import asyncio
import hashlib
import json
import logging
import time

import db
from db import _write_lock
from config import settings
from embedder import embed, embed_batch_parallel
from chroma_store import upsert_node, query_level
from graph import recompute_parent_bbox
from growth import _is_duplicate
from seed import _mother_generate, _parse_json_array

logger = logging.getLogger(__name__)


# ============================================================
# Rate Limiter — in-memory token bucket per trigger type
# ============================================================

# (max_calls, window_seconds)
RATE_LIMITS: dict[str, tuple[int, int]] = {
    "gap":              (3, 60),    # 3 searches per 60s
    "post_seed":        (5, 120),   # 5 per 2 min
    "sparse_evidence":  (2, 300),   # 2 per 5 min
    "curiosity":        (3, 600),   # 3 per 10 min
    "post_distill":     (5, 120),   # 5 per 2 min
    "manual":           (10, 60),   # 10/min for manual MCP tool
}

_rate_log: dict[str, list[float]] = {}  # trigger_type -> [timestamps]


def _check_rate_limit(trigger_type: str) -> bool:
    """Returns True if the call is allowed, False if rate-limited."""
    if trigger_type not in RATE_LIMITS:
        return True

    max_calls, window = RATE_LIMITS[trigger_type]
    now = time.time()
    timestamps = _rate_log.setdefault(trigger_type, [])

    # Prune old entries
    _rate_log[trigger_type] = [t for t in timestamps if now - t < window]
    timestamps = _rate_log[trigger_type]

    if len(timestamps) >= max_calls:
        return False

    timestamps.append(now)
    return True


# ============================================================
# Dedup — never search the same query twice in a session
# ============================================================

_searched_queries: set[str] = set()


def _query_hash(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


# ============================================================
# Mother Prompt — structures search results into L4/L5 only
# ============================================================

STRUCTURE_SEARCH_PROMPT = """You are structuring web search results into evidence-layer
knowledge graph nodes. Extract ONLY L4 (facts) and L5 (evidence) nodes.

Parent context:
  "{parent_content}" (L{parent_level})

Search results:
{search_results}

Rules:
- Output ONLY L4 (verifiable facts with dates/numbers) and L5 (direct quotes, cited statistics)
- Each node must be grounded in the search results above — not invented
- Include source URL in the content when possible
- Be precise and factual — no vague statements
- Generate 3-8 nodes

Return JSON array:
[
  {{"content": "precise fact or evidence with source reference", "resolution_level": 4|5}}
]"""


# ============================================================
# Core Function
# ============================================================

async def search_triggered(
    reason: str,
    query: str,
    node_context: dict,
    parent_node_id: int = None,
    max_urls: int = None,
) -> dict:
    """Centralized search trigger for graph expansion.

    All expansion paths call this to fill L4-L5 evidence with real sources.
    Rate-limited and deduped per query.

    Args:
        reason: Trigger type (gap, post_seed, sparse_evidence, curiosity, post_distill, manual)
        query: Search query string
        node_context: Dict with at least 'content' and optionally 'resolution_level', 'id'
        parent_node_id: Optional parent to attach new nodes under
        max_urls: Override default max URLs

    Returns:
        {triggered: bool, reason, nodes_created: int, nodes: [...],
         skipped: bool, skip_reason: str}
    """
    max_urls = max_urls or settings.search_max_urls

    # Check if search trigger is enabled
    if not settings.search_trigger_enabled:
        return {"triggered": False, "reason": reason,
                "nodes_created": 0, "nodes": [],
                "skipped": True, "skip_reason": "disabled"}

    # Rate limit check
    if not _check_rate_limit(reason):
        logger.info("Search trigger rate-limited: %s", reason)
        return {"triggered": False, "reason": reason,
                "nodes_created": 0, "nodes": [],
                "skipped": True, "skip_reason": "rate_limited"}

    # Dedup check
    qhash = _query_hash(query)
    if qhash in _searched_queries:
        logger.info("Search trigger deduped: %s", query[:60])
        return {"triggered": False, "reason": reason,
                "nodes_created": 0, "nodes": [],
                "skipped": True, "skip_reason": "already_searched"}
    _searched_queries.add(qhash)

    # Execute search
    try:
        from ingest import search_searxng, extract_text

        search_results = await search_searxng(query, max_results=max_urls)
        if not search_results:
            return {"triggered": False, "reason": reason,
                    "nodes_created": 0, "nodes": [],
                    "skipped": True, "skip_reason": "no_search_results"}

        # Extract text from URLs in parallel
        async def _fetch(r):
            text = await extract_text(r["url"])
            if text:
                return {"url": r["url"], "title": r.get("title", ""), "text": text}
            return None

        fetch_tasks = [_fetch(r) for r in search_results[:max_urls]]
        results = await asyncio.gather(*fetch_tasks)
        url_texts = [r for r in results if r]

        if not url_texts:
            return {"triggered": False, "reason": reason,
                    "nodes_created": 0, "nodes": [],
                    "skipped": True, "skip_reason": "no_text_extracted"}

        # Format search results for mother prompt
        search_summary = "\n".join(
            f"--- [{i+1}] {r['title']}\n    URL: {r['url']}\n    {r['text'][:2000]}"
            for i, r in enumerate(url_texts)
        )

        # Determine parent context for prompt
        parent_content = node_context.get("content", query)
        parent_level = node_context.get("resolution_level", 3)

        # Mother structures results into L4/L5 nodes
        prompt = STRUCTURE_SEARCH_PROMPT.format(
            parent_content=parent_content[:200],
            parent_level=parent_level,
            search_results=search_summary[:8000],
        )

        nodes_data = _parse_json_array(await _mother_generate(prompt))
        if not nodes_data:
            return {"triggered": False, "reason": reason,
                    "nodes_created": 0, "nodes": [],
                    "skipped": True, "skip_reason": "mother_failed"}

        # Embed and insert nodes
        texts = [n["content"] for n in nodes_data if "content" in n]
        embeddings = await embed_batch_parallel(texts)

        conn = db.get_db()
        created_nodes = []

        async with _write_lock:
            for i, node_data in enumerate(nodes_data):
                if "content" not in node_data:
                    continue

                content = node_data["content"]
                raw_level = str(node_data.get("resolution_level", 4)).lstrip("Ll")
                level = max(4, min(5, int(raw_level)))  # Force L4-L5 only
                emb = embeddings[i] if i < len(embeddings) else None
                if not emb:
                    continue

                # Dedup check
                if await _is_duplicate(content, emb, None, level):
                    continue

                # Find parent — use provided or vector search
                pid = parent_node_id
                if not pid:
                    for plevel in range(level - 1, -1, -1):
                        hits = query_level(emb, plevel, n_results=1)
                        if hits:
                            pid = int(hits[0]["node_id"])
                            break

                # Detect source URL from content
                source_url = None
                for r in url_texts:
                    if r["url"] in content:
                        source_url = r["url"]
                        break

                node_id = db.insert_node(
                    conn, content, resolution_level=level, parent_id=pid,
                    confidence=0.8, source_url=source_url,
                )
                upsert_node(node_id, content, emb, level, pid, 0.8, source_url)
                created_nodes.append({
                    "id": node_id, "level": level,
                    "content": content[:100],
                    "parent_id": pid,
                    "source_url": source_url,
                })

                if pid:
                    recompute_parent_bbox(conn, node_id)

        logger.info("Search trigger [%s]: %d nodes from '%s'",
                     reason, len(created_nodes), query[:60])

        return {
            "triggered": True,
            "reason": reason,
            "query": query,
            "nodes_created": len(created_nodes),
            "nodes": created_nodes,
        }

    except Exception as e:
        logger.error("Search trigger failed [%s]: %s", reason, e)
        return {"triggered": False, "reason": reason,
                "nodes_created": 0, "nodes": [],
                "skipped": True, "skip_reason": f"error: {e}"}
