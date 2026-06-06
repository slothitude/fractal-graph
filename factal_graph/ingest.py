"""Ingest pipeline — web search -> extract -> classify -> insert."""

import importlib
import importlib.util
import re

import db
from embedder import embed
from chroma_store import upsert_node, query_level
from graph import propagate_confidence, recompute_parent_bbox

# Import search/extract from searchMCP without polluting sys.path.
# We need to temporarily make searchmcp's config.py resolvable because
# core.py imports from it, but the fractal graph has its own config.py.
_SEARCHMCP_DIR = "C:/Users/aaron/searchmcp"

def _load_searchmcp_core():
    """Load searchmcp/core.py and its dependencies in isolation."""
    import sys
    # Temporarily add searchmcp to front of path
    sys.path.insert(0, _SEARCHMCP_DIR)
    # Remove cached local config if present so searchmcp's config wins
    local_config = sys.modules.pop("config", None)

    try:
        spec = importlib.util.spec_from_file_location(
            "searchmcp_core", f"{_SEARCHMCP_DIR}/core.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["searchmcp_core"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        # Restore local config cache and remove searchmcp from path
        sys.path.pop(0)
        if local_config is not None:
            sys.modules["config"] = local_config

_searchmcp_core = _load_searchmcp_core()
search_searxng = _searchmcp_core.search
extract_text = _searchmcp_core.extract_text


def classify_resolution_heuristic(text: str) -> int:
    """Classify text into resolution level using heuristics.

    L0: Very broad domain statements (< 50 words, generic)
    L1: Topic-level statements (50-150 words, named topic)
    L2: Concept-level (specific concepts, definitions)
    L3: Entity-focused (named entities, specific references)
    L4: Fact-level (dates, numbers, specific claims)
    L5: Evidence-level (quotes, source references, URLs)
    """
    word_count = len(text.split())
    has_dates = bool(re.search(r"\b(19|20)\d{2}\b", text))
    has_numbers = bool(re.search(r"\b\d+\.?\d*\b", text))
    has_quotes = '"' in text or "'" in text
    has_urls = bool(re.search(r"https?://", text))
    has_named_entities = sum(1 for w in text.split() if w[0].isupper() and len(w) > 3)

    score = 0
    if has_urls: score += 3
    if has_quotes: score += 2
    if has_dates: score += 2
    if has_numbers: score += 1
    if has_named_entities > 3: score += 2
    elif has_named_entities > 1: score += 1
    if word_count > 200: score += 1

    if score >= 5:
        return 5  # evidence
    elif score >= 4:
        return 4  # fact
    elif score >= 3:
        return 3  # entity
    elif score >= 2:
        return 2  # concept
    elif word_count < 50:
        return 0  # domain
    else:
        return 1  # topic


async def find_parent_for_text(conn, text: str, embedding: list[float],
                               resolution_level: int) -> int | None:
    """Find the best parent node for a new piece of text."""
    if resolution_level == 0:
        return None

    parent_level = resolution_level - 1
    hits = query_level(embedding, parent_level, n_results=3)
    if hits:
        return int(hits[0]["node_id"])
    return None


async def ingest_text(conn, text: str, resolution_level: int = None,
                      source_url: str = None, confidence: float = 0.5) -> dict:
    """Ingest a single text as a node."""
    if resolution_level is None:
        resolution_level = classify_resolution_heuristic(text)

    embedding = await embed(text)
    parent_id = await find_parent_for_text(conn, text, embedding, resolution_level)

    node_id = db.insert_node(
        conn, text, resolution_level=resolution_level, parent_id=parent_id,
        confidence=confidence, source_url=source_url,
    )

    upsert_node(node_id, text, embedding, resolution_level, parent_id,
                confidence, source_url)

    if parent_id:
        propagate_confidence(conn, node_id)
        recompute_parent_bbox(conn, node_id)

    return {
        "node_id": node_id,
        "resolution_level": resolution_level,
        "parent_id": parent_id,
        "embedding_dims": len(embedding),
    }
