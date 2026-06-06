"""Ingest pipeline — web search -> extract -> classify -> insert."""

import re

import httpx

import db
from config import settings
from embedder import embed
from chroma_store import upsert_node, query_level
from graph import propagate_confidence, recompute_parent_bbox, compute_bbox


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


async def classify_resolution_llm(text: str) -> int:
    """Classify text using 2B LLM + graph structure fallback.

    No longer escalates to 9B mother model. Uses 2B classification first,
    then graph bbox containment for uncertain cases.
    """
    result = await _classify_with_model(
        settings.llm_url, settings.llm_model, text, timeout=15.0
    )

    if result is None:
        return classify_resolution_heuristic(text)

    level, confidence = result

    # Use graph structure instead of 9B escalation for uncertain cases
    if confidence < 0.7:
        try:
            from context import find_best_level_by_structure
            from embedder import embed
            embedding = await embed(text)
            return find_best_level_by_structure(embedding)
        except Exception:
            pass  # Fall through to 2B's answer

    return level


async def _classify_with_model(url: str, model: str, text: str,
                                timeout: float = 15.0) -> tuple[int, float] | None:
    """Classify text using a specific model. Returns (level, confidence) or None."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{url}/api/generate",
                json={
                    "model": model,
                    "prompt": (
                        f"Classify this text into ONE resolution level (0-5):\n"
                        f"0=domain, 1=topic, 2=concept, 3=entity, 4=fact, 5=evidence\n"
                        f"Text: {text[:500]}\n\n"
                        f"Reply with ONLY the number, nothing else."
                    ),
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 20},
                    "think": False,
                },
            )
            resp.raise_for_status()
            result = resp.json().get("response", "").strip()

            # qwen3.5 models use </think > tags — strip them
            result = re.sub(r"</?think\s*>", "", result).strip()

            match = re.search(r"[0-5]", result)
            if match:
                level = int(match.group())
                # Confidence: if the model output was clean (just a number),
                # high confidence. If surrounded by noise, lower.
                clean = re.sub(r"[^0-5]", "", result).strip()
                confidence = 1.0 if len(clean) == 1 else 0.5
                return (level, confidence)
    except Exception as e:
        print(f"Classification with {model} failed: {e}")

    return None


async def search_searxng(query: str, max_results: int = 5) -> list[dict]:
    """Search via SearXNG."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                settings.searxng_url + "/search",
                params={"q": query, "format": "json", "count": max_results},
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            return [
                {"url": r["url"], "title": r.get("title", ""),
                 "content": r.get("content", "")}
                for r in results
            ]
    except Exception as e:
        print(f"SearXNG search failed: {e}")
        return []


async def extract_text(url: str) -> str | None:
    """Extract text content from a URL (simple HTML stripping)."""
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()

            html = resp.text
            # Remove scripts and styles
            html = re.sub(r"<script[^>]*>.*?</script>", "", html,
                          flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r"<style[^>]*>.*?</style>", "", html,
                          flags=re.DOTALL | re.IGNORECASE)
            # Remove tags
            html = re.sub(r"<[^>]+>", " ", html)
            # Clean whitespace
            html = re.sub(r"\s+", " ", html).strip()

            return html[:5000] if len(html.strip()) > 50 else None
    except Exception as e:
        print(f"Text extraction failed for {url}: {e}")
        return None


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
        resolution_level = await classify_resolution_llm(text)

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


async def web_ingest(query: str, max_urls: int = 3, use_llm: bool = True) -> dict:
    """Full ingest pipeline: search -> extract -> classify -> insert."""
    search_results = await search_searxng(query, max_results=max_urls)
    if not search_results:
        return {"error": "No search results found", "query": query}

    conn = db.get_db()
    nodes_created = []
    edges_created = []

    for result in search_results[:max_urls]:
        url = result["url"]

        # Skip duplicates
        existing = conn.execute(
            "SELECT id FROM nodes WHERE source_url = ?", (url,)
        ).fetchone()
        if existing:
            continue

        text = await extract_text(url)
        if not text:
            continue

        resolution = (await classify_resolution_llm(text) if use_llm
                      else classify_resolution_heuristic(text))

        embedding = await embed(text)
        parent_id = await find_parent_for_text(conn, text, embedding, resolution)

        node_id = db.insert_node(
            conn, text, resolution_level=resolution, parent_id=parent_id,
            confidence=0.6, source_url=url,
            metadata={"title": result["title"], "search_query": query},
        )

        upsert_node(node_id, text, embedding, resolution, parent_id, 0.6, url)
        nodes_created.append(node_id)

        if parent_id:
            recompute_parent_bbox(conn, node_id)

        # Create edges between sibling nodes from same query
        for existing_node_id in nodes_created[:-1]:
            edge_id = db.insert_edge(
                conn, existing_node_id, node_id, "related", confidence=0.3,
                context=f"Co-ingested from query: {query}",
            )
            edges_created.append(edge_id)

    return {
        "query": query,
        "nodes_created": nodes_created,
        "edges_created": edges_created,
        "total_search_results": len(search_results),
    }


async def ingest_url(conn, url: str, use_llm: bool = True) -> dict:
    """Fetch a URL and ingest its content, splitting long text into chunks."""
    text = await extract_text(url)
    if not text:
        return {"error": "Could not extract meaningful text", "url": url}

    # Split into chunks if very long
    chunks = []
    if len(text) > 1500:
        words = text.split()
        chunk_size = 500
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i + chunk_size]))
    else:
        chunks = [text]

    nodes_created = []
    for chunk in chunks:
        resolution = (await classify_resolution_llm(chunk) if use_llm
                      else classify_resolution_heuristic(chunk))

        embedding = await embed(chunk)
        parent_id = await find_parent_for_text(conn, chunk, embedding, resolution)

        node_id = db.insert_node(
            conn, chunk, resolution_level=resolution, parent_id=parent_id,
            confidence=0.6, source_url=url,
        )
        upsert_node(node_id, chunk, embedding, resolution, parent_id, 0.6, url)
        nodes_created.append(node_id)

    return {
        "url": url,
        "chunks_processed": len(chunks),
        "nodes_created": nodes_created,
    }
