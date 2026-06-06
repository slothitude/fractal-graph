"""Autonomous graph expansion — mother decomposes itself.

Three expansion mechanisms:
1. Gap-driven auto-expansion — ask() detects low confidence/gaps, triggers mother to fill
2. Answer-enriches-graph — while answering, mother decomposes related knowledge into graph
3. Continuous curiosity scan — background process finds sparse areas and expands them
"""

import asyncio
import json
import re

import db
from config import settings
from embedder import embed, embed_batch_parallel
from chroma_store import upsert_node, query_level, get_collection, query_all_levels
from graph import recompute_parent_bbox
from seed import _mother_generate, _parse_json_array, _parse_json_object, LEVEL_MEANINGS


# --- Per-topic asyncio locks (prevents concurrent enrichment race) ---

_expansion_locks: dict[str, asyncio.Lock] = {}


def _get_lock(topic_key: str) -> asyncio.Lock:
    return _expansion_locks.setdefault(topic_key, asyncio.Lock())


# --- Level-aware dedup thresholds ---

DEDUP_THRESHOLD = {
    0: 0.95,  # Domain — almost identical text only
    1: 0.92,  # Topic
    2: 0.88,  # Concept
    3: 0.85,  # Entity
    4: 0.82,  # Fact
    5: 0.80,  # Evidence
}


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


async def _is_duplicate(content: str, embedding: list[float],
                        parent_id: int | None, level: int) -> bool:
    """Check if a node is too similar to existing siblings at the same level.

    Uses level-aware dedup thresholds — L0-L2 are lenient, L4-L5 are strict.
    """
    threshold = DEDUP_THRESHOLD.get(level, 0.85)
    conn = db.get_db()

    if parent_id:
        # Check siblings under same parent
        children = db.get_children(conn, parent_id)
        for child in children:
            if child["resolution_level"] != level:
                continue
            try:
                col = get_collection(level)
                result = col.get(ids=[str(child["id"])])
                if result and result["embeddings"] and result["embeddings"][0]:
                    sim = _cosine_sim(embedding, result["embeddings"][0])
                    if sim >= threshold:
                        return True
            except Exception:
                continue

    return False


# --- Gap fill prompt ---

GAP_FILL_PROMPT = """Fill a knowledge gap. A question returned low confidence due to missing knowledge.

Question: {question}
Missing knowledge: {gap}

Nearby graph nodes (candidate parents):
{nearby_nodes}

Level meanings:
{level_meanings}

Generate {n} nodes to fill this gap. Attach each to the best parent above.
Return JSON array:
[{{"content": "...", "level": N, "parent_id": X, "edge_type": "refines|supports|contradicts"}}]

Rules:
- Each node must have a numeric level (0-5)
- parent_id must match one of the candidate parents listed above
- edge_type describes relationship to parent
- Content should be factual and concise (under 100 words)"""


async def fill_gaps(gaps: list[str], question: str,
                   max_nodes: int = 5) -> dict:
    """Fill knowledge gaps identified by the 2B answer pipeline.

    For each gap, generates 1-3 nodes via the mother model at appropriate levels.
    Deduplicates against existing graph content before inserting.

    Args:
        gaps: List of gap descriptions from 2B synthesis
        question: Original question (for context)
        max_nodes: Maximum total nodes to create

    Returns:
        {nodes_created: int, nodes: [...], gaps_filled: int}
    """
    if not gaps:
        return {"nodes_created": 0, "nodes": [], "gaps_filled": 0}

    conn = db.get_db()
    created_nodes = []
    total_created = 0
    gaps_addressed = 0

    for gap in gaps[:3]:  # Max 3 gaps to process
        if total_created >= max_nodes:
            break
        gap_start_count = total_created

        # Find nearby nodes for parent assignment context
        gap_emb = await embed(gap)
        nearby = []
        all_hits = query_all_levels(gap_emb, n_results=3)
        for level_hits in all_hits.values():
            for hit in level_hits[:3]:
                nid = int(hit["node_id"])
                node = db.get_node(conn, nid)
                if node:
                    nearby.append(node)

        if not nearby:
            continue

        # Format nearby nodes for prompt
        nearby_desc = "\n".join(
            f"  [ID:{n['id']} L{n['resolution_level']}] {n['content'][:120]}"
            for n in nearby[:5]
        )

        level_meanings = "\n".join(
            f"  L{k}: {v}" for k, v in LEVEL_MEANINGS.items()
        )

        n_to_gen = min(max_nodes - total_created, 3)

        prompt = GAP_FILL_PROMPT.format(
            question=question,
            gap=gap,
            nearby_nodes=nearby_desc,
            level_meanings=level_meanings,
            n=n_to_gen,
        )

        nodes_data = _parse_json_array(
            await _mother_generate(prompt, num_predict=512)
        )
        if not nodes_data:
            continue

        # Embed and dedup
        texts = [n["content"] for n in nodes_data if "content" in n]
        embeddings = await embed_batch_parallel(texts)

        for i, node_data in enumerate(nodes_data):
            if total_created >= max_nodes:
                break
            if "content" not in node_data:
                continue

            content = node_data["content"]
            raw_level = str(node_data.get("level", 3)).lstrip("Ll")
            level = max(0, min(5, int(raw_level)))
            parent_id = node_data.get("parent_id")
            edge_type = node_data.get("edge_type", "refines")

            emb = embeddings[i] if i < len(embeddings) else None
            if not emb:
                continue

            # Validate parent exists
            if parent_id and not db.get_node(conn, parent_id):
                # Fall back to nearest parent from vector search
                hits = query_level(emb, level - 1 if level > 0 else 0, n_results=1)
                if hits:
                    parent_id = int(hits[0]["node_id"])
                else:
                    parent_id = None

            # Dedup check
            if await _is_duplicate(content, emb, parent_id, level):
                continue

            # Insert node
            node_id = db.insert_node(
                conn, content, resolution_level=level, parent_id=parent_id,
                confidence=0.6,
            )
            upsert_node(node_id, content, emb, level, parent_id, 0.6)

            # Create edge to parent if not already parent-child
            if parent_id and edge_type and edge_type.lower() != "refines":
                # refines is implicit via parent_id
                try:
                    db.insert_edge(
                        conn, parent_id, node_id,
                        edge_type=edge_type, confidence=0.5,
                        context=f"Gap fill from: {gap[:80]}",
                    )
                except (ValueError, Exception):
                    pass

            created_nodes.append({
                "id": node_id, "level": level,
                "content": content[:100], "parent_id": parent_id,
            })
            total_created += 1

            # Recompute bbox for parent
            if parent_id:
                recompute_parent_bbox(conn, node_id)

        if total_created > gap_start_count:
            gaps_addressed += 1

    search_fallback = False

    # Mother couldn't fill gaps — fall back to web search
    if total_created == 0:
        try:
            from seed import seed_from_search
            search_query = gaps[0] if gaps else question
            search_result = await asyncio.wait_for(
                seed_from_search(search_query, max_urls=3),
                timeout=45.0,
            )
            total_created += search_result.get("nodes_created", 0)
            search_fallback = total_created > 0
        except asyncio.TimeoutError:
            logger.warning("Web search fallback timed out")
        except Exception as e:
            logger.warning("Web search fallback failed: %s", e)

    return {
        "nodes_created": total_created,
        "nodes": created_nodes,
        "gaps_filled": gaps_addressed,
        "search_fallback": search_fallback,
    }


async def enrich_topic(question: str, question_type: dict,
                        max_nodes: int = 4) -> dict:
    """Decompose the question topic into 2-4 new nodes (fire-and-forget).

    Runs in parallel with 2B answer synthesis. Mother identifies key entities
    and concepts from the question and places them in the graph.

    Args:
        question: The user's question
        question_type: Classification from classify_question()
        max_nodes: Maximum nodes to create

    Returns:
        {nodes_created: int, nodes: [...]}
    """
    topic_key = question.lower()[:64]
    lock = _get_lock(topic_key)

    async with lock:
        conn = db.get_db()
        entities = question_type.get("entities", [])
        q_type = question_type.get("type", "factual")

        if not entities and q_type in ("yes_no", "simple"):
            return {"nodes_created": 0, "nodes": [], "reason": "no_entities"}

        # Build prompt for mother
        enrich_prompt = (
            "Decompose this question into key knowledge nodes for a fractal "
            "knowledge graph. Identify the core concepts, entities, or facts "
            "that would be needed to fully answer this question.\n\n"
            f"Question: {question}\n"
            f"Detected entities: {', '.join(entities) if entities else 'none'}\n"
            f"Question type: {q_type}\n\n"
            f"Level meanings:\n"
        )
        for lvl, meaning in LEVEL_MEANINGS.items():
            enrich_prompt += f"  L{lvl}: {meaning}\n"

        enrich_prompt += (
            f"\nGenerate {max_nodes} nodes. Focus on L2-L4 (concepts, entities, facts).\n"
            "Return JSON array:\n"
            '[{"content": "...", "resolution_level": N}]\n'
        )

        nodes_data = _parse_json_array(
            await _mother_generate(enrich_prompt, num_predict=512)
        )
        if not nodes_data:
            return {"nodes_created": 0, "nodes": [], "reason": "mother_failed"}

        created_nodes = []
        texts = [n["content"] for n in nodes_data if "content" in n]
        embeddings = await embed_batch_parallel(texts)

        for i, node_data in enumerate(nodes_data):
            if len(created_nodes) >= max_nodes:
                break
            if "content" not in node_data:
                continue

            content = node_data["content"]
            raw_level = str(node_data.get("resolution_level", 3)).lstrip("Ll")
            level = max(0, min(5, int(raw_level)))
            emb = embeddings[i] if i < len(embeddings) else None
            if not emb:
                continue

            # Dedup
            if await _is_duplicate(content, emb, None, level):
                continue

            # Find best parent via vector similarity
            parent_id = None
            for plevel in range(level - 1, -1, -1):
                hits = query_level(emb, plevel, n_results=1)
                if hits:
                    parent_id = int(hits[0]["node_id"])
                    break

            node_id = db.insert_node(
                conn, content, resolution_level=level, parent_id=parent_id,
                confidence=0.6,
            )
            upsert_node(node_id, content, emb, level, parent_id, 0.6)
            created_nodes.append({
                "id": node_id, "level": level,
                "content": content[:100], "parent_id": parent_id,
            })

            if parent_id:
                recompute_parent_bbox(conn, node_id)

        return {
            "nodes_created": len(created_nodes),
            "nodes": created_nodes,
        }


def detect_sparse_nodes(conn) -> list[dict]:
    """Find sparse areas in the graph that need expansion.

    Returns nodes with:
    - 0 children at non-L5 level (leaf nodes that should have children)
    - < 2 children (under-populated subtrees)
    - High confidence L3+ nodes with no L4-L5 evidence children

    Returns:
        List of {node_id, content, level, sparse_reason, gap_description}
    """
    sparse = []

    # Leaf nodes below L5 (should have children)
    leaf_rows = conn.execute("""
        SELECT n.id, n.content, n.resolution_level, n.confidence
        FROM nodes n
        LEFT JOIN nodes c ON c.parent_id = n.id
        WHERE n.resolution_level < 5 AND c.id IS NULL
        ORDER BY n.resolution_level ASC, n.confidence DESC
    """).fetchall()

    for row in leaf_rows[:10]:  # Cap at 10
        sparse.append({
            "node_id": row["id"],
            "content": row["content"][:100],
            "level": row["resolution_level"],
            "confidence": row["confidence"],
            "sparse_reason": "leaf_below_L5",
            "gap_description": f"L{row['resolution_level']} node with no children",
        })

    # Nodes with < 2 children at non-leaf levels
    few_children = conn.execute("""
        SELECT p.id, p.content, p.resolution_level, p.confidence,
               COUNT(c.id) as child_count
        FROM nodes p
        LEFT JOIN nodes c ON c.parent_id = p.id
        WHERE p.resolution_level < 4
        GROUP BY p.id
        HAVING child_count > 0 AND child_count < 2
        ORDER BY p.resolution_level ASC
    """).fetchall()

    for row in few_children[:5]:  # Cap at 5 more
        # Skip if already in sparse list
        if any(s["node_id"] == row["id"] for s in sparse):
            continue
        sparse.append({
            "node_id": row["id"],
            "content": row["content"][:100],
            "level": row["resolution_level"],
            "confidence": row["confidence"],
            "sparse_reason": "few_children",
            "gap_description": f"L{row['resolution_level']} node with only {row['child_count']} child(ren)",
        })

    return sparse


async def curiosity_scan(max_expansions: int = 3) -> dict:
    """Scan graph for sparse areas and expand them using existing seed_expand.

    Rate-limited: max N expansions per scan. Uses existing seed_expand()
    logic — this is just the detection layer.

    Args:
        max_expansions: Maximum nodes to expand per scan

    Returns:
        {sparse_nodes_found: int, expansions_attempted: int, expanded: [...]}
    """
    conn = db.get_db()
    sparse_nodes = detect_sparse_nodes(conn)

    if not sparse_nodes:
        return {
            "sparse_nodes_found": 0,
            "expansions_attempted": 0,
            "expanded": [],
        }

    expanded = []
    for sparse in sparse_nodes[:max_expansions]:
        from seed import seed_expand as seed_expand_fn
        result = await seed_expand_fn(sparse["node_id"])
        if result.get("nodes_created", 0) > 0:
            expanded.append({
                "node_id": sparse["node_id"],
                "content": sparse["content"],
                "level": sparse["level"],
                "sparse_reason": sparse["sparse_reason"],
                "result": {
                    "nodes_created": result.get("nodes_created", 0),
                },
            })

    return {
        "sparse_nodes_found": len(sparse_nodes),
        "expansions_attempted": min(len(sparse_nodes), max_expansions),
        "expanded": expanded,
    }
