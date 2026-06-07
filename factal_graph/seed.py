"""Mother model seeding pipeline — intelligent knowledge graph population.

Uses a larger LLM (e.g. lfm2.5:latest, ~8B) to generate structured multi-resolution
node hierarchies with cross-resolution edges and bounding boxes.
"""

import json
import re

import httpx

import db
from db import _write_lock
from config import settings
from embedder import embed, embed_batch_parallel
from chroma_store import upsert_node
from graph import propagate_confidence, recompute_parent_bbox, compute_bbox


import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

import db
from config import settings
from embedder import embed, embed_batch_parallel
from chroma_store import upsert_node
from graph import propagate_confidence, recompute_parent_bbox, compute_bbox


# --- Mother model LLM calls ---

async def _mother_generate(prompt: str, model: str = None) -> str:
    """Call the mother model (larger LLM) for structured generation.

    Retries once on empty response or JSON parse failure.
    Uses keep_alive=0 to free VRAM immediately for 2B model.
    """
    model = model or settings.mother_model
    url = settings.mother_url

    # Warmup: skip if already loaded (uses shared cache with TTL)
    from model_cache import ensure_model_loaded
    await ensure_model_loaded(model, url)

    # Use /api/chat — lfm2-thinking parser auto-separates thinking into .thinking field
    # No "format": "json" needed — it causes lfm2.5 to drain tokens on thinking with empty content
    # No "think": False — it breaks lfm2-thinking parser and leaks raw tags
    # keep_alive=0 — free VRAM immediately so 2B can load during auto-expand
    for attempt in range(2):
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{url}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "keep_alive": "0",
                    "options": {"temperature": 0.3},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = ""
            if "message" in data:
                content = data["message"].get("content", "").strip()
            else:
                content = data.get("content", "").strip()

            if content:
                return content

            if attempt == 0:
                logger.warning("Mother model returned empty response, retrying...")

    return ""


def _parse_json_array(text: str) -> list[dict]:
    """Extract JSON array from LLM output, tolerant of markdown fences."""
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    # Try direct parse
    # Strip thinking tags (qwen, lfm, etc.)
    text = re.sub(r"</?think\s*>", "", text).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "children" in parsed:
            return parsed["children"]
        # Single object → wrap
        return [parsed]
    except json.JSONDecodeError:
        pass

    # Try to find JSON array within the text
    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    if bracket_start != -1 and bracket_end != -1:
        try:
            return json.loads(text[bracket_start:bracket_end + 1])
        except json.JSONDecodeError:
            pass

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        try:
            obj = json.loads(text[brace_start:brace_end + 1])
            if isinstance(obj, dict) and "children" in obj:
                return obj["children"]
            return [obj]
        except json.JSONDecodeError:
            pass

    print(f"WARNING: Could not parse JSON from mother model output: {text[:200].encode('ascii','replace').decode()}")
    return []


def _parse_json_object(text: str) -> dict | None:
    """Extract a JSON object from LLM output."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    return None


# --- Level meanings for prompts ---

LEVEL_MEANINGS = {
    0: "domain — broad field of knowledge (e.g. 'NATO-Russia relations')",
    1: "topic — a named topic within the domain (e.g. 'NATO eastward expansion')",
    2: "concept — specific concept or mechanism (e.g. 'Article 5 collective defense')",
    3: "entity — a named entity or specific instance (e.g. '2014 Crimea annexation')",
    4: "fact — a verifiable fact with details (e.g. 'NATO added 4 members between 1999-2020')",
    5: "evidence — cited source, quote, or data point (e.g. a specific statistic or statement)",
}


# --- Core seeding functions ---

async def _mother_generate_keepalive(prompt: str, keep_alive: str = "15s",
                                     model: str = None) -> str:
    """Mother call with configurable keep_alive. For batch generation.

    Same warmup + retry logic as _mother_generate but passes through
    the keep_alive parameter so the mother stays hot between calls.
    Use keep_alive="0" after the last call in a batch to unload.
    """
    model = model or settings.mother_model
    url = settings.mother_url

    # No warmup polling — caller is responsible for keeping model hot
    # (warmup only in _mother_generate for single fire-and-forget calls)

    # Use /api/chat with configurable keep_alive
    for attempt in range(2):
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{url}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "keep_alive": keep_alive,
                    "options": {"temperature": 0.3, "num_predict": 4096},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = ""
            if "message" in data:
                content = data["message"].get("content", "").strip()
            else:
                content = data.get("content", "").strip()

            if content:
                return content

            if attempt == 0:
                logger.warning("Mother model returned empty response, retrying...")

    return ""


async def seed_topic(topic: str, depth: int = 3, mother_model: str = None,
                    confidence: float = 0.7) -> dict:
    """Seed a single topic end-to-end using the mother model.

    Generates a full L0-L{depth} hierarchy:
    1. Mother creates L0 domain node
    2. Expands to L1 topics (3-5)
    3. For each L1, expands to L2 concepts (2-4)
    4. Continues down to requested depth
    5. Creates parent-child edges automatically
    6. Detects cross-resolution edges against existing graph
    7. Computes bounding boxes bottom-up

    Args:
        topic: Topic to seed
        depth: Maximum resolution depth (default 3 → L0-L3)
        mother_model: Override mother model name
        confidence: Default confidence for created nodes

    Returns:
        Summary with node IDs, edges created, levels populated
    """
    conn = db.get_db()
    created_nodes = []
    created_edges = []
    depth = min(max(depth, 1), 5)  # Clamp 1-5

    # Step 1: Generate L0 domain node
    domain_prompt = (
        "You are seeding a fractal knowledge graph. Given the topic below, "
        "generate a single broad domain summary.\n\n"
        "The domain should be the broadest possible framing — a field of knowledge "
        "that encompasses this topic. Under 50 words. No quotes.\n\n"
        f"Topic: {topic}\n\n"
        'Return JSON: {"domain": "your domain summary here"}'
    )
    domain_result = _parse_json_object(await _mother_generate(domain_prompt, mother_model))
    if not domain_result or "domain" not in domain_result:
        return {"error": "Mother model failed to generate domain node", "raw": domain_result}

    domain_text = domain_result["domain"]
    domain_emb = await embed(domain_text)
    domain_id = db.insert_node(
        conn, domain_text, resolution_level=0, confidence=confidence
    )
    upsert_node(domain_id, domain_text, domain_emb, 0, None, confidence)
    created_nodes.append({"id": domain_id, "level": 0, "content": domain_text})

    # Step 2+: Expand recursively down to target depth
    await _expand_level(
        conn, topic, domain_id, domain_text, 0, depth,
        mother_model, confidence, created_nodes, created_edges
    )

    # Step 3: Compute bounding boxes bottom-up
    _compute_bboxes_for_subtree(conn, domain_id)

    return {
        "topic": topic,
        "root_id": domain_id,
        "depth_reached": depth,
        "nodes_created": len(created_nodes),
        "edges_created": len(created_edges),
        "nodes": created_nodes,
    }


async def _expand_level(conn, topic: str, parent_id: int, parent_content: str,
                       parent_level: int, max_depth: int, mother_model: str,
                       confidence: float, created_nodes: list, created_edges: list):
    """Expand a node into children at the next resolution level."""
    next_level = parent_level + 1
    if next_level > max_depth:
        return

    # Determine child count based on level
    child_counts = {1: (3, 5), 2: (2, 4), 3: (2, 4), 4: (1, 3), 5: (1, 2)}
    min_children, max_children = child_counts.get(next_level, (2, 3))

    expand_prompt = (
        f"Expand this knowledge graph node into {min_children}-{max_children} children "
        f"at resolution level {next_level}.\n\n"
        f"Level meanings:\n"
    )
    for lvl, meaning in LEVEL_MEANINGS.items():
        expand_prompt += f"  L{lvl}: {meaning}\n"

    expand_prompt += (
        f"\nParent: \"{parent_content}\" (L{parent_level})\n"
        f"Overall topic: {topic}\n\n"
        "Generate children that REFINE the parent into more specific knowledge.\n"
        "Each child should be a distinct aspect or sub-topic.\n\n"
        "Return JSON array: [\n"
        "  {\"content\": \"...\", \"edges\": [{\"type\": \"refines|contradicts|supports\", \"sibling_index\": N}]}\n"
        "]\n"
        "Edges are optional — describe relationships between siblings if any.\n"
    )

    children_data = _parse_json_array(await _mother_generate(expand_prompt, mother_model))
    if not children_data:
        return

    # Embed all children in parallel
    child_texts = [c["content"] for c in children_data if "content" in c]
    child_embeddings = await embed_batch_parallel(child_texts)

    new_node_ids = []
    for i, child_data in enumerate(children_data):
        if "content" not in child_data:
            continue
        content = child_data["content"]
        emb = child_embeddings[i] if i < len(child_embeddings) else None

        node_id = db.insert_node(
            conn, content, resolution_level=next_level, parent_id=parent_id,
            confidence=confidence,
        )
        if emb:
            upsert_node(node_id, content, emb, next_level, parent_id, confidence)

        new_node_ids.append(node_id)
        created_nodes.append({"id": node_id, "level": next_level, "content": content})

        # Create edges between siblings if specified
        sibling_edges = child_data.get("edges", [])
        for edge_info in sibling_edges:
            sib_idx = edge_info.get("sibling_index")
            edge_type = edge_info.get("type", "related")
            if sib_idx is not None and 0 <= sib_idx < len(new_node_ids) - 1:
                try:
                    edge_id = db.insert_edge(
                        conn, new_node_ids[sib_idx], node_id,
                        edge_type=edge_type, confidence=0.5,
                        context=f"Mother-identified sibling relationship",
                    )
                    created_edges.append(edge_id)
                except (ValueError, Exception):
                    pass

    # Recurse into children
    for node_id in new_node_ids:
        await _expand_level(
            conn, topic, node_id,
            [n["content"] for n in created_nodes if n["id"] == node_id][0] if any(n["id"] == node_id for n in created_nodes) else "",
            next_level, max_depth, mother_model, confidence,
            created_nodes, created_edges
        )


def _compute_bboxes_for_subtree(conn, root_id: int):
    """Compute bounding boxes bottom-up for a subtree."""
    # Get all nodes in the subtree ordered by level descending
    nodes = conn.execute(
        "WITH RECURSIVE subtree(id, resolution_level, parent_id) AS ("
        "  SELECT id, resolution_level, parent_id FROM nodes WHERE id = ?"
        "  UNION ALL"
        "  SELECT n.id, n.resolution_level, n.parent_id FROM nodes n"
        "  JOIN subtree s ON n.parent_id = s.id"
        ") SELECT id, resolution_level, parent_id FROM subtree"
        " ORDER BY resolution_level DESC",
        (root_id,)
    ).fetchall()

    for node in nodes:
        recompute_parent_bbox(conn, node["id"])


async def seed_from_search(query: str, max_urls: int = 5,
                           mother_model: str = None) -> dict:
    """Web search + mother enrichment pipeline (two-pass).

    Pass 1: Mother sees each URL individually → extracts L3-L5 nodes
            (facts, entities, evidence). One LLM call per URL.
    Pass 2: Mother sees only the compressed L3 nodes from pass 1 →
            scaffolds L0-L2 structure + cross-document edges.
            Typically under 4k tokens regardless of source volume.

    This avoids context overflow and forces intelligence compression
    at L3 — facts are dense, source text is noisy.

    Args:
        query: Search query
        max_urls: Max URLs to process
        mother_model: Override mother model

    Returns:
        Summary with nodes and edges created
    """
    from ingest import search_searxng, extract_text

    # Step 1: Search
    search_results = await search_searxng(query, max_results=max_urls)
    if not search_results:
        return {"error": "No search results", "query": query}

    # Step 2: Extract text from all URLs in parallel
    async def _fetch_url(r):
        text = await extract_text(r["url"])
        if text:
            return {"url": r["url"], "title": r.get("title", ""), "text": text}
        return None

    fetch_tasks = [_fetch_url(r) for r in search_results[:max_urls]]
    results = await asyncio.gather(*fetch_tasks)
    all_texts = [r for r in results if r]

    if not all_texts:
        return {"error": "Could not extract text from any URL", "query": query}

    conn = db.get_db()
    created_nodes = []
    created_edges = []

    # === PASS 1: Extract dense L3-L5 nodes from each URL ===
    # Each URL gets its own LLM call — no context overflow
    all_pass1_nodes = []  # list of list[dict] — nodes per URL

    for url_data in all_texts:
        extract_prompt = (
            "Extract the key facts, entities, and evidence from this article. "
            "Query context: " + query + "\n\n"
            "Article title: " + url_data["title"] + "\n"
            "Article text:\n" + url_data["text"][:5000] + "\n\n"
            "For each important fact/entity/evidence, create a node.\n"
            "Level meanings:\n"
            "  L3: entity — named entity, event, or specific instance\n"
            "  L4: fact — verifiable fact with details (dates, numbers)\n"
            "  L5: evidence — direct quote or cited statistic\n\n"
            "Return JSON array: [\n"
            '  {"content": "...", "resolution_level": 3|4|5}\n'
            "]\n"
            "Extract 3-7 of the most important nodes. Be precise and factual."
        )

        nodes_data = _parse_json_array(
            await _mother_generate(extract_prompt, mother_model)
        )
        if nodes_data:
            all_pass1_nodes.append(nodes_data)

    if not all_pass1_nodes:
        return {"error": "Mother model failed to extract nodes from any URL", "query": query}

    # Embed and store pass 1 nodes (batch embed)
    pass1_db_ids = []  # (db_id, content, level) for each node
    # Collect all nodes first, then batch embed
    flat_nodes = []
    for url_nodes in all_pass1_nodes:
        for node_data in url_nodes:
            if "content" not in node_data:
                continue
            content = node_data["content"]
            level = int(str(node_data.get("resolution_level", 4)).lstrip("Ll"))
            level = max(3, min(5, level))  # Clamp to L3-L5
            flat_nodes.append({"content": content, "level": level})

    if flat_nodes:
        all_contents = [n["content"] for n in flat_nodes]
        all_embeddings = await embed_batch_parallel(all_contents)

        async with _write_lock:
            for i, node_data in enumerate(flat_nodes):
                content = node_data["content"]
                level = node_data["level"]
                emb = all_embeddings[i] if i < len(all_embeddings) else None

                node_id = db.insert_node(
                    conn, content, resolution_level=level,
                    confidence=0.6, source_url=None,
                )
                if emb:
                    upsert_node(node_id, content, emb, level, None, 0.6)
                created_nodes.append({"id": node_id, "level": level, "content": content[:100], "pass": 1})
                pass1_db_ids.append(node_id)

    # === PASS 2: Scaffold L0-L2 + cross-document edges ===
    # Compress all L3 nodes into a summary for the mother
    pass1_summary = "\n".join(
        f"[{n['level']}] {n['content'][:150]}"
        for n in created_nodes
    )

    scaffold_prompt = (
        "Given these extracted knowledge nodes from web search, "
        "create the hierarchical structure ABOVE them.\n\n"
        f"Query: {query}\n\n"
        f"Existing nodes (L3-L5):\n{pass1_summary}\n\n"
        "Create L0 (domain), L1 (topics), and L2 (concepts) nodes that "
        "organize these existing nodes into a coherent hierarchy.\n"
        "Also identify cross-document relationships (contradicts, supports, challenges).\n\n"
        "Return JSON: {\n"
        '  "hierarchy": [\n'
        '    {"content": "...", "resolution_level": 0|1|2, "parent_index": N or null}\n'
        "  ],\n"
        '  "cross_edges": [\n'
        '    {"source_index": N, "target_index": N, "type": "contradicts|supports|challenges"}\n'
        "  ]\n"
        "}\n"
        "parent_index references other items in the hierarchy array (null for L0 root).\n"
        "cross_edges reference existing nodes by their index in the "
        "Existing nodes list above (0-based).\n"
    )

    scaffold_result = _parse_json_object(
        await _mother_generate(scaffold_prompt, mother_model)
    )
    if not scaffold_result:
        # Fallback: just return pass 1 nodes, no structure
        return {
            "query": query,
            "urls_processed": len(all_texts),
            "nodes_created": len(created_nodes),
            "edges_created": 0,
            "pass": 1,
            "note": "Pass 2 (scaffold) failed — pass 1 nodes created without L0-L2 structure",
        }

    # Create L0-L2 scaffold nodes
    hierarchy = scaffold_result.get("hierarchy", [])
    if hierarchy:
        hierarchy_embeddings = await embed_batch_parallel(
            [n["content"] for n in hierarchy]
        )

        # Sort by level to ensure parents exist first
        sorted_indices = sorted(
            range(len(hierarchy)),
            key=lambda i: int(str(hierarchy[i].get("resolution_level", 0)).lstrip("Ll"))
        )
        id_map = {}  # hierarchy array index → db node_id

        for idx in sorted_indices:
            node_data = hierarchy[idx]
            content = node_data["content"]
            level = int(str(node_data.get("resolution_level", 0)).lstrip("Ll"))
            level = max(0, min(2, level))  # Clamp to L0-L2
            parent_idx = node_data.get("parent_index")
            emb = hierarchy_embeddings[idx] if idx < len(hierarchy_embeddings) else None

            parent_id = id_map.get(parent_idx) if parent_idx is not None else None

            node_id = db.insert_node(
                conn, content, resolution_level=level, parent_id=parent_id,
                confidence=0.7,
            )
            if emb:
                upsert_node(node_id, content, emb, level, parent_id, 0.7)

            id_map[idx] = node_id
            created_nodes.append({"id": node_id, "level": level, "content": content[:100], "pass": 2})

        # Attach pass 1 L3 nodes to their nearest L2 parent
        for pass1_id in pass1_db_ids:
            pass1_node = db.get_node(conn, pass1_id)
            if not pass1_node:
                continue
            # Find best L2 parent by vector similarity
            from chroma_store import query_level
            hits = query_level(
                await embed(pass1_node["content"]), 2, n_results=1
            )
            if hits:
                parent_id = int(hits[0]["node_id"])
                conn.execute(
                    "UPDATE nodes SET parent_id = ? WHERE id = ?",
                    (parent_id, pass1_id)
                )
                conn.commit()

    # Create cross-document edges
    cross_edges = scaffold_result.get("cross_edges", [])
    for edge_info in cross_edges:
        source_idx = edge_info.get("source_index")
        target_idx = edge_info.get("target_index")
        edge_type = edge_info.get("type", "related")
        # Source/target indices reference pass 1 nodes (0-based in the existing nodes list)
        if source_idx is not None and target_idx is not None:
            # Find corresponding pass 1 db IDs
            pass1_node_ids = [n["id"] for n in created_nodes if n.get("pass") == 1]
            if (source_idx < len(pass1_node_ids)
                    and target_idx < len(pass1_node_ids)):
                try:
                    edge_id = db.insert_edge(
                        conn, pass1_node_ids[source_idx],
                        pass1_node_ids[target_idx],
                        edge_type=edge_type, confidence=0.5,
                        context="Cross-document relationship from pass 2",
                    )
                    created_edges.append(edge_id)
                except (ValueError, Exception):
                    pass

    # Recompute bboxes for all pass 2 parents
    for cn in created_nodes:
        if cn.get("pass") == 2:
            recompute_parent_bbox(conn, cn["id"])

    return {
        "query": query,
        "urls_processed": len(all_texts),
        "nodes_created": len(created_nodes),
        "edges_created": len(created_edges),
        "pass1_nodes": sum(1 for n in created_nodes if n.get("pass") == 1),
        "pass2_nodes": sum(1 for n in created_nodes if n.get("pass") == 2),
        "nodes": created_nodes,
    }


async def seed_expand(node_id: int, mother_model: str = None,
                      confidence: float = 0.7) -> dict:
    """Expand an existing sparse node — fill in missing resolution levels.

    1. Get node + its children
    2. Mother identifies gaps in resolution coverage
    3. Mother generates missing intermediate nodes
    4. Recomputes bboxes

    Args:
        node_id: Node to expand
        mother_model: Override mother model
        confidence: Default confidence

    Returns:
        Summary with new nodes created
    """
    conn = db.get_db()
    node = db.get_node(conn, node_id)
    if not node:
        return {"error": f"Node {node_id} not found"}

    children = db.get_children(conn, node_id)
    child_levels = [c["resolution_level"] for c in children]

    # Find which levels are missing
    current_level = node["resolution_level"]
    all_levels = set(range(current_level + 1, 6))
    existing_levels = set(child_levels)
    missing_levels = sorted(all_levels - existing_levels)

    if not missing_levels and not children:
        # No children at all — expand at next level
        missing_levels = [min(current_level + 1, 5)]

    if not missing_levels and children:
        return {"status": "already_expanded", "node_id": node_id,
                "message": "All resolution levels present under this node"}

    # Ask mother to fill gaps
    children_desc = "\n".join(
        f"  L{c['resolution_level']}: {c['content'][:100]}" for c in children
    )

    gap_prompt = (
        f"This knowledge graph node has gaps in its resolution hierarchy.\n\n"
        f"Node: \"{node['content']}\" (L{current_level})\n\n"
        f"Existing children:\n{children_desc}\n\n"
        f"Missing resolution levels: {missing_levels}\n"
        f"Level meanings:\n"
    )
    for lvl, meaning in LEVEL_MEANINGS.items():
        gap_prompt += f"  L{lvl}: {meaning}\n"

    gap_prompt += (
        "\nGenerate nodes to fill the missing levels. Each node should bridge "
        "between the parent and existing children.\n"
        "Return JSON array: [\n"
        "  {\"content\": \"...\", \"resolution_level\": N}\n"
        "]\n"
    )

    gap_nodes = _parse_json_array(await _mother_generate(gap_prompt, mother_model))
    if not gap_nodes:
        return {"error": "Mother model failed to generate gap nodes"}

    created_nodes = []
    gap_embeddings = await embed_batch_parallel(
        [n["content"] for n in gap_nodes]
    )

    async with _write_lock:
        for i, gap_data in enumerate(gap_nodes):
            content = gap_data["content"]
            level = int(str(gap_data.get("resolution_level", missing_levels[0] if missing_levels else current_level + 1)).lstrip("Ll"))
            emb = gap_embeddings[i] if i < len(gap_embeddings) else None

            # Attach to the expanded node
            new_id = db.insert_node(
                conn, content, resolution_level=level, parent_id=node_id,
                confidence=confidence,
            )
            if emb:
                upsert_node(new_id, content, emb, level, node_id, confidence)

            created_nodes.append({"id": new_id, "level": level, "content": content[:100]})

        # Recompute bboxes
        recompute_parent_bbox(conn, node_id)

    return {
        "node_id": node_id,
        "missing_levels_filled": missing_levels,
        "nodes_created": len(created_nodes),
        "nodes": created_nodes,
    }
