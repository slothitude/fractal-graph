"""Distillation pipeline — batch-extract mother model parametric knowledge into graph nodes.

Three-phase architecture avoids VRAM thrashing on OLLAMA_MAX_LOADED_MODELS=1:
  GENERATE phase — mother loaded once (keep_alive="15s"), ~15 calls/domain, no warmup between
  EMBED phase — embedder loaded once, batch embed all collected nodes
  STORE phase — no model needed, dedup + insert into SQLite + ChromaDB

Target: 2B+Graph confidence 0.75+ without needing auto-expand or triad.
"""

import asyncio
import json
import time
import logging

import db
from db import _write_lock
from config import settings
from embedder import embed, embed_batch_parallel
from chroma_store import upsert_node, query_level, get_collection, query_all_levels
from graph import recompute_parent_bbox
from seed import (
    _mother_generate, _mother_generate_keepalive,
    _parse_json_array, _parse_json_object, LEVEL_MEANINGS,
)
from growth import _is_duplicate, DEDUP_THRESHOLD, _cosine_sim

logger = logging.getLogger(__name__)


# ============================================================
# Prompt Templates
# ============================================================

PROMPT_ENUMERATE_DOMAINS = """List the 25-30 most important broad domains of human knowledge.
Think broadly — sciences, humanities, arts, technology, medicine, law, philosophy, etc.
Return JSON array: [{"domain": "domain name here"}]"""


PROMPT_DECOMPOSE_ASPECTS = """Given the knowledge domain "{domain}", list its 8-12 most important aspects or sub-topics.

For each aspect, include 2-3 specific facts you know about it.
Be precise — use specific numbers, dates, names where possible.

Return JSON array: [
  {{"aspect": "aspect name", "facts": ["fact1", "fact2"]}}
]"""


PROMPT_EXTRACT_FACTS = """Given domain "{domain}" and aspect "{aspect}":

1. List 3-5 specific named entities (people, events, places, discoveries, theories)
2. List 3-5 verifiable facts with dates/numbers where possible
3. List 2-3 key relationships between concepts (A influences B, X contradicts Y, C enables D)

Level meanings:
  L3: entity — named entity, event, or specific instance
  L4: fact — verifiable fact with details (dates, numbers)
  L5: evidence — direct quote, cited statistic, or data point

Return JSON array: [
  {{"content": "precise description", "resolution_level": 3|4|5,
   "edge_to": "related content (optional)", "edge_type": "refines|contradicts|supports|challenges (optional)"}}
]

Be factual and specific. Each entry should be a self-contained knowledge node."""


PROMPT_EXTRACT_FACTS_BATCH = """For the domain "{domain}", extract knowledge nodes for these aspects:

{aspects_list}

For EACH aspect, provide:
1. 3-5 specific named entities (people, events, places, discoveries)
2. 3-5 verifiable facts with dates/numbers
3. 2-3 key relationships

Level meanings: L3=entity, L4=fact, L5=evidence

Return a JSON object with aspect names as keys:
{{"aspect name": [{{"content": "...", "resolution_level": 3|4|5, "edge_to": "related content (optional)", "edge_type": "type (optional)"}}]}}"""


PROMPT_SCAFFOLD = """Given these extracted knowledge nodes from domain "{domain}":

{compressed_nodes}

Create the L0 domain node and L1 topic nodes that organize them into a coherent hierarchy.

Level meanings:
  L0: domain — broad field of knowledge
  L1: topic — a named topic within the domain

Return JSON: {{
  "domain": {{"content": "broad domain summary (under 50 words)"}},
  "topics": [
    {{"content": "topic summary", "covers_aspects": [0, 1, 2]}}
  ]
}}

covers_aspects references indices in the nodes list above (0-based)."""


# ============================================================
# Core Functions
# ============================================================

def _compress_nodes(nodes: list[dict], max_per_level: int = 15) -> str:
    """Compress collected nodes into a compact listing for the scaffold prompt."""
    lines = []
    for i, node in enumerate(nodes):
        level = node.get("resolution_level", 3)
        content = node.get("content", "")[:120]
        lines.append(f"[{i}] (L{level}) {content}")
        if len(lines) >= max_per_level * 6:
            lines.append(f"... and {len(nodes) - len(lines)} more nodes")
            break
    return "\n".join(lines)


async def _enumerate_domains(mother_model: str = None) -> list[str]:
    """Ask the mother model to enumerate knowledge domains."""
    raw = await _mother_generate(PROMPT_ENUMERATE_DOMAINS, mother_model)
    domains_data = _parse_json_array(raw)
    return [d["domain"] for d in domains_data if "domain" in d]


async def distill_domain(domain: str, mother_model: str = None) -> dict:
    """Distill all mother model knowledge about one domain into graph nodes.

    Three phases:
    1. GENERATE — mother stays hot (keep_alive="15s"), ~12-15 calls
    2. EMBED — batch embed all collected nodes
    3. STORE — dedup + insert into SQLite + ChromaDB

    Returns:
        {domain, nodes_created, edges_created, phase_times}
    """
    t0 = time.time()
    m = mother_model or settings.mother_model
    conn = db.get_db()
    collected = []  # list of dicts: content, resolution_level, edge_to, edge_type
    edges_to_create = []

    # === PHASE 1: GENERATE (mother hot) ===
    gen_start = time.time()

    # Step 1: Decompose aspects (1 call) — also serves as warmup (first call loads model)
    aspects_raw = await _mother_generate_keepalive(
        PROMPT_DECOMPOSE_ASPECTS.format(domain=domain),
        keep_alive="15s", model=m
    )
    aspects = _parse_json_array(aspects_raw)
    print(f"    Aspects: {len(aspects)}")

    # Step 2: Extract facts per aspect (1 call per aspect)
    for aspect_data in aspects:
        aspect_name = aspect_data.get("aspect", "")
        if not aspect_name:
            continue
        facts_raw = await _mother_generate_keepalive(
            PROMPT_EXTRACT_FACTS.format(domain=domain, aspect=aspect_name),
            keep_alive="15s", model=m
        )
        nodes = _parse_json_array(facts_raw)
        for node in nodes:
            if "content" in node:
                level = int(str(node.get("resolution_level", 4)).lstrip("Ll"))
                level = max(3, min(5, level))
                collected.append({
                    "content": node["content"],
                    "resolution_level": level,
                    "edge_to": node.get("edge_to"),
                    "edge_type": node.get("edge_type"),
                })
        print(f"      {aspect_name}: +{len(nodes)} nodes")

    print(f"    Collected L3-L5: {len(collected)} nodes")

    # Step 3: Scaffold L0-L1 (1 call)
    if collected:
        scaffold_raw = await _mother_generate_keepalive(
            PROMPT_SCAFFOLD.format(
                domain=domain,
                compressed_nodes=_compress_nodes(collected)
            ),
            keep_alive="15s", model=m
        )
        scaffold = _parse_json_object(scaffold_raw)

        if scaffold and isinstance(scaffold, dict):
            # L0 domain node
            domain_entry = scaffold.get("domain", {})
            if isinstance(domain_entry, dict) and domain_entry.get("content"):
                collected.append({
                    "content": domain_entry["content"],
                    "resolution_level": 0,
                })

            # L1 topic nodes
            topic_nodes = scaffold.get("topics", [])
            for topic in topic_nodes:
                if isinstance(topic, dict) and topic.get("content"):
                    collected.append({
                        "content": topic["content"],
                        "resolution_level": 1,
                    })

            print(f"    Scaffold: L0 + {len(topic_nodes)} L1 topics")

    # Step 4: Unload mother
    await _mother_generate_keepalive(".", keep_alive="0", model=m)
    gen_time = round(time.time() - gen_start, 1)

    if not collected:
        return {"domain": domain, "nodes_created": 0, "edges_created": 0,
                "phase_times": {"generate": gen_time, "embed": 0, "store": 0},
                "error": "No nodes generated"}

    # === PHASE 2: EMBED (batch, groups of 5) ===
    embed_start = time.time()
    texts = [n["content"] for n in collected]
    embeddings = []
    for i in range(0, len(texts), 20):
        batch = texts[i:i+5]
        print(f"    Embedding batch {i//5+1}/{(len(texts)-1)//5+2} ({len(batch)} nodes)...")
        batch_embs = await embed_batch_parallel(batch)
        embeddings.extend(batch_embs)
    embed_time = round(time.time() - embed_start, 1)

    # === PHASE 3: STORE (dedup + insert) ===
    store_start = time.time()
    created_nodes = []
    created_edges = []
    content_to_id = {}  # content text → db node_id (for edge references)

    async with _write_lock:
        # Sort by level so parents exist first (L0, L1, L3, L4, L5)
        collected_sorted = sorted(collected, key=lambda n: n["resolution_level"])

        for i, node_data in enumerate(collected_sorted):
            content = node_data["content"]
            level = node_data["resolution_level"]
            emb = embeddings[i] if i < len(embeddings) else None
            if not emb:
                continue

            # Dedup check
            if await _is_duplicate(content, emb, None, level):
                continue

            # Find best parent via vector similarity (skip for L0)
            parent_id = None
            if level > 0:
                for plevel in range(level - 1, -1, -1):
                    hits = query_level(emb, plevel, n_results=1)
                    if hits:
                        parent_id = int(hits[0]["node_id"])
                        break

            node_id = db.insert_node(
                conn, content, resolution_level=level, parent_id=parent_id,
                confidence=0.7,
            )
            upsert_node(node_id, content, emb, level, parent_id, 0.7)
            content_to_id[content] = node_id
            created_nodes.append({
                "id": node_id, "level": level,
                "content": content[:100],
            })

            # Queue cross-reference edges
            edge_to = node_data.get("edge_to")
            edge_type = node_data.get("edge_type")
            if edge_to and edge_type:
                edges_to_create.append({
                    "source_content": content,
                    "target_content": edge_to,
                    "edge_type": edge_type,
                })

        # Create cross-reference edges (batch embed targets, groups of 5)
        if edges_to_create:
            print(f"    Resolving {len(edges_to_create)} cross-reference edges...")
            edge_targets = [e["target_content"] for e in edges_to_create]
            target_texts = list(dict.fromkeys(edge_targets))  # dedup
            target_embeddings = []
            for i in range(0, len(target_texts), 5):
                batch = target_texts[i:i+5]
                batch_embs = await embed_batch_parallel(batch)
                target_embeddings.extend(batch_embs)

            target_text_to_emb = dict(zip(target_texts, target_embeddings))
        else:
            target_text_to_emb = {}

        for edge_info in edges_to_create:
            source_id = content_to_id.get(edge_info["source_content"])
            # Use batched target embedding
            target_emb = target_text_to_emb.get(edge_info["target_content"])
            all_hits = query_all_levels(target_emb, n_results=1)
            target_id = None
            for level_hits in all_hits.values():
                if level_hits:
                    target_id = int(level_hits[0]["node_id"])
                    break
            if source_id and target_id:
                try:
                    db.insert_edge(
                        conn, source_id, target_id,
                        edge_type=edge_info["edge_type"], confidence=0.5,
                        context=f"Mother-identified relationship in {domain}",
                    )
                    created_edges.append(1)
                except (ValueError, Exception):
                    pass

        # Recompute bboxes for L0 and L1 parents
        for cn in created_nodes:
            if cn["level"] <= 1:
                recompute_parent_bbox(conn, cn["id"])

    store_time = round(time.time() - store_start, 1)
    total_time = round(time.time() - t0, 1)

    return {
        "domain": domain,
        "nodes_created": len(created_nodes),
        "edges_created": len(created_edges),
        "phase_times": {
            "generate": gen_time,
            "embed": embed_time,
            "store": store_time,
            "total": total_time,
        },
    }


async def distill_all(domains: list[str] = None,
                      mother_model: str = None) -> dict:
    """Distill all knowledge domains from the mother model.

    If domains is None, enumerates domains from the mother first.
    Processes each domain sequentially (mother loads/unloads per domain).

    Returns:
        {domains_processed, total_nodes, total_edges, total_time, results}
    """
    t0 = time.time()

    if domains is None:
        print("Enumerating domains from mother model...")
        domains = await _enumerate_domains(mother_model)
        print(f"Found {len(domains)} domains")

    if not domains:
        return {"error": "No domains to distill"}

    results = []
    for i, domain in enumerate(domains):
        print(f"\n[{i+1}/{len(domains)}] Distilling: {domain}")
        try:
            result = await distill_domain(domain, mother_model)
            results.append(result)
            pt = result.get("phase_times", {})
            print(f"    Result: {result['nodes_created']} nodes, "
                  f"{result.get('edges_created', 0)} edges "
                  f"(gen={pt.get('generate', '?')}s, "
                  f"embed={pt.get('embed', '?')}s, "
                  f"store={pt.get('store', '?')}s)")
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"domain": domain, "error": str(e)})

    total_time = round(time.time() - t0, 1)
    total_nodes = sum(r.get("nodes_created", 0) for r in results)
    total_edges = sum(r.get("edges_created", 0) for r in results)

    return {
        "domains_processed": len(results),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "total_time": total_time,
        "results": results,
    }


def distill_coverage() -> dict:
    """Show graph coverage — nodes per level, shallow branches, gaps.

    Returns:
        {total_nodes, nodes_per_level, avg_depth, shallow_branches, domains}
    """
    conn = db.get_db()

    # Count nodes per level
    nodes_per_level = {}
    for level in range(6):
        count = conn.execute(
            "SELECT COUNT(*) as c FROM nodes WHERE resolution_level = ?",
            (level,)
        ).fetchone()["c"]
        nodes_per_level[level] = count
    total_nodes = sum(nodes_per_level.values())

    # Average depth per L0 domain
    depth_rows = conn.execute("""
        WITH RECURSIVE depth(id, level, max_depth) AS (
            SELECT id, resolution_level, resolution_level
            FROM nodes WHERE resolution_level = 0
            UNION ALL
            SELECT n.id, n.resolution_level, d.max_depth
            FROM nodes n JOIN depth d ON n.parent_id = d.id
        )
        SELECT d.id, d.max_depth,
               (SELECT MAX(n.resolution_level) FROM nodes n
                WHERE n.id IN (
                    SELECT c.id FROM nodes c
                    JOIN nodes p ON p.id = c.parent_id
                    JOIN nodes pp ON pp.id = p.parent_id
                    WHERE pp.id = d.id
                ) OR n.id = d.id) as actual_depth
        FROM depth d
        GROUP BY d.id
    """).fetchall()

    avg_depth = 0
    shallow_branches = []
    if depth_rows:
        depths = [row["actual_depth"] or row["max_depth"] for row in depth_rows]
        avg_depth = round(sum(depths) / len(depths), 1)

        for row in depth_rows:
            d = row["actual_depth"] or row["max_depth"]
            if d < 3:
                node = db.get_node(conn, row["id"])
                shallow_branches.append({
                    "node_id": row["id"],
                    "content": (node["content"][:80] if node else "?"),
                    "depth": d,
                })

    # L0 domain list
    domains = []
    for row in conn.execute(
        "SELECT id, content FROM nodes WHERE resolution_level = 0"
    ).fetchall():
        children = db.get_children(conn, row["id"])
        child_levels = [c["resolution_level"] for c in children]
        max_child_level = max(child_levels) if child_levels else 0
        domains.append({
            "node_id": row["id"],
            "content": row["content"][:80],
            "children": len(children),
            "max_level": max_child_level,
        })

    return {
        "total_nodes": total_nodes,
        "nodes_per_level": nodes_per_level,
        "avg_depth": avg_depth,
        "shallow_branches": shallow_branches,
        "shallow_count": len(shallow_branches),
        "domains": domains,
    }


# ============================================================
# CLI entry point
# ============================================================

async def main():
    """Run distillation from command line."""
    import sys
    domains = sys.argv[1:] if len(sys.argv) > 1 else None
    result = await distill_all(domains)
    print(f"\n{'='*60}")
    print(f"Distillation complete:")
    print(f"  Domains: {result.get('domains_processed', 0)}")
    print(f"  Nodes:   {result.get('total_nodes', 0)}")
    print(f"  Edges:   {result.get('total_edges', 0)}")
    print(f"  Time:    {result.get('total_time', 0)}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
