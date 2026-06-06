"""Incremental graph enrichment — compare mother knowledge vs graph, cross-link domains.

Four functions:
1. mother_knowledge_probe(topic) — compare what mother knows vs what's in the graph
2. enrich_node(node_id) — ask mother for related knowledge, insert siblings/children
3. cross_link_nodes(node_id_a, node_id_b) — ask mother if two nodes are related
4. auto_crosslink(domain_node_id, max_pairs) — batch cross-linking within a domain
"""

import json
import logging

import db
from db import _write_lock
from config import settings
from embedder import embed, embed_batch_parallel
from chroma_store import upsert_node, query_all_levels, query_level, get_collection
from graph import recompute_parent_bbox
from seed import (_mother_generate, _mother_generate_keepalive,
                 _parse_json_array, _parse_json_object, LEVEL_MEANINGS)
from growth import _is_duplicate, _cosine_sim, DEDUP_THRESHOLD

logger = logging.getLogger(__name__)


# --- Prompt templates ---

PROBE_PROMPT = """You are comparing your knowledge against an existing knowledge graph.
Given the topic below, list the most important facts, concepts, and entities you know about it.

Topic: {topic}

Return JSON array of facts:
[{{"content": "factual statement", "level": N}}]

Level meanings:
{level_meanings}

Include 8-15 of the most important things you know. Focus on facts and entities (L3-L5).
Be specific and factual — no vague generalizations.
"""


ENRICH_NODE_PROMPT = """This is a node in a fractal knowledge graph:

Node: "{content}" (L{level})
Parent: "{parent_content}" (L{parent_level})

Existing children:
{children_desc}

Existing siblings:
{siblings_desc}

What related knowledge is MISSING from this node's context? Generate 2-5 new knowledge
nodes that would enrich this area of the graph. These can be:
- Siblings (same level, same parent) — related topics not yet covered
- Children (level+1) — more specific details about this node

Return JSON array:
[{{"content": "...", "level": N, "placement": "sibling"|"child"}}]

Level meanings:
{level_meanings}

Be specific and factual. Each node should be a distinct piece of knowledge.
"""


CROSS_LINK_PROMPT = """You are assessing relationships between two knowledge graph nodes.

Node A: "{content_a}" (L{level_a})
Node B: "{content_b}" (L{level_b})

How are these related? Consider:
- supports (A provides evidence/context for B)
- contradicts (A and B are in tension)
- refines (A adds detail to B)
- exemplifies (A is an instance of B)
- generalizes (A is broader than B)
- challenges (A questions the validity of B)
- derived_from (A was derived from B)
- related (general association)

Return JSON:
{{"related": true/false, "edge_type": "type", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}

If not related, set related=false and confidence=0.
"""


AUTO_CROSSLINK_PROMPT = """You are finding cross-links between knowledge topics.

Domain: "{domain_content}" (L0)

Topics under this domain:
{topics_desc}

For each pair of topics that should be connected but aren't already related,
identify the relationship type and strength.

Return JSON array of edges:
[{{"topic_a_index": N, "topic_b_index": N, "edge_type": "type", "confidence": 0.0-1.0, "reasoning": "brief"}}]

Only include pairs with confidence > 0.3. Empty array if no cross-links found.
Edge types: supports, contradicts, refines, exemplifies, generalizes, challenges, related
"""


async def mother_knowledge_probe(topic: str) -> dict:
    """Compare what the mother knows about a topic vs what's in the graph.

    Returns gaps as structured JSON: mother facts not covered by existing nodes.

    Args:
        topic: Topic to probe

    Returns:
        {topic, mother_facts, graph_coverage, gaps}
    """
    # Search graph for existing coverage
    topic_emb = await embed(topic)
    all_hits = query_all_levels(topic_emb, n_results=5)

    existing_texts = []
    for level, hits in all_hits.items():
        for hit in hits:
            existing_texts.append(hit["content"])

    # Ask mother what it knows
    level_meanings = "\n".join(f"  L{k}: {v}" for k, v in LEVEL_MEANINGS.items())
    prompt = PROBE_PROMPT.format(topic=topic, level_meanings=level_meanings)

    response = await _mother_generate(prompt)
    mother_facts = _parse_json_array(response)

    if not mother_facts:
        return {
            "topic": topic,
            "mother_facts": 0,
            "graph_coverage": 0,
            "gaps": [],
            "error": "mother returned no facts",
        }

    # Check each mother fact against graph coverage
    gaps = []
    covered = 0

    # Pre-embed existing texts once (batch)
    existing_embeddings = []
    if existing_texts:
        existing_embeddings = await embed_batch_parallel(existing_texts[:20])

    for fact in mother_facts:
        if "content" not in fact:
            continue
        content = fact["content"]
        level = int(str(fact.get("level", 3)).lstrip("Ll"))
        level = max(0, min(5, level))

        # Embed and check similarity
        fact_emb = await embed(content)
        is_covered = False

        threshold = DEDUP_THRESHOLD.get(level, 0.85)
        for j, existing_emb in enumerate(existing_embeddings):
            if existing_emb and _cosine_sim(fact_emb, existing_emb) >= threshold:
                is_covered = True
                break

        if is_covered:
            covered += 1
        else:
            gaps.append({
                "content": content,
                "level": level,
            })

    return {
        "topic": topic,
        "mother_facts": len(mother_facts),
        "graph_coverage": covered,
        "gaps": gaps,
    }


async def enrich_node(node_id: int) -> dict:
    """Ask mother what else relates to a node and insert new siblings/children.

    Args:
        node_id: Node to enrich

    Returns:
        {node_id, nodes_created, edges_created, nodes}
    """
    conn = db.get_db()
    node = db.get_node(conn, node_id)
    if not node:
        return {"error": f"Node {node_id} not found"}

    parent = db.get_node(conn, node["parent_id"]) if node.get("parent_id") else None
    children = db.get_children(conn, node_id)
    siblings = db.get_children(conn, node["parent_id"]) if node.get("parent_id") else []

    children_desc = "\n".join(
        f"  L{c['resolution_level']}: {c['content'][:100]}"
        for c in children[:5]
    ) or "  (none)"

    siblings_desc = "\n".join(
        f"  L{s['resolution_level']}: {s['content'][:100]}"
        for s in siblings[:8] if s["id"] != node_id
    ) or "  (none)"

    parent_content = parent["content"][:150] if parent else "(root)"
    parent_level = parent["resolution_level"] if parent else -1

    level_meanings = "\n".join(f"  L{k}: {v}" for k, v in LEVEL_MEANINGS.items())
    prompt = ENRICH_NODE_PROMPT.format(
        content=node["content"][:200],
        level=node["resolution_level"],
        parent_content=parent_content,
        parent_level=parent_level,
        children_desc=children_desc,
        siblings_desc=siblings_desc,
        level_meanings=level_meanings,
    )

    response = await _mother_generate(prompt)
    new_nodes = _parse_json_array(response)

    if not new_nodes:
        return {"node_id": node_id, "nodes_created": 0, "edges_created": 0, "nodes": []}

    # Embed and dedup
    texts = [n["content"] for n in new_nodes if "content" in n]
    embeddings = await embed_batch_parallel(texts)

    created_nodes = []
    edges_created = 0

    async with _write_lock:
        for i, node_data in enumerate(new_nodes):
            if "content" not in node_data:
                continue

            content = node_data["content"]
            placement = node_data.get("placement", "child")
            raw_level = str(node_data.get("level", node["resolution_level"] + 1)).lstrip("Ll")
            level = max(0, min(5, int(raw_level)))
            emb = embeddings[i] if i < len(embeddings) else None
            if not emb:
                continue

            # Determine parent based on placement
            if placement == "sibling" and node.get("parent_id"):
                target_parent_id = node["parent_id"]
            else:
                target_parent_id = node_id

            # Dedup
            if await _is_duplicate(content, emb, target_parent_id, level):
                continue

            # Validate parent
            if target_parent_id and not db.get_node(conn, target_parent_id):
                target_parent_id = None

            # Insert
            new_id = db.insert_node(
                conn, content, resolution_level=level,
                parent_id=target_parent_id, confidence=0.6,
            )
            upsert_node(new_id, content, emb, level, target_parent_id, 0.6)
            created_nodes.append({
                "id": new_id, "level": level,
                "content": content[:100], "placement": placement,
            })

            # Create edges to top-3 most similar siblings (for sibling placements)
            if placement == "sibling":
                scored = []
                for sib in siblings:
                    if sib["id"] == new_id:
                        continue
                    try:
                        col = get_collection(level)
                        result = col.get(ids=[str(sib["id"])])
                        if result and result["embeddings"] and result["embeddings"][0]:
                            sim = _cosine_sim(emb, result["embeddings"][0])
                            scored.append((sim, sib))
                    except Exception:
                        continue
                scored.sort(key=lambda x: x[0], reverse=True)
                for sim, sib in scored[:3]:
                    try:
                        db.insert_edge(
                            conn, sib["id"], new_id,
                            edge_type="related", confidence=round(sim * 0.5, 2),
                            context="Auto-detected sibling relationship",
                        )
                        edges_created += 1
                    except (ValueError, Exception):
                        pass

            recompute_parent_bbox(conn, new_id)

    return {
        "node_id": node_id,
        "nodes_created": len(created_nodes),
        "edges_created": edges_created,
        "nodes": created_nodes,
    }


async def cross_link_nodes(node_id_a: int, node_id_b: int) -> dict:
    """Ask mother whether two nodes are related, create edge if confident.

    Args:
        node_id_a: First node ID
        node_id_b: Second node ID

    Returns:
        {from, to, edge_type, confidence, created}
    """
    conn = db.get_db()
    node_a = db.get_node(conn, node_id_a)
    node_b = db.get_node(conn, node_id_b)

    if not node_a:
        return {"error": f"Node {node_id_a} not found"}
    if not node_b:
        return {"error": f"Node {node_id_b} not found"}

    # Check if edge already exists
    existing = conn.execute(
        "SELECT id FROM edges WHERE "
        "(from_node_id = ? AND to_node_id = ?) OR "
        "(from_node_id = ? AND to_node_id = ?)",
        (node_id_a, node_id_b, node_id_b, node_id_a)
    ).fetchone()
    if existing:
        return {
            "from": node_id_a, "to": node_id_b,
            "created": False, "reason": "edge already exists",
        }

    prompt = CROSS_LINK_PROMPT.format(
        content_a=node_a["content"][:200],
        level_a=node_a["resolution_level"],
        content_b=node_b["content"][:200],
        level_b=node_b["resolution_level"],
    )

    response = await _mother_generate(prompt)
    result = _parse_json_object(response)

    if not result:
        return {
            "from": node_id_a, "to": node_id_b,
            "created": False, "reason": "mother failed to respond",
        }

    related = result.get("related", False)
    edge_type = result.get("edge_type", "related")
    confidence = float(result.get("confidence", 0.0))

    if not related or confidence < 0.3:
        return {
            "from": node_id_a, "to": node_id_b,
            "related": related, "confidence": confidence,
            "created": False,
            "reasoning": result.get("reasoning", ""),
        }

    # Create the edge
    async with _write_lock:
        try:
            db.insert_edge(
                conn, node_id_a, node_id_b,
                edge_type=edge_type, confidence=confidence,
                context=result.get("reasoning", ""),
            )
            created = True
        except (ValueError, Exception) as e:
            created = False
            result["error"] = str(e)

    return {
        "from": node_id_a, "to": node_id_b,
        "edge_type": edge_type,
        "confidence": confidence,
        "created": created,
        "reasoning": result.get("reasoning", ""),
    }


async def auto_crosslink(domain_node_id: int = None,
                        max_pairs: int = 20) -> dict:
    """Batch cross-linking: find node pairs that should be related but aren't.

    Focuses on L1 topic nodes under the same L0 domain. Uses keep_alive="15s"
    to batch through pairs efficiently.

    Args:
        domain_node_id: Optional L0 domain node to scope the scan (null = all domains)
        max_pairs: Maximum node pairs to check

    Returns:
        {pairs_checked, edges_created, details}
    """
    conn = db.get_db()

    # Get candidate pairs: L1 topics under the same L0
    if domain_node_id:
        domain = db.get_node(conn, domain_node_id)
        if not domain:
            return {"error": f"Node {domain_node_id} not found"}
        domains = [domain]
    else:
        domains = db.get_nodes_by_resolution(conn, 0)

    total_checked = 0
    total_edges = 0
    details = []

    for domain in domains:
        topics = db.get_children(conn, domain["id"])
        l1_topics = [t for t in topics if t["resolution_level"] == 1]

        if len(l1_topics) < 2:
            continue

        # Build pairs to check
        topics_desc = "\n".join(
            f"  [{i}] L{t['resolution_level']}: {t['content'][:150]}"
            for i, t in enumerate(l1_topics)
        )

        # Ask mother to find cross-links in batch (one call per domain)
        prompt = AUTO_CROSSLINK_PROMPT.format(
            domain_content=domain["content"][:200],
            topics_desc=topics_desc,
        )

        response = await _mother_generate_keepalive(prompt, keep_alive="0")
        edges_data = _parse_json_array(response)

        if not edges_data:
            total_checked += len(l1_topics)
            continue

        async with _write_lock:
            for edge_info in edges_data:
                if total_edges >= max_pairs:
                    break

                idx_a = edge_info.get("topic_a_index")
                idx_b = edge_info.get("topic_b_index")
                edge_type = edge_info.get("edge_type", "related")
                confidence = float(edge_info.get("confidence", 0.0))

                if (idx_a is None or idx_b is None
                        or idx_a == idx_b
                        or confidence < 0.3):
                    continue
                if idx_a >= len(l1_topics) or idx_b >= len(l1_topics):
                    continue

                node_a = l1_topics[idx_a]
                node_b = l1_topics[idx_b]

                # Check if edge already exists
                existing = conn.execute(
                    "SELECT id FROM edges WHERE "
                    "(from_node_id = ? AND to_node_id = ?) OR "
                    "(from_node_id = ? AND to_node_id = ?)",
                    (node_a["id"], node_b["id"], node_b["id"], node_a["id"])
                ).fetchone()
                if existing:
                    continue

                # Validate edge type
                if edge_type.lower() not in {t.lower() for t in db.EDGE_TYPES}:
                    edge_type = "related"

                try:
                    db.insert_edge(
                        conn, node_a["id"], node_b["id"],
                        edge_type=edge_type, confidence=confidence,
                        context=edge_info.get("reasoning", "")[:200],
                    )
                    total_edges += 1
                    details.append({
                        "from": node_a["id"],
                        "from_content": node_a["content"][:80],
                        "to": node_b["id"],
                        "to_content": node_b["content"][:80],
                        "edge_type": edge_type,
                        "confidence": confidence,
                    })
                except (ValueError, Exception):
                    pass

                total_checked += 1

    return {
        "domains_scanned": len(domains),
        "pairs_checked": total_checked,
        "edges_created": total_edges,
        "details": details,
    }
