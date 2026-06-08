"""LARQL batch seed pipeline — extract knowledge from vindex into fractal graph.

Data flow:
  vindex (DESCRIBE entity)
    -> LARQL edges (entity, relation, target, layer, score)
    -> fractal_graph nodes (L0 domain, L1 topic, L2 concept, L4 fact)
    -> fractal_graph edges (related, refines, etc.)
    -> ChromaDB embeddings

Mapping:
  LARQL relation type (e.g. "capital")  -> L0 domain node (resolution 0)
  LARQL entity (e.g. "France")           -> L1 topic node (child of domain, resolution 1)
  LARQL DESCRIBE edge (entity->target)   -> L2-L4 concept/fact nodes
  LARQL edge score/confidence            -> node confidence
  LARQL layer range                      -> metadata
"""

import asyncio
import json
import logging
from pathlib import Path

import httpx

import db
from db import _write_lock
from config import settings
from embedder import embed, embed_batch_parallel
from chroma_store import upsert_node
from graph import propagate_confidence, recompute_parent_bbox

logger = logging.getLogger(__name__)

LARQL_V1 = f"{settings.larql_server_url}/v1"


async def _larql_get(path: str, params: dict = None, timeout: float = 30.0) -> dict:
    """GET request to LARQL HTTP API."""
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.get(f"{LARQL_V1}{path}", params=params)
        r.raise_for_status()
        return r.json()


async def _larql_post(path: str, json_data: dict = None, timeout: float = 30.0) -> dict:
    """POST request to LARQL HTTP API."""
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(f"{LARQL_V1}{path}", json=json_data)
        r.raise_for_status()
        return r.json()


async def _larql_health() -> bool:
    """Check if LARQL server is reachable."""
    try:
        await _larql_get("/stats", timeout=5.0)
        return True
    except Exception:
        return False


async def extract_relations() -> list[dict]:
    """Run SHOW RELATIONS — list discovered relation types with examples.

    Returns list of dicts with keys: name, count, max_score, min_layer,
    max_layer, examples.
    """
    data = await _larql_get("/relations")
    return data.get("relations", [])


async def extract_entity(entity_name: str) -> dict:
    """Run DESCRIBE on an entity — get all edges.

    Returns dict with keys: entity, edges (list of dicts with target,
    gate_score, layer, feature, relation).
    """
    data = await _larql_get("/describe", params={"entity": entity_name})
    return data


async def extract_stats() -> dict:
    """Get vindex stats (layers, features, extract_level)."""
    return await _larql_get("/stats")


def _ensure_domain(conn, domain_name: str, confidence: float = 0.7) -> int:
    """Find or create an L0 domain node for a relation type."""
    existing = conn.execute(
        "SELECT id FROM nodes WHERE content = ? AND resolution_level = 0",
        (domain_name,),
    ).fetchone()
    if existing:
        return existing["id"]

    node_id = db.insert_node(
        conn, domain_name, resolution_level=0,
        confidence=confidence,
        metadata={"source": "larql", "type": "relation_domain"},
    )
    return node_id


def _ensure_topic(conn, topic_name: str, parent_id: int,
                  confidence: float = 0.6) -> int:
    """Find or create an L1 topic node under a domain."""
    existing = conn.execute(
        "SELECT id FROM nodes WHERE content = ? AND resolution_level = 1 AND parent_id = ?",
        (topic_name, parent_id),
    ).fetchone()
    if existing:
        return existing["id"]

    node_id = db.insert_node(
        conn, topic_name, resolution_level=1, parent_id=parent_id,
        confidence=confidence,
        metadata={"source": "larql"},
    )
    return node_id


def _create_fact_node(conn, content: str, parent_id: int,
                      confidence: float = 0.5,
                      metadata: dict = None) -> int:
    """Create an L4 fact node under a topic."""
    meta = {"source": "larql"}
    if metadata:
        meta.update(metadata)

    node_id = db.insert_node(
        conn, content, resolution_level=4, parent_id=parent_id,
        confidence=confidence, metadata=meta,
    )
    return node_id


async def seed_entity(entity_name: str,
                      default_domain: str = None) -> dict:
    """Extract one entity from vindex and create fractal graph nodes.

    Returns dict with created node IDs, edge count, etc.
    """
    if not await _larql_health():
        return {"error": "LARQL server unreachable",
                "url": settings.larql_server_url}

    entity_data = await extract_entity(entity_name)
    edges = entity_data.get("edges", [])

    if not edges:
        return {"entity": entity_name, "edges": 0, "note": "No edges found in vindex"}

    conn = db.get_db()
    created_nodes = []
    created_edges = []

    # Group edges by relation type to create domain structure
    relation_groups: dict[str, list] = {}
    for edge in edges:
        rel = edge.get("relation", "related")
        if not rel:
            rel = "related"
        relation_groups.setdefault(rel, []).append(edge)

    for rel_type, rel_edges in relation_groups.items():
        # L0: Domain node for relation type
        domain_id = _ensure_domain(conn, rel_type, confidence=0.7)

        # L1: Entity topic under domain
        topic_id = _ensure_topic(conn, entity_name, parent_id=domain_id,
                                  confidence=0.6)
        created_nodes.append({"id": topic_id, "content": entity_name,
                              "level": 1, "domain": rel_type})

        # L4: Fact nodes for each edge
        for edge in rel_edges:
            target = edge.get("target", "unknown")
            score = edge.get("gate_score", 0.5)
            layer = edge.get("layer")
            feature = edge.get("feature")

            fact_content = f"{entity_name} {rel_type} {target}"
            confidence = min(1.0, max(0.1, score))

            fact_id = _create_fact_node(
                conn, fact_content, parent_id=topic_id,
                confidence=confidence,
                metadata={
                    "larql_target": target,
                    "larql_layer": layer,
                    "larql_feature": feature,
                    "larql_score": score,
                },
            )
            created_nodes.append({"id": fact_id, "content": fact_content,
                                  "level": 4, "confidence": confidence})
            created_edges.append({
                "from": topic_id, "to": fact_id, "type": "refines",
            })

    # Embed new nodes in ChromaDB
    texts = [n["content"] for n in created_nodes]
    try:
        embeddings = await embed_batch_parallel(texts)
        for node, emb in zip(created_nodes, embeddings):
            if emb:
                upsert_node(
                    node["id"], node["content"], emb,
                    node["level"], confidence=node.get("confidence", 0.5),
                    metadata={"source": "larql"},
                )
    except Exception as e:
        logger.warning(f"ChromaDB embedding failed for {entity_name}: {e}")

    # Propagate confidence
    for node in created_nodes:
        propagate_confidence(conn, node["id"])
        recompute_parent_bbox(conn, node["id"])

    db._retry_commit(conn)

    return {
        "entity": entity_name,
        "edges_found": len(edges),
        "nodes_created": len(created_nodes),
        "relation_types": list(relation_groups.keys()),
        "nodes": created_nodes,
    }


async def batch_seed(topics: list[str],
                     max_per_topic: int = 20) -> dict:
    """Seed multiple topics from the vindex into the fractal graph.

    Args:
        topics: List of entity names to extract (e.g. ["France", "Einstein"])
        max_per_topic: Max edges to process per topic

    Returns:
        Summary dict with per-topic results.
    """
    if not await _larql_health():
        return {"error": "LARQL server unreachable",
                "url": settings.larql_server_url}

    results = []
    total_nodes = 0
    total_edges = 0

    for topic in topics:
        try:
            result = await seed_entity(topic)
            results.append(result)
            if "nodes_created" in result:
                total_nodes += result["nodes_created"]
            if "edges_found" in result:
                total_edges += result["edges_found"]
        except Exception as e:
            logger.error(f"Failed to seed {topic}: {e}")
            results.append({"entity": topic, "error": str(e)})

    return {
        "total_topics": len(topics),
        "total_nodes_created": total_nodes,
        "total_edges_found": total_edges,
        "results": results,
    }


# CLI entry point
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Batch seed fractal graph from LARQL vindex")
    parser.add_argument("--topics", type=str, default="",
                        help="Comma-separated entity names (e.g. 'France,Einstein,Python')")
    parser.add_argument("--vindex", type=str, default="",
                        help="Path to vindex directory")
    parser.add_argument("--url", type=str, default="",
                        help="LARQL server URL (default from config)")
    parser.add_argument("--all-relations", action="store_true",
                        help="Seed all discovered relation types")
    args = parser.parse_args()

    if args.url:
        settings.larql_server_url = args.url
    if args.vindex:
        settings.larql_vindex_path = args.vindex

    async def main():
        health = await _larql_health()
        if not health:
            print(f"ERROR: LARQL server unreachable at {settings.larql_server_url}")
            return

        stats = await extract_stats()
        print(f"Vindex stats: {json.dumps(stats, indent=2)}")

        if args.all_relations:
            relations = await extract_relations()
            print(f"Found {len(relations)} relation types")
            for rel in relations[:5]:
                print(f"  - {rel.get('name')}: {rel.get('count', '?')} edges")
            return

        if args.topics:
            topics = [t.strip() for t in args.topics.split(",") if t.strip()]
        else:
            topics = ["France", "Einstein", "Python", "Quantum mechanics",
                      "Photosynthesis", "Shakespeare"]

        print(f"Seeding {len(topics)} topics from vindex...")
        result = await batch_seed(topics)
        print(json.dumps(result, indent=2, default=str))

    asyncio.run(main())
