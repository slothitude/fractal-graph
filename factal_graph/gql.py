"""Structured graph query primitives — LaRQL-inspired.

Composable query primitives called internally by the answer pipeline
based on question classification. Not a full query language — the 2B
classifies intent from natural language; these cover the 80% cases.
"""

import db


def find_contradictions_of(entity_id: int) -> list[dict]:
    """DESCRIBE WHERE contradicts — find all contradictions for an entity.

    Returns contradiction edges with full node content on both sides.
    """
    conn = db.get_db()
    node = db.get_node(conn, entity_id)
    if not node:
        return []

    contradictions = []
    interesting = {"contradicts", "challenges", "resolution_conflict"}

    edges = db.get_edges(conn, entity_id)
    for edge in edges:
        if edge.get("edge_type", "").lower() not in interesting:
            continue

        # Get the OTHER node in this edge
        other_id = (edge["to_node_id"]
                     if edge["from_node_id"] == entity_id
                     else edge["from_node_id"])
        other = db.get_node(conn, other_id)
        if not other:
            continue

        contradictions.append({
            "node_id": other_id,
            "content": other["content"],
            "resolution_level": other["resolution_level"],
            "confidence": other["confidence"],
            "edge_type": edge["edge_type"],
            "edge_confidence": edge.get("confidence", 0.5),
            "context": edge.get("context", ""),
        })

    return contradictions


def find_evidence_for(fact_id: int) -> list[dict]:
    """WALK FOLLOW supports/contradicts — find evidence for or against a fact.

    Walks edges from a fact node to find supporting and contradicting evidence.
    """
    conn = db.get_db()
    node = db.get_node(conn, fact_id)
    if not node:
        return []

    evidence = []
    edges = db.get_edges(conn, fact_id)

    for edge in edges:
        etype = edge.get("edge_type", "").lower()
        if etype not in ("supports", "contradicts", "challenges"):
            continue

        other_id = (edge["to_node_id"]
                     if edge["from_node_id"] == fact_id
                     else edge["from_node_id"])
        other = db.get_node(conn, other_id)
        if not other:
            continue

        evidence.append({
            "node_id": other_id,
            "content": other["content"],
            "resolution_level": other["resolution_level"],
            "confidence": other["confidence"],
            "relationship": etype,
            "edge_confidence": edge.get("confidence", 0.5),
            "context": edge.get("context", ""),
        })

    # Sort: supports first, then contradicts
    evidence.sort(key=lambda x: (0 if x["relationship"] == "supports" else 1,
                                  -x["confidence"]))
    return evidence


def compare_entities(entity_a_id: int, entity_b_id: int) -> dict:
    """DESCRIBE "X" vs "Y" — compare two entities using graph context.

    Gathers facts, evidence, and contradictions for both entities,
    then checks for direct edges between them.
    """
    conn = db.get_db()
    node_a = db.get_node(conn, entity_a_id)
    node_b = db.get_node(conn, entity_b_id)

    if not node_a or not node_b:
        missing = []
        if not node_a:
            missing.append(entity_a_id)
        if not node_b:
            missing.append(entity_b_id)
        return {"error": f"Nodes not found: {missing}"}

    # Gather evidence for each
    evidence_a = find_evidence_for(entity_a_id)
    evidence_b = find_evidence_for(entity_b_id)

    # Check for direct edges between them
    direct_edges = []
    for edge in db.get_edges(conn, entity_a_id):
        if edge["to_node_id"] == entity_b_id or edge["from_node_id"] == entity_b_id:
            direct_edges.append({
                "type": edge["edge_type"],
                "confidence": edge.get("confidence", 0.5),
                "context": edge.get("context", ""),
            })

    # Get hierarchy context
    def _get_ancestry(nid):
        chain = []
        current = db.get_node(conn, nid)
        while current and current.get("parent_id"):
            parent = db.get_node(conn, current["parent_id"])
            if parent:
                chain.append({"id": parent["id"], "content": parent["content"],
                              "level": parent["resolution_level"]})
            current = parent
        return chain

    return {
        "entity_a": {
            "id": entity_a_id,
            "content": node_a["content"],
            "level": node_a["resolution_level"],
            "confidence": node_a["confidence"],
            "evidence_count": len(evidence_a),
            "supporting": [e for e in evidence_a if e["relationship"] == "supports"],
            "contradicting": [e for e in evidence_a if e["relationship"] in ("contradicts", "challenges")],
            "ancestry": _get_ancestry(entity_a_id),
        },
        "entity_b": {
            "id": entity_b_id,
            "content": node_b["content"],
            "level": node_b["resolution_level"],
            "confidence": node_b["confidence"],
            "evidence_count": len(evidence_b),
            "supporting": [e for e in evidence_b if e["relationship"] == "supports"],
            "contradicting": [e for e in evidence_b if e["relationship"] in ("contradicts", "challenges")],
            "ancestry": _get_ancestry(entity_b_id),
        },
        "direct_edges": direct_edges,
    }
