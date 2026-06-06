"""Graph operations — traversal, contradictions, confidence propagation."""

import db


def compute_bbox(children_embeddings: list[list[float]]) -> list[float] | None:
    """Compute bounding box from child embeddings as [min_x, min_y, ..., max_x, max_y, ...]."""
    if not children_embeddings:
        return None
    dims = len(children_embeddings[0])
    bbox = [float("inf")] * dims + [float("-inf")] * dims
    for emb in children_embeddings:
        for i in range(dims):
            bbox[i] = min(bbox[i], emb[i])
            bbox[i + dims] = max(bbox[i + dims], emb[i])
    return bbox


def point_in_bbox(point: list[float], bbox: list[float]) -> bool:
    """Check if a point is within a bounding box."""
    dims = len(point)
    for i in range(dims):
        if point[i] < bbox[i] or point[i] > bbox[i + dims]:
            return False
    return True


def get_subtree(conn, node_id: int, max_depth: int = 10) -> dict | None:
    """Extract a subtree rooted at a node (self-similar property)."""

    def _build(nid, depth):
        node = db.get_node(conn, nid)
        if not node or depth > max_depth:
            return None
        children = db.get_children(conn, nid)
        return {
            "node": node,
            "children": [_build(c["id"], depth + 1) for c in children]
        }

    return _build(node_id, 0)


def find_contradictions(conn) -> list[dict]:
    """Find all CONTRADICTS and CHALLENGES edges with full node context."""
    rows = conn.execute(
        """SELECT e.*, nf.content as from_content, nf.resolution_level as from_level,
                  nt.content as to_content, nt.resolution_level as to_level
           FROM edges e
           JOIN nodes nf ON e.from_node_id = nf.id
           JOIN nodes nt ON e.to_node_id = nt.id
           WHERE e.edge_type IN ('contradicts', 'challenges')"""
    ).fetchall()
    return [dict(row) for row in rows]


def propagate_confidence(conn, node_id: int):
    """Push confidence changes up the parent chain."""
    node = db.get_node(conn, node_id)
    if not node or node["parent_id"] is None:
        return

    chain = []
    current_id = node_id
    while current_id:
        current = db.get_node(conn, current_id)
        if not current:
            break
        chain.append(current)
        current_id = current.get("parent_id")

    # Propagate: parent confidence = average of children
    for parent in chain[1:]:
        children = db.get_children(conn, parent["id"])
        if children:
            avg_conf = sum(c["confidence"] for c in children) / len(children)
            if abs(avg_conf - parent["confidence"]) > 0.01:
                db.update_node_confidence(conn, parent["id"], round(avg_conf, 3))


def full_propagation(conn):
    """Run confidence propagation across entire graph, bottom-up."""
    nodes = conn.execute(
        "SELECT * FROM nodes ORDER BY resolution_level DESC"
    ).fetchall()

    for node in nodes:
        parent_id = node["parent_id"]
        if parent_id is None:
            continue

        children = db.get_children(conn, parent_id)
        if children:
            avg_conf = sum(c["confidence"] for c in children) / len(children)
            db.update_node_confidence(conn, parent_id, round(avg_conf, 3))


def recompute_parent_bbox(conn, node_id: int):
    """Recompute bounding box for the parent of the given node.

    Walks up the parent chain, recomputing bbox from children embeddings
    at each level. Uses ChromaDB to retrieve child embeddings.
    """
    current_id = node_id
    while current_id:
        node = db.get_node(conn, current_id)
        if not node or node["parent_id"] is None:
            break

        parent_id = node["parent_id"]
        children = db.get_children(conn, parent_id)
        if not children:
            break

        # Gather embeddings for all children
        from chroma_store import get_collection
        child_embeddings = []
        for child in children:
            try:
                col = get_collection(child["resolution_level"])
                result = col.get(ids=[str(child["id"])])
                if result and result["embeddings"] and result["embeddings"][0]:
                    child_embeddings.append(result["embeddings"][0])
            except Exception:
                pass

        if child_embeddings:
            bbox = compute_bbox(child_embeddings)
            if bbox:
                db.update_node_bbox(conn, parent_id, bbox)

        current_id = parent_id
