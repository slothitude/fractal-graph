"""Context assembler — LaRQL DESCRIBE/WALK equivalent.

Gathers structured context from the fractal graph for the 2B reasoning engine.
The 2B model doesn't need to recall facts — it reasons over provided graph data.
"""

import db
from config import settings
from chroma_store import query_level, query_all_levels
from graph import point_in_bbox


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def gather_context(prompt_embedding: list[float], top_k: int = 10,
                    max_tokens: int = 3000) -> dict:
    """Gather structured context from the graph for a query embedding.

    Steps:
    1. Search all 6 levels via query_all_levels
    2. Fetch full node data from SQLite (content, confidence, edges)
    3. Walk parent chains for hierarchy context
    4. Follow contradiction/support edges to connected nodes
    5. Deduplicate, rank by composite score
    6. Token-budget-aware truncation

    Returns:
        {direct_hits: [...], parents: [...], edges: [...], total_tokens: int}
    """
    conn = db.get_db()
    max_tokens = max_tokens or settings.max_context_tokens

    # Step 1: Search all levels
    level_hits = query_all_levels(prompt_embedding, n_results=top_k)

    # Step 2: Fetch full nodes + edges, collect IDs
    direct_hits = []
    seen_ids = set()
    nodes_by_id = {}

    for level, hits in level_hits.items():
        for hit in hits:
            node_id = int(hit["node_id"])
            if node_id in seen_ids:
                continue
            node = db.get_node(conn, node_id)
            if not node:
                continue
            seen_ids.add(node_id)
            edges = db.get_edges(conn, node_id)
            nodes_by_id[node_id] = node
            direct_hits.append({
                "node": node,
                "distance": hit.get("distance", 1.0),
                "edges": edges,
                "score": 0.0,  # computed below
            })

    # Step 3: Walk parent chains
    parents = []
    parent_seen = set()
    for hit in direct_hits:
        node = hit["node"]
        current = node
        chain = []
        while current.get("parent_id") and current["parent_id"] not in parent_seen:
            pid = current["parent_id"]
            parent = db.get_node(conn, pid)
            if not parent:
                break
            parent_seen.add(pid)
            chain.append(parent)
            nodes_by_id[pid] = parent
            current = parent
        parents.extend(chain)

    # Step 4: Follow contradiction/support edges
    edge_nodes = []
    edge_seen = set()
    interesting_types = {"contradicts", "supports", "challenges", "resolution_conflict"}
    for hit in direct_hits:
        for edge in hit["edges"]:
            edge_type = edge.get("edge_type", "")
            if edge_type.lower() not in interesting_types:
                continue
            for node_id_key in ("from_node_id", "to_node_id"):
                eid = edge[node_id_key]
                if eid in seen_ids or eid in edge_seen:
                    continue
                enode = db.get_node(conn, eid)
                if not enode:
                    continue
                edge_seen.add(eid)
                nodes_by_id[eid] = enode
                edge_nodes.append({
                    "node": enode,
                    "edge_type": edge_type,
                    "context": edge.get("context", ""),
                })

    # Step 5: Rank by composite score
    # 0.4*vector_sim + 0.3*confidence + 0.2*edge_count + 0.1*resolution_relevance
    for hit in direct_hits:
        node = hit["node"]
        vector_sim = max(0, 1.0 - hit["distance"])
        confidence = node.get("confidence", 0.5)
        edge_count = len([e for e in hit["edges"]
                          if e.get("edge_type", "").lower() in interesting_types])
        # Resolution relevance: L4-L5 are most useful for factual answers
        res_relevance = min(node.get("resolution_level", 2) / 5.0, 1.0)
        hit["score"] = (
            0.4 * vector_sim +
            0.3 * confidence +
            0.2 * min(edge_count / 3.0, 1.0) +
            0.1 * res_relevance
        )

    direct_hits.sort(key=lambda x: x["score"], reverse=True)

    # Step 6: Token-budget-aware truncation
    # Priority: direct hits -> parents -> edge-connected nodes
    result = {"direct_hits": [], "parents": [], "edges": [], "total_tokens": 0}
    budget_remaining = max_tokens

    def _add_node(entry, category, node_list):
        nonlocal budget_remaining
        node = entry["node"] if isinstance(entry, dict) and "node" in entry else entry
        content = node.get("content", "")
        edge_info = ""
        if isinstance(entry, dict) and "edge_type" in entry:
            edge_info = f" [{entry['edge_type'].upper()}]"

        text = f"- [ID:{node['id']} L{node['resolution_level']} conf:{node['confidence']:.2f}]{edge_info} {content}"
        tokens = _estimate_tokens(text)
        if tokens <= budget_remaining:
            node_list.append(text)
            budget_remaining -= tokens
            return True
        return False

    # Add direct hits (highest priority)
    for hit in direct_hits:
        if not _add_node(hit, "direct", result["direct_hits"]):
            break

    # Add parent context (lower priority)
    for parent in parents[:5]:
        if not _add_node(parent, "parent", result["parents"]):
            break

    # Add edge-connected nodes (lowest priority)
    for enode in edge_nodes[:5]:
        if not _add_node(enode, "edge", result["edges"]):
            break

    result["total_tokens"] = max_tokens - budget_remaining
    return result


def format_context_for_llm(context: dict) -> str:
    """Format gathered context into structured text for 2B reasoning.

    Uses natural language with section headers — easier for 2B to parse
    than JSON. Organized by resolution level.
    """
    parts = ["=== KNOWLEDGE CONTEXT ==="]

    if context.get("direct_hits"):
        # Group by level
        by_level = {}
        for line in context["direct_hits"]:
            # Parse level from line
            import re
            m = re.search(r"L(\d)", line)
            level = int(m.group(1)) if m else 2
            by_level.setdefault(level, []).append(line)

        level_names = {
            0: "DOMAINS",
            1: "TOPICS",
            2: "CONCEPTS",
            3: "ENTITIES",
            4: "KEY FACTS",
            5: "EVIDENCE",
        }
        for level in sorted(by_level.keys()):
            name = level_names.get(level, f"LEVEL {level}")
            parts.append(f"\n{name}:")
            for line in by_level[level]:
                parts.append(f"  {line}")

    if context.get("parents"):
        parts.append("\nHIERARCHY CONTEXT:")
        for line in context["parents"]:
            parts.append(f"  {line}")

    if context.get("edges"):
        parts.append("\nRELATIONSHIPS:")
        for line in context["edges"]:
            parts.append(f"  {line}")

    parts.append("=== END CONTEXT ===")
    return "\n".join(parts)


def walk_graph(node_id: int, max_hops: int = 3,
               follow_edge_types: list[str] | None = None) -> dict:
    """LaRQL WALK equivalent — traverse edges from a node.

    Args:
        node_id: Starting node
        max_hops: Maximum edge hops
        follow_edge_types: Edge types to follow (None = all)

    Returns:
        {visited_nodes: [...], traversed_edges: [...], root: node}
    """
    conn = db.get_db()
    root = db.get_node(conn, node_id)
    if not root:
        return {"error": f"Node {node_id} not found"}

    visited = {node_id: root}
    visited_edges = []
    frontier = [node_id]

    for _ in range(max_hops):
        next_frontier = []
        for nid in frontier:
            edges = db.get_edges(conn, nid)
            for edge in edges:
                edge_type = edge.get("edge_type", "")
                if follow_edge_types and edge_type.lower() not in [t.lower() for t in follow_edge_types]:
                    continue

                # Follow to connected node
                for key in ("to_node_id", "from_node_id"):
                    other_id = edge[key]
                    if other_id == nid or other_id in visited:
                        continue
                    other = db.get_node(conn, other_id)
                    if not other:
                        continue
                    visited[other_id] = other
                    visited_edges.append(edge)
                    next_frontier.append(other_id)

        if not next_frontier:
            break
        frontier = next_frontier

    return {
        "root": root,
        "visited_nodes": list(visited.values()),
        "traversed_edges": visited_edges,
        "hop_count": max_hops,
    }


def find_best_level_by_structure(embedding: list[float]) -> int:
    """Place text in graph using bbox containment — replaces mother escalation.

    Walks from L0 down, checking if the embedding falls within each
    node's bounding box. Returns the deepest level where containment
    is confirmed, or falls back to heuristic if no bboxes exist.

    Args:
        embedding: Text embedding vector

    Returns:
        Best resolution level (0-5)
    """
    conn = db.get_db()

    # Try structural placement: find deepest bbox that contains this point
    best_level = 0
    for level in range(5, 0, -1):  # Check deepest first
        nodes = db.get_nodes_by_resolution(conn, level)
        for node in nodes:
            bbox = node.get("bbox")
            if bbox and point_in_bbox(embedding, bbox):
                # This node's parent bbox contains us — we belong at this level or deeper
                if level > best_level:
                    best_level = level
                break  # Found containment at this level, stop checking

    # If we found structural containment, place one level deeper
    # (if containment at L2, this content belongs at L3)
    if best_level > 0:
        return min(best_level + 1, 5)

    # No bboxes found — fall back to heuristic
    from ingest import classify_resolution_heuristic
    return classify_resolution_heuristic("embedding-based fallback")
