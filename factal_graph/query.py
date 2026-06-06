"""Resolution-aware query engine — the core innovation."""

import db
from embedder import embed
from chroma_store import query_level, query_all_levels
from graph import point_in_bbox


def classify_specificity(prompt: str) -> str:
    """Classify prompt specificity using heuristics.

    Returns: 'vague', 'moderate', 'specific', 'very_specific'
    """
    has_dates = any(
        w in prompt.lower()
        for w in ["2020", "2021", "2022", "2023", "2024", "2025", "2026",
                   "jan", "feb", "mar", "apr", "may", "jun",
                   "jul", "aug", "sep", "oct", "nov", "dec"]
    )
    has_names = any(w[0].isupper() for w in prompt.split() if len(w) > 3)
    has_quotes = '"' in prompt or "'" in prompt or "``" in prompt
    short_prompt = len(prompt.split()) < 5
    long_prompt = len(prompt.split()) > 20
    has_what_why = any(
        w in prompt.lower()
        for w in ["what is", "what are", "why", "how does", "explain", "describe"]
    )
    has_specific_q = any(
        w in prompt.lower()
        for w in ["who", "when", "where", "which", "how many", "how much"]
    )

    score = 0
    if has_dates: score += 2
    if has_names: score += 1
    if has_quotes: score += 2
    if has_specific_q: score += 1
    if short_prompt and not has_what_why: score += 1
    if long_prompt: score -= 1
    if has_what_why: score -= 1

    if score >= 3:
        return "very_specific"
    elif score >= 2:
        return "specific"
    elif score >= 1:
        return "moderate"
    else:
        return "vague"


def specificity_to_resolution(specificity: str) -> int:
    """Map specificity to target resolution level."""
    return {
        "vague": 0,
        "moderate": 2,
        "specific": 4,
        "very_specific": 5,
    }.get(specificity, 2)


async def query(prompt: str, resolution_hint: int = None, top_k: int = 5) -> dict:
    """Resolution-aware query — the core innovation.

    1. Embed the prompt
    2. Coarse search at L0
    3. Classify specificity
    4. Drill down, stay coarse, or multi-level pull
    """
    vec = await embed(prompt)

    # Coarse search
    coarse_hits = query_level(vec, 0, n_results=3)

    if not coarse_hits:
        # No L0 nodes yet — search all levels
        all_hits = query_all_levels(vec, n_results=top_k)
        return {
            "query": prompt,
            "strategy": "broad_search",
            "path": [],
            "results": all_hits,
            "resolution_used": "all",
        }

    target_resolution = resolution_hint
    if target_resolution is None:
        specificity = classify_specificity(prompt)
        target_resolution = specificity_to_resolution(specificity)

    if target_resolution == 0:
        return {
            "query": prompt,
            "strategy": "coarse",
            "path": [{"level": 0, "nodes": coarse_hits}],
            "results": {0: coarse_hits},
            "resolution_used": 0,
        }
    elif target_resolution >= 4:
        path = await _drill_down(vec, coarse_hits, target_resolution, top_k)
        return {
            "query": prompt,
            "strategy": "drill_down",
            "path": path,
            "resolution_used": target_resolution,
        }
    else:
        # Multi-level pull
        levels = [0, 2, 4] if target_resolution == 2 else [0, target_resolution]
        results = {}
        for level in levels:
            hits = query_level(vec, level, n_results=top_k)
            if hits:
                results[level] = hits
        return {
            "query": prompt,
            "strategy": "multi_level",
            "levels_pulled": levels,
            "results": results,
            "resolution_used": target_resolution,
        }


async def _drill_down(vec: list[float], start_nodes: list[dict],
                      target_resolution: int, top_k: int = 5) -> list[dict]:
    """Drill down from coarse nodes to target resolution.

    Walks children of each hit node (not the entire next level) and
    supplements with vector similarity at the target level.
    """
    path = []
    current_level = 0
    current_hits = start_nodes
    conn = db.get_db()

    while current_level < target_resolution:
        next_level = current_level + 1
        next_hits = []

        for hit in current_hits:
            node_id = int(hit["node_id"])

            # Walk children of this specific node (hierarchy-aware)
            children = db.get_children(conn, node_id)
            for child in children:
                entry = {
                    "node_id": str(child["id"]),
                    "content": child["content"],
                    "distance": 0.0 if (child.get("bbox") and point_in_bbox(vec, child["bbox"])) else 0.5,
                    "metadata": {
                        "resolution_level": child["resolution_level"],
                        "confidence": child["confidence"],
                    },
                }
                next_hits.append(entry)

        # Supplement with vector search at next level (for nodes without children)
        if not next_hits or current_level == target_resolution - 1:
            level_hits = query_level(vec, next_level, n_results=top_k)
            existing_ids = {h["node_id"] for h in next_hits}
            for h in level_hits:
                if h["node_id"] not in existing_ids:
                    next_hits.append(h)

        if not next_hits:
            break

        # Deduplicate and rank
        seen = set()
        unique = []
        for h in next_hits:
            if h["node_id"] not in seen:
                seen.add(h["node_id"])
                unique.append(h)

        unique.sort(key=lambda x: x.get("distance", 0.5))

        path.append({"level": current_level, "nodes": current_hits})
        current_hits = unique[:top_k]
        current_level = next_level

    path.append({"level": current_level, "nodes": current_hits})
    return path


async def drill_down(conn, node_id: int, target_resolution: int) -> dict:
    """Drill down from a specific node to target resolution level."""
    node = db.get_node(conn, node_id)
    if not node:
        return {"error": f"Node {node_id} not found"}

    path = [{"level": node["resolution_level"], "node": node}]
    current_id = node_id

    while node["resolution_level"] < target_resolution:
        children = db.get_children(conn, current_id)
        if not children:
            break

        best = max(children, key=lambda c: c["confidence"])
        path.append({"level": best["resolution_level"], "node": best})
        current_id = best["id"]
        node = best

    return {"root_id": node_id, "path": path}


async def search_nodes(query_text: str, resolution_level: int = None,
                      top_k: int = 10) -> list[dict]:
    """Semantic search within a resolution level or all levels."""
    vec = await embed(query_text)

    if resolution_level is not None:
        return query_level(vec, resolution_level, n_results=top_k)
    else:
        all_results = query_all_levels(vec, n_results=top_k)
        flat = []
        for level, hits in all_results.items():
            for hit in hits:
                hit["resolution_level"] = level
                flat.append(hit)
        flat.sort(key=lambda x: x["distance"])
        return flat[:top_k]
