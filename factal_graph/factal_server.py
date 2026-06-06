"""Fractal Graph MCP Server — Resolution-aware knowledge graph."""

import json

import db
import graph
from config import settings
from query import query as query_fn, drill_down as drill_down_fn, search_nodes as search_nodes_fn
from seed import seed_topic as seed_topic_fn, seed_from_search as seed_from_search_fn, seed_expand as seed_expand_fn
from distill import distill_domain as distill_domain_fn, distill_all as distill_all_fn, distill_coverage as distill_coverage_fn
from judges import judge_topic as judge_topic_fn, judge_answer as judge_answer_fn
from reasoning import answer as answer_fn
from growth import curiosity_scan as curiosity_scan_fn
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fractal-graph")


# ============================================================
# Knowledge Ingest
# ============================================================

@mcp.tool()
async def web_ingest(query: str, max_urls: int = 3) -> str:
    """Search the web and ingest results into the fractal knowledge graph.

    Uses the mother model to extract, classify, structure, and create nodes
    with proper hierarchy. Two-pass: L3-L5 extraction then L0-L2 scaffolding.

    Args:
        query: Search query to find content
        max_urls: Maximum URLs to process (default 3)
    """
    result = await seed_from_search_fn(query, max_urls=max_urls)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def add_node(content: str, resolution_level: int = 2,
                  parent_id: int = None, confidence: float = 0.5,
                  source_url: str = None, metadata: str = None) -> str:
    """Manually add a node to the knowledge graph.

    Args:
        content: Text content of the node
        resolution_level: Resolution level 0-5 (0=domain, 1=topic, 2=concept,
                         3=entity, 4=fact, 5=evidence)
        parent_id: Parent node ID (optional)
        confidence: Confidence score 0.0-1.0
        source_url: Source URL (optional)
        metadata: JSON metadata string (optional)
    """
    conn = db.get_db()
    meta = json.loads(metadata) if metadata else None
    node_id = db.insert_node(
        conn, content, resolution_level=resolution_level, parent_id=parent_id,
        confidence=confidence, source_url=source_url, metadata=meta,
    )

    # Embed and index in ChromaDB
    from embedder import embed
    from chroma_store import upsert_node as chroma_upsert
    try:
        embedding = await embed(content)
        chroma_upsert(node_id, content, embedding, resolution_level,
                      parent_id, confidence, source_url)
    except Exception as e:
        return json.dumps({
            "node_id": node_id,
            "warning": f"ChromaDB indexing failed: {e}",
        })

    return json.dumps({"node_id": node_id, "resolution_level": resolution_level})


@mcp.tool()
async def add_edge(from_node_id: int, to_node_id: int,
                  edge_type: str = "related", confidence: float = 0.5,
                  context: str = None) -> str:
    """Create an edge between two nodes.

    Args:
        from_node_id: Source node ID
        to_node_id: Target node ID
        edge_type: Edge type — related, refines, contradicts, exemplifies,
                  generalizes, challenges, supports, derived_from,
                  resolution_conflict
        confidence: Edge confidence 0.0-1.0
        context: Description of the relationship (optional)
    """
    conn = db.get_db()
    try:
        edge_id = db.insert_edge(
            conn, from_node_id, to_node_id, edge_type, confidence, context,
        )
        return json.dumps({
            "edge_id": edge_id, "from": from_node_id,
            "to": to_node_id, "type": edge_type,
        })
    except ValueError as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def ingest_url(url: str) -> str:
    """Fetch a URL and ingest via the mother model into the knowledge graph.

    Uses the domain from the URL as a search query to find related content,
    then seeds structured nodes via the mother model (two-pass extraction).

    Args:
        url: URL to fetch and ingest
    """
    from urllib.parse import urlparse
    domain = urlparse(url).netloc.replace("www.", "")
    result = await seed_from_search_fn(domain, max_urls=3)
    return json.dumps(result, indent=2, default=str)


# ============================================================
# Mother Model Seeding
# ============================================================

@mcp.tool()
async def seed_topic(topic: str, depth: int = 3, mother_model: str = None) -> str:
    """Seed a knowledge graph topic using the mother model (larger LLM).

    Generates a full multi-resolution hierarchy (L0-L{depth}) with
    proper parent-child relationships, cross-resolution edges, and
    bounding boxes computed bottom-up.

    Args:
        topic: Topic to seed (e.g. "NATO expansion", "quantum computing")
        depth: Maximum resolution depth (1-5, default 3)
        mother_model: Override mother model (default: lfm2.5:latest, ~8B)
    """
    result = await seed_topic_fn(topic, depth=depth, mother_model=mother_model)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def seed_from_search(query: str, max_urls: int = 5,
                           mother_model: str = None) -> str:
    """Search the web and seed structured knowledge via the mother model.

    Searches SearXNG, extracts text from URLs, then feeds everything
    to the mother model which classifies, structures, finds relationships,
    and creates nodes with proper hierarchy including missing level scaffolds.

    Args:
        query: Search query
        max_urls: Maximum URLs to process (default 5)
        mother_model: Override mother model (default: lfm2.5:latest, ~8B)
    """
    result = await seed_from_search_fn(query, max_urls=max_urls,
                                       mother_model=mother_model)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def seed_expand(node_id: int, mother_model: str = None) -> str:
    """Expand an existing sparse node using the mother model.

    Identifies gaps in resolution coverage under the node, generates
    missing intermediate nodes, and recomputes bounding boxes.

    Args:
        node_id: Node ID to expand
        mother_model: Override mother model (default: lfm2.5:latest, ~8B)
    """
    result = await seed_expand_fn(node_id, mother_model=mother_model)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def curiosity_scan(max_expansions: int = settings.curiosity_max_expansions) -> str:
    """Scan the graph for sparse areas and expand them automatically.

    Finds nodes with no children below L5, nodes with few children,
    and high-confidence nodes lacking evidence. Uses the mother model
    via seed_expand to fill gaps. Rate-limited per scan.

    Args:
        max_expansions: Maximum nodes to expand per scan (default 3)
    """
    result = await curiosity_scan_fn(max_expansions=max_expansions)
    return json.dumps(result, indent=2, default=str)


# ============================================================
# Distillation — Mother Model Knowledge Extraction
# ============================================================

@mcp.tool()
async def distill_topic(topic: str, mother_model: str = "") -> str:
    """Distill mother model knowledge about a topic into graph nodes.

    Extracts entities, facts, evidence, and relationships from the mother
    model's parametric knowledge and stores them as structured graph nodes.
    Uses keep_alive to keep mother hot during generation, then batch embeds.

    Args:
        topic: Topic to distill (e.g. "quantum computing", "Roman Empire")
        mother_model: Override mother model (default: lfm2.5:latest)
    """
    model = mother_model or None
    result = await distill_domain_fn(topic, mother_model=model)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def distill_domains(domains: str = "", mother_model: str = "") -> str:
    """Distill multiple domains. Comma-separated list or empty for all.

    If domains is empty, asks the mother to enumerate all knowledge domains
    first (~25-30), then distills each one sequentially.

    Each domain takes ~2 min (15 LLM calls with mother hot, then batch embed).
    Total: ~60 min for all ~30 domains, ~9-15K nodes.

    Args:
        domains: Comma-separated domain list (empty = enumerate all)
        mother_model: Override mother model (default: lfm2.5:latest)
    """
    model = mother_model or None
    domain_list = [d.strip() for d in domains.split(",") if d.strip()] if domains else None
    result = await distill_all_fn(domains=domain_list, mother_model=model)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def distill_coverage() -> str:
    """Show graph coverage — nodes per level, shallow branches, gaps.

    Returns statistics about graph depth, identifies domains that lack
    deep resolution coverage, and lists L0 domains with their children.
    """
    result = distill_coverage_fn()
    return json.dumps(result, indent=2, default=str)


# ============================================================
# Judge Triad
# ============================================================

@mcp.tool()
async def judge_topic(topic: str, top_k: int = 5) -> str:
    """Run the Angel/Devil/Neutral judge triad on a topic (batched).

    All three judges query the graph at their native resolution levels,
    then a single LLM call produces all verdicts + conflict detection.
    A second 2B call synthesizes a final answer from the verdicts.
    Disagreements are logged as resolution_conflict edges.

    Angel sees the forest (L0-L1, optimistic summaries).
    Devil sees the trees (L4-L5, adversarial evidence).
    Neutral bridges the gap (L2-L3, cross-resolution coherence).

    Args:
        topic: Topic to judge (e.g. "NATO expansion", "climate policy")
        top_k: Number of nodes to retrieve per resolution level (default 5)
    """
    triad = await judge_topic_fn(topic, top_k)
    answer = await judge_answer_fn(topic, triad)
    triad["answer"] = answer["answer"]
    triad["answer_confidence"] = answer["confidence"]
    return json.dumps(triad, indent=2, default=str)


# ============================================================
# Query / Retrieval
# ============================================================

@mcp.tool()
async def query_graph(prompt: str, resolution_hint: int = None) -> str:
    """Resolution-aware query — the core innovation.

    Embeds your prompt, finds semantic matches, then drills down
    or stays coarse based on query specificity. Returns the zoom path.

    Args:
        prompt: Your question or query
        resolution_hint: Force a specific resolution level 0-5 (optional)
    """
    result = await query_fn(prompt, resolution_hint)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def drill_down(node_id: int, target_resolution: int = 4) -> str:
    """Drill down from a node to a specific resolution level.

    Returns the full path from the starting node to the deepest reachable
    node at or near the target resolution.

    Args:
        node_id: Starting node ID
        target_resolution: Target resolution level 0-5
    """
    conn = db.get_db()
    result = await drill_down_fn(conn, node_id, target_resolution)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def search_nodes(query_text: str, resolution_level: int = None) -> str:
    """Semantic search for nodes within a resolution level or across all levels.

    Args:
        query_text: Search query
        resolution_level: Specific level 0-5 to search (null = search all)
    """
    results = await search_nodes_fn(query_text, resolution_level)
    return json.dumps(results, indent=2, default=str)


@mcp.tool()
async def get_node(node_id: int) -> str:
    """Get full node details with children, parents, and edges.

    Args:
        node_id: Node ID to retrieve
    """
    conn = db.get_db()
    node = db.get_node(conn, node_id)
    if not node:
        return json.dumps({"error": f"Node {node_id} not found"})

    children = db.get_children(conn, node_id)
    edges = db.get_edges(conn, node_id)
    parent = (db.get_node(conn, node["parent_id"])
              if node.get("parent_id") else None)

    return json.dumps({
        "node": node,
        "parent": parent,
        "children": children,
        "edges": edges,
    }, indent=2, default=str)


# ============================================================
# Ask — 2B Reasoning Pipeline
# ============================================================

@mcp.tool()
async def ask(question: str, auto_expand: bool = True) -> str:
    """Ask a question and get an answer from the 2B reasoning engine.

    The core user-facing tool. The 2B model reasons over structured graph
    context (not recall). Pipeline: embed -> classify (2B) -> gather context
    -> synthesize answer (2B). ~5-8s, 2B only, no mother touched.

    When auto_expand is True and confidence is low, the mother model fills
    knowledge gaps then re-answers with richer context.

    Args:
        question: Your question (e.g. "Why did Russia oppose NATO expansion?")
        auto_expand: If True, fill gaps on low confidence and re-answer (default True)
    """
    result = await answer_fn(question, auto_expand=auto_expand)
    return json.dumps(result, indent=2, default=str)


# ============================================================
# Graph Operations
# ============================================================

@mcp.tool()
async def get_subtree(node_id: int, max_depth: int = 10) -> str:
    """Extract a subtree rooted at a node.

    Self-similar property: any subtree is itself a valid knowledge graph.

    Args:
        node_id: Root node ID
        max_depth: Maximum depth to traverse
    """
    conn = db.get_db()
    subtree = graph.get_subtree(conn, node_id, max_depth)
    if not subtree:
        return json.dumps({"error": f"Node {node_id} not found"})
    return json.dumps(subtree, indent=2, default=str)


@mcp.tool()
async def find_contradictions() -> str:
    """Find all cross-resolution contradictions in the graph.

    Scans for CONTRADICTS, CHALLENGES, and RESOLUTION_CONFLICT edges
    with full node context.
    """
    conn = db.get_db()
    contradictions = graph.find_contradictions(conn)
    return json.dumps({
        "count": len(contradictions),
        "contradictions": contradictions,
    }, indent=2, default=str)


@mcp.tool()
async def propagate_confidence(node_id: int = None) -> str:
    """Propagate confidence changes up the parent chain.

    Args:
        node_id: Starting node ID (optional — null = full bottom-up propagation)
    """
    conn = db.get_db()
    if node_id:
        graph.propagate_confidence(conn, node_id)
        return json.dumps({"status": "propagated", "from_node": node_id})
    else:
        graph.full_propagation(conn)
        return json.dumps({"status": "full_propagation_complete"})


@mcp.tool()
async def graph_stats() -> str:
    """Show graph statistics — nodes and edges per resolution level, embedding coverage."""
    conn = db.get_db()

    node_counts = {}
    for level in range(6):
        count = conn.execute(
            "SELECT COUNT(*) as c FROM nodes WHERE resolution_level = ?", (level,)
        ).fetchone()["c"]
        node_counts[level] = count
    total_nodes = sum(node_counts.values())

    edge_counts = {}
    for row in conn.execute(
        "SELECT edge_type, COUNT(*) as c FROM edges GROUP BY edge_type"
    ):
        edge_counts[row["edge_type"]] = row["c"]
    total_edges = sum(edge_counts.values())

    from chroma_store import get_stats
    chroma_stats = get_stats()

    return json.dumps({
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "nodes_per_level": node_counts,
        "edges_by_type": edge_counts,
        "chromadb": chroma_stats,
    }, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
