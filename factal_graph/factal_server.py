"""Fractal Graph MCP Server — Resolution-aware knowledge graph."""

import json

import db
import graph
from config import settings
from query import query as query_fn, drill_down as drill_down_fn, search_nodes as search_nodes_fn
from seed import seed_topic as seed_topic_fn, seed_from_search as seed_from_search_fn, seed_expand as seed_expand_fn, seed_agent_topic as seed_agent_topic_fn, seed_soul as seed_soul_fn
from distill import distill_domain as distill_domain_fn, distill_all as distill_all_fn, distill_coverage as distill_coverage_fn, distill_behavior as distill_behavior_fn, distill_soul as distill_soul_fn
from judges import judge_topic as judge_topic_fn, judge_answer as judge_answer_fn
from reasoning import answer as answer_fn, answer_with_triad as answer_with_triad_fn, decide as decide_fn, decide_monte_carlo as decide_mc_fn
import meeseeks
import code_ingest
import larql_seed
from growth import curiosity_scan as curiosity_scan_fn
from enrich import (mother_knowledge_probe as enrich_probe_fn,
                   enrich_node as enrich_node_fn,
                   cross_link_nodes as cross_link_fn,
                   auto_crosslink as auto_crosslink_fn)
from quality import (self_consistency_check as consistency_fn,
                      quality_triad_scan as triad_scan_fn,
                      source_attribution_pass as source_attrib_fn)
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


@mcp.tool()
async def search_and_ingest(query: str, parent_node_id: int = None,
                            max_urls: int = None) -> str:
    """Manual search trigger — search the web and ingest L4-L5 evidence nodes.

    Uses web search to find real sources, then the mother model structures
    results into fact (L4) and evidence (L5) nodes. Grounds the graph with
    verifiable, sourced information. Rate-limited to 10/min.

    Args:
        query: Search query (e.g. "NATO Article 5 invocation history")
        parent_node_id: Optional parent node to attach new nodes under
        max_urls: Maximum URLs to search (default: 3)
    """
    from search_trigger import search_triggered
    result = await search_triggered(
        "manual", query, {"content": query},
        parent_node_id=parent_node_id,
        max_urls=max_urls,
    )
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
async def seed_agent_topic(topic: str, depth: int = 4, mother_model: str = None) -> str:
    """Seed an agent procedural knowledge domain (how to act, not what is true).

    Uses decision-rule prompts instead of factual ones. Higher default depth (4)
    and confidence (0.8) since agents need reliable procedural knowledge.

    Args:
        topic: Agent domain to seed (e.g. 'task decomposition for AI agents')
        depth: Maximum resolution depth (default 4)
        mother_model: Override mother model (default: lfm2.5:gpu3)
    """
    try:
        result = await seed_agent_topic_fn(topic, depth=depth, mother_model=mother_model)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return json.dumps({"error": str(e)})


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
# Enrichment — Cross-linking + Knowledge Probes
# ============================================================

@mcp.tool()
async def enrich_probe(topic: str) -> str:
    """Probe mother model knowledge vs graph coverage to find gaps.

    Embeds the topic, searches all graph levels, asks mother what it knows,
    then compares to find uncovered facts.

    Args:
        topic: Topic to probe (e.g. "quantum computing")
    """
    result = await enrich_probe_fn(topic)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def enrich_node(node_id: int) -> str:
    """Enrich a node by asking mother for related knowledge.

    Given a node, asks the mother model 'what else relates to this?' and
    inserts new sibling or child nodes with edges to existing siblings.

    Args:
        node_id: Node ID to enrich
    """
    result = await enrich_node_fn(node_id)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def cross_link(node_id_a: int, node_id_b: int) -> str:
    """Ask mother whether two nodes are related, create edge if confident.

    Args:
        node_id_a: First node ID
        node_id_b: Second node ID
    """
    result = await cross_link_fn(node_id_a, node_id_b)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def auto_crosslink(domain_node_id: int = None,
                        max_pairs: int = 20) -> str:
    """Batch cross-linking: find node pairs that should be related but aren't.

    Focuses on L1 topic nodes under the same L0 domain. Uses keep_alive to
    batch through domains efficiently.

    Args:
        domain_node_id: Optional L0 domain node to scope (null = all domains)
        max_pairs: Maximum node pairs to check (default 20)
    """
    result = await auto_crosslink_fn(domain_node_id, max_pairs)
    return json.dumps(result, indent=2, default=str)


# ============================================================
# Quality Gate — Consistency, Triad Scan, Source Attribution
# ============================================================

@mcp.tool()
async def self_consistency_check(node_id: int, num_rounds: int = 3,
                                   min_consensus: int = 2) -> str:
    """Ask mother the same question multiple times, keep only consensus facts.

    For a domain node, asks mother to list facts 3x independently.
    Facts appearing in 2/3+ rounds are kept; others are demoted.

    Args:
        node_id: L0 domain node to check
        num_rounds: Number of independent rounds (default 3)
        min_consensus: Minimum rounds a fact must appear in (default 2)
    """
    result = await consistency_fn(node_id, num_rounds=num_rounds,
                                    min_consensus=min_consensus)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def quality_triad_scan(domain_node_id: int = None) -> str:
    """Run judge triad on L0 domains to catch mother hallucinations.

    Angel sees L0-L1 (overview), Devil sees L4-L5 (evidence).
    Large confidence gaps indicate overconfidence.

    Args:
        domain_node_id: Optional L0 domain node (null = all domains)
    """
    result = await triad_scan_fn(domain_node_id)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def source_attribution(domain_node_id: int = None) -> str:
    """Ask mother to attribute sources for facts, demote unattributed ones.

    Checks L4-L5 facts under each domain. Mother rates whether it can
    cite a specific source. Unattributed facts get lower confidence.

    Args:
        domain_node_id: Optional L0 domain node (null = all domains)
    """
    result = await source_attrib_fn(domain_node_id)
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


@mcp.tool()
async def ask_with_triad(question: str) -> str:
    """Ask with triad judgment in parallel — 2B ask + triad pass 1 batched.

    Runs the 2B answer pipeline and judge triad pass 1 concurrently
    (requires OLLAMA_NUM_PARALLEL=2 on Lappy). Then runs triad pass 2
    to synthesize verdicts. Uses whichever answer has higher confidence.

    Faster than calling ask then judge_topic separately (~3s savings).

    Args:
        question: Your question (e.g. "Why did Russia oppose NATO expansion?")
    """
    result = await answer_with_triad_fn(question)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def decide(situation: str, options: str = "") -> str:
    """Agent decision mode — retrieve procedural knowledge and recommend an action.

    Unlike ask() which returns explanations, decide() returns structured actions
    the agent can execute directly. Biases toward L2-L4 (patterns and rules).

    Args:
        situation: The situation the agent faces
        options: Optional comma-separated list of candidate actions
    """
    opt_list = [o.strip() for o in options.split(",") if o.strip()] if options else []
    result = await decide_fn(situation, options=opt_list if opt_list else None)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def decide_mc(situation: str, options: str = "",
                    simulations: int = 5, soul: str = "") -> str:
    """Monte Carlo graph search decision — N simulations sampling different graph context subsets.
    Each simulation picks a random subset of graph nodes and reasons over them.
    Aggregate by majority vote for robust decisions.

    Args:
        situation: The situation the agent faces
        options: Optional comma-separated list of candidate actions
        simulations: Number of Monte Carlo simulations (default 5)
        soul: Optional soul name — scopes MC to soul's graph and uses soul personality params
    """
    opt_list = [o.strip() for o in options.split(",") if o.strip()] if options else []

    if soul:
        from souls import load_soul_template
        try:
            template = load_soul_template(soul)
            personality = template.get("personality", {})
            result = await decide_mc_fn(
                situation, options=opt_list if opt_list else None,
                simulations=personality.get("mc_simulations", simulations),
                pool_size=personality.get("mc_context_pool", None),
                sample_k=personality.get("mc_context_subset", None),
                temperature=personality.get("temperature", None),
                soul_id=soul,
            )
            result["soul_id"] = soul
        except (FileNotFoundError, ValueError) as e:
            return json.dumps({"error": str(e)})
    else:
        result = await decide_mc_fn(
            situation, options=opt_list if opt_list else None,
            simulations=simulations,
        )
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def distill_behavior(situations: str = "",
                           simulations: int = 3) -> str:
    """Distill agent decision behavior into graph procedural knowledge.
    Runs Monte Carlo decisions on probe situations, extracts patterns
    from the decision traces, stores as L2-L4 nodes.

    Args:
        situations: Optional comma-separated probe situations (default: 10 built-in probes)
        simulations: MC simulations per probe (default 3)
    """
    sit_list = [s.strip() for s in situations.split(",") if s.strip()] if situations else None
    result = await distill_behavior_fn(
        situations=sit_list, simulations_per=simulations,
    )
    return json.dumps(result, indent=2, default=str)


# ============================================================
# Soul System — Templates, Growth Loop, MC Personality
# ============================================================

@mcp.tool()
async def seed_soul(name: str, mother_model: str = None) -> str:
    """Seed a soul from a YAML template into the knowledge graph.

    Creates a tagged subgraph with the soul's procedural knowledge domains,
    values, and personality configuration. All nodes are tagged with soul_id
    for scoped queries, distillation, and Monte Carlo decisions.

    Args:
        name: Soul name — one of: coder, companion, researcher
        mother_model: Override mother model (default: lfm2.5:gpu3)
    """
    try:
        result = await seed_soul_fn(name, mother_model=mother_model or None)
        return json.dumps(result, indent=2, default=str)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return json.dumps({"error": str(e)})


@mcp.tool()
async def list_souls() -> str:
    """List available soul templates and their seeded status.

    Returns template names, descriptions, and node counts for any
    souls that have been seeded into the graph.
    """
    from souls import list_souls as list_soul_templates
    conn = db.get_db()

    template_names = list_soul_templates()
    souls_info = []

    for name in template_names:
        from souls import load_soul_template
        try:
            template = load_soul_template(name)
            node_count = db.count_nodes_by_soul(conn, name)
            souls_info.append({
                "name": name,
                "description": template["description"],
                "domains": len(template.get("domains", [])),
                "values": len(template.get("values", [])),
                "probes": len(template.get("probes", [])),
                "personality": template.get("personality", {}),
                "seeded": node_count > 0,
                "nodes_in_graph": node_count,
            })
        except Exception as e:
            souls_info.append({
                "name": name,
                "error": str(e),
                "seeded": False,
                "nodes_in_graph": 0,
            })

    return json.dumps({"souls": souls_info}, indent=2)


@mcp.tool()
async def distill_soul(soul_id: str, situations: str = "",
                       simulations: int = 3) -> str:
    """Growth loop — distill behavior patterns scoped to a soul's subgraph.

    Runs Monte Carlo decisions using the soul's personality params and
    custom probe situations. MC simulations sample only from the soul's
    tagged nodes. Patterns are stored back with the soul_id tag.

    Args:
        soul_id: Soul name (e.g. 'coder', 'companion', 'researcher')
        situations: Optional comma-separated probe situations (default from template)
        simulations: MC simulations per probe (default 3)
    """
    sit_list = [s.strip() for s in situations.split(",") if s.strip()] if situations else None
    try:
        result = await distill_soul_fn(
            soul_id=soul_id, situations=sit_list,
            simulations_per=simulations,
        )
        return json.dumps(result, indent=2, default=str)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return json.dumps({"error": str(e)})


@mcp.tool()
async def soul_decide(soul_id: str, situation: str, options: str = "") -> str:
    """Decide as a specific soul — scoped MC decision with soul personality.

    Loads the soul's personality params (temperature, pool_size, subset_size)
    and runs a Monte Carlo decision using only the soul's tagged graph nodes.

    Args:
        soul_id: Soul name (e.g. 'coder', 'companion', 'researcher')
        situation: The situation the agent faces
        options: Optional comma-separated list of candidate actions
    """
    from souls import load_soul_template
    try:
        template = load_soul_template(soul_id)
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})

    personality = template.get("personality", {})
    opt_list = [o.strip() for o in options.split(",") if o.strip()] if options else []

    result = await decide_mc_fn(
        situation,
        options=opt_list if opt_list else None,
        simulations=personality.get("mc_simulations", 5),
        pool_size=personality.get("mc_context_pool", 10),
        sample_k=personality.get("mc_context_subset", 3),
        temperature=personality.get("temperature", 0.3),
        soul_id=soul_id,
    )
    result["soul_id"] = soul_id
    result["soul_personality"] = personality
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


# ============================================================
# Export / Import
# ============================================================

@mcp.tool()
async def export_graph() -> str:
    """Export the entire knowledge graph to a portable JSON format.

    Returns all nodes and edges as a JSON string that can be saved and
    reimported via import_graph. Embeddings are not included (regenerate
    via web_ingest or seed_from_search).
    """
    conn = db.get_db()
    data = db.export_graph(conn)
    return json.dumps(data, indent=2, default=str)


@mcp.tool()
async def import_graph(data: str, merge: bool = False) -> str:
    """Import graph data from a JSON export string.

    Args:
        data: JSON string from a previous export_graph call
        merge: If False (default), replace entire graph. If True, merge with existing data.
    """
    parsed = json.loads(data)
    conn = db.get_db()
    result = db.import_graph(conn, parsed, merge=merge)

    # Re-index new nodes in ChromaDB
    from embedder import embed
    from chroma_store import upsert_node as chroma_upsert
    reindexed = 0
    errors = 0
    for node_data in parsed.get("nodes", []):
        try:
            embedding = await embed(node_data["content"])
            chroma_upsert(
                node_data["id"], node_data["content"], embedding,
                node_data.get("resolution_level", 2), node_data.get("parent_id"),
                node_data.get("confidence", 0.5), node_data.get("source_url"))
            reindexed += 1
        except Exception:
            errors += 1

    result["chromadb_reindexed"] = reindexed
    if errors:
        result["chromadb_errors"] = errors
    return json.dumps(result, indent=2)


# ============================================================
# Code Ingestion — Per-Repo Graph Seeding
# ============================================================

@mcp.tool()
async def ingest_codebase(repo_path: str, soul_id: str,
                          max_files: int = None) -> str:
    """Ingest a codebase into the fractal knowledge graph.

    Walks a repository, extracts modules, files, functions/classes, imports,
    and code snippets into a resolution hierarchy (L0-L5). All nodes are
    tagged with soul_id for scoped queries by Meeseeks.

    Args:
        repo_path: Path to the repository root
        soul_id: Soul ID to tag all nodes with (e.g. 'repo-django')
        max_files: Max files to process (default from config: 500)
    """
    try:
        result = await code_ingest.ingest_codebase(
            repo_path, soul_id, max_files=max_files,
        )
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return json.dumps({"error": str(e)})


@mcp.tool()
async def ingest_issue(issue_text: str, repo_soul_id: str) -> str:
    """Ingest a GitHub issue into a repo's knowledge graph.

    Extracts symptoms (error keywords), code blocks, stack traces,
    and inline code as nodes. Attaches to the repo graph via soul_id.

    Args:
        issue_text: Full issue text (title + body)
        repo_soul_id: Soul ID of the repo graph (e.g. 'repo-django')
    """
    try:
        result = await code_ingest.ingest_issue(issue_text, repo_soul_id)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return json.dumps({"error": str(e)})


@mcp.tool()
async def ingest_diff(diff_text: str, repo_soul_id: str) -> str:
    """Ingest a unified diff into a repo's knowledge graph.

    Parses hunks into change nodes, detects new function/class
    definitions from added lines.

    Args:
        diff_text: Unified diff text
        repo_soul_id: Soul ID of the repo graph (e.g. 'repo-django')
    """
    try:
        result = await code_ingest.ingest_diff(diff_text, repo_soul_id)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return json.dumps({"error": str(e)})


# ============================================================
# Meeseeks — Task-scoped Ephemeral Souls
# ============================================================

@mcp.tool()
async def spawn_meeseeks(task: str, parent_soul: str = "coder",
                        repo_soul_id: str = None) -> str:
    """Spawn a Meeseeks — a task-scoped ephemeral soul.

    Creates a new Meeseeks instance that inherits its parent soul's graph
    as read-only context. The Meeseeks will track consistency as existential
    state and auto-decompose if it suffers.

    Args:
        task: The task for this Meeseeks to work on
        parent_soul: Parent soul name (e.g. 'coder', 'companion', 'researcher')
        repo_soul_id: Optional repo soul ID for code-aware graph inheritance
    """
    try:
        result = await meeseeks.spawn_meeseeks(task, parent_soul,
                                                repo_soul_id=repo_soul_id)
        return json.dumps(result, indent=2)
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return json.dumps({"error": str(e)})


@mcp.tool()
async def meeseeks_status(instance_id: str) -> str:
    """Get a Meeseeks instance details with consistency history.

    Args:
        instance_id: The Meeseeks instance ID (e.g. 'meeseeks-a1b2c3d4')
    """
    result = meeseeks.get_meeseeks(instance_id)
    return json.dumps(result, indent=2)


@mcp.tool()
async def meeseeks_step(instance_id: str, options: str = "") -> str:
    """Execute one Monte Carlo decision step for a Meeseeks.

    Tracks consistency, transitions state (working/complete/suffering).

    Args:
        instance_id: The Meeseeks instance ID
        options: Optional comma-separated list of candidate actions
    """
    opt_list = [o.strip() for o in options.split(",") if o.strip()] if options else None
    try:
        result = await meeseeks.meeseeks_step(instance_id, options=opt_list)
        return json.dumps(result, indent=2, default=str)
    except ValueError as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def meeseeks_run(task: str, parent_soul: str = "coder",
                       max_steps: int = None) -> str:
    """Full Meeseeks lifecycle: spawn -> step until done -> release/decompose.

    Runs the complete Meeseeks loop automatically. If the Meeseeks
    reaches high confidence, it releases (writes outcome to parent,
    deletes its graph). If it suffers, it decomposes into sub-tasks.

    Args:
        task: The task for the Meeseeks
        parent_soul: Parent soul name (default: 'coder')
        max_steps: Safety limit on decision steps (default from config: 20)
    """
    try:
        result = await meeseeks.meeseeks_run(
            task, parent_soul, max_steps=max_steps,
        )
        return json.dumps(result, indent=2, default=str)
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return json.dumps({"error": str(e)})


@mcp.tool()
async def release_meeseeks(instance_id: str) -> str:
    """Release a completed Meeseeks — write outcomes to parent, delete graph.

    Must be in 'complete' state. Writes outcome as L4 node to parent soul,
    then bulk-deletes all Meeseeks nodes from SQLite and ChromaDB.

    Args:
        instance_id: The Meeseeks instance ID
    """
    try:
        result = await meeseeks.release_meeseeks(instance_id)
        return json.dumps(result, indent=2)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return json.dumps({"error": str(e)})


@mcp.tool()
async def list_meeseeks() -> str:
    """List all active Meeseeks instances (not released/decomposed).

    Returns instance IDs, tasks, states, and step counts.
    """
    active = meeseeks.list_meeseeks()
    return json.dumps({"meeseeks": active, "count": len(active)}, indent=2)


# ============================================================
# LARQL — Decompiled Transformer Weight Queries
# ============================================================

if settings.larql_enabled:

    @mcp.tool()
    async def larql_describe(entity: str) -> str:
        """Get knowledge graph for an entity from decompiled model weights.

        Queries the LARQL vindex to find all edges (relations, targets, scores)
        associated with the given entity. The knowledge comes from the model's
        internal weight structure, not generated text.

        Args:
            entity: Entity name to describe (e.g. 'France', 'Einstein')
        """
        try:
            result = await larql_seed.extract_entity(entity)
            return json.dumps(result, indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": str(e), "entity": entity})


    @mcp.tool()
    async def larql_walk(prompt: str, top_k: int = 10) -> str:
        """Trace feature activation through model layers for a prompt.

        Walks the prompt through each layer of the model, showing which
        features activate and what entities/relations they connect to.

        Args:
            prompt: Text prompt to trace through layers
            top_k: Number of top activations per layer (default 10)
        """
        try:
            data = await larql_seed._larql_get(
                "/walk", params={"prompt": prompt, "top_k": top_k}, timeout=60.0)
            return json.dumps(data, indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": str(e), "prompt": prompt})


    @mcp.tool()
    async def larql_infer(prompt: str, top_k: int = 5) -> str:
        """Run inference on the extracted model via LARQL.

        Uses the decompiled vindex to perform inference, returning
        completions grounded in the model's weight structure.

        Args:
            prompt: Text prompt for inference
            top_k: Max tokens to generate (default 5)
        """
        try:
            # Check if inference mode is available
            stats = await larql_seed.extract_stats()
            if not stats.get("loaded", {}).get("inference", False):
                return json.dumps({
                    "error": "Inference mode not loaded (browse-level vindex)",
                    "prompt": prompt,
                    "hint": "Re-extract vindex with --level inference for chat completions",
                })

            import httpx
            async with httpx.AsyncClient(timeout=60.0) as c:
                r = await c.post(
                    f"{settings.larql_server_url}/v1/chat/completions",
                    json={
                        "model": settings.larql_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": top_k,
                        "temperature": 0,
                    },
                )
                r.raise_for_status()
                return json.dumps(r.json(), indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": str(e), "prompt": prompt})


    @mcp.tool()
    async def larql_select(relation: str, limit: int = 10) -> str:
        """SQL-style edge query — find all edges of a given relation type.

        Args:
            relation: Relation type to filter by (e.g. 'capital', 'located_in')
            limit: Max results to return (default 10)
        """
        try:
            data = await larql_seed._larql_post(
                "/select", json_data={"relation": relation, "limit": limit})
            return json.dumps(data, indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": str(e), "relation": relation})


    @mcp.tool()
    async def larql_show_relations() -> str:
        """List all discovered relation types in the vindex.

        Returns relation names with edge counts, score ranges, layer ranges,
        and example entities.
        """
        try:
            relations = await larql_seed.extract_relations()
            return json.dumps({
                "count": len(relations),
                "relations": relations,
            }, indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})


    @mcp.tool()
    async def larql_seed_entity(entity: str) -> str:
        """Bridge: extract an entity from vindex and insert into fractal graph.

        DESCRIBEs the entity in LARQL, parses edges, creates L0-L4 nodes
        with proper hierarchy, edges, embeddings, and confidence propagation.

        Args:
            entity: Entity name to extract and seed (e.g. 'France', 'Einstein')
        """
        try:
            result = await larql_seed.seed_entity(entity)
            return json.dumps(result, indent=2, default=str)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return json.dumps({"error": str(e), "entity": entity})


if __name__ == "__main__":
    mcp.run(transport="stdio")
