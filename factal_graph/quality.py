"""Quality gate — self-consistency checks, contradiction detection, source attribution.

Three quality mechanisms:
1. self_consistency_check(domain_node_id) — ask mother same question 3x, keep 2/3+ consensus
2. quality_triad_scan(domain_node_id) — run judge triad on each L0 to catch hallucinations
3. source_attribution_pass(domain_node_id) — lower confidence for unattributed facts
"""

import asyncio
import json
import logging

import db
from db import _write_lock
from config import settings
from embedder import embed, embed_batch_parallel
from chroma_store import upsert_node, query_all_levels
from graph import recompute_parent_bbox
from seed import (_mother_generate_keepalive, _parse_json_array,
                 _parse_json_object, LEVEL_MEANINGS)
from growth import _cosine_sim, DEDUP_THRESHOLD
from judges import judge_topic as judge_topic_fn

logger = logging.getLogger(__name__)


# --- Prompt templates ---

CONSENSUS_PROMPT = """You are a fact-checking system. Given the topic below,
list the most important facts you know about it.

Topic: {topic}
Focus area: {focus}

Return JSON array of facts:
[{{"content": "precise factual statement", "level": N}}]

Level meanings:
  L3: entity — named entity, event, or specific instance
  L4: fact — verifiable fact with details (dates, numbers)
  L5: evidence — direct quote, cited statistic, or data point

Include 5-8 of the most important facts. Be specific and factual.
Repeat the same facts consistently — do not invent new ones across calls."""


SOURCE_ATTRIBUTION_PROMPT = """For each fact below, indicate whether you can cite a specific
source for it. Be honest — if you cannot recall a specific source, say so.

Facts:
{facts_list}

Return JSON array:
[{{"index": N, "has_source": true/false, "source_hint": "book/paper/URL or null", "confidence_adjustment": 0.0-1.0}}]

confidence_adjustment: how much you trust this fact without a verifiable source.
0.0 = certain (well-known fact), 0.3 = high confidence but no citation,
0.5 = moderate, 0.7 = uncertain, 1.0 = likely hallucinated.
Be strict — if you can't name a source, be conservative."""


# --- Self-consistency check ---

async def self_consistency_check(node_id: int, num_rounds: int = 3,
                                 min_consensus: int = 2) -> dict:
    """Ask mother the same question multiple times, keep only consensus facts.

    For a given L0 domain node, asks mother to list facts 3x independently.
    Facts appearing in 2/3+ rounds are kept at current confidence;
    facts appearing only once are demoted or flagged for removal.

    Args:
        node_id: L0 domain node to check
        num_rounds: Number of independent rounds (default 3)
        min_consensus: Minimum rounds a fact must appear in (default 2)

    Returns:
        {node_id, rounds, facts_kept, facts_demoted, demoted_ids, details}
    """
    conn = db.get_db()
    node = db.get_node(conn, node_id)
    if not node:
        return {"error": f"Node {node_id} not found"}

    # Collect facts from multiple rounds
    all_round_facts = []

    for round_i in range(num_rounds):
        prompt = CONSENSUS_PROMPT.format(
            topic=node["content"][:200],
            focus=node["content"][:200],
        )
        raw = await _mother_generate_keepalive(prompt, keep_alive="15s")
        facts = _parse_json_array(raw)
        if facts:
            all_round_facts.append(facts)

    if len(all_round_facts) < min_consensus:
        return {
            "node_id": node_id,
            "rounds": len(all_round_facts),
            "error": f"Only {len(all_round_facts)} successful rounds (need {min_consensus})",
        }

    # Embed all facts from all rounds
    all_texts = []
    all_meta = []  # (round_index, fact_index_in_round)
    for ri, round_facts in enumerate(all_round_facts):
        for fi, fact in enumerate(round_facts):
            if "content" in fact:
                all_texts.append(fact["content"])
                all_meta.append((ri, fi, fact.get("level", 4)))

    if not all_texts:
        return {"node_id": node_id, "rounds": len(all_round_facts),
                "facts_kept": 0, "facts_demoted": 0, "details": []}

    embeddings = await embed_batch_parallel(all_texts)

    # For each fact, count how many rounds contain a similar fact (consensus)
    fact_consensus = []  # (text, level, consensus_count, indices)
    checked = set()

    for i, (text, emb, meta) in enumerate(zip(all_texts, embeddings, all_meta)):
        ri, fi, level = meta

        # Skip if we already counted a similar fact
        is_dup = False
        for prev in fact_consensus:
            prev_emb = prev["embedding"]
            if _cosine_sim(emb, prev_emb) >= 0.88:
                # This is a duplicate of a previously counted fact — increment its round count
                prev_rounds = prev["round_indices"]
                if ri not in prev_rounds:
                    prev["round_indices"].add(ri)
                    prev["consensus"] = len(prev["round_indices"])
                is_dup = True
                break

        if is_dup:
            continue

        fact_consensus.append({
            "content": text,
            "level": int(str(level).lstrip("Ll")),
            "round_indices": {ri},
            "consensus": 1,
            "embedding": emb,
        })

    # Find children of the domain node to check against
    children = db.get_children(conn, node_id)
    child_embeddings = []
    for child in children:
        try:
            from chroma_store import get_collection
            col = get_collection(child["resolution_level"])
            result = col.get(ids=[str(child["id"])])
            if result and result["embeddings"] and result["embeddings"][0]:
                child_embeddings.append((child, result["embeddings"][0]))
        except Exception:
            child_embeddings.append((child, None))

    # Demote facts in the graph that lack consensus (appear only once)
    facts_kept = 0
    facts_demoted = 0
    demoted_ids = []
    details = []

    async with _write_lock:
        for child, child_emb in child_embeddings:
            if not child_emb:
                continue

            # Find the best matching fact from consensus check
            best_match = None
            best_sim = 0
            for fc in fact_consensus:
                sim = _cosine_sim(child_emb, fc["embedding"])
                if sim > best_sim:
                    best_sim = sim
                    best_match = fc

            if best_match and best_sim >= 0.85:
                if best_match["consensus"] >= min_consensus:
                    facts_kept += 1
                    details.append({
                        "id": child["id"],
                        "content": child["content"][:80],
                        "consensus": best_match["consensus"],
                        "action": "kept",
                    })
                else:
                    # Demote — reduce confidence
                    facts_demoted += 1
                    demoted_ids.append(child["id"])
                    db.update_node_confidence(
                        conn, child["id"],
                        max(0.2, child["confidence"] - 0.3),
                    )
                    details.append({
                        "id": child["id"],
                        "content": child["content"][:80],
                        "consensus": best_match["consensus"],
                        "action": "demoted",
                        "old_confidence": child["confidence"],
                        "new_confidence": max(0.2, child["confidence"] - 0.3),
                    })

    return {
        "node_id": node_id,
        "rounds": len(all_round_facts),
        "facts_kept": facts_kept,
        "facts_demoted": facts_demoted,
        "demoted_ids": demoted_ids,
        "details": details,
    }


# --- Contradiction detection via triad ---

async def quality_triad_scan(domain_node_id: int = None) -> dict:
    """Run judge triad on each L0 domain to catch mother hallucinations.

    For each L0 domain (or a specific one), runs the Angel/Devil/Neutral
    triad to find resolution conflicts — disagreements between broad
    overview and detailed evidence that indicate hallucinations.

    Args:
        domain_node_id: Optional L0 node to scan (null = all L0 domains)

    Returns:
        {domains_scanned, total_conflicts, results: [...]}
    """
    conn = db.get_db()

    if domain_node_id:
        domains = [db.get_node(conn, domain_node_id)]
        domains = [d for d in domains if d and d["resolution_level"] == 0]
    else:
        domains = db.get_nodes_by_resolution(conn, 0)

    if not domains:
        return {"error": "No L0 domains found"}

    results = []
    total_conflicts = 0

    for domain in domains:
        # Use domain content as the topic for the triad
        topic = domain["content"][:200]
        triad = await judge_topic_fn(topic)

        conflicts = triad.get("conflicts", [])
        edges_created = triad.get("edges_created", [])

        # Also check: if Devil's confidence is much lower than Angel's,
        # it might indicate the overview is inflated
        verdicts = triad.get("verdicts", {})
        angel_conf = verdicts.get("angel", {}).get("confidence", 0.5)
        devil_conf = verdicts.get("devil", {}).get("confidence", 0.5)

        quality_warning = None
        if angel_conf > 0.7 and devil_conf < 0.4 and angel_conf - devil_conf > 0.3:
            quality_warning = (
                f"High Angel confidence ({angel_conf}) vs low Devil confidence "
                f"({devil_conf}) — possible overconfidence in overview"
            )

        result = {
            "domain_id": domain["id"],
            "topic": topic[:100],
            "conflicts_found": len(conflicts),
            "edges_created": len(edges_created),
            "angel_confidence": angel_conf,
            "devil_confidence": devil_conf,
            "quality_warning": quality_warning,
        }

        # If there are conflicts and a quality warning, demote the domain
        if quality_warning and conflicts:
            db.update_node_confidence(
                conn, domain["id"],
                max(0.3, domain["confidence"] - 0.1),
            )
            result["domain_demoted"] = True
            result["new_confidence"] = max(0.3, domain["confidence"] - 0.1)

        results.append(result)
        total_conflicts += len(conflicts)

    return {
        "domains_scanned": len(domains),
        "total_conflicts": total_conflicts,
        "results": results,
    }


# --- Source attribution ---

async def source_attribution_pass(domain_node_id: int = None) -> dict:
    """Ask mother to attribute sources for facts, demote unattributed ones.

    For each L0 domain (or a specific one), collects L4-L5 fact nodes,
    asks mother to rate source attributability, and adjusts confidence
    based on whether the mother can cite a specific source.

    Args:
        domain_node_id: Optional L0 node to scan (null = all L0 domains)

    Returns:
        {domains_scanned, facts_checked, adjusted: [...]}
    """
    conn = db.get_db()

    if domain_node_id:
        domains = [db.get_node(conn, domain_node_id)]
        domains = [d for d in domains if d and d["resolution_level"] == 0]
    else:
        domains = db.get_nodes_by_resolution(conn, 0)

    if not domains:
        return {"error": "No L0 domains found"}

    total_checked = 0
    all_adjusted = []

    for domain in domains:
        # Get all L4-L5 descendants
        descendants = conn.execute("""
            WITH RECURSIVE subtree(id) AS (
                SELECT id FROM nodes WHERE id = ?
                UNION ALL
                SELECT n.id FROM nodes n JOIN subtree s ON n.parent_id = s.id
            )
            SELECT * FROM nodes
            WHERE id IN (SELECT id FROM subtree)
            AND resolution_level >= 4
            ORDER BY resolution_level DESC
        """, (domain["id"],)).fetchall()

        if not descendants:
            continue

        # Batch facts for source attribution (max 10 per domain)
        facts_list = []
        fact_nodes = descendants[:10]
        for i, node in enumerate(fact_nodes):
            facts_list.append(
                f"  [{i}] {node['content'][:150]}"
            )

        prompt = SOURCE_ATTRIBUTION_PROMPT.format(
            facts_list="\n".join(facts_list)
        )

        raw = await _mother_generate_keepalive(prompt, keep_alive="0")
        attributions = _parse_json_array(raw)

        if not attributions:
            total_checked += len(fact_nodes)
            continue

        adjusted = []

        async with _write_lock:
            for attr in attributions:
                idx = attr.get("index")
                if idx is None or idx >= len(fact_nodes):
                    continue

                node = fact_nodes[idx]
                has_source = attr.get("has_source", False)
                adj = float(attr.get("confidence_adjustment", 0.5))
                source_hint = attr.get("source_hint")

                if not has_source and adj > 0.3:
                    # Demote unattributed facts
                    new_conf = max(0.2, node["confidence"] - adj * 0.3)
                    db.update_node_confidence(conn, node["id"], new_conf)

                    adjusted.append({
                        "id": node["id"],
                        "content": node["content"][:80],
                        "has_source": has_source,
                        "source_hint": source_hint,
                        "old_confidence": node["confidence"],
                        "new_confidence": new_conf,
                        "adjustment": adj,
                    })
                elif has_source:
                    adjusted.append({
                        "id": node["id"],
                        "content": node["content"][:80],
                        "has_source": True,
                        "source_hint": source_hint,
                        "action": "kept",
                    })

        all_adjusted.extend(adjusted)
        total_checked += len(fact_nodes)

    return {
        "domains_scanned": len(domains),
        "facts_checked": total_checked,
        "adjustments": all_adjusted,
    }
