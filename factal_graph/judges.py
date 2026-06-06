"""Judge Triad — Angel / Devil / Neutral resolution-conflict detection.

Three judges operate at different resolution levels:
  Angel  (L0-L1): optimistic, sees broad consensus
  Devil  (L4-L5): adversarial, finds contradictions in fine evidence
  Neutral (L2-L3): balanced, checks cross-resolution coherence

A single batched LLM call produces all three verdicts + conflict edges.
Disagreements are logged as resolution_conflict edges, not factual contradictions.
"""

import asyncio
import json

import db
from config import settings
from embedder import embed
from chroma_store import query_level
from seed import _mother_generate, _parse_json_object


JUDGES = {
    "angel": {
        "name": "Angel",
        "resolution_range": (0, 1),
        "bias": "optimistic",
        "prompt_style": "summarize the broad consensus across domain and topic nodes",
    },
    "devil": {
        "name": "Devil",
        "resolution_range": (4, 5),
        "bias": "adversarial",
        "prompt_style": "find contradictions, weak evidence, and unsupported claims in facts and evidence",
    },
    "neutral": {
        "name": "Neutral",
        "resolution_range": (2, 3),
        "bias": "balanced",
        "prompt_style": "check coherence across resolution levels — bridge angel's overview with devil's findings",
    },
}


async def judge_topic(topic: str, top_k: int = 5) -> dict:
    """Run the Angel/Devil/Neutral judge triad on a topic.

    1. Embed the topic query
    2. All three judges query their resolution ranges in parallel
    3. Single batched LLM call produces all verdicts + conflict detection
    4. Resolution-conflict edges created for disagreements (severity > 0.3)

    Returns:
        {topic, verdicts: {angel, devil, neutral}, conflicts: [...], edges_created: [...]}
    """
    query_emb = await embed(topic)
    conn = db.get_db()

    # Step 2: Query all three resolution ranges in parallel
    async def _query_judge(key: str) -> tuple[str, list[dict]]:
        judge = JUDGES[key]
        lo, hi = judge["resolution_range"]
        all_hits = []
        for level in range(lo, hi + 1):
            hits = query_level(query_emb, level, n_results=top_k)
            for h in hits:
                node_id = int(h["node_id"])
                node = db.get_node(conn, node_id)
                if node:
                    all_hits.append({
                        "node_id": node_id,
                        "content": node["content"],
                        "resolution_level": node["resolution_level"],
                        "confidence": node["confidence"],
                        "distance": h["distance"],
                    })
        return key, all_hits

    results = await asyncio.gather(
        _query_judge("angel"),
        _query_judge("devil"),
        _query_judge("neutral"),
    )
    judge_contexts = {k: v for k, v in results}

    # Check if we have any nodes at all
    total_nodes = sum(len(v) for v in judge_contexts.values())
    if total_nodes == 0:
        return {
            "topic": topic,
            "verdicts": {},
            "answer": "No graph data available for this topic.",
            "conflicts": [],
            "edges_created": [],
            "nodes_examined": {k: len(v) for k, v in judge_contexts.items()},
            "error": "No nodes found in graph for this topic. Seed it first with seed_topic().",
        }

    # Step 3: Build batched prompt for all three judges
    context_parts = []
    for key in ("angel", "devil", "neutral"):
        judge = JUDGES[key]
        nodes = judge_contexts[key]
        node_lines = "\n".join(
            f"  [ID:{n['node_id']} L{n['resolution_level']} conf:{n['confidence']}] {n['content'][:200]}"
            for n in nodes
        ) if nodes else "  (no nodes found)"

        context_parts.append(
            f"=== {judge['name']} (L{judge['resolution_range'][0]}-L{judge['resolution_range'][1]}, "
            f"bias: {judge['bias']}) ===\n"
            f"Role: {judge['prompt_style']}\n"
            f"Nodes found:\n{node_lines}"
        )

    triad_prompt = (
        "You are a Judge Triad analyzing a knowledge graph topic. "
        "Three judges have examined the graph at different resolution levels.\n\n"
        f"Topic: {topic}\n\n"
        "Here is what each judge found:\n\n"
        + "\n\n".join(context_parts) + "\n\n"
        "Produce verdicts for all three judges AND identify conflicts between them.\n"
        "A conflict means Angel's optimistic overview disagrees with Devil's adversarial findings — "
        "this is a RESOLUTION MISMATCH, not necessarily a factual contradiction. "
        "It means the coarse and fine views of the topic diverge.\n\n"
        "Return JSON:\n"
        "{\n"
        '  "angel": {"assessment": "1-2 sentence verdict", "confidence": 0.0-1.0, '
        '"key_node_ids": [id1, id2], "stance": "for/against/neutral on topic"},\n'
        '  "devil": {"assessment": "1-2 sentence verdict", "confidence": 0.0-1.0, '
        '"key_node_ids": [id1, id2], "stance": "for/against/neutral on topic"},\n'
        '  "neutral": {"assessment": "1-2 sentence verdict", "confidence": 0.0-1.0, '
        '"key_node_ids": [id1, id2], "stance": "for/against/neutral on topic",\n'
        '    "bridges": [{"from": node_id, "to": node_id, "explanation": "how these connect"}]},\n'
        '  "conflicts": [{"angel_node": id, "devil_node": id, '
        '"gap_type": "resolution_mismatch|evidence_gap|perspective_divergence", '
        '"severity": 0.0-1.0, "description": "why they disagree"}]\n'
        "}\n\n"
        "If Angel and Devil agree, return empty conflicts. Only report real disagreements.\n"
        "Severity > 0.3 means a meaningful gap worth tracking."
    )

    raw = await _mother_generate(triad_prompt, settings.llm_model, num_predict=2048)
    parsed = _parse_json_object(raw)

    if not parsed:
        return {
            "topic": topic,
            "verdicts": {},
            "answer": "",
            "conflicts": [],
            "edges_created": [],
            "nodes_examined": {k: len(v) for k, v in judge_contexts.items()},
            "error": f"Failed to parse judge triad response. Raw: {raw[:500]}",
        }

    # Step 4: Create resolution_conflict edges for conflicts with severity > 0.3
    conflicts = parsed.get("conflicts", [])
    edges_created = []

    for conflict in conflicts:
        severity = conflict.get("severity", 0)
        if severity <= 0.3:
            continue

        angel_node = conflict.get("angel_node")
        devil_node = conflict.get("devil_node")
        if not angel_node or not devil_node:
            continue

        # Verify both nodes exist
        angel_exists = db.get_node(conn, angel_node)
        devil_exists = db.get_node(conn, devil_node)
        if not angel_exists or not devil_exists:
            continue

        context_text = (
            f"Judge triad resolution conflict: {conflict.get('gap_type', 'unknown')}\n"
            f"Severity: {severity}\n"
            f"{conflict.get('description', '')}"
        )

        try:
            edge_id = db.insert_edge(
                conn, angel_node, devil_node,
                edge_type="resolution_conflict",
                confidence=severity,
                context=context_text,
            )
            edges_created.append({
                "edge_id": edge_id,
                "from": angel_node,
                "to": devil_node,
                "severity": severity,
                "gap_type": conflict.get("gap_type"),
            })
        except (ValueError, Exception) as e:
            pass  # Edge already exists or nodes missing

    verdicts = {}
    for key in ("angel", "devil", "neutral"):
        if key in parsed:
            verdicts[key] = parsed[key]

    # Step 5: Synthesize final answer from triad verdicts
    verdicts_summary = "\n".join(
        f"{k.upper()}: {v.get('assessment', 'N/A')} "
        f"(confidence={v.get('confidence', 'N/A')}, stance={v.get('stance', 'N/A')})"
        for k, v in verdicts.items()
    )
    bridges_summary = ""
    neutral = verdicts.get("neutral", {})
    for b in neutral.get("bridges", []):
        bridges_summary += f"  - {b.get('explanation', 'N/A')}\n"
    conflicts_summary = ""
    for c in conflicts:
        conflicts_summary += f"  - {c.get('description', 'N/A')[:150]}\n"

    synthesis_prompt = (
        "You are the final arbiter of a Judge Triad. Three judges examined "
        "a knowledge graph topic at different resolution levels.\n\n"
        f"Topic: {topic}\n\n"
        f"Judge Verdicts:\n{verdicts_summary}\n\n"
    )
    if bridges_summary:
        synthesis_prompt += f"Cross-resolution bridges:\n{bridges_summary}\n"
    if conflicts_summary:
        synthesis_prompt += f"Conflicts found:\n{conflicts_summary}\n"
    if not bridges_summary and not conflicts_summary:
        synthesis_prompt += "The judges largely agree — no significant conflicts or bridges.\n"

    synthesis_prompt += (
        "\nSynthesize a final answer to the topic question. "
        "Weigh Angel's broad view against Devil's detailed critique, "
        "using Neutral's bridges to reconcile where possible.\n"
        "Be concise (3-5 sentences). State the overall conclusion clearly.\n\n"
        "Return JSON: {\"answer\": \"your synthesized answer here\"}"
    )

    synthesis_raw = await _mother_generate(
        synthesis_prompt, settings.llm_model, num_predict=512
    )
    synthesis_parsed = _parse_json_object(synthesis_raw)
    final_answer = ""
    if synthesis_parsed and "answer" in synthesis_parsed:
        final_answer = synthesis_parsed["answer"]
    else:
        # Fallback: use raw text if JSON parse failed
        final_answer = synthesis_raw.strip()[:500] if synthesis_raw else "Synthesis failed"

    return {
        "topic": topic,
        "verdicts": verdicts,
        "answer": final_answer,
        "conflicts": conflicts,
        "edges_created": edges_created,
        "nodes_examined": {k: len(v) for k, v in judge_contexts.items()},
    }
