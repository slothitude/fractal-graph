"""2B reasoning engine — LaRQL INFER equivalent.

Where "2B speed + ~8B quality" is realized. The 2B model doesn't need to
KNOW facts — it reasons over structured graph context provided by context.py.
Factual correctness comes from the graph, coherence from the 2B.
"""

import asyncio
import logging
import re
import json

import httpx

logger = logging.getLogger(__name__)

from config import settings
from embedder import embed
from context import gather_context, format_context_for_llm


async def _warmup_model(model: str, url: str) -> float:
    """Trigger model load with a dummy generate. Uses shared model cache.

    Returns:
        Load time in seconds.
    """
    from model_cache import ensure_model_loaded
    return await ensure_model_loaded(model, url)


async def _llm_call(prompt: str, model: str = None, num_predict: int = 512,
                    timeout: float = 60.0, temperature: float = None) -> str:
    """Call the 2B LLM. Returns raw response text."""
    model = model or settings.llm_model
    url = settings.llm_url
    temp = temperature if temperature is not None else 0.2
    try:
        await _warmup_model(model, url)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "5m",
                    "options": {
                        "temperature": temp,
                        "num_ctx": 8192,
                    },
                    "think": False,
                },
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
    except Exception as e:
        logger.warning("2B LLM call failed: %s", e)
        return ""


def _parse_json(text: str) -> dict | None:
    """Extract JSON object from LLM output."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text).strip()
    # Strip qwen3.5 thinking tags
    text = re.sub(r"</?think\s*>", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass
    return None


async def synthesize(question: str, context_str: str) -> dict:
    """2B generates answer from structured graph context.

    The key insight: 2B doesn't recall facts, it reasons over provided data.
    Prompt instructs it to stay grounded in the context.

    Returns:
        {answer: str, confidence: float, key_facts: [...], gaps: [...]}
    """
    prompt = (
        "You are answering a question using ONLY the knowledge context provided below. "
        "Do NOT use any outside knowledge. If the context doesn't contain enough "
        "information, say so explicitly.\n\n"
        f"Question: {question}\n\n"
        f"{context_str}\n\n"
        "Based ONLY on the context above, answer the question.\n"
        "Return JSON:\n"
        '{\n'
        '  "answer": "your 2-4 sentence answer here",\n'
        '  "confidence": 0.0-1.0,\n'
        '  "key_facts": ["fact1", "fact2"],\n'
        '  "gaps": ["what information is missing"]\n'
        "}\n\n"
        "Rules:\n"
        "- Only use facts present in the context\n"
        "- If context is empty or insufficient, set confidence < 0.3\n"
        "- Be concise — 2-4 sentences max\n"
        "- key_facts should reference specific information from context\n"
    )

    raw = await _llm_call(prompt, num_predict=settings.max_answer_tokens, timeout=30.0)
    parsed = _parse_json(raw)

    if not parsed or "answer" not in parsed:
        # Try extracting answer from raw text — 2B sometimes wraps JSON badly
        raw = re.sub(r"</?think\s*>", "", raw).strip()
        if raw.startswith("{") or raw.startswith('"answer"'):
            parsed = _parse_json(raw)
        if parsed and "answer" in parsed:
            parsed.setdefault("confidence", 0.5)
            parsed.setdefault("key_facts", [])
            parsed.setdefault("gaps", [])
            return parsed

        # Last fallback: strip JSON framing if present, use inner text
        inner = re.sub(r'^\s*\{[^}]*"answer"\s*:\s*"', '', raw)
        inner = re.sub(r'"\s*\}\s*$', '', inner)
        if inner and len(inner) > 10:
            return {
                "answer": inner.strip(),
                "confidence": 0.4,
                "key_facts": [],
                "gaps": ["JSON parsing failed, extracted raw answer"],
            }

        answer = raw.strip()[:500] if raw else "Failed to generate answer"
        return {
            "answer": answer,
            "confidence": 0.3,
            "key_facts": [],
            "gaps": ["Answer generation parsing failed"],
        }

    parsed.setdefault("confidence", 0.5)
    parsed.setdefault("key_facts", [])
    parsed.setdefault("gaps", [])
    return parsed


async def answer(question: str, auto_expand: bool = True,
                 _round: int = 0) -> dict:
    """Top-level pipeline: embed -> gather -> synthesize.

    With auto_expand, low-confidence answers trigger gap fill via the mother
    model, then re-answer with enriched graph context.

    Args:
        question: The user's question
        auto_expand: If True, fill gaps on low confidence and re-answer
        _round: Internal recursion counter (max max_expansion_rounds)

    Returns:
        {question, answer, confidence, key_facts, gaps,
         context_tokens, timing: {...},
         expanded: bool, nodes_added: int}
    """
    import time
    t0 = time.time()

    # Step 1: Embed
    t1 = time.time()
    embedding = await embed(question)
    embed_time = time.time() - t1

    # Step 2: Gather context from graph
    t2 = time.time()
    context = gather_context(embedding, top_k=10, max_tokens=settings.max_context_tokens)
    context_str = format_context_for_llm(context)
    context_time = time.time() - t2

    # Step 3: Synthesize answer (2B, single call)
    t3 = time.time()
    result = await synthesize(question, context_str)
    synthesize_time = time.time() - t3

    total_time = time.time() - t0
    response = {
        "question": question,
        "answer": result["answer"],
        "confidence": result["confidence"],
        "key_facts": result["key_facts"],
        "gaps": result["gaps"],
        "context_tokens": context["total_tokens"],
        "timing": {
            "embed_s": round(embed_time, 2),
            "context_gather_s": round(context_time, 2),
            "synthesize_s": round(synthesize_time, 2),
            "total_s": round(total_time, 2),
        },
        "expanded": False,
        "nodes_added": 0,
        "search_fallback": False,
    }

    # Step 4: Auto-expand if confidence is low or gaps detected
    conf = result.get("confidence", 1.0)
    gaps = result.get("gaps", [])
    llm_failed = any("parsing failed" in g for g in gaps)
    if (auto_expand
            and (conf < settings.auto_expand_threshold or len(gaps) > 0)
            and _round < settings.max_expansion_rounds):

        # Try search-triggered gap fill first (faster, grounded in real sources)
        from search_trigger import search_triggered
        search_nodes_added = 0
        if settings.search_trigger_enabled and gaps:
            for gap_str in gaps[:2]:
                sr = await search_triggered(
                    "gap", gap_str, {"content": question},
                )
                if sr["triggered"] and sr["nodes_created"] > 0:
                    search_nodes_added += sr["nodes_created"]
                    if sr.get("search_fallback"):
                        response["search_fallback"] = True

        # Fall back to mother-based gap fill if search didn't help
        fill_result = {"nodes_created": 0}
        if search_nodes_added == 0:
            from growth import fill_gaps
            fill_result = await fill_gaps(
                gaps, question, max_nodes=settings.max_gap_fill_nodes,
            )
        else:
            fill_result["nodes_created"] = search_nodes_added

        total_added = fill_result["nodes_created"]
        if total_added > 0:
            # Track if web search fallback was used
            if fill_result.get("search_fallback"):
                response["search_fallback"] = True

            # Re-gather context with enriched graph
            new_context = gather_context(
                embedding, top_k=10, max_tokens=settings.max_context_tokens,
            )
            new_context_str = format_context_for_llm(new_context)

            # Re-synthesize with richer context
            new_result = await synthesize(question, new_context_str)

            response["answer"] = new_result["answer"]
            response["confidence"] = new_result["confidence"]
            response["key_facts"] = new_result["key_facts"]
            response["gaps"] = new_result["gaps"]
            response["expanded"] = True
            response["nodes_added"] = total_added
            response["context_tokens"] = new_context["total_tokens"]
            response["timing"]["total_s"] = round(time.time() - t0, 2)
            response["timing"]["gap_fill_s"] = round(
                response["timing"]["total_s"] - total_time, 2,
            )

            # If still low confidence after fill, recurse once more
            # But NOT if the LLM call itself failed (parsing error)
            new_conf = new_result.get("confidence", 1.0)
            new_gaps = new_result.get("gaps", [])
            new_llm_failed = any("parsing failed" in g for g in new_gaps)
            if (total_added > 0
                    and not new_llm_failed
                    and new_conf < settings.auto_expand_threshold
                    and len(new_gaps) > 0
                    and _round + 1 < settings.max_expansion_rounds):
                response["timing"]["total_s"] = round(time.time() - t0, 2)
                return await answer(
                    question, auto_expand=True, _round=_round + 1,
                )

    return response


async def answer_with_triad(question: str) -> dict:
    """Run ask + triad pass 1 in parallel, then triad pass 2.

    With OLLAMA_NUM_PARALLEL=2, the 2B model can batch two inference
    requests into one forward pass — saving ~3s vs sequential.

    Returns:
        {question, answer, confidence, key_facts, gaps,
         context_tokens, timing, triad, expanded, nodes_added}
    """
    import time
    from judges import judge_topic, judge_answer

    t0 = time.time()

    # Both use the 2B model — OLLAMA_NUM_PARALLEL=2 batches them
    answer_task = answer(question, auto_expand=True)
    triad_task = judge_topic(question)

    result, triad = await asyncio.gather(answer_task, triad_task)

    # Pass 2: synthesize from triad verdicts
    triad_answer = await judge_answer(question, triad)

    # Use triad answer if higher confidence
    if triad_answer.get("confidence", 0) > result.get("confidence", 0):
        result["answer"] = triad_answer["answer"]
        result["confidence"] = triad_answer["confidence"]
        result["key_facts"] = triad_answer.get("key_facts", result["key_facts"])

    result["timing"]["total_with_triad_s"] = round(time.time() - t0, 2)
    result["triad"] = triad
    return result


# --- Agent Decision Mode ---

def _gather_procedural_context(prompt_embedding: list[float], top_k: int = 5,
                               max_tokens: int = 4000,
                               soul_id: str = None) -> dict:
    """Gather context biased toward L2-L4 (patterns, rules, examples).

    Unlike gather_context which hits all 6 levels equally, this focuses
    on the resolution levels most useful for agent decisions:
    - L2: decision patterns (IF/THEN rule types)
    - L3: concrete examples (specific scenarios)
    - L4: specific rules (exact conditions, actions, outcomes)

    Weighting: 0.5*L2 + 0.3*L3 + 0.2*L4

    Args:
        soul_id: Optional soul filter — only gather from this soul's nodes

    Returns:
        Same format as gather_context: {direct_hits, parents, edges, total_tokens}
    """
    from chroma_store import query_level
    import db as db_mod

    conn = db_mod.get_db()
    max_tokens = max_tokens or settings.max_context_tokens

    # Query L2, L3, L4 specifically
    level_weights = {2: 0.5, 3: 0.3, 4: 0.2}
    direct_hits = []
    seen_ids = set()

    for level, weight in level_weights.items():
        hits = query_level(prompt_embedding, level, n_results=top_k,
                          soul_id=soul_id)
        for hit in hits:
            node_id = int(hit["node_id"])
            if node_id in seen_ids:
                continue
            node = db_mod.get_node(conn, node_id)
            if not node:
                continue
            seen_ids.add(node_id)
            edges = db_mod.get_edges(conn, node_id)
            direct_hits.append({
                "node": node,
                "distance": hit.get("distance", 1.0),
                "edges": edges,
                "score": 0.0,
                "level_weight": weight,
            })

    # Walk parent chains
    parents = []
    parent_seen = set()
    for hit in direct_hits:
        node = hit["node"]
        current = node
        while current.get("parent_id") and current["parent_id"] not in parent_seen:
            pid = current["parent_id"]
            parent = db_mod.get_node(conn, pid)
            if not parent:
                break
            parent_seen.add(pid)
            parents.append(parent)
            current = parent

    # Rank: 0.4*vector_sim + 0.3*confidence + 0.2*level_weight + 0.1*edge_count
    for hit in direct_hits:
        vector_sim = max(0, 1.0 - hit["distance"])
        confidence = hit["node"].get("confidence", 0.5)
        edge_count = len(hit["edges"])
        hit["score"] = (
            0.4 * vector_sim +
            0.3 * confidence +
            0.2 * hit["level_weight"] +
            0.1 * min(edge_count / 3.0, 1.0)
        )

    direct_hits.sort(key=lambda x: x["score"], reverse=True)

    # Token-budget truncation
    budget_remaining = max_tokens
    result = {"direct_hits": [], "parents": [], "edges": [], "total_tokens": 0}

    def _add_node(entry, node_list):
        nonlocal budget_remaining
        node = entry["node"] if isinstance(entry, dict) and "node" in entry else entry
        content = node.get("content", "")
        text = f"- [ID:{node['id']} L{node['resolution_level']} conf:{node['confidence']:.2f}] {content}"
        tokens = len(text) // 4
        if tokens <= budget_remaining:
            node_list.append(text)
            budget_remaining -= tokens
            return True
        return False

    for hit in direct_hits:
        if not _add_node(hit, result["direct_hits"]):
            break
    for parent in parents[:5]:
        if not _add_node(parent, result["parents"]):
            break

    result["total_tokens"] = max_tokens - budget_remaining
    return result


async def decide_synthesize(situation: str, options: list[str],
                            context_str: str,
                            temperature: float = None) -> dict:
    """2B generates a decision from procedural context.

    Returns:
        {action, reasoning, confidence, relevant_patterns, risks, option_scores}
    """
    options_text = ""
    if options:
        options_text = (
            "\n\nCandidate actions to evaluate:\n" +
            "".join(f"  {i+1}. {o}\n" for i, o in enumerate(options))
        )

    prompt = (
        "You are a decision-making agent. Given the situation and procedural "
        "knowledge context below, recommend the BEST action to take.\n\n"
        "Do NOT use outside knowledge. Base your decision ONLY on the provided context.\n\n"
        f"Situation: {situation}\n"
        f"{options_text}\n\n"
        f"{context_str}\n\n"
        "Based ONLY on the context above, recommend an action.\n"
        "Return JSON:\n"
        '{\n'
        '  "action": "the recommended action (concise, executable)",\n'
        '  "reasoning": "why this action is best, referencing specific patterns from context",\n'
        '  "confidence": 0.0-1.0,\n'
        '  "relevant_patterns": ["pattern1 from context", "pattern2"],\n'
        '  "risks": ["risk1", "risk2"]'
    )
    if options:
        prompt += (
            ',\n'
            '  "option_scores": {"action text": 0.0-1.0}'
        )
    prompt += (
        "\n}\n\n"
        "Rules:\n"
        "- Only use patterns/rules present in the context\n"
        "- If context is empty, propose a reasonable default action with low confidence\n"
        "- Always list at least one risk\n"
        "- Keep reasoning under 3 sentences\n"
    )

    raw = await _llm_call(prompt, num_predict=settings.max_answer_tokens, timeout=30.0,
                          temperature=temperature)
    parsed = _parse_json(raw)

    if not parsed or "action" not in parsed:
        # Fallback extraction
        raw = re.sub(r"</?think\s*>", "", raw).strip()
        answer_text = raw.strip()[:500] if raw else "Failed to generate decision"
        return {
            "action": answer_text,
            "reasoning": "JSON parsing failed",
            "confidence": 0.3,
            "relevant_patterns": [],
            "risks": ["Decision generation failed"],
            "option_scores": {},
        }

    parsed.setdefault("reasoning", "")
    parsed.setdefault("confidence", 0.5)
    parsed.setdefault("relevant_patterns", [])
    parsed.setdefault("risks", [])
    parsed.setdefault("option_scores", {})
    return parsed


async def decide(situation: str, options: list[str] = None) -> dict:
    """Given a situation and optional options, retrieve procedural knowledge
    and recommend an action via 2B.

    Pipeline: embed -> gather_procedural_context (L2-L4 biased) -> decide_synthesize (2B).
    Triad is NOT used here — it queries factual levels (L0-L5) which are
    irrelevant to procedural decisions. Use decide_mc() for Monte Carlo
    enhanced decisions.

    Args:
        situation: Description of the situation the agent faces
        options: Optional list of candidate actions to evaluate

    Returns:
        {situation, action, reasoning, confidence, relevant_patterns, risks,
         option_scores, timing}
    """
    import time

    t0 = time.time()

    # Step 1: Embed
    t1 = time.time()
    embedding = await embed(situation)
    embed_time = time.time() - t1

    # Step 2: Gather procedural context (L2-L4 biased)
    t2 = time.time()
    context = _gather_procedural_context(
        embedding, top_k=5, max_tokens=settings.max_context_tokens,
    )
    context_str = format_context_for_llm(context)
    context_time = time.time() - t2

    # Step 3: Synthesize decision (2B)
    t3 = time.time()
    decision = await decide_synthesize(situation, options or [], context_str)
    synthesize_time = time.time() - t3

    total_time = time.time() - t0

    return {
        "situation": situation,
        "action": decision["action"],
        "reasoning": decision["reasoning"],
        "confidence": decision["confidence"],
        "relevant_patterns": decision["relevant_patterns"],
        "risks": decision["risks"],
        "option_scores": decision["option_scores"],
        "context_tokens": context["total_tokens"],
        "timing": {
            "embed_s": round(embed_time, 2),
            "context_gather_s": round(context_time, 2),
            "synthesize_s": round(synthesize_time, 2),
            "total_s": round(total_time, 2),
        },
    }


# --- Monte Carlo Graph Search ---

def _gather_procedural_context_sampled(
    prompt_embedding: list[float],
    pool_size: int = 10,
    sample_k: int = 3,
    max_tokens: int = 3000,
    soul_id: str = None,
    soul_ids: list[str] = None,
) -> dict:
    """Gather a large candidate pool from L2/L3/L4, return a random k-subset.

    Unlike _gather_procedural_context which returns the best-ranked nodes,
    this gathers a larger pool then randomly samples k nodes for one
    Monte Carlo simulation. Each simulation sees different context.

    Args:
        soul_id: Optional soul filter — only sample from this soul's nodes
        soul_ids: Optional list of soul_ids — sample from multiple souls (Meeseeks inheritance).
                  When provided, takes precedence over soul_id.

    Returns:
        {nodes: [list of formatted node strings], pool_size: int}
    """
    import random
    from chroma_store import query_level
    import db as db_mod

    conn = db_mod.get_db()
    level_weights = {2: 0.5, 3: 0.3, 4: 0.2}
    pool = []
    seen_ids = set()

    # Build where clause for soul filtering
    # soul_ids (Meeseeks inheritance) takes precedence over single soul_id
    effective_soul_id = None
    if soul_ids:
        effective_soul_id = soul_ids  # passed as list, handled below
    elif soul_id:
        effective_soul_id = soul_id

    for level, weight in level_weights.items():
        if isinstance(effective_soul_id, list) and len(effective_soul_id) > 1:
            # Query each soul_id separately and merge results
            hits = []
            for sid in effective_soul_id:
                hits.extend(query_level(prompt_embedding, level, n_results=pool_size,
                                        soul_id=sid))
            # Dedup by node_id — keep best (lowest distance)
            best_by_id = {}
            for hit in hits:
                nid = hit["node_id"]
                if nid not in best_by_id or hit["distance"] < best_by_id[nid]["distance"]:
                    best_by_id[nid] = hit
            hits = list(best_by_id.values())
        else:
            single_id = effective_soul_id[0] if isinstance(effective_soul_id, list) else effective_soul_id
            hits = query_level(prompt_embedding, level, n_results=pool_size,
                              soul_id=single_id)
        for hit in hits:
            node_id = int(hit["node_id"])
            if node_id in seen_ids:
                continue
            node = db_mod.get_node(conn, node_id)
            if not node:
                continue
            seen_ids.add(node_id)
            edges = db_mod.get_edges(conn, node_id)
            vector_sim = max(0, 1.0 - hit.get("distance", 1.0))
            confidence = node.get("confidence", 0.5)
            score = 0.4 * vector_sim + 0.3 * confidence + 0.3 * weight
            pool.append({
                "node": node,
                "score": score,
                "level_weight": weight,
            })

    # Shuffle for randomness, take first k
    random.shuffle(pool)
    sampled = pool[:sample_k]

    # Format sampled nodes
    budget_remaining = max_tokens
    nodes = []
    for entry in sampled:
        node = entry["node"]
        content = node.get("content", "")
        text = f"- [ID:{node['id']} L{node['resolution_level']} conf:{node['confidence']:.2f}] {content}"
        tokens = len(text) // 4
        if tokens <= budget_remaining:
            nodes.append(text)
            budget_remaining -= tokens

    return {"nodes": nodes, "pool_size": len(pool), "sampled_count": len(nodes)}


def _word_overlap(s1: str, s2: str) -> float:
    """Ratio of shared words between two strings."""
    if not s1 or not s2:
        return 0.0
    words1 = set(s1.lower().split())
    words2 = set(s2.lower().split())
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / min(len(words1), len(words2))


def _cluster_actions(decisions: list[dict]) -> list[list[dict]]:
    """Cluster decisions by action word overlap >= 0.5.

    Returns groups of decisions that agree on action.
    """
    clusters = []
    assigned = set()

    for i, d in enumerate(decisions):
        if i in assigned:
            continue
        cluster = [d]
        assigned.add(i)
        action_i = d.get("action", "")
        for j in range(i + 1, len(decisions)):
            if j in assigned:
                continue
            action_j = decisions[j].get("action", "")
            if _word_overlap(action_i, action_j) >= 0.5:
                cluster.append(decisions[j])
                assigned.add(j)
        clusters.append(cluster)

    return clusters


def _aggregate_decisions(clusters: list[list[dict]]) -> dict:
    """Aggregate clusters: largest cluster wins. Merge risks + patterns."""
    if not clusters:
        return {
            "action": "no decision reached",
            "reasoning": "all simulations failed",
            "confidence": 0.0,
            "relevant_patterns": [],
            "risks": [],
        }

    # Largest cluster = winner
    winner = max(clusters, key=len)
    winner_action = winner[0].get("action", "unknown")

    # Average confidence across winning cluster
    confs = [d.get("confidence", 0.5) for d in winner]
    avg_conf = sum(confs) / len(confs)

    # Union risks (dedup by substring)
    all_risks = []
    seen_risk_substr = set()
    for d in winner:
        for risk in d.get("risks", []):
            # Dedup: if any existing risk has >60% word overlap, skip
            is_dup = any(
                _word_overlap(risk, existing) > 0.6
                for existing in all_risks
            )
            if not is_dup:
                all_risks.append(risk)

    # Union relevant_patterns (dedup)
    all_patterns = []
    seen_pat_substr = set()
    for d in winner:
        for pat in d.get("relevant_patterns", []):
            is_dup = any(
                _word_overlap(pat, existing) > 0.6
                for existing in all_patterns
            )
            if not is_dup:
                all_patterns.append(pat)

    return {
        "action": winner_action,
        "reasoning": winner[0].get("reasoning", ""),
        "confidence": round(avg_conf, 2),
        "relevant_patterns": all_patterns,
        "risks": all_risks,
        "cluster_sizes": [len(c) for c in clusters],
        "consistency": round(len(winner) / sum(len(c) for c in clusters), 2),
    }


async def decide_monte_carlo(
    situation: str,
    options: list[str] = None,
    simulations: int = None,
    model: str = None,
    pool_size: int = None,
    sample_k: int = None,
    temperature: float = None,
    soul_id: str = None,
    soul_ids: list[str] = None,
) -> dict:
    """Monte Carlo graph search decision — N simulations sampling different
    graph context subsets. Each simulation picks a random subset of graph
    nodes and reasons over them. Aggregate by majority vote.

    Args:
        situation: The situation the agent faces
        options: Optional list of candidate actions
        simulations: Number of MC simulations (default from config)
        model: Model to use (default from config)
        pool_size: Candidate nodes per level (default from config)
        sample_k: Nodes per simulation subset (default from config)
        temperature: LLM temperature (default 0.2)
        soul_id: Optional soul filter — scope context to this soul
        soul_ids: Optional list of soul_ids — scope to multiple souls (Meeseeks inheritance)

    Returns:
        {situation, action, reasoning, confidence, relevant_patterns, risks,
         consistency, cluster_sizes, simulations_detail, timing}
    """
    import time
    import random

    t0 = time.time()
    sims = simulations or settings.mc_simulations
    m = model or settings.llm_model
    pool_size = pool_size or settings.mc_context_pool
    sample_k = sample_k or settings.mc_context_subset

    # Step 1: Embed once
    t1 = time.time()
    embedding = await embed(situation)
    embed_time = round(time.time() - t1, 2)

    # Step 2: Run N simulations
    decisions = []
    sim_details = []

    # Batch into groups of 2 (OLLAMA_NUM_PARALLEL=2)
    for batch_start in range(0, sims, 2):
        batch_sims = range(batch_start, min(batch_start + 2, sims))

        async def _run_one_sim(_sim_idx: int) -> dict:
            """Run one Monte Carlo simulation. Retry once on failure."""
            for attempt in range(2):
                sampled = _gather_procedural_context_sampled(
                    embedding, pool_size=pool_size, sample_k=sample_k,
                    soul_id=soul_id, soul_ids=soul_ids,
                )
                context_str = format_context_for_llm({
                    "direct_hits": sampled["nodes"],
                    "parents": [],
                    "edges": [],
                    "total_tokens": sum(len(n) // 4 for n in sampled["nodes"]),
                })
                decision = await decide_synthesize(
                    situation, options or [], context_str,
                    temperature=temperature,
                )
                # If decision has real action (not a fallback), use it
                if (decision.get("confidence", 0) > 0.3
                        and decision.get("action", "")
                        and "Failed" not in decision.get("action", "")):
                    break
                # Retry with different random sample
            return {
                "decision": decision,
                "pool_size": sampled["pool_size"],
                "sampled_count": sampled["sampled_count"],
            }

        batch_tasks = [_run_one_sim(i) for i in batch_sims]
        batch_results = await asyncio.gather(*batch_tasks)

        for sim_idx, result in zip(batch_sims, batch_results):
            d = result["decision"]
            decisions.append(d)
            sim_details.append({
                "sim": sim_idx,
                "action": d.get("action", ""),
                "confidence": d.get("confidence", 0),
                "pool_size": result["pool_size"],
                "sampled_count": result["sampled_count"],
            })

    # Step 3: Cluster + aggregate
    t_cluster = time.time()
    clusters = _cluster_actions(decisions)
    aggregated = _aggregate_decisions(clusters)
    cluster_time = round(time.time() - t_cluster, 2)

    # Step 4: Collect ALL risks across all simulations (not just winner)
    all_sim_risks = []
    seen_risks = set()
    for d in decisions:
        for risk in d.get("risks", []):
            is_dup = any(
                _word_overlap(risk, existing) > 0.6
                for existing in all_sim_risks
            )
            if not is_dup:
                all_sim_risks.append(risk)

    total_time = round(time.time() - t0, 2)

    return {
        "situation": situation,
        "action": aggregated["action"],
        "reasoning": aggregated["reasoning"],
        "confidence": aggregated["confidence"],
        "relevant_patterns": aggregated["relevant_patterns"],
        "risks": aggregated["risks"],
        "all_simulation_risks": all_sim_risks,
        "consistency": aggregated["consistency"],
        "cluster_sizes": aggregated["cluster_sizes"],
        "simulations_detail": sim_details,
        "model": m,
        "timing": {
            "embed_s": embed_time,
            "simulations_s": round(total_time - embed_time - cluster_time, 2),
            "cluster_s": cluster_time,
            "total_s": total_time,
        },
    }
