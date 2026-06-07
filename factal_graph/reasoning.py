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
                    timeout: float = 60.0) -> str:
    """Call the 2B LLM. Returns raw response text."""
    model = model or settings.llm_model
    url = settings.llm_url
    try:
        await _warmup_model(model, url)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "30s",
                    "options": {
                        "temperature": 0.2,
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
                               max_tokens: int = 4000) -> dict:
    """Gather context biased toward L2-L4 (patterns, rules, examples).

    Unlike gather_context which hits all 6 levels equally, this focuses
    on the resolution levels most useful for agent decisions:
    - L2: decision patterns (IF/THEN rule types)
    - L3: concrete examples (specific scenarios)
    - L4: specific rules (exact conditions, actions, outcomes)

    Weighting: 0.5*L2 + 0.3*L3 + 0.2*L4

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
        hits = query_level(prompt_embedding, level, n_results=top_k)
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
                            context_str: str) -> dict:
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

    raw = await _llm_call(prompt, num_predict=settings.max_answer_tokens, timeout=30.0)
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
    and recommend an action via 2B + triad.

    Pipeline: embed -> gather_procedural_context (L2-L4 biased) -> decide_synthesize (2B)
    -> judge_topic (triad) -> judge_answer -> return structured decision.

    Args:
        situation: Description of the situation the agent faces
        options: Optional list of candidate actions to evaluate

    Returns:
        {situation, action, reasoning, confidence, relevant_patterns, risks,
         option_scores, timing, triad}
    """
    import time
    from judges import judge_topic, judge_answer

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

    # Step 4: Run triad automatically (every decide() gets triad judgment)
    t4 = time.time()
    triad = await judge_topic(situation)
    triad_time = time.time() - t4

    # Step 5: Judge answer with triad context
    t5 = time.time()
    triad_answer = await judge_answer(situation, triad)
    judge_time = time.time() - t5

    # Use triad answer confidence if higher
    triad_conf = triad_answer.get("confidence", 0)
    if triad_conf > decision.get("confidence", 0):
        decision["action"] = triad_answer["answer"]
        decision["reasoning"] = triad_answer.get("reasoning", decision["reasoning"])
        decision["confidence"] = triad_conf

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
            "triad_s": round(triad_time, 2),
            "judge_s": round(judge_time, 2),
            "total_s": round(total_time, 2),
        },
        "triad": triad,
    }
