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

        # Synchronous gap fill (this IS the enrichment — no background task
        # because with keep_alive=0 only one model fits in VRAM at a time)
        from growth import fill_gaps
        fill_result = await fill_gaps(
            gaps, question, max_nodes=settings.max_gap_fill_nodes,
        )

        if fill_result["nodes_created"] > 0:
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
            response["nodes_added"] = fill_result["nodes_created"]
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
            if (fill_result["nodes_created"] > 0
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
