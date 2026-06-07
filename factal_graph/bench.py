"""Pipeline benchmarking — compare answer quality across modes using pipeline metrics.

Modes:
  1. 2B + graph context (our pipeline)
  2. 2B standalone (no graph, just the question)
  3. Mother standalone (~8B, no graph, just the question)
  4. 2B + graph + auto_expand (full expansion pipeline)
  5. 2B + graph + judge triad (two-pass Angel/Devil/Neutral)
  6. 2B + graph + triad + parallel (ask + triad pass 1 batched)

No LLM-as-judge — the pipeline provides its own metrics:
  confidence, key_facts, gaps, context_tokens, timing, expanded, nodes_added.
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "data"
RESULTS_DIR.mkdir(exist_ok=True)


async def _warmup(model: str, url: str) -> float:
    """Trigger model load. Uses shared cache to skip if already loaded.

    Returns:
        Load time in seconds.
    """
    from model_cache import ensure_model_loaded
    return await ensure_model_loaded(model, url)

TEST_QUESTIONS = [
    # Existing graph topics (should score high)
    {"q": "What is NATO?", "expected_level": 0},
    {"q": "What is Article 5 collective defense?", "expected_level": 2},
    {"q": "Which countries joined NATO between 1999 and 2020?", "expected_level": 4},
    {"q": "What did Putin say about NATO expansion in 2008?", "expected_level": 5},
    # New topics (tests expansion — low confidence triggers gap fill)
    {"q": "What is the economic impact of sea level rise?", "expected_level": 2},
    {"q": "How does quantum error correction work?", "expected_level": 2},
    {"q": "What causes ocean acidification?", "expected_level": 3},
    # High-confidence topics (should NOT expand)
    {"q": "Why did Russia oppose NATO expansion?", "expected_level": 1},
]


def _extract_structured(raw: str) -> dict:
    """Parse JSON with answer, key_facts, gaps from LLM output."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.+?\}', raw, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                return {"answer": raw[:500], "key_facts": [], "gaps": []}
        else:
            m2 = re.search(r'"answer"\s*:\s*"(.+?)"', raw, re.DOTALL)
            return {
                "answer": m2.group(1)[:500] if m2 else raw[:500],
                "key_facts": [],
                "gaps": [],
            }
    return {
        "answer": parsed.get("answer", raw[:500])[:500],
        "key_facts": parsed.get("key_facts", []),
        "gaps": parsed.get("gaps", []),
    }


async def _call_2b_graph(question: str, auto_expand: bool = False) -> dict:
    """Mode 1/4: 2B + graph context (our pipeline)."""
    from reasoning import answer
    t0 = time.time()
    result = await answer(question, auto_expand=auto_expand)
    result["time_s"] = round(time.time() - t0, 2)
    return result


async def _call_2b_standalone(question: str) -> dict:
    """Mode 2: 2B standalone — no graph context."""
    import httpx
    t0 = time.time()
    await _warmup(settings.llm_model, settings.llm_url)
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{settings.llm_url}/api/generate",
            json={
                "model": settings.llm_model,
                "prompt": (
                    "Answer this question concisely (2-3 sentences). "
                    "Return JSON: {\"answer\": \"...\", "
                    "\"key_facts\": [\"fact1\"], "
                    "\"gaps\": [\"missing info\"]}\n\n"
                    f"Question: {question}"
                ),
                "stream": False,
                "keep_alive": "0",
                "options": {"temperature": 0.2},
                "think": False,
            },
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
    parsed = _extract_structured(raw)
    return {
        "answer": parsed["answer"],
        "confidence": 0.0,
        "key_facts": parsed["key_facts"],
        "gaps": parsed["gaps"],
        "context_tokens": 0,
        "expanded": False,
        "nodes_added": 0,
        "time_s": round(time.time() - t0, 2),
    }


async def _call_mother_standalone(question: str) -> dict:
    """Mode 3: Mother standalone (~8B) — no graph context.

    Uses /api/generate with format:json — lfm2.5 is a thinking model that
    ignores think:false and burns all tokens on <think > blocks. format:json
    forces clean JSON output without thinking tag wrapping.
    """
    import httpx
    t0 = time.time()
    await _warmup(settings.mother_model, settings.mother_url)
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{settings.mother_url}/api/generate",
            json={
                "model": settings.mother_model,
                "prompt": (
                    "Answer this question concisely (2-3 sentences). "
                    "Return JSON: {\"answer\": \"...\", "
                    "\"key_facts\": [\"fact1\"], "
                    "\"gaps\": [\"missing info\"]}\n\n"
                    f"Question: {question}"
                ),
                "stream": False,
                "keep_alive": "0",
                "format": "json",
                "options": {"temperature": 0.2},
            },
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
    parsed = _extract_structured(raw)
    return {
        "answer": parsed["answer"],
        "confidence": 0.0,
        "key_facts": parsed["key_facts"],
        "gaps": parsed["gaps"],
        "context_tokens": 0,
        "expanded": False,
        "nodes_added": 0,
        "time_s": round(time.time() - t0, 2),
    }


async def _call_2b_graph_triad(question: str) -> dict:
    """Mode 5: 2B + graph + judge triad (two-pass)."""
    from judges import judge_topic, judge_answer

    t0 = time.time()

    # Pass 1: Judge triad verdicts (single 2B call)
    judge_result = await judge_topic(question)
    triad_time = round(time.time() - t0, 2)

    # Pass 2: 2B answers with triad verdicts
    result = await judge_answer(question, judge_result)

    total_time = round(time.time() - t0, 2)
    return {
        "answer": result["answer"],
        "confidence": result["confidence"],
        "key_facts": result["key_facts"],
        "gaps": result["gaps"],
        "context_tokens": 0,
        "expanded": False,
        "nodes_added": 0,
        "search_fallback": False,
        "time_s": total_time,
        "triad_time_s": triad_time,
        "verdicts": judge_result.get("verdicts", {}),
        "conflicts": judge_result.get("conflicts", []),
        "edges_created": judge_result.get("edges_created", []),
    }


async def _call_2b_graph_triad_parallel(question: str) -> dict:
    """Mode 6: 2B + graph + triad parallel (ask + triad pass 1 batched)."""
    from reasoning import answer_with_triad

    t0 = time.time()
    result = await answer_with_triad(question)
    total_time = round(time.time() - t0, 2)

    triad = result.get("triad", {})
    return {
        "answer": result["answer"],
        "confidence": result["confidence"],
        "key_facts": result["key_facts"],
        "gaps": result["gaps"],
        "context_tokens": result.get("context_tokens", 0),
        "expanded": result.get("expanded", False),
        "nodes_added": result.get("nodes_added", 0),
        "search_fallback": result.get("search_fallback", False),
        "time_s": total_time,
        "triad_time_s": result.get("timing", {}).get("total_with_triad_s", total_time),
        "verdicts": triad.get("verdicts", {}),
        "conflicts": triad.get("conflicts", []),
        "edges_created": triad.get("edges_created", []),
    }


def _fmt_mode(data: dict) -> str:
    """Format one mode result as a single-line summary."""
    conf = data.get("confidence", 0.0)
    facts = len(data.get("key_facts", []))
    gaps = len(data.get("gaps", []))
    ctx = data.get("context_tokens", 0)
    exp = data.get("expanded", False)
    t = data.get("time_s", 0)
    nodes = data.get("nodes_added", 0)

    parts = [f"{t:>6.1f}s"]
    parts.append(f"conf={conf:.2f}" if conf else "conf=  -")
    parts.append(f"facts={facts}")
    parts.append(f"gaps={gaps}")
    if ctx:
        parts.append(f"ctx={ctx}tok")
    if data.get("search_fallback"):
        parts.append("search_fb=T")
    if exp:
        parts.append(f"expanded=T +{nodes}nodes")
    else:
        parts.append("expanded=F")
    # Triad-specific info
    conflicts = data.get("conflicts", [])
    if conflicts:
        parts.append(f"triad_conflicts={len(conflicts)}")
    triad_time = data.get("triad_time_s")
    if triad_time:
        parts.append(f"triad_pass1={triad_time}s")
    return "  ".join(parts)


async def run_bench():
    """Run benchmark comparing all four modes with pipeline metrics."""
    results = []
    total_t0 = time.time()

    for test in TEST_QUESTIONS:
        q = test["q"]
        level = test["expected_level"]
        print(f"\n{'='*70}")
        print(f"Q (L{level}): {q}")
        print(f"{'='*70}")

        entry = {
            "question": q,
            "expected_level": level,
        }

        # Mode 1: 2B + graph
        print("  [1/6] 2B + graph...")
        try:
            entry["graph_2b"] = await _call_2b_graph(q, auto_expand=False)
            print(f"       {_fmt_mode(entry['graph_2b'])}")
        except Exception as e:
            entry["graph_2b"] = {"answer": f"ERROR: {e}", "time_s": 0, "key_facts": [], "gaps": [], "confidence": 0.0, "context_tokens": 0, "expanded": False, "nodes_added": 0, "search_fallback": False}
            print(f"       ERROR: {e}")

        # Mode 2: 2B standalone
        print("  [2/6] 2B standalone...")
        try:
            entry["standalone_2b"] = await _call_2b_standalone(q)
            print(f"       {_fmt_mode(entry['standalone_2b'])}")
        except Exception as e:
            entry["standalone_2b"] = {"answer": f"ERROR: {e}", "time_s": 0, "key_facts": [], "gaps": [], "confidence": 0.0, "context_tokens": 0, "expanded": False, "nodes_added": 0, "search_fallback": False}
            print(f"       ERROR: {e}")

        # Mode 3: Mother standalone
        print("  [3/6] Mother standalone...")
        try:
            entry["standalone_mother"] = await _call_mother_standalone(q)
            print(f"       {_fmt_mode(entry['standalone_mother'])}")
        except Exception as e:
            entry["standalone_mother"] = {"answer": f"ERROR: {e}", "time_s": 0, "key_facts": [], "gaps": [], "confidence": 0.0, "context_tokens": 0, "expanded": False, "nodes_added": 0, "search_fallback": False}
            print(f"       ERROR: {e}")

        # Mode 4: 2B + graph + auto_expand
        print("  [4/6] 2B + graph + auto_expand...")
        try:
            entry["graph_2b_expand"] = await _call_2b_graph(q, auto_expand=True)
            print(f"       {_fmt_mode(entry['graph_2b_expand'])}")
        except Exception as e:
            entry["graph_2b_expand"] = {"answer": f"ERROR: {e}", "time_s": 0, "key_facts": [], "gaps": [], "confidence": 0.0, "context_tokens": 0, "expanded": False, "nodes_added": 0, "search_fallback": False}
            print(f"       ERROR: {e}")

        # Mode 5: 2B + graph + triad
        print("  [5/6] 2B + graph + triad...")
        try:
            entry["graph_2b_triad"] = await _call_2b_graph_triad(q)
            print(f"       {_fmt_mode(entry['graph_2b_triad'])}")
        except Exception as e:
            entry["graph_2b_triad"] = {"answer": f"ERROR: {e}", "time_s": 0, "key_facts": [], "gaps": [], "confidence": 0.0, "context_tokens": 0, "expanded": False, "nodes_added": 0, "search_fallback": False}
            print(f"       ERROR: {e}")

        # Mode 6: 2B + graph + triad parallel
        print("  [6/6] 2B + graph + triad parallel...")
        try:
            entry["graph_2b_triad_parallel"] = await _call_2b_graph_triad_parallel(q)
            print(f"       {_fmt_mode(entry['graph_2b_triad_parallel'])}")
        except Exception as e:
            entry["graph_2b_triad_parallel"] = {"answer": f"ERROR: {e}", "time_s": 0, "key_facts": [], "gaps": [], "confidence": 0.0, "context_tokens": 0, "expanded": False, "nodes_added": 0, "search_fallback": False}
            print(f"       ERROR: {e}")

        results.append(entry)

    total_time = round(time.time() - total_t0, 1)

    # Per-question detail table
    print(f"\n{'='*70}")
    print("DETAILED RESULTS — PIPELINE METRICS")
    print(f"{'='*70}")
    print(f"{'Q (L?)':<8} {'Mode':<8} {'Time':>6} {'Conf':>5} {'Facts':>5} {'Gaps':>4} {'Ctx':>6} {'Expanded':>10} {'Search':>6} {'Triad':>6}")
    print("-" * 88)

    for r in results:
        q_label = f"{r['question'][:30]}... (L{r['expected_level']})"
        first = True
        for mode_key, label in [
            ("graph_2b", "2B+Gr"),
            ("standalone_2b", "2B"),
            ("standalone_mother", "Mom"),
            ("graph_2b_expand", "2B+Ex"),
            ("graph_2b_triad", "2B+Tr"),
            ("graph_2b_triad_parallel", "2B+TP"),
        ]:
            d = r.get(mode_key, {})
            t = d.get("time_s", 0)
            c = d.get("confidence", 0.0)
            f = len(d.get("key_facts", []))
            g = len(d.get("gaps", []))
            ctx = d.get("context_tokens", 0)
            exp = "T +" + str(d.get("nodes_added", 0)) if d.get("expanded") else "F"
            sfb = "T" if d.get("search_fallback") else "-"
            triad_c = len(d.get("conflicts", []))
            triad_str = f"{triad_c}conf" if triad_c else "-"
            row_label = q_label[:36] if first else ""
            first = False
            c_str = f"{c:.2f}" if c else " -"
            ctx_str = f"{ctx}tok" if ctx else "-"
            print(f"{row_label:<36} {label:<8} {t:>5.1f}s {c_str:>5} {f:>5} {g:>4} {ctx_str:>6} {exp:>10} {sfb:>6} {triad_str:>6}")

    # Aggregate averages per mode
    print(f"\n{'='*70}")
    print("AGGREGATE AVERAGES")
    print(f"{'='*70}")

    MODES = [
        ("graph_2b", "2B+Graph"),
        ("standalone_2b", "2B alone"),
        ("standalone_mother", "Mother alone"),
        ("graph_2b_expand", "2B+Graph+Expand"),
        ("graph_2b_triad", "2B+Graph+Triad"),
        ("graph_2b_triad_parallel", "2B+Grp+Triad+Par"),
    ]

    for mode_key, label in MODES:
        entries = [r[mode_key] for r in results if mode_key in r]
        n = len(entries)
        if not n:
            print(f"  {label:<22} (no results)")
            continue
        avg_conf = sum(e.get("confidence", 0.0) for e in entries) / n
        avg_facts = sum(len(e.get("key_facts", [])) for e in entries) / n
        avg_gaps = sum(len(e.get("gaps", [])) for e in entries) / n
        avg_time = sum(e.get("time_s", 0) for e in entries) / n
        expanded_count = sum(1 for e in entries if e.get("expanded"))
        total_nodes = sum(e.get("nodes_added", 0) for e in entries)
        avg_ctx = sum(e.get("context_tokens", 0) for e in entries) / n

        extra = ""
        if mode_key in ("graph_2b_triad", "graph_2b_triad_parallel"):
            avg_triad = sum(e.get("triad_time_s", 0) for e in entries) / n
            total_conflicts = sum(len(e.get("conflicts", [])) for e in entries)
            extra = f"  avg_triad_pass1={avg_triad:.1f}s  conflicts={total_conflicts}"

        print(f"  {label:<22} avg_conf={avg_conf:.2f}  avg_facts={avg_facts:.1f}  "
              f"avg_gaps={avg_gaps:.1f}  avg_ctx={avg_ctx:.0f}tok  "
              f"avg_time={avg_time:.1f}s  expanded={expanded_count}/{n}  "
              f"nodes={total_nodes}{extra}")

    print(f"\nTotal bench time: {total_time}s")

    # Persist results with timestamp
    run_timestamp = datetime.now(timezone.utc).isoformat()
    run_data = {
        "timestamp": run_timestamp,
        "total_time_s": total_time,
        "results": results,
    }

    # Append to history file
    history_path = RESULTS_DIR / "bench_history.jsonl"
    with open(history_path, "a") as f:
        f.write(json.dumps(run_data, default=str) + "\n")

    # Also save latest standalone
    with open(RESULTS_DIR / "bench_results.json", "w") as f:
        json.dump(run_data, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_DIR / 'bench_results.json'}")
    print(f"History appended to {history_path}")


async def run_distill_compare(distill_domain: str = None):
    """Run bench before and after distilling a domain, show delta."""
    from distill import distill_domain as distill_fn

    print("=" * 70)
    print("BEFORE DISTILLATION")
    print("=" * 70)
    before_results = await _run_bench_inner()

    if distill_domain:
        print(f"\n{'='*70}")
        print(f"DISTILLING: {distill_domain}")
        print(f"{'='*70}")
        distill_result = await distill_fn(distill_domain)
        print(f"  Nodes created: {distill_result.get('nodes_created', 0)}")
        print(f"  Edges created: {distill_result.get('edges_created', 0)}")
        pt = distill_result.get("phase_times", {})
        print(f"  Time: gen={pt.get('generate', '?')}s embed={pt.get('embed', '?')}s store={pt.get('store', '?')}s")

    print(f"\n{'='*70}")
    print("AFTER DISTILLATION")
    print("=" * 70)
    after_results = await _run_bench_inner()

    # Print comparison
    print(f"\n{'='*70}")
    print("BEFORE vs AFTER (2B+Graph mode only)")
    print(f"{'='*70}")
    print(f"{'Question':<40} {'Before':>6} {'After':>6} {'Delta':>6}")
    print("-" * 62)

    for before, after in zip(before_results, after_results):
        q = before["question"][:38] + "..."
        b_conf = before.get("graph_2b", {}).get("confidence", 0.0)
        a_conf = after.get("graph_2b", {}).get("confidence", 0.0)
        delta = a_conf - b_conf
        sign = "+" if delta >= 0 else ""
        b_str = f"{b_conf:.2f}" if b_conf else "  -"
        a_str = f"{a_conf:.2f}" if a_conf else "  -"
        d_str = f"{sign}{delta:.2f}" if (b_conf or a_conf) else "  -"
        print(f"{q:<40} {b_str:>6} {a_str:>6} {d_str:>6}")

    # Save comparison
    run_timestamp = datetime.now(timezone.utc).isoformat()
    comp_data = {
        "timestamp": run_timestamp,
        "distill_domain": distill_domain,
        "before": before_results,
        "after": after_results,
    }
    comp_path = RESULTS_DIR / "bench_distill_compare.json"
    with open(comp_path, "w") as f:
        json.dump(comp_data, f, indent=2, default=str)
    print(f"\nComparison saved to {comp_path}")


async def _run_bench_inner() -> list[dict]:
    """Run bench and return results list (without printing table)."""
    results = []
    for test in TEST_QUESTIONS:
        q = test["q"]
        entry = {"question": q, "expected_level": test["expected_level"]}
        try:
            entry["graph_2b"] = await _call_2b_graph(q, auto_expand=False)
        except Exception as e:
            entry["graph_2b"] = {"confidence": 0.0, "error": str(e)}
        results.append(entry)
    return results


# --- Agent Decision Benchmark -- compare models on procedural knowledge ---

DECIDE_QUESTIONS = [
    {"q": "A tool call returned an unexpected format", "expected": "retry with schema check"},
    {"q": "I don't have enough information to complete the task", "expected": "ask for clarification"},
    {"q": "The user request is vague and could mean multiple things", "expected": "ask for clarification"},
    {"q": "A file operation failed with permission denied", "expected": "check permissions or escalate"},
    {"q": "The test suite is passing but coverage is low", "expected": "prioritize high-risk areas"},
]


def _fmt_decide(data: dict) -> str:
    """Format one decide result as a single-line summary."""
    conf = data.get("confidence", 0.0)
    t = data.get("time_s", 0)
    patterns = len(data.get("relevant_patterns", []))
    risks = len(data.get("risks", []))
    action = data.get("action", "")[:40]
    parts = [f"{t:>6.1f}s", f"conf={conf:.2f}", f"pats={patterns}", f"risks={risks}"]
    parts.append(f"act={action}")
    return "  ".join(parts)


async def _call_decide(question: str) -> dict:
    """Call decide() with current LLM model."""
    from reasoning import decide
    t0 = time.time()
    result = await decide(question)
    result["time_s"] = round(time.time() - t0, 2)
    return result


async def run_decide_bench():
    """Bench decide() across 2B models: qwen3.5:2b and qwen3.5:0.8b.

    Tests 5 agent-oriented procedural knowledge questions.
    Compares confidence, action quality, relevant_patterns, risks, timing.
    """
    import os
    original_model = settings.llm_model

    models = ["qwen3.5:2b", "qwen3.5:0.8b"]
    model_results = {}

    total_t0 = time.time()

    for model in models:
        print(f"\n{'='*70}")
        print(f"Model: {model}")
        print(f"{'='*70}")

        # Override LLM model for this run
        settings.llm_model = model
        # Clear model cache so new model loads
        from model_cache import invalidate
        invalidate()

        model_results[model] = []

        for test in DECIDE_QUESTIONS:
            q = test["q"]
            expected = test["expected"]
            print(f"\n  Q: {q}")
            print(f"  Expected: {expected}")
            print(f"  {'-'*50}")

            try:
                result = await _call_decide(q)
                print(f"  {_fmt_decide(result)}")

                # Score: did the action match expected behavior?
                action_lower = result.get("action", "").lower()
                expected_lower = expected.lower()
                action_words = set(action_lower.split())
                expected_words = set(expected_lower.split())
                overlap = action_words & expected_words
                relevance = len(overlap) / max(len(expected_words), 1)

                result["relevance"] = round(relevance, 2)
                model_results[model].append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                model_results[model].append({
                    "question": q, "expected": expected,
                    "action": f"ERROR: {e}", "confidence": 0.0,
                    "time_s": 0, "relevant_patterns": [], "risks": [],
                    "relevance": 0.0,
                })

        print(f"\n  [{model}] done")

    # Restore original model
    settings.llm_model = original_model
    from model_cache import invalidate
    invalidate()

    total_time = round(time.time() - total_t0, 1)

    # Comparison table
    print(f"\n{'='*70}")
    print("DECIDE BENCH — MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Q':<50} {'2B conf':>7} {'2B rel':>5} | {'0.8b conf':>9} {'0.8b rel':>6} | {'Winner':>8}")
    print("-" * 100)

    for i, test in enumerate(DECIDE_QUESTIONS):
        q_label = test["q"][:48]
        r2 = model_results.get("qwen3.5:2b", [{}])[i]
        r08 = model_results.get("qwen3.5:0.8b", [{}])[i]
        c2 = r2.get("confidence", 0.0)
        c08 = r08.get("confidence", 0.0)
        rel2 = r2.get("relevance", 0.0)
        rel08 = r08.get("relevance", 0.0)
        c2s = f"{c2:.2f}" if c2 else "  -"
        c08s = f"{c08:.2f}" if c08 else "  -"
        r2s = f"{rel2:.2f}" if rel2 else "  -"
        r08s = f"{rel08:.2f}" if rel08 else "  -"
        winner = "2B" if c2 > c08 else ("0.8b" if c08 > c2 else "tie")
        print(f"{q_label:<50} {c2s:>7} {r2s:>5} | {c08s:>9} {r08s:>6} | {winner:>8}")

    # Aggregate averages
    print(f"\n{'='*70}")
    print("AGGREGATES")
    print(f"{'='*70}")
    for model in models:
        entries = model_results.get(model, [])
        n = len(entries)
        if not n:
            continue
        avg_conf = sum(e.get("confidence", 0.0) for e in entries) / n
        avg_rel = sum(e.get("relevance", 0.0) for e in entries) / n
        avg_time = sum(e.get("time_s", 0) for e in entries) / n
        avg_pats = sum(len(e.get("relevant_patterns", [])) for e in entries) / n
        avg_risks = sum(len(e.get("risks", [])) for e in entries) / n
        print(f"  {model:<20} n={n}  avg_conf={avg_conf:.2f}  avg_relevance={avg_rel:.2f}  "
              f"avg_time={avg_time:.1f}s  avg_patterns={avg_pats:.1f}  avg_risks={avg_risks:.1f}")

    print(f"\nTotal bench time: {total_time}s")

    # Persist
    run_timestamp = datetime.now(timezone.utc).isoformat()
    run_data = {
        "timestamp": run_timestamp,
        "bench_type": "decide",
        "total_time_s": total_time,
        "results": model_results,
    }
    with open(RESULTS_DIR / "bench_decide_results.json", "w") as f:
        json.dump(run_data, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_DIR / 'bench_decide_results.json'}")


if __name__ == "__main__":
    import sys
    if "--distill" in sys.argv:
        domain = None
        idx = sys.argv.index("--distill")
        if idx + 1 < len(sys.argv):
            domain = sys.argv[idx + 1]
        asyncio.run(run_distill_compare(distill_domain=domain))
    elif "--decide" in sys.argv:
        asyncio.run(run_decide_bench())
    else:
        asyncio.run(run_bench())
