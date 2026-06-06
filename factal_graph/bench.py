"""Quality benchmarking — compare answer quality across modes.

Modes:
  1. 2B + graph context (our pipeline)
  2. 2B standalone (no graph, just the question)
  3. 9B standalone (no graph, just the question)

Test questions at L0/L2/L4/L5. Logs answer text, time, token usage, confidence.
"""

import asyncio
import json
import time

from config import settings


TEST_QUESTIONS = [
    # L0 — domain level
    {"q": "What is NATO?", "expected_level": 0},
    # L2 — concept level
    {"q": "What is Article 5 collective defense?", "expected_level": 2},
    # L4 — fact level
    {"q": "Which countries joined NATO between 1999 and 2020?", "expected_level": 4},
    # L5 — evidence level
    {"q": "What did Putin say about NATO expansion in 2008?", "expected_level": 5},
]


async def _call_2b_graph(question: str) -> dict:
    """Mode 1: 2B + graph context (our pipeline)."""
    from reasoning import answer
    t0 = time.time()
    result = await answer(question)
    result["time_s"] = round(time.time() - t0, 2)
    return result


async def _call_2b_standalone(question: str) -> dict:
    """Mode 2: 2B standalone — no graph context."""
    import httpx
    t0 = time.time()
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{settings.llm_url}/api/generate",
            json={
                "model": settings.llm_model,
                "prompt": f"Answer this question concisely (2-3 sentences):\n{question}",
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 256},
                "think": False,
            },
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
    return {
        "answer": raw[:500],
        "confidence": 0.0,  # Can't measure without structured output
        "time_s": round(time.time() - t0, 2),
    }


async def _call_9b_standalone(question: str) -> dict:
    """Mode 3: 9B standalone — no graph context."""
    import httpx
    t0 = time.time()
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{settings.mother_url}/api/generate",
            json={
                "model": settings.mother_model,
                "prompt": f"Answer this question concisely (2-3 sentences):\n{question}",
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 256},
                "think": False,
            },
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
    return {
        "answer": raw[:500],
        "confidence": 0.0,
        "time_s": round(time.time() - t0, 2),
    }


async def run_bench():
    """Run benchmark comparing all three modes."""
    results = []

    for test in TEST_QUESTIONS:
        q = test["q"]
        level = test["expected_level"]
        print(f"\n{'='*60}")
        print(f"Q (L{level}): {q}")
        print(f"{'='*60}")

        entry = {"question": q, "expected_level": level}

        # Mode 1: 2B + graph
        print("  [1/3] 2B + graph context...")
        try:
            entry["graph_2b"] = await _call_2b_graph(q)
            print(f"       {entry['graph_2b']['time_s']}s, conf={entry['graph_2b'].get('confidence', 'N/A')}")
        except Exception as e:
            entry["graph_2b"] = {"answer": f"ERROR: {e}", "time_s": 0}
            print(f"       ERROR: {e}")

        # Mode 2: 2B standalone
        print("  [2/3] 2B standalone...")
        try:
            entry["standalone_2b"] = await _call_2b_standalone(q)
            print(f"       {entry['standalone_2b']['time_s']}s")
        except Exception as e:
            entry["standalone_2b"] = {"answer": f"ERROR: {e}", "time_s": 0}
            print(f"       ERROR: {e}")

        # Mode 3: 9B standalone
        print("  [3/3] 9B standalone...")
        try:
            entry["standalone_9b"] = await _call_9b_standalone(q)
            print(f"       {entry['standalone_9b']['time_s']}s")
        except Exception as e:
            entry["standalone_9b"] = {"answer": f"ERROR: {e}", "time_s": 0}
            print(f"       ERROR: {e}")

        results.append(entry)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Question':<40} {'2B+Graph':>10} {'2B':>6} {'9B':>6}")
    print("-" * 64)

    for r in results:
        q_short = r["question"][:38]
        t_graph = r.get("graph_2b", {}).get("time_s", "ERR")
        t_2b = r.get("standalone_2b", {}).get("time_s", "ERR")
        t_9b = r.get("standalone_9b", {}).get("time_s", "ERR")
        print(f"{q_short:<40} {str(t_graph):>10} {str(t_2b):>6} {str(t_9b):>6}")

    # Save full results
    with open("data/bench_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to data/bench_results.json")


if __name__ == "__main__":
    asyncio.run(run_bench())
