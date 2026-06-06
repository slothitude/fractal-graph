"""Quick speed test — compare mother model candidates side-by-side.

Tests: lfm2.5 vs granite4.1:8b on seeding prompt generation.
Records to data/speed_test_results.json and appends to bench_history.jsonl.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

MODELS = [
    ("granite4.1:8b", "http://100.84.161.63:11434"),
    ("lfm2.5:latest", "http://100.84.161.63:11434"),
]

TEST_PROMPTS = [
    # Short (classify-like)
    'Classify: What is NATO? Return JSON: {"type": "factual", "entities": ["NATO"]}',
    # Medium (seed-like)
    'Generate a knowledge hierarchy for "NATO expansion" at 3 levels (L0-L2). Return JSON array of nodes with content, level, and parent_id.',
    # Long (seed_from_search extraction)
    'Extract 5 key facts about the 2014 Crimea annexation from the following context. For each fact, provide: content, resolution_level (4=fact), and confidence (0.0-1.0). Context: In February 2014, Russian forces occupied Crimea. A referendum was held on March 16, 2014, with 97% voting to join Russia. The international community largely did not recognize the annexation. NATO condemned the action and increased its presence in Eastern Europe. Russia maintained that the referendum was legal and that Crimeans exercised their right to self-determination.',
]

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


async def warmup(model: str, url: str) -> float:
    """Warmup model and return load time."""
    import time as _t
    t0 = _t.time()
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            await client.post(
                f"{url}/api/generate",
                json={"model": model, "prompt": ".", "stream": False,
                      "options": {"num_predict": 1}, "think": False},
            )
    except Exception:
        pass
    for _ in range(120):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{url}/api/ps")
                if any(m["name"] == model for m in resp.json().get("models", [])):
                    return round(_t.time() - t0, 1)
        except Exception:
            pass
        await asyncio.sleep(1.0)
    return round(_t.time() - t0, 1)


async def test_model(model: str, url: str, prompt: str, label: str) -> dict:
    """Run one prompt through a model and return timing + output."""
    load_time = await warmup(model, url)
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "0",
                    "options": {"temperature": 0.3, "num_predict": 512},
                    "think": False,
                },
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
    except Exception as e:
        raw = f"ERROR: {e}"
    gen_time = round(time.time() - t0, 2)
    return {
        "model": model,
        "label": label,
        "load_time_s": load_time,
        "gen_time_s": gen_time,
        "total_time_s": round(load_time + gen_time, 2),
        "output_len": len(raw),
        "output_preview": raw[:200],
    }


async def run_speed_test():
    results = []
    print("=" * 70)
    print("SPEED TEST — Mother Model Comparison")
    print("=" * 70)

    for i, prompt in enumerate(TEST_PROMPTS):
        labels = ["short (classify)", "medium (seed)", "long (extract)"]
        print(f"\n--- Test {i+1}: {labels[i]} ---")
        for model, url in MODELS:
            print(f"  {model}...", end="", flush=True)
            r = await test_model(model, url, prompt, labels[i])
            print(f" load={r['load_time_s']}s  gen={r['gen_time_s']}s  total={r['total_time_s']}s  out={r['output_len']}ch")
            results.append(r)

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Test':<20} {'Load':>6} {'Gen':>6} {'Total':>7} {'Output':>7}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<20} {r['label']:<20} {r['load_time_s']:>5.1f}s {r['gen_time_s']:>5.1f}s {r['total_time_s']:>6.1f}s {r['output_len']:>6}ch")

    # Per-model averages
    print(f"\n--- Averages ---")
    for model, _ in MODELS:
        model_results = [r for r in results if r["model"] == model]
        avg_load = sum(r["load_time_s"] for r in model_results) / len(model_results)
        avg_gen = sum(r["gen_time_s"] for r in model_results) / len(model_results)
        avg_total = sum(r["total_time_s"] for r in model_results) / len(model_results)
        avg_out = sum(r["output_len"] for r in model_results) / len(model_results)
        print(f"  {model:<20} avg_load={avg_load:.1f}s  avg_gen={avg_gen:.1f}s  avg_total={avg_total:.1f}s  avg_out={avg_out:.0f}ch")

    # Save results
    run_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "speed_test",
        "results": results,
    }

    with open(DATA_DIR / "speed_test_results.json", "w") as f:
        json.dump(run_data, f, indent=2, default=str)

    # Also append to bench_history.jsonl
    with open(DATA_DIR / "bench_history.jsonl", "a") as f:
        f.write(json.dumps(run_data, default=str) + "\n")

    print(f"\nResults saved to data/speed_test_results.json")
    print(f"Appended to data/bench_history.jsonl")


if __name__ == "__main__":
    asyncio.run(run_speed_test())
