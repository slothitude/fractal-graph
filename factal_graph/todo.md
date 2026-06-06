# Fractal Graph — TODO

## Completed (GPU Fix + Benchmark)

- [x] Kill Epic Games Launcher on Lappy (frees GPU resources)
- [x] Remove `OLLAMA_GPU_OVERHEAD` from registry (was causing miscalculation)
- [x] Restart Ollama clean (no stale llama-server processes)
- [x] Fix qwen3.5 thinking tokens — root cause of timeouts, NOT GPU
  - `seed.py` — `_mother_generate()`: added `"think": false`, `num_predict: 512`, timeout 300s→120s
  - `seed.py` — pass 2 scaffold: `num_predict=1024` (longer JSON output)
  - `ingest.py` — `_classify_with_model()`: added `"think": false`, `num_predict: 20`
- [x] Fix `query.py` — `top_k` param mismatch with `chroma_store.query_level(n_results)`
- [x] Create `start_ollama_gpu.bat` on Lappy with all GPU env vars inline
- [x] Benchmark seed_topic("NATO expansion", depth=3)
  - With 4b model, think:false: 86 nodes, 53 edges, ~160-180s
- [x] Benchmark seed_from_search("quantum computing", max_urls=3)
  - 85s, 25 nodes (13 pass1 + 12 pass2), 4 edges. Pass 2 fixed with num_predict=1024.
- [x] Test MCP server functions (query, search_nodes, drill_down, get_node) — all pass
- [x] Test OLLAMA_MAX_LOADED_MODELS=1 with qwen3.5:9b
  - 9b loaded in 2.8GB VRAM (evicted 4b). Works with 6GB card.
- [x] Upgrade mother model to qwen3.5:9b in config.py
- [x] Profiled seed_topic: LLM calls are 93% of time (not embedding)
  - depth=3 = 16 LLM calls × ~5s = ~80s LLM + 4s embed + 8s DB = ~160s total
- [x] Commit as `9ccd684` on master

## Outstanding

### 1. Judge triad (Angel / Devil / Neutral) — not implemented
Design doc (`factal_graph.md:128-136`) describes a judge triad that operates the
fractal graph at different resolution levels:
- **Angel** — coarse, optimistic summary level
- **Devil** — fine, adversarial evidence level
- **Neutral** — mid, cross-resolution coherence checker

Their disagreements become *resolution disagreements* — the deliberating model
learns that Angel/Devil conflict often means "zoom level mismatch" rather than
genuine contradiction.

**Action**: Implement judge triad. Each judge queries the graph at a different
default resolution. Disagreements are logged as resolution-conflict edges rather
than contradiction edges.

### 2. Parallelize LLM calls in seed pipeline
Current seed_topic makes sequential LLM calls per node expansion. Children at
the same level are independent — could use `asyncio.gather` to parallelize.
With 16 calls at ~5s each, parallelizing could cut LLM time from 80s to ~20s.

### 3. seed_expand untested
`seed_expand` was not benchmarked. It uses `_mother_generate` and should work
with the think:false fix.

### 4. Judge triad integration with Pantheon council/court
The design doc mentions connecting judges to the Pantheon council/court system
for multi-perspective deliberation. This is a future integration point.
