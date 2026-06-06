# Fractal Graph — TODO

## Completed (GPU Fix + Benchmark)
- [x] Kill Epic Games Launcher on Lappy (frees GPU resources)
- [x] Remove `OLLAMA_GPU_OVERHEAD` from registry (was causing miscalculation)
- [x] Restart Ollama clean (no stale llama-server processes)
- [x] Fix qwen3.5 thinking tokens — root cause of timeouts, NOT GPU
- [x] Fix `query.py` — `top_k` param mismatch with `chroma_store.query_level(n_results)`
- [x] Upgrade mother model to qwen3.5:9b in config.py
- [x] Judge triad (Angel/Devil/Neutral) — implemented in judges.py
- [x] Commit as `c720a92` and `2b9f49c` on master

---

## Graph-Context Injection — Make the 2B Model Actually Reason

### Phase 1: Fix Foundations
- [x] Fix `embed_sync()` in embedder.py — broken `get_event_loop().run_until_complete()`, use `asyncio.run()` with fallback
- [x] Fix `drill_down()` in query.py — walks children of current node, not entire next level; move `db.get_db()` outside loop
- [x] Fix ingest 9B escalation in ingest.py — replace `classify_resolution_llm()` 9B escalation with `find_best_level_by_structure()` (bbox containment)

### Phase 2: Create context.py
- [x] `gather_context()` — search all 6 levels, fetch nodes, walk parents, dedup, rank
- [x] `format_context_for_llm()` — structured text for 2B
- [x] `walk_graph()` — LaRQL WALK equivalent
- [x] `find_best_level_by_structure()` — place text via bbox containment

### Phase 3: Create reasoning.py
- [x] `classify_question()` — 2B classifies question type
- [x] `synthesize()` — 2B generates answer from graph context
- [x] `answer()` — top-level pipeline: embed → classify → gather → synthesize

### Phase 4: gql.py + config.py
- [x] `find_contradictions_of()`, `find_evidence_for()`, `compare_entities()`
- [x] Config: max_context_tokens, max_answer_tokens

### Phase 5: Server + Benchmark
- [x] `ask()` MCP tool — primary user-facing tool
- [x] bench.py — 4 modes: 2B+graph, 2B standalone, 8B standalone, 2B+graph+auto_expand
- [x] Fix ChromaDB concurrent access corruption
- [x] Verify seed_topic, web_ingest, seed_expand

---

## Autonomous Graph Self-Expansion — Mother Decomposes Itself

### Phase 1: growth.py
- [x] `fill_gaps()` — mother fills gaps with level-aware dedup
- [x] `enrich_topic()` — fire-and-forget topic decomposition
- [x] `curiosity_scan()` — find sparse nodes, expand via seed_expand
- [x] `detect_sparse_nodes()` — SQL for leaf-below-L5, few-children
- [x] Asyncio locks keyed on topic (prevents concurrent enrichment race)
- [x] Level-aware dedup thresholds (L0:0.95 → L5:0.80)

### Phase 2: reasoning.py
- [x] `answer(auto_expand=True, _round=0)` — auto-expand on low confidence
- [x] Background enrichment via asyncio.ensure_future
- [x] max_expansion_rounds guard

### Phase 3: Server + config
- [x] `ask(auto_expand=True)` — pass through to pipeline
- [x] `curiosity_scan()` MCP tool
- [x] Config: threshold, max nodes, max rounds

---

## Web Search Fallback + Model Loading Stability

### Problem
1. Cold-start model loading (~26s) kills httpx timeouts
2. `keep_alive` default holds VRAM, blocking subsequent model calls
3. Mother hallucinates when graph lacks info — populates with fake nodes
4. Background enrich races with gap-fill for model slot → RecursionError

### Fixes Applied
- [x] `_warmup_model()` — trigger load, poll `/api/ps` until ready, no timeouts
- [x] `_warmup()` in bench.py standalone calls — timed output
- [x] `keep_alive: "0"` on all 4 LLM call sites — free VRAM after each call
- [x] Web search fallback in `fill_gaps()` — `seed_from_search()` when mother creates 0 nodes
- [x] Recursion guards — background enrich round 0 only, skip recursion on parse failure
- [x] `search_fallback` tracking in response dict + bench `search_fb` column

### Verification
- [ ] Run `python -u bench.py` — all 4 modes complete, no crashes
- [ ] Non-graph topics (quantum error correction) trigger search fallback
- [ ] Graph-native topics (NATO) do NOT trigger search fallback
- [ ] No RecursionError in background enrichment

---

## Outstanding (future)
- Parallelize LLM calls in seed pipeline with asyncio.gather
- Judge triad integration with Pantheon council/court
- Cache model load state to avoid cold-starts on repeated calls
