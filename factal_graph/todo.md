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

### Goal
Pipeline: User Question -> Embed+Retrieve -> Gather Context -> Classify (2B) -> Synthesize (2B) -> Answer
9B only needed for seeding. All runtime operations are 2B-only.

### Phase 1: Fix Foundations
- [x] Fix `embed_sync()` in embedder.py — broken `get_event_loop().run_until_complete()`, use `asyncio.run()` with fallback
- [x] Fix `drill_down()` in query.py — walks children of current node, not entire next level; move `db.get_db()` outside loop
- [x] Fix ingest 9B escalation in ingest.py — replace `classify_resolution_llm()` 9B escalation with `find_best_level_by_structure()` (bbox containment)

### Phase 2: Create context.py (~250 lines)
- [x] `gather_context(prompt_embedding, top_k=10, max_tokens=3000) -> dict`
  - Search all 6 levels via `query_all_levels()`
  - Fetch nodes from SQLite (content, confidence, edges)
  - Walk parent chains for hierarchy context
  - Follow contradiction/support edges
  - Deduplicate, rank: 0.4*vector_sim + 0.3*confidence + 0.2*edge_count + 0.1*resolution_relevance
  - Token-budget-aware truncation (prioritize: direct hits > parents > edges > siblings)
- [x] `format_context_for_llm(context: dict) -> str` — structured text for 2B (not JSON)
- [x] `walk_graph(node_id, max_hops=3, follow_edge_types=None) -> dict` — LaRQL WALK equivalent
- [x] `find_best_level_by_structure(embedding) -> int` — place text in graph using bbox containment

### Phase 3: Create reasoning.py (~200 lines)
- [x] `classify_question(question: str) -> dict` — 2B classifies: factual|analytical|comparative|exploratory|yes_no, extracts entities
- [x] `synthesize(question, context, question_type) -> dict` — 2B generates answer from structured graph context
- [x] `answer(question) -> dict` — top-level pipeline: embed -> classify -> gather -> synthesize

### Phase 4: Create gql.py (~150 lines) + Update config.py
- [x] `find_contradictions_of(entity) -> list[dict]` — DESCRIBE WHERE contradicts
- [x] `find_evidence_for(fact_id) -> list[dict]` — WALK FOLLOW supports/contradicts
- [x] `compare_entities(entity_a, entity_b) -> dict` — DESCRIBE "X" vs "Y"
- [x] Add to config.py: `max_context_tokens: int = 2048`, `max_answer_tokens: int = 512`

### Phase 5: Server + Benchmark
- [x] Add `ask(question) -> str` MCP tool to factal_server.py — primary user-facing tool
- [x] Create bench.py — compare 2B+graph vs 9B standalone vs 2B standalone
- [x] Test: `ask("Why did Russia oppose NATO expansion?")` via MCP — works, 4.64s, conf 0.75
- [x] Fix query_graph top_k→n_results param mismatch
- [x] Fix ChromaDB concurrent access corruption (fresh client per query_all_levels)
- [x] Commit and push as `077578a`
- [ ] Verify seed_topic still works (9B untouched)
- [ ] Verify web_ingest no longer escalates to 9B

---

## Outstanding (from before)

### Parallelize LLM calls in seed pipeline
Current seed_topic makes sequential LLM calls per node expansion. Children at
the same level are independent — could use `asyncio.gather` to parallelize.
With 16 calls at ~5s each, parallelizing could cut LLM time from 80s to ~20s.

### seed_expand untested
`seed_expand` was not benchmarked. It uses `_mother_generate` and should work
with the think:false fix.

### Judge triad integration with Pantheon council/court
The design doc mentions connecting judges to the Pantheon council/court system
for multi-perspective deliberation. This is a future integration point.
