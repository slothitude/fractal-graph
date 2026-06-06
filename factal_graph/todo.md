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
- [x] Verify seed_topic still works — granite4.1:8b seeds 26 nodes (climate change, depth 2)
- [x] Verify web_ingest no longer escalates to 9B — 2B classifier + graph structure, L5 placement confirmed

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

---

## Autonomous Graph Self-Expansion — Mother Decomposes Itself

### Goal
Make the graph grow organically without explicit user seeding. Three mechanisms:
1. Gap-driven auto-expansion — ask() triggers mother on low confidence
2. Answer-enriches-graph — background enrichment during answering
3. Continuous curiosity scan — finds sparse areas and expands them

### Phase 1: growth.py
- [x] `fill_gaps(gaps, question, max_nodes=5)` — mother fills identified gaps with dedup
- [x] `enrich_topic(question, question_type, max_nodes=4)` — fire-and-forget topic decomposition
- [x] `curiosity_scan(max_expansions=3)` — scan for sparse nodes, use seed_expand
- [x] `detect_sparse_nodes(conn)` — SQL queries for leaf-below-L5, few-children nodes
- [x] Level-aware dedup thresholds (L0:0.95 → L5:0.80)
- [x] Asyncio locks keyed on topic (prevents concurrent enrichment race)
- [x] Gap fill prompt with nearby nodes for parent assignment

### Phase 2: reasoning.py
- [x] `answer(question, auto_expand=True, _round=0)` — auto_expand param, backward compat
- [x] Low confidence (< 0.4) or gaps detected → fill_gaps → re-answer
- [x] Background enrichment via asyncio.ensure_future (fire-and-forget)
- [x] max_expansion_rounds guard (default 2, prevents infinite loop)
- [x] Returns `expanded: True`, `nodes_added: N`, `expansion_limit_reached` flag

### Phase 3: factal_server.py + config.py
- [x] `ask(question, auto_expand=True)` — pass auto_expand to answer_fn
- [x] `curiosity_scan(max_expansions=3)` — new MCP tool
- [x] Config: auto_expand_threshold, max_gap_fill_nodes, max_enrich_nodes,
      curiosity_max_expansions, max_expansion_rounds

### Verification
- [ ] `ask("What is the economic impact of sea level rise?")` — detect gaps, fill, re-answer
- [ ] `curiosity_scan()` — find and expand sparse nodes
- [ ] `ask("What is NATO?")` — should NOT trigger expansion (high confidence)
- [ ] `graph_stats()` — more nodes after expansion
- [ ] No duplicate nodes (level-aware dedup)
- [ ] Concurrent `ask()` calls on same topic — no duplicates (asyncio Lock)
- [ ] max_expansion_rounds=2 — stops after 2 rounds, returns expansion_limit_reached
