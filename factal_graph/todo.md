# Fractal Graph — TODO

## Completed

### Phase 0: GPU + Embedding Fixes
- [x] Kill Epic Games Launcher on Lappy (frees GPU resources)
- [x] Fix qwen3.5 thinking tokens — root cause of timeouts
- [x] Fix `embed_sync()` — broken `get_event_loop().run_until_complete()`
- [x] Fix ChromaDB concurrent access corruption
- [x] Upgrade mother model to lfm2.5:latest (~8B)

### Phase 1: 2B Reasoning Pipeline (reasoning.py + context.py)
- [x] `gather_context()` — search all 6 levels, walk parents, dedup, rank
- [x] `format_context_for_llm()` — structured text for 2B
- [x] `walk_graph()` — LaRQL WALK equivalent
- [x] `find_best_level_by_structure()` — place text via bbox containment
- [x] `classify_question()` + `synthesize()` + `answer()` — full pipeline
- [x] `gql.py` — contradictions, evidence, entity comparison
- [x] `ask()` MCP tool — primary user-facing tool

### Phase 2: Mother Seeding (seed.py)
- [x] `seed_topic()` — L0-L{depth} hierarchy via mother JSON generation
- [x] `_mother_generate()` — warmup polling, keep_alive:0, retry on empty
- [x] `_compute_bboxes_for_subtree()` — bottom-up bbox computation
- [x] `seed_from_search()` — SearXNG search + mother structuring
- [x] `seed_expand()` — expand sparse nodes with missing levels
- [x] Level-aware dedup thresholds (L0:0.95 -> L5:0.80)

### Phase 3: Autonomous Expansion (growth.py)
- [x] `fill_gaps()` — mother fills detected knowledge gaps
- [x] `enrich_topic()` — fire-and-forget topic decomposition
- [x] `curiosity_scan()` — find sparse areas, expand via seed_expand
- [x] `detect_sparse_nodes()` — SQL for leaf-below-L5, few-children
- [x] Per-topic asyncio locks (race prevention)
- [x] Web search fallback when mother creates 0 nodes

### Phase 4: Judge Triad (judges.py)
- [x] Angel/Devil/Neutral two-pass triad
- [x] Resolution-aware: Angel L0-L1, Devil L4-L5, Neutral L2-L3
- [x] Disagreements logged as `resolution_conflict` edges
- [x] `judge_topic()` + `judge_answer()` MCP tools

### Phase 5: Benchmark + Stability
- [x] bench.py — 5 modes x 8 questions, 846s, no crashes
- [x] All warmup/keep_alive/recursion fixes verified
- [x] **Triad mode winner**: 0.84 conf at 11.9s

### Phase 6: Code Dedup
- [x] Replace duplicated search/extract with searchMCP core imports
- [x] Remove ~150 lines of SearXNG fan-out + trafilatura from ingest.py
- [x] `searchmcp/core.py` — public API (search, extract_text, search_and_read)

### Phase 7: Distillation Pipeline (distill.py)
- [x] `distill_coverage()` — nodes per level, shallow branches, L0 domain list
- [x] `distill_domain()` — 3-phase: GENERATE (mother hot, keep_alive=15s), EMBED (batches of 5), STORE (dedup+insert)
- [x] `distill_all()` — enumerate domains from mother, distill each
- [x] `_mother_generate_keepalive()` — configurable keep_alive, no warmup polling, 300s timeout, num_predict=4096
- [x] Batch edge embedding (groups of 5) instead of sequential
- [x] `PROMPT_EXTRACT_FACTS_BATCH` template (ready for larger context windows)
- [x] Remove ALL `/api/ps` polling loops — bench.py, reasoning.py, seed.py
- [x] Custom model `lfm2.5:gpu3` (num_gpu=-1, num_ctx=4096)
- [x] Unicode fix in `_parse_json_array` (cp1252 → ascii replace)
- [x] FK constraint and database locked error handling
- [x] Test: quantum computing → 99 collected, 25 created (75% dedup), 6 edges, 713s
- [x] Bench verified: distillation helps quantum questions but not other topics
- [x] MCP tools: `distill_topic`, `distill_domains`, `distill_coverage`

---

## Next: Enrichment + Quality

### Phase 8: Incremental Graph Enrichment
- [x] `mother_knowledge_probe(topic)` — structured comparison: ask mother "what do you know about X?" vs existing graph nodes
- [x] `enrich_node(node_id)` — given a node, ask mother "what else relates to this?" and insert as sibling/child edges
- [x] Cross-linking: after distillation, run mother over pairs of nodes to detect missed edges (supports, contradicts, refines)
- [x] Edge confidence from mother: "how strongly does A relate to B?" -> edge confidence score
- [x] MCP tools: `enrich_probe`, `enrich_node`, `cross_link`, `auto_crosslink`

### Phase 9: Quality Gate
- [ ] Mother self-consistency check: ask same question 3x, only insert facts that appear in 2/3+ responses
- [ ] Contradiction detection: after distillation, run triad on each L0 to catch mother hallucinations
- [ ] Source attribution: mother generates "source" hints — mark unattributed facts as lower confidence
- [x] Fix FK constraint errors in auto-expand (parent_id validation in insert_node + _write_lock)
- [x] Fix "database is locked" errors (busy_timeout=30000 + asyncio _write_lock)

### Key Decisions Made
- **Distillation depth**: L3-L5 (entity/fact/evidence) — working with single-aspect calls
- **Batch size**: 1 aspect per call (4K context limit) — batch template ready for future larger context
- **keep_alive**: 15s between calls — fast enough for sequential aspect calls
- **Warmup**: NO polling — just fire the API call, model loads naturally (300s timeout)
- **num_ctx**: 4096 (default) — 32K caused 5GB VRAM usage, model slower

---

## Outstanding (future, lower priority)
- Cache model load state to avoid cold-starts on repeated calls
- Parallelize LLM calls in seed pipeline with asyncio.gather
- Investigate sea level rise triad anomaly (24.8s pass1, low confidence)
- Judge triad integration with Pantheon council/court
- Distill more domains to lift avg confidence across all topics
