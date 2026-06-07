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
- [x] Mother self-consistency check: ask same question 3x, only insert facts that appear in 2/3+ responses
- [x] Contradiction detection: after distillation, run triad on each L0 to catch mother hallucinations
- [x] Source attribution: mother generates "source" hints — mark unattributed facts as lower confidence
- [x] MCP tools: `self_consistency_check`, `quality_triad_scan`, `source_attribution`
- [x] Fix FK constraint errors in auto-expand (parent_id validation in insert_node + _write_lock)
- [x] Fix "database is locked" errors (busy_timeout=30000 + asyncio _write_lock)

### Key Decisions Made
- **Distillation depth**: L3-L5 (entity/fact/evidence) — working with single-aspect calls
- **Batch size**: 1 aspect per call (4K context limit) — batch template ready for future larger context
- **keep_alive**: 15s between calls — fast enough for sequential aspect calls
- **Warmup**: NO polling — just fire the API call, model loads naturally (300s timeout)
- **num_ctx**: 4096 (default) — 32K caused 5GB VRAM usage, model slower

---

### Phase 10: Performance + Polish
- [x] Cache model load state to avoid cold-starts on repeated calls
- [x] Parallelize LLM calls in seed pipeline with asyncio.gather
- [x] Investigate sea level rise triad anomaly (24.8s pass1, low confidence)
- [x] Judge triad integration with Pantheon council/court (MCP tools already exposed — Pantheon side needs update to call `judge_topic`/`judge_answer` as research source)
- [x] Distill more domains to lift avg confidence across all topics (NATO-Russia +30, Climate +25, AI/ML +25 → 172 nodes, 201 edges, 5 L0 domains)

---

### Phase 13: NVIDIA Nemotron 550B Cloud Mother
- [x] Add NVIDIA Integrate API support (OpenAI-compatible, thinking enabled)
- [x] `nvidia/` model prefix routing in `_mother_generate` / `_mother_generate_keepalive`
- [x] 180s timeout, 16384 max_tokens + reasoning_budget
- [x] Direct model override: `seed_topic("...", mother_model="nvidia/...")`

### Phase 14: Mother Ladder + Timeout Fix
- [x] Add grandmother config: `grandmother_model`, `grandmother_enabled`, `grandmother_max_retries_before_escalate`
- [x] Extract `_ollama_call()` helper — DRY up duplicated Ollama httpx boilerplate
- [x] Ladder in `_mother_generate()`: mother → retry N times → escalate to grandmother
- [x] Ladder in `_mother_generate_keepalive()`: same pattern, keep_alive + num_predict passed through
- [x] Anti-cascade guard: grandmother never calls back through ladder
- [x] Escalation trigger: empty response only (not JSON parse failures)
- [x] `grandmother_enabled=False` preserves all existing behavior

---

---

## Phase 16: Monte Carlo Graph Search + Behavior Distillation (IN PROGRESS)

### Audit Fixes
- [x] Fix distill.py line 229: batch step bug — `range(0, len, 20)` with inner batch size 5 skips 75% of nodes. Fixed: step=5.
- [x] Fix model_cache.py: warmup call has no `keep_alive` — model unloads immediately. Fixed: added `"keep_alive": "30s"`.
- [x] Fix reasoning.py decide(): triad queries ALL levels (L0-L5) including factual knowledge irrelevant to decisions. Fixed: removed triad from decide(), MC handles its own multi-simulation.
- [x] Fix bench.py: relevance scoring uses exact word match — fragile. Fixed: use `difflib.SequenceMatcher` ratio.

### Monte Carlo Decision Engine
- [x] `config.py` — add mc_simulations, mc_context_pool, mc_context_subset, behavior_probes, behavior_simulations_per
- [x] `reasoning.py` — `_gather_procedural_context_sampled()` — large pool + random k-subset per sim
- [x] `reasoning.py` — `_cluster_actions()` — word overlap >= 0.5 clustering
- [x] `reasoning.py` — `_aggregate_decisions()` — majority vote + risk/pattern union
- [x] `reasoning.py` — `decide_monte_carlo()` — N simulations in parallel batches of 2
- [x] `reasoning.py` — `decide()` simplified: removed triad (irrelevant factual queries)
- [x] `factal_server.py` — register `decide_mc` + `distill_behavior` MCP tools (37 total)

### Behavior Distillation
- [x] `distill.py` — `distill_behavior()` — 10 probe situations, MC per probe, mother extracts patterns
- [x] `distill.py` — BEHAVIOR_PROBE_SITUATIONS — tool error, ambiguous request, file not found, test failing, network timeout, git conflict, schema mismatch, rate limit, memory pressure, missing config
- [x] `distill.py` — PROMPT_EXTRACT_BEHAVIOR_PATTERNS — mother extracts L2/L3 behavior patterns from traces
- [x] `distill.py` — stores under "observed_agent_behavior" L0 domain with metadata

### Benchmark
- [x] `bench.py` — `run_decide_mc_bench()` — 6 test groups (2 models x 3 modes)
- [x] `bench.py` — `--decide-mc` CLI flag
- [x] Metrics: avg_confidence, action_consistency, risk_coverage, time per model x mode

### Verification (PENDING)
- [ ] `python bench.py --decide-mc` — 6 test groups, compare single vs MC
- [ ] `distill_behavior()` via MCP — should create nodes under "observed_agent_behavior"
- [ ] `decide_mc("A tool call returned an unexpected format")` — return action + simulations_detail
- [ ] Verify action consistency: simulations_detail shows all N actions and cluster assignments

---

## Outstanding (future, lower priority)

### Phase 18: Remove NVIDIA, z.ai GLM-5.1 only cloud provider
- [x] Remove nvidia_api_key, nvidia_base_url from config.py
- [x] Delete `_mother_generate_nvidia()` from seed.py
- [x] Remove nvidia/ prefix from `_is_cloud_model()` and `_mother_generate_cloud()`
- [x] Update README: architecture diagram, config table, setup comment
- [x] z.ai GLM-5.1 is now the sole cloud provider

### Phase 17: z.ai GLM5.1 Grandmother
- [x] Add z.ai API config (base_url, api_key, model)
- [x] Add `_mother_generate_zai()` — OpenAI-compatible API call with reasoning_content fallback
- [x] Generalize cloud model routing (`_is_cloud_model` + `_mother_generate_cloud`)
- [x] Fix prefix stripping: `zai/` → bare model name for API, same for `nvidia/`
- [x] Update grandmother ladder to use generalized dispatcher
- [x] Test: `_mother_generate("...", model="glm-5.1")` — works, JSON parsed correctly
- [x] WARNING: z.ai ToS restricts usage to supported coding tools only — grandmother use may trigger restrictions
- [x] Switch grandmother default to glm-5.1 (grandmother_enabled=True)

- [x] Web UI for graph visualization (Flask + D3.js, port 8018)
- [x] Export/import graph to JSON format (db.py + MCP tools + Web UI endpoints + download button)
- Persistent cron-based auto-distillation (distill new domains periodically)
