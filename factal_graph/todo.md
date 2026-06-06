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

---

## Next: Distill Mother Knowledge Into the Graph

### Problem
The mother model (lfm2.5, ~8B) is only called during seeding. Once seeded, the graph is static — the 2B model can only reason over what was pre-loaded. If the graph lacks a fact, the 2B can't fill it.

Current workflow for knowledge acquisition:
1. `seed_topic("X")` — mother generates L0-L5 hierarchy from its parametric memory
2. `seed_from_search("X")` — SearXNG -> extract -> mother structures extracted text
3. `seed_expand(node_id)` — mother expands a sparse node

All three go: **mother generates -> parse JSON -> insert nodes**. The mother's knowledge is only captured at seed time. If a user asks about something the mother "knows" but isn't in the graph, the 2B gets low confidence and tries auto-expand (65.6s) or returns a weak answer.

### Goal
**Distill the mother model's parametric knowledge into the graph structure proactively**, not reactively on low confidence. The graph should be rich enough that the 2B rarely needs to trigger auto-expand.

### Phase 1: Analyze What the Mother Knows vs What the Graph Has
- [ ] `graph_coverage_report()` — for each L0 domain node, count children per level, identify shallow branches
- [ ] `mother_knowledge_probe(topic)` — ask mother "what do you know about X?" in structured format, compare against existing nodes
- [ ] Quantify gap: % of seeded topics that have full L0-L5 coverage vs shallow L0-L2

### Phase 2: Structured Distillation Pipeline
- [ ] `distill_topic(topic)` — send mother a distillation prompt that extracts ALL knowledge it has about a topic into structured nodes+edges, not just a top-down hierarchy
- [ ] Prompt engineering: instead of "generate a hierarchy", ask "list every fact, entity, relationship, and source you know about X" — then classify and insert each piece
- [ ] Batch distillation: `distill_all(depth=5)` — iterate all L0 nodes, distill each to L5

### Phase 3: Incremental Graph Enrichment
- [ ] `enrich_node(node_id)` — given a node, ask mother "what else relates to this?" and insert as sibling/child edges
- [ ] Cross-linking: after distillation, run mother over pairs of nodes to detect missed edges (supports, contradicts, refines)
- [ ] Edge confidence from mother: "how strongly does A relate to B?" -> edge confidence score

### Phase 4: Quality Gate
- [ ] Mother self-consistency check: ask same question 3x, only insert facts that appear in 2/3+ responses
- [ ] Contradiction detection: after distillation, run triad on each L0 to catch mother hallucinations
- [ ] Source attribution: mother generates "source" hints (wikipedia, general knowledge, etc.) — mark unattributed facts as lower confidence

### Key Decisions Needed
- **Distillation depth**: L3 (entity/fact) vs L5 (evidence) — L5 needs real sources, mother will hallucinate them
- **Batch size**: mother generates ~10-20 nodes per call — how many calls per topic?
- **Dedup strategy**: cosine sim on embeddings vs text overlap — current L0:0.95/L5:0.80 thresholds
- **When to distill**: on-demand (user triggers) vs background (curiosity_scan enhanced)

---

## Outstanding (future, lower priority)
- Cache model load state to avoid cold-starts on repeated calls
- Parallelize LLM calls in seed pipeline with asyncio.gather
- Investigate sea level rise triad anomaly (24.8s pass1, low confidence)
- Judge triad integration with Pantheon council/court
