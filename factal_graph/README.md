# Fractal Graph — Resolution-Aware Knowledge Graph

A multi-resolution knowledge graph where knowledge is organized by specificity level (L0-L5), with semantic bounding boxes enabling zoom-based queries, a **mother model seeding** system for one-time quality, and a **2B reasoning engine** that answers questions over graph context at runtime.

## Core Idea

Most knowledge graphs are flat. Fractal Graph organizes knowledge as a hierarchy of specificity:

| Level | Name       | Description       | Example                                          |
|-------|------------|-------------------|--------------------------------------------------|
| L0    | Domain     | Broadest field    | "NATO-Russia relations"                          |
| L1    | Topic      | Named sub-topic   | "NATO eastward expansion"                       |
| L2    | Concept    | Specific mechanism | "Article 5 collective defense"                  |
| L3    | Entity     | Named instance    | "2014 Crimea annexation"                        |
| L4    | Fact       | Verifiable claim  | "NATO added 4 members 1999-2020"                |
| L5    | Evidence   | Cited source/data | "According to NATO Secretary General..."        |

**Any subtree is itself a valid knowledge graph** (self-similarity). Query specificity determines zoom depth.

## Architecture

```
                    SEEDING (one-time)
          +-----------------------------------+
          |  Mother Model Ladder               |
          |  L1: lfm2.5:gpu3 (~8B, local)      |
          |  L2: z.ai GLM-5.1 (cloud)       |
          |  Generates L0->L{depth} hierarchy    |
          |  Computes bboxes + cross-res edges   |
          +-----------------+-----------------+
                            |
                            v
         +------------------------------------------+
         |              Fractal Graph                 |
         |  SQLite (nodes, edges, bbox) + ChromaDB   |
         |  Edges: supports, contradicts, refines     |
         +-----------------+------------------------+
                           |
         +----------+----------+----------+
         |          |          |          |
         v          v          v          v
   +-----------+ +----------+ +---------+ +-----------+
   | 2B Runtime | | Distill  | | Enrich  | | Quality   |
   | classify,  | | mother   | | cross-  | | triad,    |
   | ask, decide| | extract  | | link    | | consistency|
   +-----------+ +----------+ +---------+ +-----------+
   +-----------+ +-----------+ +-----------+
   | Web UI    | | Export/   | | Search    |
   | D3.js     | | Import    | | Trigger   |
   +-----------+ +-----------+ +-----------+

   RUNTIME — 2B for queries, mother only for seeding/expansion
   SEARCH TRIGGER — fills L4-L5 evidence from web sources at all expansion points
   AGENT MODE — procedural knowledge (decide()) returns structured actions
```

### The Key Insight

**The model's job is inference over structured data, not recall.** The mother model seeds once. The 2B model handles all runtime by reasoning over pre-built graph structure -- classification via bbox containment, context assembly from vector search + graph traversal, answer synthesis from structured facts.

## Files

| File              | Purpose |
|-------------------|---------|
| `config.py`       | Settings -- DB paths, Ollama URLs, model config, token budgets |
| `db.py`           | SQLite -- nodes, edges, bbox, metadata, export/import |
| `graph.py`        | Graph ops -- bbox computation, subtree, contradictions, propagation |
| `seed.py`         | Mother seeding -- hierarchy generation, agent procedural knowledge, gap fill |
| `distill.py`      | Distillation -- 3-phase mother knowledge extraction (GENERATE, EMBED, STORE) |
| `enrich.py`       | Enrichment -- knowledge probes, node enrichment, cross-linking |
| `quality.py`      | Quality gate -- self-consistency, triad scan, source attribution |
| `ingest.py`       | Ingest pipeline -- search/extract (from searchMCP), heuristic classification |
| `embedder.py`     | Ollama embedding -- async embed + sync fallback + batch |
| `chroma_store.py` | ChromaDB -- one collection per level, concurrent-access safe |
| `model_cache.py`  | Model load state cache -- avoids cold-starts on repeated calls |
| `context.py`      | Context assembler -- gather, walk graph, format for LLM, bbox level placement |
| `reasoning.py`    | 2B reasoning -- ask (explain), decide (act), classify, auto-expand |
| `growth.py`       | Autonomous expansion -- gap fill, enrichment, curiosity scan |
| `judges.py`       | Judge triad -- Angel/Devil/Neutral two-pass with conflict detection |
| `query.py`        | Query engine -- specificity classification, drill-down, multi-level search |
| `gql.py`          | Structured queries -- contradictions, evidence, entity comparison |
| `bench.py`        | Benchmark -- ask (5 modes), decide (2B vs 0.8b) |
| `web_ui.py`       | Flask Web UI -- graph visualization, search, export/import endpoints |
| `templates/index.html` | D3.js force-directed graph, side panel, search, export/import buttons |
| `meeseeks.py`     | Meeseeks system -- task-scoped ephemeral souls with lifecycle management |
| `code_ingest.py`  | Code ingestion -- regex-based repo parsing into L0-L5 hierarchy, issue/diff ingestion |
| `souls/`          | Soul templates -- YAML configs for named tagged subgraphs (coder, companion, researcher) |
| `factal_server.py`| MCP server -- 49 tools via FastMCP stdio |
| `search_trigger.py`| Search trigger layer -- rate-limited, deduped web-grounded evidence |

### External Dependencies
- **searchMCP** (`C:/Users/aaron/searchmcp/core.py`) -- search + text extraction via SearXNG fan-out and trafilatura/BeautifulSoup.

## MCP Tools (49 total)

### Ask -- 2B Reasoning
- `ask(question, auto_expand)` -- Primary tool. embed -> classify (2B) -> gather context -> synthesize (2B). Optional auto-expand on low confidence.
- `ask_with_triad(question)` -- 2B answer + judge triad batched in parallel, faster (~3s savings).
- `decide(situation, options)` -- Agent decision mode. Returns structured action (not prose). L2-L4 biased context + automatic triad.

### Knowledge Ingest
- `add_node(content, resolution_level, parent_id, confidence, source_url, metadata)` -- Manual node
- `add_edge(from_node_id, to_node_id, edge_type, confidence, context)` -- Manual edge
- `web_ingest(query, max_urls)` -- SearXNG -> extract -> classify -> insert (via searchMCP)
- `ingest_url(url)` -- Fetch URL, chunk, classify, insert

### Mother Model Seeding
- `seed_topic(topic, depth, mother_model)` -- L0-L{depth} hierarchy from scratch
- `seed_agent_topic(topic, depth, mother_model)` -- Procedural knowledge domain (decision rules, patterns, failure modes)
- `seed_from_search(query, max_urls, mother_model)` -- Web search + mother structuring
- `seed_expand(node_id, mother_model)` -- Expand sparse node, fill missing levels

### Search Trigger -- Web-Grounded Evidence
- `search_and_ingest(query, parent_node_id, max_urls)` -- Search web, mother structures L4-L5 evidence nodes. Rate-limited 10/min.

### Distillation -- Mother Model Knowledge Extraction
- `distill_topic(topic, mother_model)` -- Extract entities/facts/evidence from mother's parametric knowledge
- `distill_domains(domains, mother_model)` -- Batch distill multiple domains (or all ~30)
- `distill_coverage()` -- Show graph coverage, nodes per level, shallow branches, gaps

### Enrichment -- Cross-linking + Knowledge Probes
- `enrich_probe(topic)` -- Compare mother model knowledge vs graph coverage
- `enrich_node(node_id)` -- Ask mother "what else relates?" and insert siblings/children
- `cross_link(node_id_a, node_id_b)` -- Ask mother if two nodes are related, create edge
- `auto_crosslink(domain_node_id, max_pairs)` -- Batch cross-link L1 topics under same L0

### Quality Gate
- `self_consistency_check(node_id, num_rounds, min_consensus)` -- Ask mother 3x, keep consensus facts
- `quality_triad_scan(domain_node_id)` -- Angel/Devil on L0 domains to catch hallucinations
- `source_attribution(domain_node_id)` -- Mother rates sourceability, demotes unattributed facts

### Judge Triad
- `judge_topic(topic, top_k)` -- Angel (L0-L1, optimistic) / Devil (L4-L5, adversarial) / Neutral (L2-L3, bridging). Disagreements -> resolution_conflict edges.

### Query / Retrieval
- `query_graph(prompt, resolution_hint)` -- Resolution-aware zoom query
- `drill_down(node_id, target_resolution)` -- Traverse hierarchy downward
- `search_nodes(query_text, resolution_level)` -- Semantic search within level or all
- `get_node(node_id)` -- Full node with children, parents, edges

### Graph Operations
- `get_subtree(node_id, max_depth)` -- Extract self-similar subgraph
- `find_contradictions()` -- All CONTRADICTS/CHALLENGES/resolution_conflict edges
- `propagate_confidence(node_id)` -- Bottom-up confidence propagation
- `graph_stats()` -- Nodes/edges per level, ChromaDB coverage

### Agent Decision Mode — Monte Carlo
- `decide_mc(situation, options, simulations, soul)` -- Monte Carlo graph search decision. N simulations sampling random graph subsets, aggregate by majority vote.
- `distill_behavior(situations, simulations)` -- Distill MC decision traces into procedural knowledge nodes.

### Soul System — Named Tagged Subgraphs
- `seed_soul(name, mother_model)` -- Seed a soul from YAML template into graph (coder, companion, researcher)
- `list_souls()` -- List soul templates with seeded status and node counts
- `distill_soul(soul_id, situations, simulations)` -- Growth loop: distill behavior scoped to soul's subgraph
- `soul_decide(soul_id, situation, options)` -- Scoped MC decision using soul's personality params and graph

### Meeseeks — Task-scoped Ephemeral Souls
- `spawn_meeseeks(task, parent_soul, repo_soul_id)` -- Create a Meeseeks instance for a task (inherits parent soul + optional repo graph)
- `meeseeks_status(instance_id)` -- Get instance details with consistency history
- `meeseeks_step(instance_id, options)` -- Execute one MC decision step (tracks consistency as existential state)
- `meeseeks_run(task, parent_soul, max_steps)` -- Full lifecycle: spawn → step until done → release or decompose
- `release_meeseeks(instance_id)` -- Write outcome to parent soul, delete ephemeral graph
- `list_meeseeks()` -- List all active (not released/decomposed) instances

### Code Ingestion — Per-Repo Graph Seeding (Phase 22)
- `ingest_codebase(repo_path, soul_id, max_files)` -- Walk a repo, extract modules/files/functions/imports into L0-L5 hierarchy. Tags all nodes with soul_id for scoped Meeseeks queries.
- `ingest_issue(issue_text, repo_soul_id)` -- Ingest a GitHub issue: extract symptoms (error keywords), code blocks, stack traces as L3-L5 nodes.
- `ingest_diff(diff_text, repo_soul_id)` -- Parse unified diff: hunk change nodes (L4), new function/class definitions (L3).

#### Code Resolution Mapping

| Level | Code Concept | Example |
|-------|-------------|---------|
| L0 | Repository identity | "fractal-graph — resolution-aware knowledge graph" |
| L1 | Module/package | "fractal-graph/query — query engine module" |
| L2 | File | "db.py — SQLite storage layer" |
| L3 | Function/class | "def insert_node(conn, content, ...) — Insert a node into the graph" |
| L4 | Import/dependency | "imports: sqlite3, json, time, datetime" |
| L5 | Code snippet | "db.py:98-108 — INSERT INTO nodes (...) VALUES (...)" |

#### SWE-bench Integration

```
ingest_codebase("django/django", soul_id="repo-django")
ingest_issue(issue_text, "repo-django")
spawn_meeseeks("Fix the empty filter bug", parent_soul="coder", repo_soul_id="repo-django")
# Meeseeks soul_ids = [meeseeks_id, coder, repo-django] → code-aware decisions
```

### Export / Import
- `export_graph()` -- Export entire graph as portable JSON (no embeddings)
- `import_graph(data, merge)` -- Import from JSON. merge=True preserves existing data, re-indexes ChromaDB

## Web UI

Flask + D3.js force-directed graph on port 8018:

```bash
python web_ui.py
```

Features:
- **Interactive graph** -- zoom, pan, click nodes for detail panel (children, connections, metadata)
- **Search** -- text search across all nodes with instant highlighting
- **Color-coded levels** -- L0 Domain (red) through L5 Evidence (purple)
- **Edge coloring** -- related (gray), refines (teal), contradicts (red), supports (green), resolution_conflict (orange dashed)
- **Export button** -- downloads graph as date-stamped JSON file
- **Import button** -- upload JSON file, replaces graph, auto-reloads

### Web UI API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | D3.js graph visualization |
| `/api/graph` | GET | Full graph as `{nodes, links}` |
| `/api/stats` | GET | Node/edge counts per level |
| `/api/node/:id` | GET | Node details with children, parent, edges |
| `/api/search?q=` | GET | Text search (LIKE, case-insensitive) |
| `/api/domains` | GET | L0 domains with child counts |
| `/api/export` | GET | Download full graph as JSON |
| `/api/import` | POST | Upload JSON to import (multipart file) |

## Search Trigger Layer

All graph expansion paths automatically search the web for L4-L5 evidence nodes
when the graph is sparse or confidence is low. This grounds mother-model hallucinations
(~40% on L4-L5) with real, sourced information.

### Trigger Types & Rate Limits

| Type | Max Calls | Window | When |
|------|-----------|--------|------|
| `gap` | 3 | 60s | Low confidence answer, gap detected |
| `post_seed` | 5 | 120s | After seed_topic completes for L3 nodes |
| `sparse_evidence` | 2 | 300s | Curiosity scan finds L3+ leaf nodes |
| `post_distill` | 5 | 120s | After distillation for low-conf L4 nodes |
| `manual` | 10 | 60s | MCP tool `search_and_ingest` calls |

### Wiring Points

1. **reasoning.py** -- gap strings → search before mother fill (saves ~30s warmup)
2. **seed.py** -- L3 nodes after `_compute_bboxes_for_subtree()`
3. **growth.py** -- L3+ sparse nodes in `curiosity_scan()` (L0-L2 still use mother)
4. **distill.py** -- L4 nodes with confidence < 0.7 after STORE phase

### Dedup

Same query (case-insensitive MD5 hash) is never searched twice in a session.

## Export/Import Format

```json
{
  "version": 1,
  "exported_at": "2026-06-07T...",
  "stats": { "total_nodes": 172, "total_edges": 201 },
  "nodes": [
    { "id": 1, "content": "...", "resolution_level": 0, "parent_id": null,
      "confidence": 0.5, "bbox": [...], "source_url": null, "metadata": {...} }
  ],
  "edges": [
    { "id": 1, "from_node_id": 1, "to_node_id": 2, "edge_type": "related",
      "from_resolution": 0, "to_resolution": 1, "confidence": 0.5, "context": null }
  ]
}
```

Embeddings are excluded -- regenerate via `web_ingest`, `seed_from_search`, or the MCP `import_graph` tool (which re-indexes ChromaDB automatically).

## Reasoning Pipeline

```
User Question
    |
    v
[1] EMBED -- nomic-embed-text
    |
    v
[2] CLASSIFY -- 2B classifies: factual | analytical | comparative | exploratory | yes_no
    |
    v
[3] GATHER -- search all 6 levels, walk graph, rank by sim+confidence+edges
    |  Token budget: 2048 tokens
    |
    v
[4] SYNTHESIZE -- 2B generates answer from structured context
    |  Returns: answer, confidence, key_facts, gaps
    |
    v
[5] AUTO-EXPAND (if confidence < threshold)
    |  search trigger (gap) -> if no nodes -> mother gap fill -> if fails -> web search fallback
    |
    v
[6] RE-SYNTHESIZE with enriched context
    |
    v
Answer + confidence + gaps + timing
```

## Benchmark Results (2026-06-07)

### Ask Bench

| Mode | Avg Time | Avg Conf | Notes |
|------|----------|----------|-------|
| 2B+Graph | 4.9s | 0.60 | Fast, decent |
| 2B alone | 3.6s | -- | No confidence metric |
| Mother alone | 19.7s | -- | Empty JSON ~40% |
| 2B+Graph+Expand | 65.6s | 0.77 | Mother warmup dominates |
| **2B+Graph+Triad** | **11.9s** | **0.84** | Best quality/speed |

### Decide Bench (Agent Procedural Knowledge)

| Model | Avg Conf | Avg Relevance | Avg Time | Notes |
|-------|----------|---------------|----------|-------|
| qwen3.5:2b | 0.76 | 0.30 | 60.8s | Better relevance, slower |
| qwen3.5:0.8b | 0.84 | 0.22 | 36.6s | Higher confidence, 1.7x faster |

Note: Both models scored low relevance because no procedural knowledge was seeded in the graph. Scores will improve after `seed_agent_topic()`.

## Configuration

All settings via environment variables with `FRACTAL_` prefix:

| Setting | Default | Description |
|---------|---------|-------------|
| `FRACTAL_DB_PATH` | `data/fractal.db` | SQLite path |
| `FRACTAL_CHROMA_PATH` | `data/chroma` | ChromaDB path |
| `FRACTAL_OLLAMA_URL` | `http://100.84.161.63:11434` | Ollama endpoint |
| `FRACTAL_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `FRACTAL_LLM_URL` | `http://100.84.161.63:11434` | 2B classifier endpoint |
| `FRACTAL_LLM_MODEL` | `qwen3.5:2b` | Runtime classifier |
| `FRACTAL_MOTHER_URL` | `http://100.84.161.63:11434` | Mother model endpoint |
| `FRACTAL_MOTHER_MODEL` | `lfm2.5:gpu3` | Mother model (seeding) |
| `FRACTAL_SEARCH_TRIGGER_ENABLED` | `true` | Enable search trigger layer |
| `FRACTAL_SEARCH_MAX_URLS` | `3` | Max URLs per search trigger call |

## Edge Types

| Type | Direction | Meaning |
|------|-----------|---------|
| `related` | bidirectional | General association |
| `refines` | child->parent | Refinement of parent |
| `contradicts` | either | Nodes contradict |
| `exemplifies` | child->parent | Example of parent |
| `generalizes` | parent->child | Generalizes child |
| `challenges` | either | Challenges the other |
| `supports` | either | Supports the other |
| `derived_from` | child->parent | Derived from source |
| `resolution_conflict` | judge->judges | Judges disagree |
| `contains` | parent->child | Module contains file, file contains function |
| `imports_from` | function->module | Function imports from another module |
| `mentions` | issue->symptom | Issue mentions a symptom |

## Setup

```bash
pip install -r requirements.txt

# Ollama needs: nomic-embed-text, qwen3.5:2b, lfm2.5:gpu3 (num_ctx 12288)
# z.ai GLM-5.1: grandmother model for cloud escalation
# searchMCP needs: running at C:/Users/aaron/searchmcp/ (for core.py import)

python factal_server.py  # FastMCP stdio (MCP server)
python web_ui.py        # Flask Web UI on port 8018
```

## Quick Start

```python
# 1. Seed a topic with the mother model
seed_topic("climate change", depth=3)

# 2. Ask a question -- 2B reasoning over graph context
ask("Why does climate change affect biodiversity?")

# 3. Run ask with triad for higher confidence (parallel, ~3s faster)
ask_with_triad("Is nuclear power a viable climate solution?")

# 4. Agent decision mode -- returns structured action
decide("A tool call returned an unexpected format")
decide("Not enough info", options="ask,proceed,abort")

# 5. Seed procedural knowledge (how to act, not what is true)
seed_agent_topic("task decomposition for AI agents", depth=4)

# 6. Web search + mother structuring
seed_from_search("quantum computing breakthroughs 2026")

# 7. Expand a sparse node
seed_expand(node_id=7)

# 8. Export/backup the graph
export_graph()

# 9. Import (restores + re-indexes embeddings)
import_graph(json_string, merge=True)

# 10. Manual search trigger -- ground a topic with real sources
search_and_ingest("NATO Article 5 invocation history")
```
