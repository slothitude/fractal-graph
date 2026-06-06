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
               ┌─────────────────────────────────┐
               │  Mother Model (lfm2.5:latest ~8B) │
               │  Generates L0→L{depth} hierarchy  │
               │  Computes bboxes + cross-res edges│
               └───────────────┬─────────────────┘
                               │
                               ▼
              ┌──────────────────────────────────────────┐
              │              Fractal Graph                 │
              │  SQLite (nodes, edges, bbox) + ChromaDB   │
              │  Edges: supports, contradicts, refines     │
              └──────────────────┬───────────────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
          ▼                      ▼                      ▼
   ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
   │ 2B Classifier│    │ 2B Reasoning     │    │ Judge Triad   │
   │ (ingestion)  │    │ (ask pipeline)    │    │ Angel/Devil/  │
   └──────────────┘    │ embed→classify→   │    │ Neutral       │
                       │ gather→synthesize │    └──────────────┘
                       └──────────────────┘

              RUNTIME — all 2B, no mother model touched
```

### The Key Insight

**The model's job is inference over structured data, not recall.** The mother model seeds once. The 2B model handles all runtime by reasoning over pre-built graph structure — classification via bbox containment, context assembly from vector search + graph traversal, answer synthesis from structured facts.

### Performance

| Operation        | Model | Time  |
|------------------|-------|-------|
| `ask()` (graph only) | 2B | ~4.9s |
| `ask()` (with triad) | 2B x3 | ~11.9s |
| Ingest classify  | 2B | ~2s |
| `seed_topic()`   | Mother (lfm2.5) | ~5s |
| `seed_expand()`  | Mother (lfm2.5) | ~15s |

## Files

| File              | Purpose |
|-------------------|---------|
| `config.py`       | Settings — DB paths, Ollama URLs, model config, token budgets |
| `db.py`           | SQLite — nodes, edges, bbox, metadata |
| `graph.py`        | Graph ops — bbox computation, subtree, contradictions, propagation |
| `seed.py`         | Mother seeding — hierarchy generation, gap fill, search+seed |
| `ingest.py`       | Ingest pipeline — search/extract (from searchMCP), heuristic classification, node insertion |
| `embedder.py`     | Ollama embedding — async embed + sync fallback + batch |
| `chroma_store.py` | ChromaDB — one collection per level, concurrent-access safe |
| `context.py`      | Context assembler — gather, walk graph, format for LLM, bbox level placement |
| `reasoning.py`    | 2B reasoning — classify question, synthesize answer, auto-expand |
| `growth.py`       | Autonomous expansion — gap fill, enrichment, curiosity scan |
| `judges.py`       | Judge triad — Angel/Devil/Neutral two-pass with conflict detection |
| `query.py`        | Query engine — specificity classification, drill-down, multi-level search |
| `gql.py`          | Structured queries — contradictions, evidence, entity comparison |
| `bench.py`        | Benchmark — 5 modes x 8 questions |
| `factal_server.py`| MCP server — 20 tools via FastMCP stdio |

### External Dependencies
- **searchMCP** (`C:/Users/aaron/searchmcp/core.py`) — search + text extraction via SearXNG fan-out and trafilatura/BeautifulSoup. Loaded via isolated import to avoid `config.py` naming conflict.

## MCP Tools (20 total)

### Ask — 2B Reasoning
- `ask(question, auto_expand)` — Primary tool. embed -> classify (2B) -> gather context -> synthesize (2B). Optional auto-expand on low confidence.

### Knowledge Ingest
- `add_node(content, resolution_level, parent_id, confidence, source_url, metadata)` — Manual node
- `add_edge(from_node_id, to_node_id, edge_type, confidence, context)` — Manual edge
- `web_ingest(query, max_urls)` — SearXNG -> extract -> classify -> insert (via searchMCP)
- `ingest_url(url)` — Fetch URL, chunk, classify, insert

### Mother Model Seeding
- `seed_topic(topic, depth, mother_model)` — L0-L{depth} hierarchy from scratch
- `seed_from_search(query, max_urls, mother_model)` — Web search + mother structuring
- `seed_expand(node_id, mother_model)` — Expand sparse node, fill missing levels

### Query / Retrieval
- `query_graph(prompt, resolution_hint)` — Resolution-aware zoom query
- `drill_down(node_id, target_resolution)` — Traverse hierarchy downward
- `search_nodes(query_text, resolution_level)` — Semantic search within level or all
- `get_node(node_id)` — Full node with children, parents, edges

### Graph Operations
- `get_subtree(node_id, max_depth)` — Extract self-similar subgraph
- `find_contradictions()` — All CONTRADICTS/CHALLENGES/resolution_conflict edges
- `propagate_confidence(node_id)` — Bottom-up confidence propagation
- `graph_stats()` — Nodes/edges per level, ChromaDB coverage

### Judge Triad
- `judge_topic(topic, top_k)` — Angel (L0-L1, optimistic) / Devil (L4-L5, adversarial) / Neutral (L2-L3, bridging). Disagreements -> resolution_conflict edges.

## Reasoning Pipeline

```
User Question
    │
    ▼
[1] EMBED — nomic-embed-text
    │
    ▼
[2] CLASSIFY — 2B classifies: factual | analytical | comparative | exploratory | yes_no
    │
    ▼
[3] GATHER — search all 6 levels, walk graph, rank by sim+confidence+edges
    │  Token budget: 2048 tokens
    │
    ▼
[4] SYNTHESIZE — 2B generates answer from structured context
    │  Returns: answer, confidence, key_facts, gaps
    │
    ▼
[5] AUTO-EXPAND (if confidence < threshold)
    │  gap fill -> if mother fails -> web search fallback
    │
    ▼
[6] RE-SYNTHESIZE with enriched context
    │
    ▼
Answer + confidence + gaps + timing
```

## Benchmark Results (2026-06-07)

| Mode | Avg Time | Avg Conf | Notes |
|------|----------|----------|-------|
| 2B+Graph | 4.9s | 0.60 | Fast, decent |
| 2B alone | 3.6s | — | No confidence metric |
| Mother alone | 19.7s | — | Empty JSON ~40% |
| 2B+Graph+Expand | 65.6s | 0.77 | Mother warmup dominates |
| **2B+Graph+Triad** | **11.9s** | **0.84** | Best quality/speed |

## Configuration

All settings via environment variables with `FRACTAL_` prefix:

| Setting | Default | Description |
|---------|---------|-------------|
| `FRACTAL_DB_PATH` | `data/fractal.db` | SQLite path |
| `FRACTAL_CHROMA_PATH` | `data/chroma` | ChromaDB path |
| `FRACTAL_OLLAMA_URL` | `http://100.84.161.63:11434` | Ollama endpoint |
| `FRACTAL_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `FRACTAL_SEARXNG_URL` | `http://100.84.161.63:8888` | SearXNG (unused — uses searchMCP) |
| `FRACTAL_LLM_URL` | `http://100.84.161.63:11434` | 2B classifier endpoint |
| `FRACTAL_LLM_MODEL` | `qwen3.5:2b` | Runtime classifier |
| `FRACTAL_MOTHER_URL` | `http://100.84.161.63:11434` | Mother model endpoint |
| `FRACTAL_MOTHER_MODEL` | `lfm2.5:latest` | Mother model (seeding) |

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

## Setup

```bash
pip install fastmcp chromadb httpx pydantic-settings trafilatura

# Ollama needs: nomic-embed-text, qwen3.5:2b, lfm2.5:latest
# searchMCP needs: running at C:/Users/aaron/searchmcp/ (for core.py import)

python factal_server.py  # FastMCP stdio
```

## Quick Start

```python
# 1. Seed a topic with the mother model
seed_topic("climate change", depth=3)

# 2. Ask a question — 2B reasoning over graph context
ask("Why does climate change affect biodiversity?")

# 3. Run judge triad for higher confidence
judge_topic("climate change")

# 4. Web search + mother structuring
seed_from_search("quantum computing breakthroughs 2026")

# 5. Expand a sparse node
seed_expand(node_id=7)
```
