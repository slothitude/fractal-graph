# Fractal Graph — Resolution-Aware Knowledge Graph

A multi-resolution knowledge graph where knowledge is organized by specificity level (L0–L5), with semantic bounding boxes enabling zoom-based queries, a **mother model seeding** system for one-time quality, and a **2B reasoning engine** that answers questions over graph context at runtime — no large model needed.

## Core Idea

Most knowledge graphs are flat. Fractal Graph organizes knowledge as a **hierarchy of specificity**:

| Level | Name       | Description       | Example                                          |
|-------|------------|-------------------|--------------------------------------------------|
| L0    | Domain     | Broadest field    | "NATO-Russia relations"                          |
| L1    | Topic      | Named sub-topic   | "NATO eastward expansion"                       |
| L2    | Concept    | Specific mechanism | "Article 5 collective defense"                   |
| L3    | Entity     | Named instance    | "2014 Crimea annexation"                         |
| L4    | Fact       | Verifiable claim  | "NATO added 4 members 1999-2020"                 |
| L5    | Evidence   | Cited source/data | "According to NATO Secretary General..."         |

**Any subtree is itself a valid knowledge graph** (self-similarity). Query specificity determines zoom depth — vague queries stay coarse, specific queries drill to evidence.

## Architecture

```
                              SEEDING (one-time)
                    ┌─────────────────────────────┐
                    │  Mother Model (granite4.1:8b) │
                    │  Builds L0→L{depth} hierarchy │
                    │  Computes bboxes + edges      │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
              ┌─────────────────────────────────────────┐
              │           Fractal Graph                 │
              │  SQLite nodes + ChromaDB vectors        │
              │  Bounding boxes per parent              │
              │  Edges: supports, contradicts, refines   │
              └──────────────────┬──────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
  ┌──────────────┐     ┌──────────────────┐    ┌──────────────┐
  │  2B Classifier│     │ 2B Reasoning     │    │ Judge Triad   │
  │  (classify    │     │  (ask pipeline)   │    │ Angel/Devil/  │
  │   ingestion)  │     │  embed→classify→  │    │ Neutral       │
  └──────────────┘     │  gather→synthesize│    └──────────────┘
                        └──────────────────┘

              RUNTIME — all 2B, no mother model touched
```

### "2B speed + 9B quality"

The key insight (inspired by LaRQL): **the model's job is inference over structured data, not recall**.

1. **Mother model seeds once** — builds full L0→L{depth} hierarchy, detects cross-resolution edges, computes bounding boxes. Uses granite4.1:8b (or qwen3.5:9b).
2. **2B handles all runtime** — the tiny 2B model reasons over the pre-built graph structure:
   - **Classification**: maps text to the right level using bbox containment + 2B fallback
   - **Context gathering**: assembles structured nodes + edges from vector search + graph traversal
   - **Answer synthesis**: 2B generates answers from graph context, not recall (LaRQL INFER)

| Operation        | Model Used | Time  |
|------------------|------------|-------|
| `ask()` pipeline | 2B only    | ~5-8s |
| Ingest classify  | 2B only    | ~2s   |
| Judge triad      | 2B only    | ~3s   |
| `seed_topic()`   | Mother (8B)| ~30s  |
| `seed_expand()`  | Mother (8B)| ~15s  |

## Files

| File              | Purpose                                                                 |
|-------------------|-------------------------------------------------------------------------|
| `config.py`       | Settings — DB paths, Ollama URLs, model config, token budgets           |
| `db.py`           | SQLite storage — nodes, edges, bbox, metadata                           |
| `graph.py`        | Graph ops — bbox computation, subtree extraction, contradiction finding |
| `seed.py`         | Mother seeding — structured hierarchy generation via larger LLM          |
| `ingest.py`       | Ingest pipeline — SearXNG search, text extraction, 2B classification      |
| `embedder.py`     | Ollama embedding client — async embed + sync fallback                    |
| `chroma_store.py` | ChromaDB — one collection per level, concurrent-access safe               |
| `context.py`      | Context assembler — LaRQL DESCRIBE/WALK, gather + format for LLM         |
| `reasoning.py`    | 2B reasoning engine — LaRQL INFER, classify + synthesize + answer        |
| `growth.py`       | Autonomous expansion — gap fill, enrichment, curiosity scan, web search fallback |
| `query.py`        | Query engine — specificity classification, drill-down, multi-level pull   |
| `gql.py`          | Structured graph queries — contradictions, evidence, entity comparison      |
| `judges.py`       | Judge triad — Angel/Devil/Neutral verdicts with resolution_conflict      |
| `bench.py`        | Benchmark — compare 2B+graph vs 2B standalone vs 9B standalone             |
| `factal_server.py`| MCP server — 20 tools exposed via FastMCP stdio                           |

## MCP Tools (20 total)

### Ask — 2B Reasoning (1 tool)
- `ask(question)` — The primary user-facing tool. 2B reasons over structured graph context to answer questions. Pipeline: embed → classify (2B) → gather context → synthesize answer (2B). Returns answer, confidence, key facts, gaps, and timing.

### Knowledge Ingest (4 tools)
- `web_ingest(query, max_urls)` — Search SearXNG → extract → classify → insert nodes
- `add_node(content, resolution_level, parent_id, confidence, source_url, metadata)` — Manual node creation
- `add_edge(from_node_id, to_node_id, edge_type, confidence, context)` — Create edge between nodes
- `ingest_url(url)` — Fetch URL, chunk, classify, insert

### Mother Model Seeding (3 tools)
- `seed_topic(topic, depth, mother_model)` — Seed a topic from scratch (L0→L{depth})
- `seed_from_search(query, max_urls, mother_model)` — Web search + mother enrichment
- `seed_expand(node_id, mother_model)` — Expand sparse node, fill missing levels

### Query / Retrieval (4 tools)
- `query_graph(prompt, resolution_hint)` — Resolution-aware query (zoom based on specificity)
- `drill_down(node_id, target_resolution)` — Traverse hierarchy from a node
- `search_nodes(query_text, resolution_level)` — Semantic search within a level or all
- `get_node(node_id)` — Full node details with children, parents, edges

### Graph Operations (5 tools)
- `get_subtree(node_id, max_depth)` — Extract subtree (self-similar subgraph)
- `find_contradictions()` — All CONTRADICTS/CHALLENGES/resolution_conflict edges
- `propagate_confidence(node_id)` — Bottom-up confidence propagation
- `graph_stats()` — Nodes/edges per level, ChromaDB coverage

### Judge Triad (1 tool)
- `judge_topic(topic, top_k)` — Run Angel/Devil/Neutral judge triad. Angel sees L0-L1 (optimistic), Devil sees L4-L5 (adversarial), Neutral bridges L2-L3. Disagreements logged as `resolution_conflict` edges.

### Bounding Boxes

Parent nodes store a **semantic bounding box** — the min/max of all child embeddings in vector space. Enables containment queries, fast subtree routing, and bottom-up recomputation.

## Reasoning Pipeline

The `ask()` tool implements a LaRQL-inspired pipeline:

```
User Question
    │
    ▼
[1] EMBED — embed question via nomic-embed-text
    │
    ▼
[2] CLASSIFY — 2B classifies question type
    │  factual | analytical | comparative | exploratory | yes_no
    │  Extracts entities, time focus, complexity
    │
    ▼
[3] GATHER CONTEXT — search all 6 levels, walk graph
    │  LaRQL DESCRIBE — assemble nodes + edges + parent chains
    │  Rank: 0.4*vector_sim + 0.3*confidence + 0.2*edge_count + 0.1*resolution
    │  Token-budget truncation (2048 tokens for 2B context window)
    │
    ▼
[4] SYNTHESIZE — 2B generates answer from structured context
    │  LaRQL INFER — reasons over facts, not recall
    │  Returns: answer, confidence, key_facts, gaps
    │
    ▼
[5] AUTO-EXPAND (if low confidence or gaps)
    │
    ├─[a] Background enrich — fire-and-forget topic decomposition
    │
    └─[b] Gap fill — mother creates nodes for missing knowledge
         │  If mother fails (no graph info) →
         └──── web search fallback (seed_from_search)
              SearXNG → extract URLs → mother structures into L0-L5
    │
    ▼
[6] RE-SYNTHESIZE — 2B re-answers with enriched context
    │
    ▼
Answer + search_fallback flag (was web search needed?)
```

## Benchmarking

```bash
# Compare answer quality across modes
python bench.py

# Modes tested:
#   1. 2B + graph context (our pipeline)
#   2. 2B standalone (no graph)
#   3. 8B standalone (no graph)
#
# Test questions at L0/L2/L4/L5
# Logs: answer text, time, confidence
```

## Configuration

All settings via environment variables with `FRACTAL_` prefix:

| Setting             | Default                          | Description                    |
|---------------------|----------------------------------|--------------------------------|
| `FRACTAL_DB_PATH`   | `data/fractal.db`                | SQLite database path           |
| `FRACTAL_CHROMA_PATH` | `data/chroma`                  | ChromaDB persistent storage    |
| `FRACTAL_OLLAMA_URL` | `http://100.84.161.63:11434`  | Ollama embedding endpoint      |
| `FRACTAL_EMBED_MODEL` | `nomic-embed-text`             | Embedding model                |
| `FRACTAL_SEARXNG_URL` | `http://100.84.161.63:8888`   | SearXNG search endpoint        |
| `FRACTAL_LLM_URL`   | `http://100.84.161.63:11434`    | Runtime 2B classifier endpoint  |
| `FRACTAL_LLM_MODEL` | `qwen3.5:2b`                     | Runtime classifier (fast, tiny) |
| `FRACTAL_MOTHER_URL` | `http://100.84.161.63:11434`   | Mother model endpoint           |
| `FRACTAL_MOTHER_MODEL` | `granite4.1:8b`               | Mother model (for seeding)     |
| `FRACTAL_SERVER_PORT` | `8018`                         | MCP server port                |

## Setup

```bash
# Install dependencies
pip install fastmcp chromadb httpx pydantic-settings

# Configure Ollama endpoint (or set FRACTAL_OLLAMA_URL)
# The server expects Ollama with:
#   - nomic-embed-text (embeddings)
#   - qwen3.5:2b (runtime classification + reasoning)
#   - granite4.1:8b (mother model for seeding)

# Start the MCP server (stdio mode for Claude Code)
python factal_server.py
```

## Edge Types

| Type                  | Direction  | Meaning                                  |
|-----------------------|------------|------------------------------------------|
| `related`             | bidirectional | General association                    |
| `refines`            | child→parent | Child is a refinement of parent        |
| `contradicts`        | either     | Nodes contradict each other             |
| `exemplifies`         | child→parent | Child is an example of parent concept  |
| `generalizes`        | parent→child | Parent generalizes the child           |
| `challenges`         | either     | One node challenges the other          |
| `supports`           | either     | One node supports the other            |
| `derived_from`        | child→parent | Child was derived from parent source  |
| `resolution_conflict` | judge→judges | Judges disagree on a topic verdict    |

## Quick Start

```python
# 1. Seed a topic with the mother model
seed_topic("climate change", depth=3)

# 2. Ask a question — 2B reasoning over graph context
ask("Why does climate change affect biodiversity?")

# 3. Check stats
graph_stats()

# 4. Drill down from a node
drill_down(node_id=1, target_resolution=4)

# 5. Find contradictions
find_contradictions()

# 6. Run judge triad
judge_topic("climate change")

# 7. Web search + mother structuring
seed_from_search("quantum computing breakthroughs 2026")

# 8. Expand a sparse node
seed_expand(node_id=7)
```
