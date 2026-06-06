# Fractal Graph — Resolution-Aware Knowledge Graph

A multi-resolution knowledge graph where knowledge is organized by specificity level (L0–L5), with semantic bounding boxes enabling zoom-based queries, and a **mother model seeding** system that lets a tiny 2B model perform at 9B quality by leveraging pre-built structure.

## Core Idea

Most knowledge graphs are flat. Fractal Graph organizes knowledge as a **hierarchy of specificity**:

| Level | Name | Description | Example |
|-------|------|-------------|---------|
| L0 | Domain | Broadest field | "NATO-Russia relations" |
| L1 | Topic | Named sub-topic | "NATO eastward expansion" |
| L2 | Concept | Specific mechanism | "Article 5 collective defense" |
| L3 | Entity | Named instance | "2014 Crimea annexation" |
| L4 | Fact | Verifiable claim | "NATO added 4 members 1999-2020" |
| L5 | Evidence | Cited source/data | "According to NATO Secretary General..." |

**Any subtree is itself a valid knowledge graph** (self-similarity). Query specificity determines zoom depth — vague queries stay coarse, specific queries drill to evidence.

## Architecture

```
User Query
    │
    ▼
┌──────────────┐     ┌──────────────┐
│ 2B Classifier│     │ 9B Mother    │ ← Seeds high-quality
│ (runtime)    │     │ (one-time)   │   structure offline
└──────┬───────┘     └──────┬───────┘
       │                    │
       ▼                    ▼
┌──────────────────────────────────┐
│         Fractal Graph            │
│  SQLite nodes + ChromaDB vectors │
│  Bounding boxes per parent       │
└──────────┬───────────────────────┘
           │
           ▼
    Resolution-Aware Query
    (zoom based on specificity)
```

### How the 2B Model Performs at 9B Level

The mother model (qwen3.5:9b) runs **once** to seed a topic — building the full L0→L5 hierarchy, detecting cross-resolution edges, computing bounding boxes. After seeding, the tiny 2B model handles all runtime operations:

- **Classification**: 2B maps text to the right level using the pre-built structure (bbox containment) rather than needing deep understanding
- **Parent finding**: Uses vector similarity + bbox containment in pre-computed bounding boxes — no reasoning needed
- **Query routing**: Specificity heuristics (word count, dates, quotes) determine zoom depth — trivial for any model

The 2B model is effectively a **routing layer** over intelligence that was front-loaded by the 9B mother.

## Files

| File | Purpose |
|------|---------|
| `config.py` | Settings — DB paths, Ollama URLs, mother model config |
| `db.py` | SQLite storage — nodes, edges, bbox, metadata |
| `graph.py` | Graph ops — bbox computation, subtree extraction, contradiction finding, confidence propagation |
| `ingest.py` | Ingest pipeline — SearXNG search, text extraction, LLM classification, node insertion |
| `embedder.py` | Ollama embedding client — sequential + parallel batch embedding |
| `chroma_store.py` | ChromaDB — one collection per resolution level, vector search |
| `query.py` | Query engine — specificity classification, drill-down, multi-level retrieval |
| `seed.py` | **Mother seeding** — structured hierarchy generation via larger LLM |
| `judges.py` | Judge triad engine — Angel/Devil/Neutral verdicts with synthesis |
| `factal_server.py` | MCP server — all tools exposed via stdio |

## MCP Tools (19 total)

### Knowledge Ingest (4 tools)
- `web_ingest(query, max_urls)` — Search SearXNG → extract → classify → insert nodes
- `add_node(content, resolution_level, parent_id, confidence, source_url, metadata)` — Manual node creation
- `add_edge(from_node_id, to_node_id, edge_type, confidence, context)` — Create edge between nodes
- `ingest_url(url)` — Fetch URL, chunk, classify, insert

### Mother Model Seeding (3 tools)
- `seed_topic(topic, depth, mother_model)` — Seed a topic end-to-end from scratch (L0→L{depth})
- `seed_from_search(query, max_urls, mother_model)` — Web search + mother enrichment (structures extracted content)
- `seed_expand(node_id, mother_model)` — Expand sparse node, fill missing resolution levels

### Query / Retrieval (4 tools)
- `query_graph(prompt, resolution_hint)` — Resolution-aware query (zoom based on specificity)
- `drill_down(node_id, target_resolution)` — Traverse L0→L5 from a node
- `search_nodes(query_text, resolution_level)` — Semantic search within a level or all levels
- `get_node(node_id)` — Full node details with children, parents, edges

### Graph Operations (5 tools)
- `get_subtree(node_id, max_depth)` — Extract subtree (self-similar subgraph)
- `find_contradictions()` — All CONTRADICTS/CHALLENGES edges with context
- `propagate_confidence(node_id)` — Bottom-up confidence propagation
- `graph_stats()` — Nodes/edges per level, ChromaDB coverage
- `ingest_url(url)` — Fetch and classify URL content

### Judge Triad (2 tools)
- `judge_topic(topic, top_k)` — Run Angel/Devil/Neutral judge triad on a topic. Angel sees the forest (L0-L1, optimistic), Devil sees the trees (L4-L5, adversarial), Neutral bridges (L2-L3). Disagreements logged as `resolution_conflict` edges.
- `find_contradictions()` — Lists `resolution_conflict` edges from judge triad disagreements (listed above under Graph Operations)

### Bounding Boxes

Parent nodes store a **semantic bounding box** — the min/max of all child embeddings in vector space. This enables:

- **Containment queries**: Is a query embedding within a parent's bbox? → drill into that subtree
- **Fast routing**: Skip vector similarity checks for entire subtrees
- **Bottom-up computation**: After any node insert, bboxes recompute up the parent chain

## Seeding Modes

### 1. `seed_topic("NATO expansion", depth=3)`
```
Mother generates:
  L0: "NATO-Russia relations" (domain)
  ├── L1: "NATO enlargement policy" (topic)
  │   ├── L2: "Open Doors policy" (concept)
  │   └── L2: "Membership Action Plans" (concept)
  ├── L1: "Russian security concerns" (topic)
  │   ├── L2: "Near abroad doctrine" (concept)
  │   └── L2: "Red line declarations" (concept)
  └── L1: "Eastern European perspectives" (topic)
      ├── L2: "Baltic state accession" (concept)
      └── L2: "Polish security doctrine" (concept)
+ cross-resolution edges (contradicts, supports)
+ bounding boxes computed
```

### 2. `seed_from_search("quantum computing", max_urls=5)`
```
1. SearXNG search → 5 URLs
2. Extract text from each
3. Mother structures ALL content at once:
   - Detects resolution levels present
   - Scaffolds missing L0-L3 if content is mostly L4-L5
   - Identifies cross-document relationships
4. Creates nodes with proper hierarchy
```

### 3. `seed_expand(node_id=7)`
```
1. Gets node + children
2. Detects missing levels (e.g. has L4 children but no L2/L3)
3. Mother generates bridge nodes at missing levels
4. Recomputes bboxes
```

## Configuration

All settings via environment variables with `FRACTAL_` prefix:

| Setting | Default | Description |
|---------|---------|-------------|
| `FRACTAL_DB_PATH` | `data/fractal.db` | SQLite database path |
| `FRACTAL_CHROMA_PATH` | `data/chroma` | ChromaDB persistent storage |
| `FRACTAL_OLLAMA_URL` | `http://100.84.161.63:11434` | Ollama embedding endpoint |
| `FRACTAL_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `FRACTAL_SEARXNG_URL` | `http://100.84.161.63:8888` | SearXNG search endpoint |
| `FRACTAL_LLM_URL` | `http://100.84.161.63:11434` | Runtime classifier LLM |
| `FRACTAL_LLM_MODEL` | `qwen3.5:2b` | Runtime classifier model (fast, tiny) |
| `FRACTAL_MOTHER_URL` | `http://100.84.161.63:11434` | Mother model endpoint |
| `FRACTAL_MOTHER_MODEL` | `qwen3.5:9b` | Mother model (slow, smart, for seeding) |
| `FRACTAL_SERVER_PORT` | `8018` | MCP server port |

## Setup

```bash
# Install dependencies
pip install fastmcp chromadb httpx pydantic-settings

# Start the MCP server (stdio mode for Claude Code)
python factal_server.py
```

## Edge Types

| Type | Direction | Meaning |
|------|-----------|---------|
| `related` | bidirectional | General association |
| `refines` | child→parent | Child is a refinement of parent |
| `contradicts` | either | Nodes contradict each other |
| `exemplifies` | child→parent | Child is an example of parent concept |
| `generalizes` | parent→child | Parent generalizes the child |
| `challenges` | either | One node challenges the other |
| `supports` | either | One node supports the other |
| `derived_from` | child→parent | Child was derived from parent source |
| `resolution_conflict` | judge→judges | Judges disagree on a topic verdict |

## Verification

```python
# 1. Seed a topic
seed_topic("NATO expansion", depth=3)

# 2. Check stats — should show nodes at L0-L3
graph_stats()

# 3. Query — should return drill-down path
query_graph("Why did Russia invade Ukraine?")

# 4. Drill down — should traverse L0→L4
drill_down(node_id=1, target_resolution=4)

# 5. Find contradictions — should surface cross-resolution tensions
find_contradictions()

# 6. Expand a sparse node
seed_expand(node_id=7)

# 7. Web search + mother structuring
seed_from_search("quantum computing breakthroughs 2025")

# 8. Judge triad — Angel/Devil/Neutral verdicts on a topic
judge_topic("NATO expansion")

# 9. Check judge disagreements (resolution_conflict edges)
find_contradictions()
```
