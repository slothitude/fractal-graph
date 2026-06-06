Let's think about this from first principles.

**The fundamental data structure:**

```
Node {
    id
    resolution_level: int      # 0 = coarsest, N = finest
    content
    children: [node_id]        # finer resolution expansions
    parent: node_id            # coarser resolution summary
    lateral_edges: [Edge]      # same-resolution relationships
    cross_edges: [Edge]        # inter-resolution relationships
    confidence: float
    bbox: [float]              # position in semantic space
}
```

The `bbox` is critical — it's what makes it fractal rather than just hierarchical. Each node occupies a *region* of semantic space, and children subdivide that region. Like a quadtree but in embedding space.

**The resolution levels semantically:**

```
L0  →  domains         (geopolitics, science, history)
L1  →  topics          (French foreign policy, WW2)
L2  →  concepts        (Maginot Line, appeasement)
L3  →  entities        (Chamberlain, specific treaty)
L4  →  facts           (signed 1938-09-30, Munich)
L5  →  evidence        (source, quote, confidence)
```

Not rigid — a node decides its own resolution by its semantic granularity.

**The cross-resolution edges are the novel bit:**

```
Edge {
    from_id
    to_id
    from_resolution: int
    to_resolution: int
    edge_type: REFINES | CONTRADICTS | EXEMPLIFIES | GENERALIZES | CHALLENGES
    confidence: float
}
```

This is what flat graphs and even RAPTOR miss. A L2 concept node can have a `CONTRADICTS` edge to a L4 fact node — that tension is *first-class data*, not a query artifact.

**Storage layer:**

You don't actually need a new DB engine. You layer this on top of something like:

```
SQLite or DuckDB     → node/edge metadata, resolution index
ChromaDB or Qdrant   → embeddings per node (one per resolution level)
JSON files or LMDB   → raw content blobs
```

The fractal behavior comes from the **query engine**, not the storage.

**The query engine — resolution-aware traversal:**

```python
def query(prompt, resolution_hint=None):
    # embed the prompt
    vec = embed(prompt)
    
    # find semantic bbox match at coarse level first
    coarse = vector_search(vec, level=0, top_k=3)
    
    # drill down based on specificity signal
    if is_specific(prompt):
        return drill_down(coarse, target_resolution=4)
    elif is_vague(prompt):
        return coarse  # stay at summary level
    else:
        return multi_resolution_pull(coarse, levels=[0,2,4])
```

**The drill-down operation:**

```python
def drill_down(nodes, target_resolution):
    current = nodes
    path = [current]
    
    while current.resolution < target_resolution:
        children = get_children(current)
        # pick children whose bbox contains query vector
        current = bbox_filter(children, query_vec)
        path.append(current)
    
    return path  # entire zoom path, not just leaf
```

Returning the *path* is important — your judge triad sees the coarse context AND the fine detail simultaneously.

**The fractal insert — how new knowledge lands:**

```
new fact arrives
    ↓
embed it
    ↓
find bbox match at L0 (which domain?)
    ↓
descend until bbox no longer contains it
    ↓
insert at that level
    ↓
propagate confidence up to parent summary nodes
    ↓
check for cross-resolution contradictions
    ↓
flag conflicts as first-class edges
```

**The self-similar property:**

Any subtree is itself a valid complete graph. You can detach the "French foreign policy" subgraph and it has its own L0→L5 structure internally. This is what makes it fractal rather than just hierarchical — **scale invariance**.

**What makes this hard:**

The bbox boundaries in embedding space are fuzzy. "Where does geopolitics end and economics begin" has no clean answer. You need soft bbox membership — a node can partially belong to two parent regions, which means your resolution levels overlap rather than partition cleanly.

That's actually fine — it models reality better. But it means your query engine needs to handle **resolution overlap** gracefully, returning blended results from competing parent contexts.

**The connection to your judge triad:**

Each judge could operate at a different default resolution:

- **Angel** — coarse, optimistic summary level
- **Devil** — fine, adversarial evidence level  
- **Neutral** — mid, cross-resolution coherence checker

Their disagreements become *resolution disagreements* as much as factual ones. The deliberating model learns that Angel/Devil conflict often just means "zoom level mismatch" rather than genuine contradiction.

This is a genuinely novel architecture. The closest existing work is Microsoft's GraphRAG hierarchical communities but they don't have cross-resolution edges, bbox semantics, or the fractal self-similarity property.