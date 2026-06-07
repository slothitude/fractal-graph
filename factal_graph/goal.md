# Fractal Graph — Goal

## Main Objective

A 2B model that answers questions with ~8B-level knowledge quality, at 2B speed, using ultra-low VRAM.

### How It Works
1. **Mother seeds once** — Mother model (lfm2.5:latest, ~8B) pre-computes knowledge into a multi-resolution graph (L0-L5 hierarchy with semantic bounding boxes). ~5s per topic, done once.
2. **2B reasons at runtime** — qwen3.5:2b receives structured graph context and reasons over it. Never "knows" facts — like a lawyer with a fact sheet. ~5-8s per answer.

### VRAM Profile
| Phase | Model | VRAM |
|-------|-------|------|
| Seeding (one-time) | lfm2.5:latest (~8B) | ~5GB |
| Runtime (every call) | qwen3.5:2b | ~2.4GB |
| Embeddings | nomic-embed-text | ~0.3GB |

With `OLLAMA_MAX_LOADED_MODELS=1`, only one model in VRAM at a time. Runtime cost: **2.4GB**.

### Benchmarked Performance
| Approach | Model | Time/Answer | Confidence |
|---------|-------|-------------|------------|
| Direct mother query | lfm2.5 (~8B) | ~15-20s | — (empty JSON ~40%) |
| 2B + graph (fast) | qwen3.5:2b + graph | ~4.9s | 0.60 |
| 2B + graph + triad | qwen3.5:2b + graph + judges | ~11.9s | **0.84** |
| 2B + graph + auto-expand | qwen3.5:2b + mother | ~65.6s | 0.77 |

**Winner: Triad mode** — 0.84 confidence at 11.9s. Auto-expand is too expensive for default use.

---

## Current Sprint: Phase 16 — Monte Carlo Graph Search + Agent Behavior Distillation

### Problem
Phase 15's `decide()` had issues:
- **2B decide()**: avg_conf=0.76, avg_time=60.8s — slow, only 0.30 relevance
- **0.8b decide()**: avg_conf=0.84, avg_time=36.6s — faster but unreliable JSON
- **No procedural knowledge seeded** — reasoned over factual context, not decision rules
- **Single-pass decision** — one call to `decide_synthesize()`, one shot at the right answer

### Solution: Monte Carlo Graph Search
Run N simulations per model, each sampling a different subset of graph nodes as context.
The graph IS the search space. Each simulation picks a random walk through the graph,
gets a different context window, and the 2B/0.8b reasons over it. Aggregate by majority vote.

Plus: **Behavior distillation** — run MC decisions on probe situations, extract patterns
from the traces, store as L2-L4 procedural knowledge nodes.

### Success Metrics
- [ ] MC lifts 2B decide confidence from 0.76 to >= 0.85
- [ ] MC lifts 0.8b decide confidence from 0.84 to >= 0.90
- [ ] Action consistency >= 80% across simulations (most sims agree)
- [ ] MC finds 2x more risks than single pass
- [ ] Behavior distillation creates 20+ L2-L4 procedural nodes per model
- [ ] MC decide time < 20s for 0.8b, < 30s for 2b

---

## Previous Sprint: Distill Mother Into the Graph

### Problem
The graph is static after seeding. The 2B model can only reason over pre-loaded nodes. If it encounters a gap, it either:
- Returns low confidence (0.60 without triad)
- Triggers auto-expand (65.6s — too slow)
- Returns a weak/empty answer

The mother model's parametric knowledge is rich but only sampled at seed time in a top-down hierarchy. Most of what the mother "knows" never makes it into the graph.

### Solution: Proactive Distillation
Instead of waiting for the 2B to hit a gap and reactively calling the mother, **proactively distill the mother's full knowledge into graph nodes**.

### Key Insight
Current seeding creates a hierarchy (domain -> topic -> concept -> entity -> fact -> evidence). But the mother knows far more than a hierarchy — it knows relationships between concepts, specific facts, contradictions, and nuances. We need extraction prompts that capture this, not just taxonomic structure.

### Success Metrics
- [ ] Graph coverage: every L0 node has children down to L4 (not just L0-L2)
- [ ] 2B+Graph confidence averages 0.75+ without triad or auto-expand
- [ ] Auto-expand trigger rate drops from 87% to <20% of questions
- [ ] End-to-end answer time stays under 15s for 90% of questions

---

## Completed Sprints

### Sprint 1: Core Pipeline (DONE)
2B reasoning pipeline with context gathering, classification, synthesis. Bench: 4.9s/0.60 conf.

### Sprint 2: Mother Seeding + Stability (DONE)
Structured hierarchy generation, warmup polling, keep_alive:0, web search fallback. No crashes in 846s bench.

### Sprint 3: Judge Triad (DONE)
Angel/Devil/Neutral two-pass with resolution_conflict edges. Bench: 11.9s/0.84 conf.

### Sprint 4: Code Dedup (DONE)
Replaced ~150 lines of duplicated search/extract with searchMCP core imports. Clean separation.

---

## Future (after distillation)
- Cache model load state to skip cold-starts
- Parallelize LLM calls in seed pipeline
- Judge triad integration with Pantheon council/court
- Graph persistence + versioning (snapshot/rollback)
