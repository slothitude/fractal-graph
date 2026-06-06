# Fractal Graph — Goal

## Main Objective: 2B Speed + 8B Knowledge Quality

A 2B model that answers questions with 8B-level knowledge, at 2B speed, using ultra-low VRAM.

### How It Works
1. **8B seeds once** — Mother model (granite4.1:8b) pre-computes knowledge into a multi-resolution graph (L0-L5 hierarchy with bounding boxes). This is the expensive part — ~30s per topic. Done once.
2. **2B reasons at runtime** — The tiny 2B model (qwen3.5:2b) never "knows" facts. It receives structured graph context and **reasons over it** — like a lawyer with a fact sheet. No 8B at inference time. ~5-8s per answer.

### VRAM Profile
| Phase | Model | VRAM |
|-------|-------|------|
| Seeding (one-time) | granite4.1:8b | ~5.3GB |
| Runtime (every call) | qwen3.5:2b | ~2.4GB |
| Embeddings | nomic-embed-text | ~0.3GB |

With `OLLAMA_MAX_LOADED_MODELS=1`, only one model in VRAM at a time. Runtime cost: **2.4GB** (the 2B). The 8B is only loaded during seeding.

### The Speedup
| Approach | Model | VRAM | Time/Answer |
|---------|-------|------|-------------|
| Direct 8B query | granite4.1:8b | ~5.3GB | ~10-15s |
| Our pipeline | qwen3.5:2b + graph | ~2.4GB | ~5-8s |
| 8B standalone (baseline) | granite4.1:8b | ~5.3GB | ~15-30s |

**~2x faster, ~2x less VRAM**, with the quality of 8B-level structured knowledge.

---

## Current Sprint: Reliability Fixes

### Problem
The pipeline works in theory but crashes in practice due to Ollama model loading races and hallucination.

### Fixes (in progress)
1. **Model warmup polling** — `_warmup_model()` polls `/api/ps` until model is loaded, no timeout-based killing
2. **`keep_alive: "0"`** — Free VRAM after every call so models don't block each other
3. **Web search fallback** — When graph lacks info (mother would hallucinate), fetch real data from the internet via `seed_from_search()`
4. **Recursion guards** — Background enrich only on round 0, skip recursion on LLM parse failure

### Status
- [x] All code changes committed (`8ec9e20`)
- [ ] Run `python -u bench.py` — verify all 4 modes, no crashes
- [ ] Verify search fallback on non-graph topics, no fallback on graph-native topics

---

## Future
- Cache model load state to skip cold-starts on repeated calls
- Seed more topics to expand graph coverage
- Parallelize LLM calls in seed pipeline
