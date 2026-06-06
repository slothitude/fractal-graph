# Fractal Graph — Current Goal

## Goal: Web Search Fallback + Model Loading Stability

### Problem
The auto-expand pipeline in `reasoning.py:answer()` had two failures:

1. **Model loading race** — With `OLLAMA_MAX_LOADED_MODELS=1`, only one model fits in VRAM at a time. The pipeline calls the 2B model, then the mother model for gap-fill, then the 2B again for re-synthesis. Each call needs a cold-start load (~26s for 2B, ~30s for 8B). The httpx timeouts were too short to cover cold starts, and the `keep_alive` default kept models loaded blocking subsequent calls.

2. **Mother hallucination** — When `fill_gaps()` fails to find info in the graph (e.g. "quantum error correction"), the 8B mother fabricates plausible-sounding nodes. The graph gets populated with hallucinations.

### Solution (implementing)
1. **Model warmup polling** — `_warmup_model()` triggers a load, then polls `/api/ps` until the model appears loaded. No timeouts. Timed for observability.
2. **`keep_alive: "0"` everywhere** — Models unload immediately after use, freeing the VRAM slot for the next model.
3. **Web search fallback** — When `fill_gaps()` creates 0 nodes (mother hallucination or failure), falls back to `seed_from_search()` which fetches real info from the internet.
4. **Recursion guards** — Background enrich only on round 0, don't recurse on LLM parse failure.

### Status
- [x] `_warmup_model()` with `/api/ps` polling in `reasoning.py` and `seed.py`
- [x] `_warmup()` in `bench.py` standalone calls
- [x] `keep_alive: "0"` on all 4 LLM call sites (2B reasoning, 2B bench, 8B bench, mother)
- [x] Web search fallback in `growth.py:fill_gaps()` → `seed_from_search()` on failure
- [x] Recursion guard: background enrich only round 0, skip recursion on parse failure
- [x] `search_fallback` tracking in response dict and bench output
- [ ] Run `python -u bench.py` to verify — all 4 modes complete, no crashes, no recursion errors
- [ ] Verify search fallback fires on non-graph topics (quantum error correction)
- [ ] Verify no search fallback on graph-native topics (NATO)
- [ ] Commit and push

### Files Modified
| File | Change |
|------|--------|
| `reasoning.py` | `_warmup_model()` poll, `keep_alive: "0"`, recursion guards, `search_fallback` tracking |
| `growth.py` | Web search fallback in `fill_gaps()`, `search_fallback` in return dict |
| `seed.py` | `keep_alive: "0"`, warmup poll in `_mother_generate()` |
| `bench.py` | `_warmup()` for standalone calls, `search_fb` column, timeout bumps |

### Next After This
- Run bench to validate all fixes
- Seed more topics to expand graph coverage
- Consider caching loaded model state to avoid cold-starts on every call
