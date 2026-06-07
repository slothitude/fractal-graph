"""Model load state cache — avoid redundant cold-start warmup calls.

Ollama's /api/ps endpoint tells us if a model is already loaded in VRAM.
Checking it is ~0.2s over Tailscale. Firing a dummy generate to trigger
loading is ~15-60s. This module caches the result so repeated calls within
a TTL window skip the network round-trip entirely.

Usage:
    from model_cache import ensure_model_loaded
    await ensure_model_loaded("lfm2.5:gpu3", "http://100.84.161.63:11434")
"""

import asyncio
import logging
import time

import httpx

logger = logging.getLogger(__name__)

# In-memory cache: {(model, url): expiry_timestamp}
_cache: dict[tuple[str, str], float] = {}
_cache_lock = asyncio.Lock()

TTL = 120.0  # seconds — how long we trust a "loaded" check


async def is_model_loaded(model: str, url: str) -> bool:
    """Check /api/ps to see if model is in VRAM. Uses cache if fresh."""
    key = (model, url)
    now = time.time()

    async with _cache_lock:
        expiry = _cache.get(key)
        if expiry and now < expiry:
            return True  # cached as loaded

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{url}/api/ps")
            models = resp.json().get("models", [])
            loaded = any(m["name"] == model for m in models)
            if loaded:
                async with _cache_lock:
                    _cache[key] = now + TTL
            return loaded
    except Exception:
        return False


async def ensure_model_loaded(model: str, url: str) -> float:
    """Ensure model is in VRAM. Returns elapsed seconds.

    - If cached as loaded within TTL: instant return (0s).
    - If /api/ps says loaded: cache and return.
    - Otherwise: fire dummy generate to trigger load, cache, return.
    """
    import time as _time
    t0 = _time.time()

    if await is_model_loaded(model, url):
        elapsed = round(_time.time() - t0, 1)
        logger.debug("warmup %s: %ss (cached)", model, elapsed)
        return elapsed

    # Kick off a dummy generate to trigger loading
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            await client.post(
                f"{url}/api/chat",
                json={"model": model,
                      "messages": [{"role": "user", "content": "."}],
                      "stream": False,
                      "keep_alive": "5m",
                      "options": {"num_predict": 1}},
            )
    except Exception:
        pass

    # Mark as loaded regardless — either it loaded or we shouldn't retry
    async with _cache_lock:
        _cache[(model, url)] = _time.time() + TTL

    elapsed = round(_time.time() - t0, 1)
    logger.debug("warmup %s: %ss", model, elapsed)
    return elapsed


def invalidate(model: str = None, url: str = None) -> None:
    """Invalidate cache entry or all entries."""
    import time as _time
    if model and url:
        _cache.pop((model, url), None)
    else:
        _cache.clear()
