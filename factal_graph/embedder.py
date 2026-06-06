"""Ollama embedding client for nomic-embed-text."""

import asyncio

import httpx
from config import settings


async def embed(text: str) -> list[float]:
    """Embed text using Ollama nomic-embed-text. Returns 768-dim vector."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{settings.ollama_url}/api/embeddings",
            json={"model": settings.embed_model, "prompt": text}
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


async def embed_batch(texts: list[str]) -> list[list[float] | None]:
    """Embed multiple texts sequentially. Returns list of vectors (None on failure)."""
    results = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for text in texts:
            try:
                resp = await client.post(
                    f"{settings.ollama_url}/api/embeddings",
                    json={"model": settings.embed_model, "prompt": text}
                )
                resp.raise_for_status()
                results.append(resp.json()["embedding"])
            except Exception as e:
                print(f"Embedding failed: {e}")
                results.append(None)
    return results


async def embed_batch_parallel(texts: list[str], batch_size: int = 10) -> list[list[float] | None]:
    """Embed multiple texts concurrently in batches."""
    results: list[list[float] | None] = [None] * len(texts)

    async def _embed_one(client: httpx.AsyncClient, index: int, text: str):
        try:
            resp = await client.post(
                f"{settings.ollama_url}/api/embeddings",
                json={"model": settings.embed_model, "prompt": text}
            )
            resp.raise_for_status()
            results[index] = resp.json()["embedding"]
        except Exception as e:
            print(f"Embedding failed for index {index}: {e}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        for start in range(0, len(texts), batch_size):
            tasks = [
                _embed_one(client, i, texts[i])
                for i in range(start, min(start + batch_size, len(texts)))
            ]
            await asyncio.gather(*tasks)

    return results


def embed_sync(text: str) -> list[float]:
    """Synchronous embed for use in sync contexts."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, embed(text)).result()
    return asyncio.run(embed(text))
