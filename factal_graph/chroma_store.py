"""ChromaDB storage — one collection per resolution level."""

import chromadb
from config import settings

_client = None
_collections: dict[int, chromadb.Collection] = {}


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(settings.chroma_path))
    return _client


def get_collection(level: int) -> chromadb.Collection:
    """Get or create a ChromaDB collection for a resolution level."""
    if level not in _collections:
        client = get_client()
        _collections[level] = client.get_or_create_collection(
            name=f"level_{level}",
            metadata={"resolution_level": level}
        )
    return _collections[level]


def upsert_node(node_id: int, text: str, embedding: list[float],
                resolution_level: int, parent_id: int = None,
                confidence: float = 0.5, source_url: str = None):
    """Add or update a node embedding in ChromaDB."""
    col = get_collection(resolution_level)
    metadata = {
        "resolution_level": resolution_level,
        "confidence": confidence,
    }
    if parent_id is not None:
        metadata["parent_id"] = parent_id
    if source_url:
        metadata["source_url"] = source_url

    col.upsert(
        ids=[str(node_id)],
        documents=[text],
        embeddings=[embedding],
        metadatas=[metadata]
    )


def delete_node(node_id: int, resolution_level: int):
    """Remove a node from ChromaDB."""
    col = get_collection(resolution_level)
    col.delete(ids=[str(node_id)])


def query_level(embedding: list[float], level: int, n_results: int = 10) -> list[dict]:
    """Vector search within a single resolution level."""
    col = get_collection(level)
    if col.count() == 0:
        return []
    results = col.query(
        query_embeddings=[embedding],
        n_results=min(n_results, col.count())
    )
    nodes = []
    for i in range(len(results["ids"][0])):
        nodes.append({
            "node_id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "distance": results["distances"][0][i],
            "metadata": results["metadatas"][0][i],
        })
    return nodes


def query_all_levels(embedding: list[float], n_results: int = 5) -> dict[int, list[dict]]:
    """Vector search across all resolution levels."""
    results = {}
    for level in range(6):
        hits = query_level(embedding, level, n_results)
        if hits:
            results[level] = hits
    return results


def get_stats() -> dict:
    """Get ChromaDB stats per collection."""
    client = get_client()
    stats = {}
    for level in range(6):
        try:
            col = client.get_collection(f"level_{level}")
            stats[level] = {"count": col.count()}
        except Exception:
            stats[level] = {"count": 0}
    return stats
