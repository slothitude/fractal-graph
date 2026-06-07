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


def reset_collections():
    """Clear the collection cache — forces fresh handles on next access."""
    global _collections
    _collections = {}


def get_collection(level: int) -> chromadb.Collection:
    """Get or create a ChromaDB collection for a resolution level."""
    if level not in _collections:
        client = get_client()
        try:
            _collections[level] = client.get_collection(
                name=f"level_{level}",
            )
        except Exception:
            # Collection corrupt or missing — recreate
            _collections[level] = client.get_or_create_collection(
                name=f"level_{level}",
                metadata={"resolution_level": level}
            )
    return _collections[level]


def upsert_node(node_id: int, text: str, embedding: list[float],
                resolution_level: int, parent_id: int = None,
                confidence: float = 0.5, source_url: str = None,
                soul_id: str = None):
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
    if soul_id:
        metadata["soul_id"] = soul_id

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


def query_level(embedding: list[float], level: int, n_results: int = 10,
                soul_id: str = None) -> list[dict]:
    """Vector search within a single resolution level.

    Args:
        embedding: Query vector
        level: Resolution level to search
        n_results: Max results
        soul_id: Optional soul filter — only return nodes from this soul
    """
    where = {"soul_id": soul_id} if soul_id else None
    try:
        col = get_collection(level)
        count = col.count()
        if count == 0:
            return []
        kwargs = dict(
            query_embeddings=[embedding],
            n_results=min(n_results, count),
        )
        if where:
            kwargs["where"] = where
        results = col.query(**kwargs)
    except Exception:
        # HNSW corruption — reset and retry with fresh client
        reset_collections()
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(settings.chroma_path))
        try:
            col = client.get_collection(f"level_{level}")
            count = col.count()
            if count == 0:
                return []
            results = col.query(
                query_embeddings=[embedding],
                n_results=min(n_results, count),
                **({"where": where} if where else {})
            )
        except Exception:
            return []

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
    """Vector search across all resolution levels.

    Uses a single fresh client connection to avoid HNSW corruption
    from concurrent access across collections.
    """
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    results = {}
    for level in range(6):
        try:
            col = client.get_collection(f"level_{level}")
            count = col.count()
            if count == 0:
                continue
            qr = min(n_results, count)
            res = col.query(query_embeddings=[embedding], n_results=qr)
            hits = []
            for i in range(len(res["ids"][0])):
                hits.append({
                    "node_id": res["ids"][0][i],
                    "content": res["documents"][0][i],
                    "distance": res["distances"][0][i],
                    "metadata": res["metadatas"][0][i],
                })
            results[level] = hits
        except Exception:
            continue
    return results


def delete_by_soul(soul_id: str) -> int:
    """Delete all embeddings for a given soul_id across all resolution levels."""
    total = 0
    for level in range(6):
        try:
            col = get_collection(level)
            # Get IDs to delete (where clause)
            existing = col.get(where={"soul_id": soul_id})
            if existing and existing["ids"]:
                col.delete(ids=existing["ids"])
                total += len(existing["ids"])
        except Exception:
            continue
    return total


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
