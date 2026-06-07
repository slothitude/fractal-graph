"""SQLite storage layer for Fractal Graph."""

import asyncio
import json
import sqlite3
import time
from datetime import datetime, timezone
from config import settings


_MAX_RETRIES = 5
_BUSY_DELAY = 0.2  # seconds

# Module-level asyncio Lock for write serialization across concurrent async operations
_write_lock = asyncio.Lock()


def _retry_commit(conn):
    """Commit with SQLITE_BUSY retry."""
    for attempt in range(_MAX_RETRIES):
        try:
            conn.commit()
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < _MAX_RETRIES - 1:
                time.sleep(_BUSY_DELAY * (attempt + 1))
            else:
                raise


def get_db() -> sqlite3.Connection:
    """Get a SQLite connection, creating the DB and schema if needed."""
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS nodes (
            id INTEGER PRIMARY KEY,
            content TEXT NOT NULL,
            resolution_level INTEGER NOT NULL DEFAULT 2,
            parent_id INTEGER REFERENCES nodes(id),
            confidence REAL DEFAULT 0.5,
            bbox TEXT,
            source_url TEXT,
            metadata JSON,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS edges (
            id INTEGER PRIMARY KEY,
            from_node_id INTEGER NOT NULL REFERENCES nodes(id),
            to_node_id INTEGER NOT NULL REFERENCES nodes(id),
            edge_type TEXT NOT NULL DEFAULT 'related',
            from_resolution INTEGER,
            to_resolution INTEGER,
            confidence REAL DEFAULT 0.5,
            context TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_nodes_resolution ON nodes(resolution_level);
        CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);
        CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_node_id);
        CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_node_id);
        CREATE INDEX IF NOT EXISTS idx_nodes_confidence ON nodes(confidence);
        CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
        CREATE INDEX IF NOT EXISTS idx_nodes_source ON nodes(source_url);
    """)

    # Migration: add soul_id column to existing nodes table
    try:
        conn.execute("ALTER TABLE nodes ADD COLUMN soul_id TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_soul ON nodes(soul_id)")
    except sqlite3.OperationalError:
        pass  # Column already exists


# --- Node CRUD ---

def insert_node(conn, content, resolution_level=2, parent_id=None,
                confidence=0.5, bbox=None, source_url=None, metadata=None,
                soul_id=None) -> int:
    # Validate parent exists before insert — prevent FK constraint failures
    if parent_id is not None:
        parent = conn.execute("SELECT id FROM nodes WHERE id = ?", (parent_id,)).fetchone()
        if not parent:
            parent_id = None

    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """INSERT INTO nodes (content, resolution_level, parent_id, confidence,
           bbox, source_url, metadata, soul_id, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (content, resolution_level, parent_id, confidence,
         json.dumps(bbox) if bbox else None, source_url,
         json.dumps(metadata) if metadata else None, soul_id, now, now)
    )
    _retry_commit(conn)
    return cursor.lastrowid


def get_node(conn, node_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
    if row:
        d = dict(row)
        if d.get("bbox"):
            d["bbox"] = json.loads(d["bbox"])
        if d.get("metadata"):
            d["metadata"] = json.loads(d["metadata"])
        return d
    return None


def get_children(conn, node_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM nodes WHERE parent_id = ?", (node_id,)
    ).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        if d.get("bbox"):
            d["bbox"] = json.loads(d["bbox"])
        if d.get("metadata"):
            d["metadata"] = json.loads(d["metadata"])
        results.append(d)
    return results


def get_nodes_by_resolution(conn, level: int) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM nodes WHERE resolution_level = ?", (level,)
    ).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        if d.get("bbox"):
            d["bbox"] = json.loads(d["bbox"])
        if d.get("metadata"):
            d["metadata"] = json.loads(d["metadata"])
        results.append(d)
    return results


def get_nodes_by_soul(conn, soul_id: str) -> list[dict]:
    """Get all nodes belonging to a soul."""
    rows = conn.execute(
        "SELECT * FROM nodes WHERE soul_id = ?", (soul_id,)
    ).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        if d.get("bbox"):
            d["bbox"] = json.loads(d["bbox"])
        if d.get("metadata"):
            d["metadata"] = json.loads(d["metadata"])
        results.append(d)
    return results


def count_nodes_by_soul(conn, soul_id: str) -> int:
    """Count nodes belonging to a soul."""
    row = conn.execute(
        "SELECT COUNT(*) as c FROM nodes WHERE soul_id = ?", (soul_id,)
    ).fetchone()
    return row["c"] if row else 0


def update_node_confidence(conn, node_id: int, confidence: float):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE nodes SET confidence = ?, updated_at = ? WHERE id = ?",
        (confidence, now, node_id)
    )
    _retry_commit(conn)


def update_node_bbox(conn, node_id: int, bbox: list):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE nodes SET bbox = ?, updated_at = ? WHERE id = ?",
        (json.dumps(bbox), now, node_id)
    )
    _retry_commit(conn)


def delete_node(conn, node_id: int):
    conn.execute(
        "DELETE FROM edges WHERE from_node_id = ? OR to_node_id = ?",
        (node_id, node_id)
    )
    conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
    _retry_commit(conn)


def delete_nodes_by_soul(conn, soul_id: str) -> int:
    """Delete all nodes and their edges for a given soul_id."""
    conn.execute("""
        DELETE FROM edges WHERE from_node_id IN
            (SELECT id FROM nodes WHERE soul_id = ?)
        OR to_node_id IN
            (SELECT id FROM nodes WHERE soul_id = ?)
    """, (soul_id, soul_id))
    count = conn.execute("DELETE FROM nodes WHERE soul_id = ?", (soul_id,)).rowcount
    _retry_commit(conn)
    return count


# --- Edge CRUD ---

EDGE_TYPES = {
    "related", "refines", "contradicts", "exemplifies",
    "generalizes", "challenges", "supports", "derived_from",
    "resolution_conflict",
}


def insert_edge(conn, from_node_id, to_node_id, edge_type="related",
                confidence=0.5, context=None) -> int:
    if edge_type.lower() not in {t.lower() for t in EDGE_TYPES}:
        raise ValueError(f"Unknown edge type: {edge_type}. Valid: {EDGE_TYPES}")

    from_node = get_node(conn, from_node_id)
    to_node = get_node(conn, to_node_id)
    if not from_node or not to_node:
        raise ValueError("Both nodes must exist to create an edge")

    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """INSERT INTO edges (from_node_id, to_node_id, edge_type,
           from_resolution, to_resolution, confidence, context, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (from_node_id, to_node_id, edge_type.lower(),
         from_node["resolution_level"], to_node["resolution_level"],
         confidence, context, now)
    )
    _retry_commit(conn)
    return cursor.lastrowid


def get_edges(conn, node_id: int, direction: str = "both") -> list[dict]:
    if direction == "out":
        rows = conn.execute(
            "SELECT * FROM edges WHERE from_node_id = ?", (node_id,)
        ).fetchall()
    elif direction == "in":
        rows = conn.execute(
            "SELECT * FROM edges WHERE to_node_id = ?", (node_id,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM edges WHERE from_node_id = ? OR to_node_id = ?",
            (node_id, node_id)
        ).fetchall()
    return [dict(row) for row in rows]


# --- Export / Import ---

def export_graph(conn) -> dict:
    """Export the entire graph to a portable JSON structure."""
    rows = conn.execute("SELECT * FROM nodes ORDER BY id").fetchall()
    nodes = []
    for row in rows:
        d = dict(row)
        if d.get("bbox"):
            d["bbox"] = json.loads(d["bbox"])
        if d.get("metadata"):
            d["metadata"] = json.loads(d["metadata"])
        nodes.append(d)

    edge_rows = conn.execute("SELECT * FROM edges ORDER BY id").fetchall()
    edges = [dict(row) for row in edge_rows]

    return {
        "version": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "stats": {"total_nodes": len(nodes), "total_edges": len(edges)},
        "nodes": nodes,
        "edges": edges,
    }


def import_graph(conn, data: dict, merge: bool = False) -> dict:
    """Import graph data from a JSON export.

    Args:
        conn: SQLite connection
        data: Parsed JSON export dict
        merge: If False (default), clear existing data first. If True, skip nodes with existing IDs.
    Returns:
        {"imported_nodes": N, "imported_edges": N}
    """
    version = data.get("version", 0)
    if version != 1:
        raise ValueError(f"Unsupported export version: {version}. Only version 1 is supported.")

    nodes_data = data.get("nodes", [])
    edges_data = data.get("edges", [])

    if not merge:
        conn.execute("DELETE FROM edges")
        conn.execute("DELETE FROM nodes")
        _retry_commit(conn)

    imported_nodes = 0
    for n in nodes_data:
        if merge:
            existing = conn.execute("SELECT id FROM nodes WHERE id = ?", (n["id"],)).fetchone()
            if existing:
                continue

        conn.execute(
            """INSERT OR IGNORE INTO nodes
               (id, content, resolution_level, parent_id, confidence,
                bbox, source_url, metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (n["id"], n["content"], n.get("resolution_level", 2),
             n.get("parent_id"), n.get("confidence", 0.5),
             json.dumps(n["bbox"]) if n.get("bbox") else None,
             n.get("source_url"),
             json.dumps(n["metadata"]) if n.get("metadata") else None,
             n.get("created_at"), n.get("updated_at")))
        imported_nodes += 1

    imported_edges = 0
    for e in edges_data:
        if merge:
            existing = conn.execute("SELECT id FROM edges WHERE id = ?", (e["id"],)).fetchone()
            if existing:
                continue

        conn.execute(
            """INSERT OR IGNORE INTO edges
               (id, from_node_id, to_node_id, edge_type,
                from_resolution, to_resolution, confidence, context, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (e["id"], e["from_node_id"], e["to_node_id"], e.get("edge_type", "related"),
             e.get("from_resolution"), e.get("to_resolution"),
             e.get("confidence", 0.5), e.get("context"), e.get("created_at")))
        imported_edges += 1

    _retry_commit(conn)
    return {"imported_nodes": imported_nodes, "imported_edges": imported_edges}
