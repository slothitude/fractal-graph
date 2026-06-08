"""GAT Knowledge Graph — SQLite graph for Godot engine + project data."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class GodotGraph:
    """SQLite-backed graph database for Godot engine schema and project data.

    Node types: class, method, property, signal, constant, enum, project_class,
                 project_node, project_script, project_resource
    Edge relations: INHERITS, HAS_METHOD, HAS_PROPERTY, HAS_SIGNAL,
                     HAS_CONSTANT, HAS_ENUM, RETURNS, ACCEPTS,
                     INSTANCE_OF, CONTAINS_CHILD, CONNECTS_SIGNAL,
                     HAS_SCRIPT, USES_METHOD, REFERENCES
    """

    def __init__(self, db_path: str | Path = "data/gat.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_schema()
        self._node_cache: dict[str, dict[str, int]] = {}  # type_name → {name: id}
        self._class_cache: dict[str, int] = {}  # class_name → node_id

    def close(self) -> None:
        self.conn.close()

    def _create_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY,
                type TEXT NOT NULL,
                name TEXT NOT NULL,
                data JSON,
                UNIQUE(type, name)
            );
            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
            CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);
            CREATE INDEX IF NOT EXISTS idx_nodes_type_name ON nodes(type, name);

            CREATE TABLE IF NOT EXISTS edges (
                from_id INTEGER NOT NULL REFERENCES nodes(id),
                to_id INTEGER NOT NULL REFERENCES nodes(id),
                relation TEXT NOT NULL,
                data JSON,
                UNIQUE(from_id, to_id, relation)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_id);
            CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_id);
            CREATE INDEX IF NOT EXISTS idx_edges_rel ON edges(relation);
            CREATE INDEX IF NOT EXISTS idx_edges_from_rel ON edges(from_id, relation);
            CREATE INDEX IF NOT EXISTS idx_edges_to_rel ON edges(to_id, relation);
        """)

    # --- Node CRUD ---

    def add_node(self, type: str, name: str, data: dict | None = None) -> int:
        """Insert a node, return its ID. Upserts on (type, name) conflict."""
        data_json = json.dumps(data, separators=(",", ":")) if data else None
        cur = self.conn.execute(
            "INSERT INTO nodes (type, name, data) VALUES (?, ?, ?) "
            "ON CONFLICT(type, name) DO UPDATE SET data=excluded.data",
            (type, name, data_json),
        )
        self.conn.commit()
        node_id = cur.lastrowid
        # Update caches
        self._node_cache.setdefault(type, {})[name] = node_id
        if type == "class":
            self._class_cache[name] = node_id
        return node_id

    def get_node(self, node_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT id, type, name, data FROM nodes WHERE id=?", (node_id,)
        ).fetchone()
        if not row:
            return None
        return {"id": row[0], "type": row[1], "name": row[2],
                "data": json.loads(row[3]) if row[3] else {}}

    def get_node_by_name(self, type: str, name: str) -> dict | None:
        row = self.conn.execute(
            "SELECT id, type, name, data FROM nodes WHERE type=? AND name=?",
            (type, name),
        ).fetchone()
        if not row:
            return None
        return {"id": row[0], "type": row[1], "name": row[2],
                "data": json.loads(row[3]) if row[3] else {}}

    def find_nodes(self, type: str | None = None, name_pattern: str | None = None,
                   limit: int = 50) -> list[dict]:
        query = "SELECT id, type, name, data FROM nodes WHERE 1=1"
        params: list = []
        if type:
            query += " AND type=?"
            params.append(type)
        if name_pattern:
            query += " AND name LIKE ?"
            params.append(f"%{name_pattern}%")
        query += f" LIMIT {limit}"
        rows = self.conn.execute(query, params).fetchall()
        return [{"id": r[0], "type": r[1], "name": r[2],
                 "data": json.loads(r[3]) if r[3] else {}} for r in rows]

    # --- Edge CRUD ---

    def add_edge(self, from_id: int, to_id: int, relation: str,
                 data: dict | None = None) -> None:
        data_json = json.dumps(data, separators=(",", ":")) if data else None
        self.conn.execute(
            "INSERT OR IGNORE INTO edges (from_id, to_id, relation, data) "
            "VALUES (?, ?, ?, ?)",
            (from_id, to_id, relation, data_json),
        )
        self.conn.commit()

    def get_edges(self, from_id: int | None = None, to_id: int | None = None,
                  relation: str | None = None) -> list[dict]:
        query = "SELECT from_id, to_id, relation, data FROM edges WHERE 1=1"
        params: list = []
        if from_id is not None:
            query += " AND from_id=?"
            params.append(from_id)
        if to_id is not None:
            query += " AND to_id=?"
            params.append(to_id)
        if relation:
            query += " AND relation=?"
            params.append(relation)
        rows = self.conn.execute(query, params).fetchall()
        return [{"from_id": r[0], "to_id": r[1], "relation": r[2],
                 "data": json.loads(r[3]) if r[3] else {}} for r in rows]

    def get_neighbors(self, node_id: int, relation: str | None = None,
                      outgoing: bool = True) -> list[dict]:
        """Get nodes connected by edges from (or to) this node."""
        col = "from_id" if outgoing else "to_id"
        other_col = "to_id" if outgoing else "from_id"
        query = f"""
            SELECT n.id, n.type, n.name, n.data, e.relation
            FROM edges e JOIN nodes n ON n.id = e.{other_col}
            WHERE e.{col} = ?
        """
        params: list = [node_id]
        if relation:
            query += " AND e.relation=?"
            params.append(relation)
        rows = self.conn.execute(query, params).fetchall()
        return [{"id": r[0], "type": r[1], "name": r[2],
                 "data": json.loads(r[3]) if r[3] else {},
                 "relation": r[4]} for r in rows]

    # --- Engine Schema Loading ---

    def load_engine_schema(self, schema_path: str | Path) -> None:
        """Load engine_schema.json into the graph."""
        schema_path = Path(schema_path)
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        print(f"[GAT] Loading {schema['total_classes']} classes from {schema_path}")

        for cls in schema["classes"]:
            class_id = self._load_class(cls)
            self._load_class_members(class_id, cls)

        # Fix parent references (INHERITS edges) — classes are loaded in order
        for cls in schema["classes"]:
            if cls.get("inherits"):
                parent_node = self.get_node_by_name("class", cls["inherits"])
                child_node = self.get_node_by_name("class", cls["class"])
                if parent_node and child_node:
                    self.add_edge(child_node["id"], parent_node["id"], "INHERITS")

        print(f"[GAT] Engine schema loaded. Graph: {self.stats()}")

    def _load_class(self, cls: dict) -> int:
        class_data = {
            "is_abstract": cls.get("is_abstract", False),
            "is_singleton": cls.get("is_singleton", False),
        }
        return self.add_node("class", cls["class"], class_data)

    def _load_class_members(self, class_id: int, cls: dict) -> int:
        for prop in cls.get("properties", []):
            prop_id = self.add_node("property", prop["name"], {
                "type": prop.get("type", ""),
                "hint": prop.get("hint", ""),
                "hint_string": prop.get("hint_string", ""),
                "usage": prop.get("usage", []),
            })
            self.add_edge(class_id, prop_id, "HAS_PROPERTY")

        for method in cls.get("methods", []):
            method_id = self.add_node("method", method["name"], {
                "return_type": method.get("return_type", ""),
                "is_virtual": method.get("is_virtual", False),
                "is_vararg": method.get("is_vararg", False),
                "args": method.get("args", []),
            })
            self.add_edge(class_id, method_id, "HAS_METHOD")
            if method.get("return_type"):
                # Create a type reference if it's an engine type
                type_name = method["return_type"]
                if type_name and type_name not in ("null", "void", "int", "float", "bool", "String"):
                    type_node = self.get_node_by_name("class", type_name)
                    if type_node:
                        self.add_edge(method_id, type_node["id"], "RETURNS")

        for sig in cls.get("signals", []):
            sig_id = self.add_node("signal", sig["name"], {
                "args": sig.get("args", []),
            })
            self.add_edge(class_id, sig_id, "HAS_SIGNAL")

        for const in cls.get("constants", []):
            const_id = self.add_node("constant", const["name"], {
                "value": const.get("value"),
            })
            self.add_edge(class_id, const_id, "HAS_CONSTANT")

        for enum in cls.get("enums", []):
            enum_id = self.add_node("enum", enum["name"], {
                "values": enum.get("values", []),
            })
            self.add_edge(class_id, enum_id, "HAS_ENUM")
            for val in enum.get("values", []):
                val_id = self.add_node("constant", val["name"], {
                    "value": val.get("value"),
                })
                self.add_edge(enum_id, val_id, "HAS_CONSTANT")

        return class_id

    # --- Query API ---

    def get_class(self, class_name: str) -> dict | None:
        """Get a class node with all its members."""
        node = self.get_node_by_name("class", class_name)
        if not node:
            return None
        node["properties"] = self.get_neighbors(node["id"], "HAS_PROPERTY")
        node["methods"] = self.get_neighbors(node["id"], "HAS_METHOD")
        node["signals"] = self.get_neighbors(node["id"], "HAS_SIGNAL")
        node["constants"] = self.get_neighbors(node["id"], "HAS_CONSTANT")
        node["enums"] = self.get_neighbors(node["id"], "HAS_ENUM")
        return node

    def get_inheritance_chain(self, class_name: str) -> list[str]:
        """Return full ancestry: [class_name, parent, grandparent, ..., Object/null]."""
        chain: list[str] = [class_name]
        current = self.get_node_by_name("class", class_name)
        while current:
            parents = self.get_neighbors(current["id"], "INHERITS")
            if not parents:
                break
            parent_name = parents[0]["name"]
            chain.append(parent_name)
            if parent_name == "Object" or parent_name == "":
                break
            current = self.get_node_by_name("class", parent_name)
        return chain

    def find_children(self, class_name: str) -> list[dict]:
        """Find all classes that inherit from the given class (direct + transitive)."""
        # BFS from the given class via reverse INHERITS edges
        start = self.get_node_by_name("class", class_name)
        if not start:
            return []
        children: list[dict] = []
        queue = [start["id"]]
        visited = {start["id"]}
        while queue:
            current_id = queue.pop(0)
            # Find nodes that have an INHERITS edge pointing to current
            rows = self.conn.execute(
                "SELECT n.id, n.type, n.name, n.data "
                "FROM edges e JOIN nodes n ON n.id = e.from_id "
                "WHERE e.to_id=? AND e.relation='INHERITS'",
                (current_id,),
            ).fetchall()
            for r in rows:
                if r[0] not in visited:
                    visited.add(r[0])
                    children.append({
                        "id": r[0], "type": r[1], "name": r[2],
                        "data": json.loads(r[3]) if r[3] else {},
                    })
                    queue.append(r[0])
        return children

    def find_method(self, class_name: str, method_name: str) -> dict | None:
        """Find a method on a class, walking up the inheritance chain."""
        current_name = class_name
        while current_name and current_name != "":
            node = self.get_node_by_name("class", current_name)
            if not node:
                break
            methods = self.get_neighbors(node["id"], "HAS_METHOD")
            for m in methods:
                if m["name"] == method_name:
                    return m
            # Walk up
            parents = self.get_neighbors(node["id"], "INHERITS")
            if not parents:
                break
            current_name = parents[0]["name"]
        return None

    def find_signal(self, class_name: str, signal_name: str) -> dict | None:
        """Find a signal on a class, walking up the inheritance chain."""
        current_name = class_name
        while current_name and current_name != "":
            node = self.get_node_by_name("class", current_name)
            if not node:
                break
            signals = self.get_neighbors(node["id"], "HAS_SIGNAL")
            for s in signals:
                if s["name"] == signal_name:
                    return s
            parents = self.get_neighbors(node["id"], "INHERITS")
            if not parents:
                break
            current_name = parents[0]["name"]
        return None

    def find_who_has_signal(self, signal_name: str) -> list[dict]:
        """Find all classes that have a signal with the given name (direct or inherited)."""
        # Find all signal nodes with this name
        signal_nodes = self.find_nodes("signal", signal_name)
        if not signal_nodes:
            return []
        # For each signal, find which class owns it (or inherits it)
        classes: list[dict] = []
        seen: set[int] = set()
        for sig in signal_nodes:
            # Walk up from signal to class via HAS_SIGNAL edge
            owners = self.get_neighbors(sig["id"], "HAS_SIGNAL", outgoing=False)
            for owner in owners:
                if owner["id"] not in seen and owner["type"] == "class":
                    seen.add(owner["id"])
                    classes.append(owner)
                    # Also add all children (who inherit this signal)
                    for child in self.find_children(owner["name"]):
                        if child["id"] not in seen:
                            seen.add(child["id"])
                            classes.append(child)
        return classes

    def find_who_has_method(self, method_name: str) -> list[dict]:
        """Find all classes that have a method with the given name (direct or inherited)."""
        method_nodes = self.find_nodes("method", method_name)
        if not method_nodes:
            return []
        classes: list[dict] = []
        seen: set[int] = set()
        for m in method_nodes:
            owners = self.get_neighbors(m["id"], "HAS_METHOD", outgoing=False)
            for owner in owners:
                if owner["id"] not in seen and owner["type"] == "class":
                    seen.add(owner["id"])
                    classes.append(owner)
                    for child in self.find_children(owner["name"]):
                        if child["id"] not in seen:
                            seen.add(child["id"])
                            classes.append(child)
        return classes

    def search(self, query: str, type: str | None = None,
               limit: int = 20) -> list[dict]:
        """Search nodes by name substring."""
        return self.find_nodes(type=type, name_pattern=query, limit=limit)

    def stats(self) -> dict:
        """Return graph statistics."""
        total_nodes = self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        total_edges = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        type_counts = {}
        for row in self.conn.execute(
            "SELECT type, COUNT(*) FROM nodes GROUP BY type"
        ).fetchall():
            type_counts[row[0]] = row[1]
        rel_counts = {}
        for row in self.conn.execute(
            "SELECT relation, COUNT(*) FROM edges GROUP BY relation"
        ).fetchall():
            rel_counts[row[0]] = row[1]
        return {
            "nodes": total_nodes,
            "edges": total_edges,
            "by_type": type_counts,
            "by_relation": rel_counts,
        }
