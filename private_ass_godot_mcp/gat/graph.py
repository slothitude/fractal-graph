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
        self.conn.execute(
            "INSERT INTO nodes (type, name, data) VALUES (?, ?, ?) "
            "ON CONFLICT(type, name) DO UPDATE SET data=excluded.data",
            (type, name, data_json),
        )
        self.conn.commit()
        # Fetch the actual ID (lastrowid is unreliable after ON CONFLICT UPDATE)
        row = self.conn.execute(
            "SELECT id FROM nodes WHERE type=? AND name=?", (type, name)
        ).fetchone()
        node_id = row[0]
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
        """Load engine_schema.json into the graph (batched for speed)."""
        schema_path = Path(schema_path)
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        print(f"[GAT] Loading {schema['total_classes']} classes from {schema_path}")

        # Batch insert: defer all FK checks and commit once
        self.conn.execute("PRAGMA defer_foreign_keys = ON")

        for cls in schema["classes"]:
            self._load_class(cls)
            self._load_class_members(cls)

        # INHERITS edges
        for cls in schema["classes"]:
            if cls.get("inherits"):
                child_id = self._class_cache.get(cls["class"])
                parent_id = self._class_cache.get(cls["inherits"])
                if child_id and parent_id:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO edges (from_id, to_id, relation) "
                        "VALUES (?, ?, ?)",
                        (child_id, parent_id, "INHERITS"),
                    )

        self.conn.commit()
        print(f"[GAT] Engine schema loaded. Graph: {self.stats()}")

    def _load_class(self, cls: dict) -> int:
        class_data = json.dumps({
            "is_abstract": cls.get("is_abstract", False),
            "is_singleton": cls.get("is_singleton", False),
        }, separators=(",", ":"))
        self.conn.execute(
            "INSERT INTO nodes (type, name, data) VALUES (?, ?, ?) "
            "ON CONFLICT(type, name) DO UPDATE SET data=excluded.data",
            ("class", cls["class"], class_data),
        )
        # Use cached id or fetch
        if cls["class"] in self._class_cache:
            return self._class_cache[cls["class"]]
        row = self.conn.execute(
            "SELECT id FROM nodes WHERE type='class' AND name=?",
            (cls["class"],),
        ).fetchone()
        node_id = row[0]
        self._class_cache[cls["class"]] = node_id
        self._node_cache.setdefault("class", {})[cls["class"]] = node_id
        return node_id

    def _load_class_members(self, cls: dict) -> None:
        class_id = self._class_cache[cls["class"]]

        for prop in cls.get("properties", []):
            prop_id = self._upsert_node("property", prop["name"], {
                "type": prop.get("type", ""),
                "hint": prop.get("hint", ""),
                "hint_string": prop.get("hint_string", ""),
                "usage": prop.get("usage", []),
            })
            self.conn.execute(
                "INSERT OR IGNORE INTO edges (from_id, to_id, relation) "
                "VALUES (?, ?, ?)",
                (class_id, prop_id, "HAS_PROPERTY"),
            )

        for method in cls.get("methods", []):
            method_id = self._upsert_node("method", method["name"], {
                "return_type": method.get("return_type", ""),
                "is_virtual": method.get("is_virtual", False),
                "is_vararg": method.get("is_vararg", False),
                "args": method.get("args", []),
            })
            self.conn.execute(
                "INSERT OR IGNORE INTO edges (from_id, to_id, relation) "
                "VALUES (?, ?, ?)",
                (class_id, method_id, "HAS_METHOD"),
            )

        for sig in cls.get("signals", []):
            sig_id = self._upsert_node("signal", sig["name"], {
                "args": sig.get("args", []),
            })
            self.conn.execute(
                "INSERT OR IGNORE INTO edges (from_id, to_id, relation) "
                "VALUES (?, ?, ?)",
                (class_id, sig_id, "HAS_SIGNAL"),
            )

        for const in cls.get("constants", []):
            const_id = self._upsert_node("constant", const["name"], {
                "value": const.get("value"),
            })
            self.conn.execute(
                "INSERT OR IGNORE INTO edges (from_id, to_id, relation) "
                "VALUES (?, ?, ?)",
                (class_id, const_id, "HAS_CONSTANT"),
            )

        for enum in cls.get("enums", []):
            enum_id = self._upsert_node("enum", enum["name"], {
                "values": enum.get("values", []),
            })
            self.conn.execute(
                "INSERT OR IGNORE INTO edges (from_id, to_id, relation) "
                "VALUES (?, ?, ?)",
                (class_id, enum_id, "HAS_ENUM"),
            )
            for val in enum.get("values", []):
                val_id = self._upsert_node("constant", val["name"], {
                    "value": val.get("value"),
                })
                self.conn.execute(
                    "INSERT OR IGNORE INTO edges (from_id, to_id, relation) "
                    "VALUES (?, ?, ?)",
                    (enum_id, val_id, "HAS_CONSTANT"),
                )

    def _upsert_node(self, type: str, name: str, data: dict | None = None) -> int:
        """Upsert a node, return its ID. No commit (caller handles txn)."""
        data_json = json.dumps(data, separators=(",", ":")) if data else None
        self.conn.execute(
            "INSERT INTO nodes (type, name, data) VALUES (?, ?, ?) "
            "ON CONFLICT(type, name) DO UPDATE SET data=excluded.data",
            (type, name, data_json),
        )
        # Check cache first
        cache = self._node_cache.get(type, {})
        if name in cache:
            return cache[name]
        row = self.conn.execute(
            "SELECT id FROM nodes WHERE type=? AND name=?", (type, name)
        ).fetchone()
        node_id = row[0]
        cache[name] = node_id
        self._node_cache[type] = cache
        if type == "class":
            self._class_cache[name] = node_id
        return node_id

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

    # --- Project Data Loading ---

    def load_project(self, parsed: "ParsedProjectFiles") -> None:
        """Insert parsed project data into the graph.

        Node types: proj_scene, proj_node, proj_script, proj_resource
        Edge types: CONTAINS_CHILD, HAS_SCRIPT, INSTANCE_OF, CONNECTS_SIGNAL,
                    HAS_SIGNAL_DECL, HAS_EXPORT, HAS_METHOD_DECL, REFERENCES
        """
        from gat.project_parser import ParsedProjectFiles

        self.conn.execute("PRAGMA defer_foreign_keys = ON")

        # --- Project node ---
        if parsed.project:
            self._upsert_node("proj_project", parsed.project.name, {
                "version": parsed.project.version,
                "main_scene": parsed.project.main_scene,
                "autoloads": parsed.project.autoloads,
                "input_actions": parsed.project.input_actions,
                "window_size": parsed.project.window_size,
            })

        # --- Scenes, nodes, connections ---
        for scene in parsed.scenes:
            # Scene node (use filename as unique name)
            scene_name = Path(scene.path).stem
            scene_id = self._upsert_node("proj_scene", scene_name, {
                "path": scene.path,
                "format": scene.format,
                "uid": scene.uid,
                "root_node": scene.root_node,
            })

            # Nodes within the scene
            for node in scene.nodes:
                # Qualified name: scene_name/node_path
                qname = f"{scene_name}/{node.name}"
                node_id = self._upsert_node("proj_node", qname, {
                    "scene": scene.path,
                    "name": node.name,
                    "type": node.type,
                    "parent": node.parent,
                    "properties": node.properties,
                })
                # CONTAINS_CHILD edge from scene or parent node
                if node.parent == "" or node.parent == ".":
                    self.conn.execute(
                        "INSERT OR IGNORE INTO edges (from_id, to_id, relation) "
                        "VALUES (?, ?, ?)",
                        (scene_id, node_id, "CONTAINS_CHILD"),
                    )
                else:
                    parent_qname = f"{scene_name}/{node.parent}"
                    parent_node = self.get_node_by_name("proj_node", parent_qname)
                    if parent_node:
                        self.conn.execute(
                            "INSERT OR IGNORE INTO edges (from_id, to_id, relation) "
                            "VALUES (?, ?, ?)",
                            (parent_node["id"], node_id, "CONTAINS_CHILD"),
                        )
                # INSTANCE_OF edge if the node type is an engine class
                engine_class = self.get_node_by_name("class", node.type)
                if engine_class:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO edges (from_id, to_id, relation) "
                        "VALUES (?, ?, ?)",
                        (node_id, engine_class["id"], "INSTANCE_OF"),
                    )

            # Signal connections
            for conn in scene.connections:
                from_qname = f"{scene_name}/{conn.from_node}"
                to_qname = f"{scene_name}/{conn.to_node}"
                from_node = self.get_node_by_name("proj_node", from_qname)
                to_node = self.get_node_by_name("proj_node", to_qname)
                if from_node and to_node:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO edges (from_id, to_id, relation, data) "
                        "VALUES (?, ?, ?, ?)",
                        (from_node["id"], to_node["id"], "CONNECTS_SIGNAL",
                         json.dumps({"signal": conn.signal, "method": conn.method})),
                    )

        # --- Scripts ---
        for script in parsed.scripts:
            script_name = Path(script.path).stem
            script_id = self._upsert_node("proj_script", script_name, {
                "path": script.path,
                "class_name": script.class_name,
                "extends": script.extends,
                "extends_type": script.extends_type,
                "signals": script.signals,
                "methods": script.methods,
                "exports": script.exports,
                "onready_vars": script.onready_vars,
                "constants": script.constants,
                "engine_refs": script.engine_refs,
            })

        # --- Resources ---
        for res in parsed.resources:
            res_name = Path(res.path).stem
            self._upsert_node("proj_resource", res_name, {
                "path": res.path,
                "type": res.type,
                "properties": res.properties,
            })

        self.conn.commit()

    def get_project(self) -> dict | None:
        """Get the loaded project node."""
        node = self.find_nodes("proj_project", limit=1)
        return node[0] if node else None

    def get_project_scenes(self) -> list[dict]:
        """Get all scene nodes for the loaded project."""
        return self.find_nodes("proj_scene", limit=100)

    def get_project_scripts(self) -> list[dict]:
        """Get all script nodes for the loaded project."""
        return self.find_nodes("proj_script", limit=500)

    def get_project_nodes(self, scene_name: str) -> list[dict]:
        """Get all nodes in a scene (by scene name, not path)."""
        qname_prefix = f"{scene_name}/"
        rows = self.conn.execute(
            "SELECT id, type, name, data FROM nodes WHERE type='proj_node' AND name LIKE ?",
            (f"{qname_prefix}%",),
        ).fetchall()
        return [{"id": r[0], "type": r[1], "name": r[2],
                 "data": json.loads(r[3]) if r[3] else {}} for r in rows]

    def find_nodes_by_type(self, node_type: str) -> list[dict]:
        """Find all project nodes of a given engine type."""
        rows = self.conn.execute(
            "SELECT n.id, n.type, n.name, n.data FROM nodes n "
            "WHERE n.type='proj_node' AND json_extract(n.data, '$.type') = ?",
            (node_type,),
        ).fetchall()
        return [{"id": r[0], "type": r[1], "name": r[2],
                 "data": json.loads(r[3]) if r[3] else {}} for r in rows]

    def get_signal_connections(self, scene_name: str) -> list[dict]:
        """Get all signal connections for a scene."""
        scene = self.get_node_by_name("proj_scene", scene_name)
        if not scene:
            return []
        # Find all proj_nodes in this scene, then their outgoing CONNECTS_SIGNAL edges
        nodes = self.get_project_nodes(scene_name)
        connections = []
        for node in nodes:
            edges = self.get_neighbors(node["id"], "CONNECTS_SIGNAL")
            for edge in edges:
                connections.append({
                    "from_node": node["name"],
                    "from_type": node["data"].get("type", ""),
                    "to_node": edge["name"],
                    "signal": edge["data"].get("signal", ""),
                    "method": edge["data"].get("method", ""),
                })
        return connections

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
