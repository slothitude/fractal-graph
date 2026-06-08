"""GAT MCP Server — exposes Godot engine knowledge graph as MCP tools."""

from __future__ import annotations

import json
from pathlib import Path

from fastmcp import FastMCP

from gat.graph import GodotGraph

# Paths relative to the project root
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "gat.db"
SCHEMA_PATH = PROJECT_ROOT / "data" / "engine_schema.json"

mcp = FastMCP("GAT", instructions=(
    "Godot Agent Toolkit — headless Godot IDE for AI agents. "
    "Query the Godot 4.6 engine knowledge graph: classes, methods, "
    "properties, signals, inheritance, constants, enums. "
    "All engine data comes from ClassDB extraction."
))

# Lazy-loaded graph instance
_graph: GodotGraph | None = None


def _get_graph() -> GodotGraph:
    global _graph
    if _graph is None:
        _graph = GodotGraph(DB_PATH)
        # Auto-load engine schema if DB is empty
        stats = _graph.stats()
        if stats["nodes"] == 0 and SCHEMA_PATH.exists():
            _graph.load_engine_schema(SCHEMA_PATH)
    return _graph


def _serialize_node(node: dict) -> dict:
    """Strip internal ids from a node for clean MCP output."""
    if not node:
        return None
    out = {
        "name": node["name"],
        "type": node["type"],
    }
    if node.get("data"):
        out["data"] = node["data"]
    return out


def _serialize_neighbors(neighbors: list[dict]) -> list[dict]:
    return [_serialize_node(n) for n in neighbors]


# --- Engine Query Tools ---


@mcp.tool()
def get_class(class_name: str) -> dict | None:
    """Get a Godot class with all its methods, properties, signals, constants, and enums.

    Returns the full class definition including all members.
    Walks up inheritance to include inherited members.

    Args:
        class_name: Godot class name (e.g. "CharacterBody2D", "Node", "Control")
    """
    g = _get_graph()
    cls = g.get_class(class_name)
    if not cls:
        return None
    return {
        "name": cls["name"],
        "data": cls.get("data", {}),
        "inheritance": g.get_inheritance_chain(class_name),
        "properties": _serialize_neighbors(cls.get("properties", [])),
        "methods": _serialize_neighbors(cls.get("methods", [])),
        "signals": _serialize_neighbors(cls.get("signals", [])),
        "constants": _serialize_neighbors(cls.get("constants", [])),
        "enums": _serialize_neighbors(cls.get("enums", [])),
    }


@mcp.tool()
def find_inheritance(class_name: str) -> list[str]:
    """Get the full inheritance chain from a class up to Object.

    Args:
        class_name: Godot class name
    """
    return _get_graph().get_inheritance_chain(class_name)


@mcp.tool()
def find_children(class_name: str) -> list[dict]:
    """Find all classes that inherit from the given class (direct and transitive).

    Args:
        class_name: Parent class name (e.g. "Node2D" returns Sprite2D, CharacterBody2D, etc.)
    """
    children = _get_graph().find_children(class_name)
    return [{"name": c["name"], "data": c.get("data", {})} for c in children]


@mcp.tool()
def search_engine(query: str, node_type: str | None = None) -> list[dict]:
    """Search the Godot engine graph by name.

    Find classes, methods, properties, signals, constants, or enums matching a query.

    Args:
        query: Search term (e.g. "velocity", "move_and_slide", "pressed")
        node_type: Optional filter — "class", "method", "property", "signal", "constant", "enum"
    """
    nodes = _get_graph().search(query, type=node_type, limit=30)
    return [{"name": n["name"], "type": n["type"],
             "data": n.get("data", {})} for n in nodes]


@mcp.tool()
def get_method(class_name: str, method_name: str) -> dict | None:
    """Get a method's signature, walking up the inheritance chain to find it.

    Args:
        class_name: The class to search on (e.g. "CharacterBody2D")
        method_name: Method name (e.g. "move_and_slide")
    """
    m = _get_graph().find_method(class_name, method_name)
    return _serialize_node(m) if m else None


@mcp.tool()
def get_signal(class_name: str, signal_name: str) -> dict | None:
    """Get a signal's signature, walking up the inheritance chain to find it.

    Args:
        class_name: The class to search on
        signal_name: Signal name (e.g. "body_entered")
    """
    s = _get_graph().find_signal(class_name, signal_name)
    return _serialize_node(s) if s else None


@mcp.tool()
def who_has_signal(signal_name: str) -> list[dict]:
    """Find all classes that have a signal (own or inherit it).

    Args:
        signal_name: Signal name (e.g. "pressed", "body_entered")
    """
    classes = _get_graph().find_who_has_signal(signal_name)
    return [{"name": c["name"], "data": c.get("data", {})} for c in classes]


@mcp.tool()
def who_has_method(method_name: str) -> list[dict]:
    """Find all classes that have a method (own or inherit it).

    Args:
        method_name: Method name (e.g. "queue_free", "get_node")
    """
    classes = _get_graph().find_who_has_method(method_name)
    return [{"name": c["name"], "data": c.get("data", {})} for c in classes]


@mcp.tool()
def graph_stats() -> dict:
    """Get Godot engine graph statistics — node/edge counts by type."""
    return _get_graph().stats()


# --- Documentation Search ---


@mcp.tool()
def search_documentation(query: str) -> str:
    """Search Godot documentation from the tomb vault (FTS5 index).

    Searches across all 1,078 class docs and 557 tutorials for the query.
    Returns ranked snippets with context. Use this for descriptions,
    usage examples, and detailed explanations not in ClassDB.

    Args:
        query: Search query (e.g. "CharacterBody2D collision", "TileMap usage")
    """
    try:
        import subprocess
        result = subprocess.run(
            ["C:/Python313/python.exe", "-c",
             f"from gat.tomb_search import tomb_search; print(tomb_search({query!r}))"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return f"Documentation search unavailable. Use search_engine('{query}') instead."
    except Exception:
        return f"Documentation search unavailable. Use search_engine('{query}') instead."


if __name__ == "__main__":
    mcp.run()
