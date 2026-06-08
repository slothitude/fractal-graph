"""GAT Phase 3 — Parse tomb godot docs to enrich the knowledge graph.

Three enrichment passes:
1. Property defaults — parse class doc property tables → store 'default' on property nodes
2. Virtual methods — parse overridable_functions.md → tag method nodes with is_virtual
3. Inheritance verification — parse YAML frontmatter → cross-reference ClassDB
"""

from __future__ import annotations

import re
from pathlib import Path

TOMB_GODOT_CLASSES = Path("C:/Users/aaron/Desktop/tomb/godot/classes")
TOMB_VIRTUAL_METHODS = Path("C:/Users/aaron/Desktop/tomb/godot/tutorials/scripting/overridable_functions.md")


# --- 1. Property Defaults ---


def _split_table_row(line: str) -> list[str]:
    """Split a markdown table row by |, respecting nested [[wikilinks]].

    The naive split("| ... | ... |") breaks on [[Type|Type]] because
    the inner | is interpreted as a column separator. We handle this by
    only splitting on | that is preceded by ]] or followed by [[.
    """
    # Strategy: walk the string, tracking wikilink nesting
    cells: list[str] = []
    depth = 0
    current: list[str] = []
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == '[' and i + 1 < len(line) and line[i + 1] == '[':
            depth += 2
            current.append('[[')
            i += 2
        elif ch == ']' and i + 1 < len(line) and line[i + 1] == ']':
            depth -= 2
            current.append(']]')
            i += 2
        elif ch == '|' and depth <= 0:
            cells.append(''.join(current).strip())
            current = []
            i += 1
        else:
            current.append(ch)
            i += 1
    if current:
        cells.append(''.join(current).strip())
    return cells


def parse_property_table(content: str) -> list[dict]:
    """Parse the Properties table from a godot class doc.

    Returns list of {"name": str, "type": str, "default": str|None}.

    Table format (3 columns):
    | [[Type|Type]] | [[Class#prop|prop]] | `default` |
    | --- | --- | --- |
    | [[bool|bool]] | [[Sprite2D#centered|centered]] | `true` |
    | [[String|String]] | [[Control#name|name]] | |              <-- no default
    """
    results: list[dict] = []
    # Find the Properties section
    prop_match = re.search(r"### Properties\s*\n", content)
    if not prop_match:
        return results

    table_text = content[prop_match.end():]
    past_header = False

    for line in table_text.split("\n"):
        stripped = line.strip()

        # End at next section
        if stripped.startswith("###"):
            break

        if not stripped or not stripped.startswith("|"):
            continue

        # Skip the separator row
        if re.match(r"^\|[\s\-:|]+\|$", stripped):
            past_header = True
            continue

        # Parse cells using wikilink-aware splitter
        cells = _split_table_row(stripped)
        # First cell is empty (before leading |), so real data starts at index 1
        # Format: [empty, type_col, name_col, default_col]
        if len(cells) < 3:
            continue

        raw_type = cells[1] if len(cells) > 1 else ""
        raw_name = cells[2] if len(cells) > 2 else ""
        raw_default = cells[3] if len(cells) > 3 else ""

        # Skip separator artifacts
        if not raw_type or raw_type.startswith("---"):
            continue

        # Extract clean type from wikilinks
        prop_type = _clean_type(raw_type)

        # Extract property name
        prop_name = _extract_prop_name(raw_name)

        if not prop_name or not prop_type:
            continue

        # Skip methods table rows (they have 4+ columns and return types)
        # Properties table has 3 data columns; Methods table has 4+
        if len(cells) > 4:
            continue

        # Extract default value from backticks
        default_val = None
        if raw_default:
            default_match = re.match(r"`(.*)`", raw_default)
            if default_match:
                default_val = default_match.group(1).strip()
            elif raw_default:
                default_val = raw_default

        results.append({
            "name": prop_name,
            "type": prop_type,
            "default": default_val,
        })

    return results


def _clean_type(t: str) -> str:
    r"""Clean a Godot type string from doc formatting.

    Handles [[bool|bool]], [[Vector2|Vector2]], [[Class#Enum|Enum]],
    and Array[Type] patterns.
    """
    # Strip wikilink display text: [[Target|Display]] → Display
    t = re.sub(r"\[\[.*?\|(.*?)\]\]", r"\1", t)
    # Strip remaining [[Type]] → Type
    t = re.sub(r"\[\[(.*?)\]\]", r"\1", t)
    return t.strip()


def _extract_prop_name(text: str) -> str:
    """Extract property name from a wikilink like [[Class#prop_name|display]].

    Returns the display text (which is the clean property name).
    """
    # [[Class#name|display]] → display
    m = re.match(r"\[\[.*?\|(.*?)\]\]", text)
    if m:
        return m.group(1).strip()
    # [[name]] → name
    m = re.match(r"\[\[(\w+)\]\]", text)
    if m:
        return m.group(1)
    # Plain text
    if text and not text.startswith("[["):
        return text.strip()
    return ""


# --- 2. Virtual Methods ---


def parse_virtual_methods(content: str) -> set[str]:
    """Extract virtual/lifecycle function names from overridable_functions.md.

    Parses GDScript code blocks for function definitions.
    """
    methods: set[str] = set()

    # Find all GDScript function definitions: func method_name(
    for m in re.finditer(r"func\s+(\w+)", content):
        methods.add(m.group(1))

    # Also find [[Node#_method|_method]] and :ref: references in prose
    for m in re.finditer(r"(?:Node|CanvasItem|Control|Resource)#(\w+)", content):
        methods.add(m.group(1))

    return methods


# --- 3. Inheritance Verification ---


def parse_yaml_inherits(content: str) -> str | None:
    """Extract the 'inherits' field from YAML frontmatter."""
    m = re.match(r"---\s*\n.*?inherits:\s*[\"']?(\w+)[\"']?\s*\n.*?---", content, re.DOTALL)
    if m:
        return m.group(1)
    return None


# --- Main enrichment ---


def enrich_graph(graph) -> dict:
    """Run all Phase 3 enrichment passes on the graph.

    Returns a report dict with counts and discrepancies.
    """
    report: dict = {
        "property_defaults": {"parsed": 0, "matched": 0, "unmatched": 0},
        "virtual_methods": {"found": 0, "tagged": 0},
        "inheritance": {"parsed": 0, "matched": 0, "discrepancies": []},
    }

    # Pass 2: Virtual methods (load once)
    virtual_methods = _load_virtual_methods()
    report["virtual_methods"]["found"] = len(virtual_methods)

    # Pass 1: Property defaults
    if TOMB_GODOT_CLASSES.exists():
        _enrich_property_defaults(graph, report)

    # Pass 2: Tag virtual methods
    if virtual_methods:
        _tag_virtual_methods(graph, virtual_methods, report)

    # Pass 3: Inheritance verification
    if TOMB_GODOT_CLASSES.exists():
        _verify_inheritance(graph, report)

    return report


def _load_virtual_methods() -> set[str]:
    """Load virtual method names from the overridable_functions doc."""
    if not TOMB_VIRTUAL_METHODS.exists():
        return set()
    content = TOMB_VIRTUAL_METHODS.read_text(encoding="utf-8")
    return parse_virtual_methods(content)


def _enrich_property_defaults(graph, report: dict) -> None:
    """Parse class docs and update property nodes with default values."""
    import json as _json

    for md_path in TOMB_GODOT_CLASSES.glob("*.md"):
        class_name = md_path.stem
        if class_name.startswith("@"):
            continue

        content = md_path.read_text(encoding="utf-8")
        props = parse_property_table(content)
        report["property_defaults"]["parsed"] += len(props)

        class_node = graph.get_node_by_name("class", class_name)
        if not class_node:
            report["property_defaults"]["unmatched"] += len(props)
            continue

        class_id = class_node["id"]

        for prop in props:
            prop_node = graph.get_node_by_name("property", prop["name"])
            if not prop_node:
                report["property_defaults"]["unmatched"] += 1
                continue

            # Verify HAS_PROPERTY edge from this class
            has_edge = graph.conn.execute(
                "SELECT 1 FROM edges WHERE from_id=? AND to_id=? AND relation='HAS_PROPERTY'",
                (class_id, prop_node["id"]),
            ).fetchone()

            if not has_edge:
                report["property_defaults"]["unmatched"] += 1
                continue

            existing_data = prop_node.get("data") or {}
            if prop["default"] is not None and "default" not in existing_data:
                existing_data["default"] = prop["default"]
                graph.conn.execute(
                    "UPDATE nodes SET data=? WHERE id=?",
                    (_json.dumps(existing_data, separators=(",", ":")), prop_node["id"]),
                )
                graph.conn.commit()
            report["property_defaults"]["matched"] += 1


def _tag_virtual_methods(graph, virtual_methods: set[str], report: dict) -> None:
    """Tag method nodes that match virtual method names."""
    import json as _json

    for method_name in virtual_methods:
        method_nodes = graph.find_nodes("method", method_name)
        for node in method_nodes:
            data = node.get("data") or {}
            if not data.get("is_virtual"):
                data["is_virtual"] = True
                graph.conn.execute(
                    "UPDATE nodes SET data=? WHERE id=?",
                    (_json.dumps(data, separators=(",", ":")), node["id"]),
                )
                graph.conn.commit()
                report["virtual_methods"]["tagged"] += 1


def _verify_inheritance(graph, report: dict) -> None:
    """Parse YAML frontmatter inherits and cross-reference with ClassDB."""
    import json as _json

    for md_path in TOMB_GODOT_CLASSES.glob("*.md"):
        class_name = md_path.stem
        if class_name.startswith("@"):
            continue

        content = md_path.read_text(encoding="utf-8")
        tomb_parent = parse_yaml_inherits(content)
        report["inheritance"]["parsed"] += 1

        if tomb_parent is None:
            continue

        class_node = graph.get_node_by_name("class", class_name)
        if not class_node:
            report["inheritance"]["discrepancies"].append(
                f"{class_name}: in tomb but not in ClassDB"
            )
            continue

        parents = graph.get_neighbors(class_node["id"], "INHERITS")
        graphdb_parent = parents[0]["name"] if parents else None

        if graphdb_parent and graphdb_parent != tomb_parent:
            report["inheritance"]["discrepancies"].append(
                f"{class_name}: tomb={tomb_parent}, ClassDB={graphdb_parent}"
            )
        elif graphdb_parent == tomb_parent:
            report["inheritance"]["matched"] += 1

        # Store tomb_inherits on class node
        data = class_node.get("data") or {}
        if "tomb_inherits" not in data:
            data["tomb_inherits"] = tomb_parent
            graph.conn.execute(
                "UPDATE nodes SET data=? WHERE id=?",
                (_json.dumps(data, separators=(",", ":")), class_node["id"]),
            )
            graph.conn.commit()
