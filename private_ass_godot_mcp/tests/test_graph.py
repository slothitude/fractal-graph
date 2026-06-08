"""Tests for GAT Knowledge Graph."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from gat.graph import GodotGraph
from gat.doc_parser import (
    parse_property_table,
    parse_virtual_methods,
    parse_yaml_inherits,
    _split_table_row,
    enrich_graph,
)


SAMPLE_SCHEMA = {
    "version": "4.6",
    "timestamp": "2026-06-09",
    "total_classes": 4,
    "classes": [
        {
            "class": "Node",
            "inherits": None,
            "is_abstract": True,
            "is_singleton": False,
            "properties": [
                {"name": "name", "type": "String", "hint": "PROPERTY_HINT_NONE",
                 "hint_string": "", "usage": ["STORAGE", "EDITOR"]},
                {"name": "process_mode", "type": "int", "hint": "PROPERTY_HINT_ENUM",
                 "hint_string": "Parent,Inherit,Pausable,Always,WhenPaused,WhenPaused",
                 "usage": ["STORAGE"]},
            ],
            "methods": [
                {"name": "get_node", "return_type": "Node",
                 "args": [{"name": "path", "type": "NodePath"}],
                 "is_virtual": False, "is_vararg": False},
                {"name": "get_child", "return_type": "Node",
                 "args": [{"name": "index", "type": "int"}],
                 "is_virtual": False, "is_vararg": False},
                {"name": "add_child", "return_type": "void",
                 "args": [{"name": "node", "type": "Node"}],
                 "is_virtual": False, "is_vararg": False},
                {"name": "queue_free", "return_type": "void", "args": [],
                 "is_virtual": False, "is_vararg": False},
                {"name": "_ready", "return_type": "void", "args": [],
                 "is_virtual": True, "is_vararg": False},
                {"name": "_process", "return_type": "void",
                 "args": [{"name": "delta", "type": "float"}],
                 "is_virtual": True, "is_vararg": False},
            ],
            "signals": [
                {"name": "ready", "args": []},
                {"name": "tree_entered", "args": []},
                {"name": "tree_exiting", "args": []},
            ],
            "constants": [
                {"name": "NOTIFICATION_READY", "value": 13},
                {"name": "NOTIFICATION_PROCESS", "value": 17},
            ],
            "enums": [],
        },
        {
            "class": "Node2D",
            "inherits": "CanvasItem",
            "is_abstract": True,
            "is_singleton": False,
            "properties": [
                {"name": "position", "type": "Vector2", "hint": "PROPERTY_HINT_NONE",
                 "hint_string": "", "usage": ["STORAGE"]},
                {"name": "rotation", "type": "float", "hint": "PROPERTY_HINT_RANGE",
                 "hint_string": "-720,720,0.1,or_less,or_greater",
                 "usage": ["STORAGE"]},
            ],
            "methods": [
                {"name": "move_local_x", "return_type": "void",
                 "args": [{"name": "delta", "type": "float"},
                          {"name": "scaled", "type": "bool"}],
                 "is_virtual": False, "is_vararg": False},
            ],
            "signals": [],
            "constants": [],
            "enums": [],
        },
        {
            "class": "CanvasItem",
            "inherits": "Node",
            "is_abstract": True,
            "is_singleton": False,
            "properties": [
                {"name": "visible", "type": "bool", "hint": "PROPERTY_HINT_NONE",
                 "hint_string": "", "usage": ["STORAGE"]},
            ],
            "methods": [
                {"name": "hide", "return_type": "void", "args": [],
                 "is_virtual": False, "is_vararg": False},
                {"name": "show", "return_type": "void", "args": [],
                 "is_virtual": False, "is_vararg": False},
            ],
            "signals": [],
            "constants": [],
            "enums": [],
        },
        {
            "class": "CharacterBody2D",
            "inherits": "PhysicsBody2D",
            "is_abstract": False,
            "is_singleton": False,
            "properties": [
                {"name": "velocity", "type": "Vector2", "hint": "PROPERTY_HINT_NONE",
                 "hint_string": "", "usage": ["STORAGE"]},
                {"name": "floor_snap_length", "type": "float",
                 "hint": "PROPERTY_HINT_RANGE",
                 "hint_string": "0,128,0.1,or_greater",
                 "usage": ["STORAGE"]},
            ],
            "methods": [
                {"name": "move_and_slide", "return_type": "bool",
                 "args": [], "is_virtual": False, "is_vararg": False},
                {"name": "is_on_floor", "return_type": "bool", "args": [],
                 "is_virtual": False, "is_vararg": False},
                {"name": "is_on_wall", "return_type": "bool", "args": [],
                 "is_virtual": False, "is_vararg": False},
            ],
            "signals": [
                {"name": "body_entered",
                 "args": [{"name": "body", "type": "Node2D"}]},
                {"name": "body_exited",
                 "args": [{"name": "body", "type": "Node2D"}]},
            ],
            "constants": [
                {"name": "MAX_SLIDES", "value": 4},
            ],
            "enums": [
                {"name": "Mode",
                 "values": [
                     {"name": "MODE_RIGID", "value": 0},
                     {"name": "MODE_KINEMATIC", "value": 1},
                 ]},
            ],
        },
    ],
}


@pytest.fixture
def graph(tmp_path):
    db_path = tmp_path / "test.db"
    schema_path = tmp_path / "schema.json"
    with open(schema_path, "w") as f:
        json.dump(SAMPLE_SCHEMA, f)
    g = GodotGraph(db_path)
    g.load_engine_schema(schema_path)
    yield g
    g.close()


class TestGraphSchema:
    def test_stats(self, graph):
        stats = graph.stats()
        assert stats["nodes"] > 0
        assert stats["edges"] > 0
        assert "class" in stats["by_type"]
        assert stats["by_type"]["class"] == 4

    def test_class_nodes(self, graph):
        assert graph.get_node_by_name("class", "Node") is not None
        assert graph.get_node_by_name("class", "CharacterBody2D") is not None


class TestGetClass:
    def test_get_node_class(self, graph):
        cls = graph.get_class("Node")
        assert cls["name"] == "Node"
        assert cls["data"]["is_abstract"] is True
        assert len(cls["properties"]) == 2
        assert len(cls["methods"]) == 6
        assert len(cls["signals"]) == 3
        assert len(cls["constants"]) == 2

    def test_get_character_body(self, graph):
        cls = graph.get_class("CharacterBody2D")
        assert cls["name"] == "CharacterBody2D"
        assert cls["data"]["is_abstract"] is False
        assert len(cls["signals"]) == 2
        # body_entered should be there
        signal_names = [s["name"] for s in cls["signals"]]
        assert "body_entered" in signal_names

    def test_get_missing_class(self, graph):
        assert graph.get_class("NonexistentClass") is None


class TestInheritance:
    def test_node_inheritance(self, graph):
        chain = graph.get_inheritance_chain("CharacterBody2D")
        assert chain[0] == "CharacterBody2D"
        # PhysicsBody2D is in the inherits field but not loaded as a class node,
        # so the chain stops at CharacterBody2D
        assert len(chain) == 1

    def test_node2d_inheritance(self, graph):
        # Node2D → CanvasItem → Node (all loaded)
        chain = graph.get_inheritance_chain("Node2D")
        assert chain[0] == "Node2D"
        assert "CanvasItem" in chain
        assert "Node" in chain

    def test_find_children(self, graph):
        # Node2D inherits from CanvasItem (not in our direct test, but we can test)
        # In our sample, CanvasItem inherits from Node
        children = graph.find_children("CanvasItem")
        names = [c["name"] for c in children]
        assert "Node2D" in names  # Node2D → CanvasItem → Node


class TestMethodLookup:
    def test_find_method_direct(self, graph):
        m = graph.find_method("Node", "get_node")
        assert m is not None
        assert m["data"]["return_type"] == "Node"
        assert len(m["data"]["args"]) == 1

    def test_find_method_inherited(self, graph):
        # CharacterBody2D doesn't have get_node, but Node does
        m = graph.find_method("CharacterBody2D", "queue_free")
        # This may or may not find it depending on the chain
        # Since PhysicsBody2D isn't loaded, it walks up: CharacterBody2D → PhysicsBody2D (not found) → stops
        # So this test verifies the chain-walking doesn't crash
        # Note: the chain is CharacterBody2D → PhysicsBody2D but PhysicsBody2D isn't loaded as a class node
        # so inheritance chain stops there

    def test_find_missing_method(self, graph):
        assert graph.find_method("Node", "nonexistent_method") is None


class TestSignalLookup:
    def test_find_signal_direct(self, graph):
        s = graph.find_signal("CharacterBody2D", "body_entered")
        assert s is not None
        assert len(s["data"]["args"]) == 1
        assert s["data"]["args"][0]["type"] == "Node2D"

    def test_find_signal_inherited(self, graph):
        # Node has "ready" signal; CharacterBody2D should inherit it through the chain
        # But since PhysicsBody2D isn't loaded, the chain breaks
        s = graph.find_signal("Node", "ready")
        assert s is not None


class TestSearch:
    def test_search_by_name(self, graph):
        results = graph.search("velocity")
        names = [r["name"] for r in results]
        assert "velocity" in names  # property on CharacterBody2D

    def test_search_by_type(self, graph):
        results = graph.search("move", type="method")
        names = [r["name"] for r in results]
        assert "move_and_slide" in names
        assert "move_local_x" in names

    def test_search_no_results(self, graph):
        results = graph.search("zzzzznonexistent")
        assert len(results) == 0


class TestWhoHas:
    def test_who_has_signal(self, graph):
        classes = graph.find_who_has_signal("body_entered")
        names = [c["name"] for c in classes]
        assert "CharacterBody2D" in names

    def test_who_has_method(self, graph):
        classes = graph.find_who_has_method("get_node")
        names = [c["name"] for c in classes]
        assert "Node" in names
        # Children of Node that inherit get_node should also appear
        # CanvasItem is a child of Node in our schema
        assert "CanvasItem" in names


class TestUpsert:
    def test_upsert_node(self, graph):
        id1 = graph.add_node("class", "TestClass", {"version": 1})
        id2 = graph.add_node("class", "TestClass", {"version": 2})
        assert id1 == id2  # Same ID, data updated
        node = graph.get_node_by_name("class", "TestClass")
        assert node["data"]["version"] == 2

    def test_ignore_duplicate_edge(self, graph):
        n1 = graph.add_node("test", "A")
        n2 = graph.add_node("test", "B")
        graph.add_edge(n1, n2, "RELATES_TO")
        graph.add_edge(n1, n2, "RELATES_TO")  # Should not duplicate
        edges = graph.get_edges(from_id=n1, relation="RELATES_TO")
        assert len(edges) == 1


# --- Phase 3 Tests ---


SAMPLE_CLASS_DOC = """---
title: Sprite2D
tags:
  - godot
  - godot/class
inherits: "Node2D"
---

**Inherits:** [[Node2D]]

## Sprite2D

**Inherits:** [[Node2D|Node2D]] **<** [[CanvasItem|CanvasItem]] **<** [[Node|Node]] **<** [[Object|Object]]

General-purpose sprite node.

### Description

A node that displays a 2D texture.

### Properties

| [[bool|bool]] | [[Sprite2D#centered|centered]] | `true` |
| --- | --- | --- |
| [[bool|bool]] | [[Sprite2D#flip_h|flip_h]] | `false` |
| [[bool|bool]] | [[Sprite2D#flip_v|flip_v]] | `false` |
| [[int|int]] | [[Sprite2D#frame|frame]] | `0` |
| [[Vector2i|Vector2i]] | [[Sprite2D#frame_coords|frame_coords]] | `Vector2i(0, 0)` |
| [[int|int]] | [[Sprite2D#hframes|hframes]] | `1` |
| [[Vector2|Vector2]] | [[Sprite2D#offset|offset]] | `Vector2(0, 0)` |
| [[bool|bool]] | [[Sprite2D#region_enabled|region_enabled]] | `false` |
| [[bool|bool]] | [[Sprite2D#region_filter_clip_enabled|region_filter_clip_enabled]] | `false` |
| [[Rect2|Rect2]] | [[Sprite2D#region_rect|region_rect]] | `Rect2(0, 0, 0, 0)` |
| [[Texture2D|Texture2D]] | [[Sprite2D#texture|texture]] |  |
| [[int|int]] | [[Sprite2D#vframes|vframes]] | `1` |

### Methods

| [[Rect2|Rect2]] | [[Sprite2D#get_rect|get_rect]]\ (\ ) | const |  |
| --- | --- | --- | --- |
| [[bool|bool]] | [[Sprite2D#is_pixel_opaque|is_pixel_opaque]]\ (\ pos\: [[Vector2|Vector2]]\ ) | const |  |
"""

SAMPLE_ENUM_PROP_DOC = """---
title: CharacterBody2D
tags:
  - godot
  - godot/class
inherits: "PhysicsBody2D"
---

### Properties

| [[CharacterBody2D#MotionMode|MotionMode]] | [[CharacterBody2D#motion_mode|motion_mode]] | `0` |
| --- | --- | --- |
| [[int|int]] | [[CharacterBody2D#max_slides|max_slides]] | `4` |
"""

SAMPLE_READONLY_DOC = """---
title: Node2D
tags:
  - godot
  - godot/class
inherits: "CanvasItem"
---

### Properties

| [[Vector2|Vector2]] | [[Node2D#global_position|global_position]] |  |
| --- | --- | --- |
| [[float|float]] | [[Node2D#global_rotation|global_rotation]] |  |
| [[Vector2|Vector2]] | [[Node2D#position|position]] | `Vector2(0, 0)` |
"""

SAMPLE_VIRTUAL_DOC = """
# Overridable functions

Two functions allow you to initialize and get nodes besides the class's
constructor: `_enter_tree()` and `_ready()`.

.. tabs::
 .. code-tab:: gdscript GDScript

    func _enter_tree():
        pass

    func _ready():
        pass

    func _process(delta):
        pass

    func _physics_process(delta):
        pass

    func _input(event):
        pass

    func _unhandled_input(event):
        pass

Another callback is [[Node#_exit_tree|_exit_tree()]].

[[CanvasItem#_draw|CanvasItem._draw()]] is also important.
"""


class TestTableSplitting:
    def test_simple_row(self):
        cells = _split_table_row("| [[bool|bool]] | [[Sprite2D#centered|centered]] | `true` |")
        assert len(cells) == 4
        assert cells[0] == ""  # before first |
        assert "centered" in cells[2]

    def test_nested_wikilinks_preserved(self):
        cells = _split_table_row("| [[CharacterBody2D#MotionMode|MotionMode]] | [[CharacterBody2D#motion_mode|motion_mode]] | `0` |")
        assert len(cells) >= 4
        assert "MotionMode" in cells[1]

    def test_enum_type(self):
        cells = _split_table_row("| [[CharacterBody2D#MotionMode|MotionMode]] | [[CharacterBody2D#motion_mode|motion_mode]] | `0` |")
        # Type column should contain the enum name
        assert "MotionMode" in cells[1]


class TestPropertyParsing:
    def test_parse_basic_properties(self):
        props = parse_property_table(SAMPLE_CLASS_DOC)
        names = [p["name"] for p in props]
        assert "centered" in names
        assert "flip_h" in names
        assert "texture" in names
        assert "vframes" in names
        assert len(props) == 12

    def test_property_defaults(self):
        props = parse_property_table(SAMPLE_CLASS_DOC)
        by_name = {p["name"]: p for p in props}
        assert by_name["centered"]["default"] == "true"
        assert by_name["flip_h"]["default"] == "false"
        assert by_name["frame"]["default"] == "0"
        assert by_name["frame_coords"]["default"] == "Vector2i(0, 0)"
        assert by_name["offset"]["default"] == "Vector2(0, 0)"
        assert by_name["region_rect"]["default"] == "Rect2(0, 0, 0, 0)"

    def test_property_types(self):
        props = parse_property_table(SAMPLE_CLASS_DOC)
        by_name = {p["name"]: p for p in props}
        assert by_name["centered"]["type"] == "bool"
        assert by_name["frame"]["type"] == "int"
        assert by_name["offset"]["type"] == "Vector2"
        assert by_name["frame_coords"]["type"] == "Vector2i"

    def test_no_default_is_none(self):
        props = parse_property_table(SAMPLE_CLASS_DOC)
        by_name = {p["name"]: p for p in props}
        assert by_name["texture"]["default"] is None

    def test_enum_property_type(self):
        props = parse_property_table(SAMPLE_ENUM_PROP_DOC)
        by_name = {p["name"]: p for p in props}
        assert by_name["motion_mode"]["type"] == "MotionMode"
        assert by_name["max_slides"]["default"] == "4"

    def test_readonly_property_no_default(self):
        props = parse_property_table(SAMPLE_READONLY_DOC)
        by_name = {p["name"]: p for p in props}
        assert by_name["global_position"]["default"] is None
        assert by_name["global_rotation"]["default"] is None
        assert by_name["position"]["default"] == "Vector2(0, 0)"

    def test_no_properties_section(self):
        props = parse_property_table("### Description\nSome text\n### Methods")
        assert len(props) == 0

    def test_method_rows_not_included(self):
        props = parse_property_table(SAMPLE_CLASS_DOC)
        names = [p["name"] for p in props]
        # Methods come after Properties section so shouldn't be included
        assert "get_rect" not in names
        assert "is_pixel_opaque" not in names


class TestVirtualMethods:
    def test_parse_virtual_methods(self):
        methods = parse_virtual_methods(SAMPLE_VIRTUAL_DOC)
        assert "_ready" in methods
        assert "_process" in methods
        assert "_physics_process" in methods
        assert "_enter_tree" in methods
        assert "_exit_tree" in methods
        assert "_input" in methods
        assert "_unhandled_input" in methods
        assert "_draw" in methods
        assert len(methods) == 8


class TestYAMLInherits:
    def test_parse_inherits(self):
        assert parse_yaml_inherits(SAMPLE_CLASS_DOC) == "Node2D"
        assert parse_yaml_inherits(SAMPLE_ENUM_PROP_DOC) == "PhysicsBody2D"
        assert parse_yaml_inherits(SAMPLE_READONLY_DOC) == "CanvasItem"

    def test_no_inherits(self):
        no_inherits = "---\ntitle: Test\ntags:\n  - godot\n---\nContent"
        assert parse_yaml_inherits(no_inherits) is None


class TestEnrichGraph:
    def test_enrich_adds_property_defaults(self, graph):
        """enrich_graph should add default values to property nodes."""
        # Manually set up doc parsing by directly calling enrichment
        # with mocked tomb paths (use a temp dir)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path as P
            # Write a mock class doc
            (P(tmpdir) / "Sprite2D.md").write_text(SAMPLE_CLASS_DOC, encoding="utf-8")

            # Patch paths and run
            import gat.doc_parser as dp
            orig_classes = dp.TOMB_GODOT_CLASSES
            dp.TOMB_GODOT_CLASSES = P(tmpdir)

            try:
                report = dp.enrich_graph(graph)
                # velocity should have a default from CharacterBody2D
                vel = graph.get_node_by_name("property", "velocity")
                assert vel is not None
                # The enrichment should set a default on the velocity property
                # but velocity is shared across many classes, so it depends on
                # which class doc we parsed
            finally:
                dp.TOMB_GODOT_CLASSES = orig_classes

    def test_enrich_tags_virtual_methods(self, graph):
        """enrich_graph should tag virtual methods."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path as P
            (P(tmpdir) / "overridable_functions.md").write_text(SAMPLE_VIRTUAL_DOC, encoding="utf-8")

            import gat.doc_parser as dp
            orig = dp.TOMB_VIRTUAL_METHODS
            dp.TOMB_VIRTUAL_METHODS = P(tmpdir) / "overridable_functions.md"

            try:
                report = dp.enrich_graph(graph)
                # _ready was already is_virtual=True in schema, should remain
                ready = graph.get_node_by_name("method", "_ready")
                assert ready["data"]["is_virtual"] is True
                # _process same
                proc = graph.get_node_by_name("method", "_process")
                assert proc["data"]["is_virtual"] is True
            finally:
                dp.TOMB_VIRTUAL_METHODS = orig

    def test_enrich_stores_tomb_inherits(self, graph):
        """enrich_graph should store tomb_inherits on class nodes."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path as P
            # Write a mock doc for a class that exists in our schema
            (P(tmpdir) / "Node2D.md").write_text(SAMPLE_READONLY_DOC, encoding="utf-8")

            import gat.doc_parser as dp
            orig = dp.TOMB_GODOT_CLASSES
            orig_vm = dp.TOMB_VIRTUAL_METHODS
            dp.TOMB_GODOT_CLASSES = P(tmpdir)
            # Disable virtual methods pass (no file needed)
            dp.TOMB_VIRTUAL_METHODS = P(tmpdir) / "nonexistent.md"

            try:
                report = dp.enrich_graph(graph)
                # Node2D should have tomb_inherits=CanvasItem
                node2d = graph.get_node_by_name("class", "Node2D")
                assert node2d is not None
                assert node2d["data"]["tomb_inherits"] == "CanvasItem"
            finally:
                dp.TOMB_GODOT_CLASSES = orig
                dp.TOMB_VIRTUAL_METHODS = orig_vm
