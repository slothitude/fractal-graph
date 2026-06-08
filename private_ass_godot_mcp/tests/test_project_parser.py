"""Tests for GAT Phase 4 — Project Parser."""

from __future__ import annotations

import pytest
from pathlib import Path

from gat.project_parser import (
    parse_gd_sections,
    parse_tscn,
    parse_gd,
    parse_project_godot,
    parse_tres,
    ParsedScene,
    ParsedScript,
    ParsedProject,
    ParsedResource,
    scan_project,
)

# ---------------------------------------------------------------------------
# Test fixtures: inline Godot file content
# ---------------------------------------------------------------------------

SIMPLE_TSCN = """\
[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://main.gd" id="1"]

[node name="Main" type="Control"]
layout_mode = 3
anchors_preset = 15
anchor_right = 1.0
anchor_bottom = 1.0
script = ExtResource("1")

[node name="Background" type="ColorRect" parent="."]
layout_mode = 1
color = Color(0.047, 0.059, 0.094, 1)

[node name="Label" type="Label" parent="."]
layout_mode = 1
text = "Hello World"
"""

TSCN_WITH_CONNECTIONS = """\
[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://main.gd" id="1"]

[node name="Main" type="Control"]
layout_mode = 3
script = ExtResource("1")

[node name="Button" type="Button" parent="."]
layout_mode = 1

[connection signal="pressed" from="Button" to="Main" method="_on_button_pressed"]
"""

TSCN_WITH_SUB_RESOURCES = """\
[gd_scene load_steps=3 format=3]

[ext_resource type="Script" path="res://main.gd" id="1"]

[sub_resource type="Gradient" id="Gradient_1"]
interpolation_mode = 2
offsets = PackedFloat32Array(0, 0.5, 1)
colors = PackedColorArray(0.1, 0.2, 0.3, 1, 0.5, 0.6, 0.7, 1)

[sub_resource type="GradientTexture2D" id="GradientTexture2D_1"]
gradient = SubResource("Gradient_1")
fill = 1
fill_from = Vector2(0, 0)

[node name="Main" type="Control"]
layout_mode = 3
script = ExtResource("1")
"""

TSCN_WITH_UID = """\
[gd_scene load_steps=2 format=3 uid="uid://c3x0b7abc123"]

[ext_resource type="Script" path="res://main.gd" id="1"]

[node name="Main" type="Control"]
layout_mode = 3
script = ExtResource("1")
"""

SAMPLE_GD = """\
## Player controller
extends CharacterBody2D

class_name Player

signal health_changed(amount: int, source: Node)
signal died

@export var speed: float = 200.0
@export var max_health: int = 100
@export_group("Movement")
@export var jump_velocity: float = -400.0

@onready var anim_sprite: AnimatedSprite2D
@onready var collision_shape: CollisionShape2D

var health: int = 100
var velocity: Vector2 = Vector2.ZERO
var score: int = 0

const MOVE_SPEED: float = 200.0
const JUMP_FORCE: float = -400.0

func _ready() -> void:
	health = max_health
	_setup_collision()

func _physics_process(delta: float) -> void:
	var input = Input.get_vector("move_left", "move_right", "move_up", "move_down")
	velocity = input * speed
	move_and_slide()

func take_damage(amount: int, source: Node) -> void:
	health -= amount
	health_changed.emit(amount, source)
	if health <= 0:
		died.emit()
		queue_free()

func _setup_collision() -> void:
	anim_sprite.play("idle")
"""

SIMPLE_GD = """\
extends Node

signal song_changed

var song

func _ready() -> void:
	song = Song.new()
	song_changed.emit()

func new_song() -> void:
	song = Song.new()
	song_changed.emit()
"""

CLASS_NAME_GD = """\
## 4-channel mixer
class_name Synthesizer

const CT = preload("res://scripts/constants.gd")

var channels: Array = []
var master_volume: float = 0.5

func _init(note_table) -> void:
	for i in range(4):
		channels.append(SynthChannel.new(note_table))

func render(frame_count: int) -> PackedVector2Array:
	var buffer := PackedVector2Array()
	return buffer
"""

SAMPLE_PROJECT_GODOT = """\
; Engine configuration file.
; Format:
;   [section]
;   param=value

config_version=5

[application]

config/name="TestGame"
run/main_scene="res://scenes/main.tscn"
config/features=PackedStringArray("4.6")

[autoload]

GameManager="*res://scripts/game_manager.gd"
AudioManager="*res://scripts/audio_manager.gd"

[display]

window/size/viewport_width=1280
window/size/viewport_height=720
window/stretch/mode="viewport"

[input]

ui_up={
"deadzone": 0.5,
"events": []
}
"""

SAMPLE_TRES = """\
[gd_resource type="StyleBoxFlat" format=3]

bg_color = Color(0.1, 0.1, 0.2, 0.9)
border_width_left = 1
border_width_top = 1
border_width_right = 1
border_width_bottom = 1
corner_radius_top_left = 8
"""


# ---------------------------------------------------------------------------
# GD Section Parser Tests
# ---------------------------------------------------------------------------


class TestGDSectionParsing:
    def test_parse_scene_header(self):
        sections = parse_gd_sections(SIMPLE_TSCN)
        assert len(sections) == 5  # gd_scene, ext_resource, node x3 (Main, Background, Label)
        assert sections[0].name == "gd_scene"
        assert sections[0].attrs["load_steps"] == "2"
        assert sections[0].attrs["format"] == "3"

    def test_parse_ext_resource(self):
        sections = parse_gd_sections(SIMPLE_TSCN)
        ext = sections[1]
        assert ext.name == "ext_resource"
        assert ext.attrs["type"] == "Script"
        assert ext.attrs["path"] == "res://main.gd"
        assert ext.attrs["id"] == "1"

    def test_parse_node_properties(self):
        sections = parse_gd_sections(SIMPLE_TSCN)
        node = sections[2]
        assert node.name == "node"
        assert node.attrs["name"] == "Main"
        assert node.attrs["type"] == "Control"
        assert ("layout_mode", "3") in node.properties
        assert ("anchor_right", "1.0") in node.properties

    def test_parse_parent_path(self):
        sections = parse_gd_sections(SIMPLE_TSCN)
        assert sections[3].attrs["parent"] == "."

    def test_skip_comments(self):
        sections = parse_gd_sections(SAMPLE_PROJECT_GODOT)
        # Should not crash on comment lines
        assert any(s.name == "application" for s in sections)

    def test_parse_header_with_quotes(self):
        sections = parse_gd_sections(TSCN_WITH_UID)
        assert sections[0].attrs["uid"] == "uid://c3x0b7abc123"

    def test_parse_sub_resource(self):
        sections = parse_gd_sections(TSCN_WITH_SUB_RESOURCES)
        subs = [s for s in sections if s.name == "sub_resource"]
        assert len(subs) == 2
        assert subs[0].attrs["type"] == "Gradient"
        assert subs[0].attrs["id"] == "Gradient_1"
        assert ("interpolation_mode", "2") in subs[0].properties

    def test_parse_connection(self):
        sections = parse_gd_sections(TSCN_WITH_CONNECTIONS)
        conns = [s for s in sections if s.name == "connection"]
        assert len(conns) == 1
        assert conns[0].attrs["signal"] == "pressed"
        assert conns[0].attrs["from"] == "Button"
        assert conns[0].attrs["to"] == "Main"
        assert conns[0].attrs["method"] == "_on_button_pressed"


# ---------------------------------------------------------------------------
# TSCN Parser Tests
# ---------------------------------------------------------------------------


class TestTSCNParser:
    def test_parse_simple_scene(self, tmp_path):
        f = tmp_path / "test.tscn"
        f.write_text(SIMPLE_TSCN, encoding="utf-8")
        scene = parse_tscn(f)
        assert scene.format == 3
        assert len(scene.ext_resources) == 1
        assert scene.ext_resources[0].type == "Script"
        assert scene.ext_resources[0].path == "res://main.gd"
        assert len(scene.nodes) == 3
        assert scene.root_node == "Main"
        assert scene.nodes[1].name == "Background"
        assert scene.nodes[1].type == "ColorRect"
        assert scene.nodes[1].parent == "."
        assert scene.nodes[2].name == "Label"
        assert scene.nodes[2].type == "Label"
        assert scene.nodes[2].parent == "."

    def test_parse_connections(self, tmp_path):
        f = tmp_path / "test.tscn"
        f.write_text(TSCN_WITH_CONNECTIONS, encoding="utf-8")
        scene = parse_tscn(f)
        assert len(scene.connections) == 1
        conn = scene.connections[0]
        assert conn.signal == "pressed"
        assert conn.from_node == "Button"
        assert conn.to_node == "Main"
        assert conn.method == "_on_button_pressed"
        assert conn.flags == 0

    def test_parse_sub_resources(self, tmp_path):
        f = tmp_path / "test.tscn"
        f.write_text(TSCN_WITH_SUB_RESOURCES, encoding="utf-8")
        scene = parse_tscn(f)
        assert len(scene.sub_resources) == 2
        assert scene.sub_resources[0].type == "Gradient"
        assert scene.sub_resources[1].type == "GradientTexture2D"
        # Check sub_resource references sub_resource
        assert ("gradient", 'SubResource("Gradient_1")') in scene.sub_resources[1].properties

    def test_parse_uid(self, tmp_path):
        f = tmp_path / "test.tscn"
        f.write_text(TSCN_WITH_UID, encoding="utf-8")
        scene = parse_tscn(f)
        assert scene.uid == "uid://c3x0b7abc123"

    def test_node_properties_stored(self, tmp_path):
        f = tmp_path / "test.tscn"
        f.write_text(SIMPLE_TSCN, encoding="utf-8")
        scene = parse_tscn(f)
        main = scene.nodes[0]
        assert ("script", 'ExtResource("1")') in main.properties
        assert ("anchors_preset", "15") in main.properties


# ---------------------------------------------------------------------------
# GDScript Parser Tests
# ---------------------------------------------------------------------------


class TestGDParser:
    def test_parse_class_name(self, tmp_path):
        f = tmp_path / "player.gd"
        f.write_text(CLASS_NAME_GD, encoding="utf-8")
        script = parse_gd(f)
        assert script.class_name == "Synthesizer"
        assert script.extends == ""  # no extends

    def test_parse_extends(self, tmp_path):
        f = tmp_path / "song_manager.gd"
        f.write_text(SIMPLE_GD, encoding="utf-8")
        script = parse_gd(f)
        assert script.extends == "Node"
        assert script.extends_type == "engine"

    def test_parse_extends_engine_type(self, tmp_path):
        f = tmp_path / "song_manager.gd"
        f.write_text(SIMPLE_GD, encoding="utf-8")
        script = parse_gd(f)
        assert script.extends_type == "engine"

    def test_parse_signals(self, tmp_path):
        f = tmp_path / "song_manager.gd"
        f.write_text(SIMPLE_GD, encoding="utf-8")
        script = parse_gd(f)
        assert len(script.signals) == 1
        assert script.signals[0]["name"] == "song_changed"

    def test_parse_signals_with_args(self, tmp_path):
        f = tmp_path / "song_manager.gd"
        f.write_text(SIMPLE_GD, encoding="utf-8")
        script = parse_gd(f)
        assert len(script.signals) == 1
        assert script.signals[0]["name"] == "song_changed"
        assert script.signals[0]["args"] == []

    def test_parse_full_signals(self, tmp_path):
        f = tmp_path / "song_manager.gd"
        f.write_text(SIMPLE_GD, encoding="utf-8")
        script = parse_gd(f)
        assert len(script.signals) == 1
        assert script.signals[0]["name"] == "song_changed"
        assert script.signals[0]["args"] == []

    def test_parse_player_script(self, tmp_path):
        f = tmp_path / "player.gd"
        f.write_text(SAMPLE_GD_PLAYER, encoding="utf-8")
        script = parse_gd(f)
        assert script.class_name == "Player"
        assert script.extends == "CharacterBody2D"
        assert script.extends_type == "engine"
        # Signals
        assert len(script.signals) == 2
        assert script.signals[0]["name"] == "health_changed"
        assert len(script.signals[0]["args"]) == 2
        assert script.signals[0]["args"][0] == {"name": "amount", "type": "int"}
        assert script.signals[1]["name"] == "died"
        assert script.signals[1]["args"] == []
        # Exports
        assert len(script.exports) == 3
        assert script.exports[0]["name"] == "speed"
        assert script.exports[0]["type"] == "float"
        # Methods
        method_names = [m["name"] for m in script.methods]
        assert "_ready" in method_names
        assert "_physics_process" in method_names
        assert "take_damage" in method_names
        assert "_setup_collision" in method_names
        # Virtual detection
        virtuals = [m["name"] for m in script.methods if m["is_virtual"]]
        assert "_ready" in virtuals
        assert "_physics_process" in virtuals
        assert "take_damage" not in virtuals
        # Method with return type and params
        take_dmg = next(m for m in script.methods if m["name"] == "take_damage")
        assert take_dmg["return_type"] == "void"
        assert len(take_dmg["params"]) == 2
        # Constants
        assert len(script.constants) == 2
        assert script.constants[0]["name"] == "MOVE_SPEED"

    def test_parse_vars(self, tmp_path):
        f = tmp_path / "player.gd"
        f.write_text(SAMPLE_GD_PLAYER, encoding="utf-8")
        script = parse_gd(f)
        var_names = [v["name"] for v in script.vars]
        assert "health" in var_names
        assert "velocity" in var_names
        assert "score" in var_names

    def test_parse_onready(self, tmp_path):
        f = tmp_path / "player.gd"
        f.write_text(SAMPLE_GD_PLAYER, encoding="utf-8")
        script = parse_gd(f)
        assert len(script.onready_vars) == 2
        assert script.onready_vars[0]["name"] == "anim_sprite"
        assert script.onready_vars[0]["type"] == "AnimatedSprite2D"

    def test_engine_refs(self, tmp_path):
        f = tmp_path / "player.gd"
        f.write_text(SAMPLE_GD_PLAYER, encoding="utf-8")
        script = parse_gd(f)
        assert "CharacterBody2D" in script.engine_refs
        assert "Node" in script.engine_refs

    def test_parse_preloads(self, tmp_path):
        f = tmp_path / "song_manager.gd"
        f.write_text(SIMPLE_GD, encoding="utf-8")
        script = parse_gd(f)
        # No preloads in this simple script
        assert len(script.preload_paths) == 0

    def test_class_name_synthesizer(self, tmp_path):
        f = tmp_path / "synthesizer.gd"
        f.write_text(CLASS_NAME_GD, encoding="utf-8")
        script = parse_gd(f)
        assert script.class_name == "Synthesizer"
        assert script.extends == ""  # class_name with no extends
        assert len(script.methods) == 2
        method_names = [m["name"] for m in script.methods]
        assert "_init" in method_names
        assert "render" in method_names


# SAMPLE_GD above IS the full player script (CharacterBody2D with signals, exports, etc.)
# Alias it so detailed tests can reference it by name
SAMPLE_GD_PLAYER = SAMPLE_GD


# ---------------------------------------------------------------------------
# Project.godot Parser Tests
# ---------------------------------------------------------------------------


class TestProjectParser:
    def test_parse_project(self, tmp_path):
        f = tmp_path / "project.godot"
        f.write_text(SAMPLE_PROJECT_GODOT, encoding="utf-8")
        proj = parse_project_godot(f)
        assert proj.name == "TestGame"
        assert proj.version == "4.6"
        assert proj.main_scene == "res://scenes/main.tscn"

    def test_parse_autoloads(self, tmp_path):
        f = tmp_path / "project.godot"
        f.write_text(SAMPLE_PROJECT_GODOT, encoding="utf-8")
        proj = parse_project_godot(f)
        assert len(proj.autoloads) == 2
        assert proj.autoloads[0]["name"] == "GameManager"
        assert proj.autoloads[0]["path"] == "res://scripts/game_manager.gd"
        assert proj.autoloads[1]["name"] == "AudioManager"

    def test_parse_input_actions(self, tmp_path):
        f = tmp_path / "project.godot"
        f.write_text(SAMPLE_PROJECT_GODOT, encoding="utf-8")
        proj = parse_project_godot(f)
        assert "ui_up" in proj.input_actions

    def test_parse_window_size(self, tmp_path):
        f = tmp_path / "project.godot"
        f.write_text(SAMPLE_PROJECT_GODOT, encoding="utf-8")
        proj = parse_project_godot(f)
        assert proj.window_size["width"] == "1280"
        assert proj.window_size["height"] == "720"


# ---------------------------------------------------------------------------
# TRES Parser Tests
# ---------------------------------------------------------------------------


class TestTRESParser:
    def test_parse_tres(self, tmp_path):
        f = tmp_path / "test.tres"
        f.write_text(SAMPLE_TRES, encoding="utf-8")
        res = parse_tres(f)
        assert res.type == "StyleBoxFlat"
        assert len(res.properties) > 0
        assert ("bg_color", "Color(0.1, 0.1, 0.2, 0.9)") in res.properties


# ---------------------------------------------------------------------------
# Real Project Tests (chiptracker_gd)
# ---------------------------------------------------------------------------


CHIPTRACKER_ROOT = "C:/Users/aaron/Desktop/dev/chiptracker_gd"


class TestRealProject:
    """Tests against a real Godot project if available."""

    @classmethod
    def setup_class(cls):
        cls.has_real_project = Path(CHIPTRACKER_ROOT).is_dir()

    def test_scan_real_project(self):
        if not self.has_real_project:
            pytest.skip("chiptracker_gd not found")
        result = scan_project(CHIPTRACKER_ROOT)
        assert result.project is not None
        assert result.project.name == "ChipTracker"
        assert result.project.version == "4.6"
        assert len(result.scenes) >= 1
        assert len(result.scripts) >= 1

    def test_real_tscn_parse(self):
        if not self.has_real_project:
            pytest.skip("chiptracker_gd not found")
        from gat.project_parser import parse_tscn as _parse_tscn
        scene = _parse_tscn(Path(CHIPTRACKER_ROOT) / "scenes" / "main.tscn")
        assert scene.format == 3
        assert scene.root_node == "Main"
        assert len(scene.ext_resources) >= 1
        assert scene.ext_resources[0].type == "Script"
        assert scene.nodes[0].type == "Control"

    def test_real_project_godot(self):
        if not self.has_real_project:
            pytest.skip("chiptracker_gd not found")
        proj = parse_project_godot(Path(CHIPTRACKER_ROOT) / "project.godot")
        assert proj.name == "ChipTracker"
        assert len(proj.autoloads) >= 5
        autoload_names = [a["name"] for a in proj.autoloads]
        assert "SongManager" in autoload_names
        assert len(proj.input_actions) >= 10

    def test_real_gd_with_class_name(self):
        if not self.has_real_project:
            pytest.skip("chiptracker_gd not found")
        script = parse_gd(Path(CHIPTRACKER_ROOT) / "scripts" / "audio" / "synthesizer.gd")
        assert script.class_name == "Synthesizer"
        assert "render" in [m["name"] for m in script.methods]

    def test_real_gd_autoload(self):
        if not self.has_real_project:
            pytest.skip("chiptracker_gd not found")
        script = parse_gd(Path(CHIPTRACKER_ROOT) / "scripts" / "autoload" / "song_manager.gd")
        assert script.extends == "Node"
        assert len(script.signals) >= 1
        assert "song_changed" in [s["name"] for s in script.signals]

    def test_scan_collects_class_names(self):
        if not self.has_real_project:
            pytest.skip("chiptracker_gd not found")
        result = scan_project(CHIPTRACKER_ROOT)
        assert "Synthesizer" in result.class_names
