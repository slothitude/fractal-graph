"""GAT Phase 4 — Parse Godot project files into the knowledge graph.

Parsers for:
- .tscn (scene files) — node tree, properties, signal connections, resource refs
- .gd (GDScript) — class_name, extends, signals, methods, exports, onready, engine refs
- .tres / .res (resource files) — type + properties
- project.godot — autoloads, input maps, project settings

All parsers use the GD text serialization format (INI-like sections).
Godot 4.x uses format=3 with string UIDs (uid://...).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# =============================================================================
# GD Text Format Parser (shared by .tscn, .tres, .res)
# =============================================================================


@dataclass
class GDSection:
    """A single [header key=value ...] section in a GD text file."""
    name: str  # e.g. "gd_scene", "ext_resource", "node", "sub_resource", "connection"
    attrs: dict[str, str] = field(default_factory=dict)  # header attributes
    properties: list[tuple[str, str]] = field(default_factory=list)  # key=value pairs


def parse_gd_sections(text: str) -> list[GDSection]:
    """Parse a GD text file into sections.

    Handles:
    - Section headers: [name key=value key2=value2]
    - Inline comments: lines starting with ;
    - Property lines: key = value (value may contain = signs)
    - Multi-line values (continuation lines that don't start with a key)
    - String values with = signs inside them
    """
    sections: list[GDSection] = []
    current: GDSection | None = None

    for line in text.split("\n"):
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith(";"):
            continue

        # Section header: [name key=value ...]
        if stripped.startswith("["):
            m = re.match(r"\[(\w+)\s*(.*?)\]", stripped)
            if m:
                current = GDSection(name=m.group(1))
                attrs_str = m.group(2).strip()
                if attrs_str:
                    current.attrs = _parse_header_attrs(attrs_str)
                sections.append(current)
            continue

        # Property line inside a section
        if current is not None:
            m = re.match(r"(\w[\w/]*)\s*=\s*(.*)", stripped)
            if m:
                key = m.group(1)
                val = m.group(2).strip()
                current.properties.append((key, val))

    return sections


def _parse_header_attrs(text: str) -> dict[str, str]:
    """Parse header attributes like 'load_steps=2 format=3 uid="uid://..."'"""
    attrs: dict[str, str] = {}
    i = 0
    while i < len(text):
        # Match key=value, value may be quoted or unquoted
        m = re.match(r'\s*(\w[\w/]*)\s*=\s*("([^"]*)"|(\S+))', text[i:])
        if m:
            key = m.group(1)
            val = m.group(2) if m.group(2) is not None else m.group(3)
            # Strip surrounding quotes if present
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            attrs[key] = val
            i += m.end()
        else:
            i += 1
    return attrs


# =============================================================================
# TSCN Parser
# =============================================================================


@dataclass
class ExtResource:
    """An external resource reference in a scene."""
    id: str  # resource id (string in format=3, e.g. "1" or "uid://...")
    type: str  # resource type (e.g. "Script", "Texture2D")
    path: str  # res:// path
    uid: str = ""  # uid:// string if present


@dataclass
class SubResource:
    """An internal sub-resource in a scene."""
    id: str  # sub-resource id
    type: str  # resource type
    properties: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class SceneNode:
    """A node instance in a scene."""
    name: str
    type: str  # engine class (e.g. "CharacterBody2D")
    parent: str  # parent path ("" for root)
    properties: list[tuple[str, str]] = field(default_factory=list)
    instance: str = ""  # scene path if this is an instanced scene
    instance_placeholder: str = ""  # placeholder name for instanced scenes


@dataclass
class SignalConnection:
    """A signal connection in a scene."""
    signal: str  # e.g. "pressed"
    from_node: str  # source node path
    to_node: str  # target node path
    method: str  # handler method name
    flags: int = 0  # connection flags


@dataclass
class ParsedScene:
    """A fully parsed .tscn file."""
    path: str
    format: int = 3
    uid: str = ""
    load_steps: int = 0
    ext_resources: list[ExtResource] = field(default_factory=list)
    sub_resources: list[SubResource] = field(default_factory=list)
    nodes: list[SceneNode] = field(default_factory=list)
    connections: list[SignalConnection] = field(default_factory=list)
    root_node: str = ""  # name of root node


def parse_tscn(path: str | Path) -> ParsedScene:
    """Parse a .tscn scene file into a ParsedScene."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    scene = ParsedScene(path=str(path))

    sections = parse_gd_sections(text)

    for section in sections:
        if section.name == "gd_scene":
            scene.format = int(section.attrs.get("format", "3"))
            scene.load_steps = int(section.attrs.get("load_steps", "0"))
            scene.uid = section.attrs.get("uid", "")

        elif section.name == "gd_resource":
            scene.format = int(section.attrs.get("format", "3"))
            scene.uid = section.attrs.get("uid", "")

        elif section.name == "ext_resource":
            ext = ExtResource(
                id=section.attrs.get("id", ""),
                type=section.attrs.get("type", ""),
                path=section.attrs.get("path", ""),
                uid=section.attrs.get("uid", ""),
            )
            scene.ext_resources.append(ext)

        elif section.name == "sub_resource":
            sub = SubResource(
                id=section.attrs.get("id", ""),
                type=section.attrs.get("type", ""),
                properties=list(section.properties),
            )
            scene.sub_resources.append(sub)

        elif section.name == "node":
            node = _parse_node(section)
            scene.nodes.append(node)
            if not node.parent:
                scene.root_node = node.name

        elif section.name == "connection":
            conn = _parse_connection(section)
            scene.connections.append(conn)

    return scene


def _parse_node(section: GDSection) -> SceneNode:
    """Parse a [node] section into a SceneNode."""
    node = SceneNode(
        name=section.attrs.get("name", ""),
        type=section.attrs.get("type", "Node"),
        parent=section.attrs.get("parent", ""),
    )

    # Check for instance keyword
    if "instance" in section.attrs:
        node.instance = section.attrs["instance"]

    # Check for instance_placeholder
    if "instance_placeholder" in section.attrs:
        node.instance_placeholder = section.attrs["instance_placeholder"]

    # Properties
    node.properties = list(section.properties)
    return node


def _parse_connection(section: GDSection) -> SignalConnection:
    """Parse a [connection] section into a SignalConnection."""
    # Signal format: "signal_name" or "NodePath:signal_name"
    signal_raw = section.attrs.get("signal", "")

    return SignalConnection(
        signal=signal_raw,
        from_node=section.attrs.get("from", ""),
        to_node=section.attrs.get("to", ""),
        method=section.attrs.get("method", ""),
        flags=int(section.attrs.get("flags", "0")),
    )


# =============================================================================
# GDScript Parser
# =============================================================================


@dataclass
class ParsedScript:
    """A parsed GDScript file."""
    path: str
    class_name: str = ""
    extends: str = ""  # engine class or another script
    extends_type: str = ""  # "engine" or "script"
    signals: list[dict] = field(default_factory=list)  # [{name, args}]
    methods: list[dict] = field(default_factory=list)  # [{name, return_type, params, is_virtual}]
    exports: list[dict] = field(default_factory=list)  # [{name, type, hint}]
    onready_vars: list[dict] = field(default_factory=list)  # [{name, type}]
    vars: list[dict] = field(default_factory=list)  # [{name, type}]
    constants: list[dict] = field(default_factory=list)  # [{name, value}]
    engine_refs: list[str] = field(default_factory=list)  # engine types referenced
    preload_paths: list[str] = field(default_factory=list)  # preload paths


def parse_gd(path: str | Path) -> ParsedScript:
    """Parse a .gd GDScript file into a ParsedScript."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    script = ParsedScript(path=str(path))

    # Strip single-line comments for parsing (preserve docstrings)
    lines = text.split("\n")

    # Extract class_name
    for m in re.finditer(r"^class_name\s+(\w+)", text, re.MULTILINE):
        script.class_name = m.group(1)

    # Extract extends
    for m in re.finditer(r"^extends\s+([\w./]+)", text, re.MULTILINE):
        script.extends = m.group(1)
        break

    # Determine if extends is an engine class or a script
    if script.extends:
        script.extends_type = "script" if "/" in script.extends else "engine"

    # Extract preload paths
    for m in re.finditer(r'preload\("([^"]+)"\)', text):
        script.preload_paths.append(m.group(1))
    for m in re.finditer(r"preload\('([^']+)'\)", text):
        script.preload_paths.append(m.group(1))

    # Extract signal declarations: signal name(type1, type2)
    for m in re.finditer(r"^signal\s+(\w+)(?:\(([^)]*)\))?", text, re.MULTILINE):
        sig = {"name": m.group(1)}
        args_str = m.group(2)
        sig["args"] = _parse_signal_args(args_str) if args_str else []
        script.signals.append(sig)

    # Extract @export declarations (skip @export_group and @export_subgroup)
    for m in re.finditer(r'@export\s+var\s+(\w+)(?:\s*:\s*(\w[\w\[\], ]*))?(\s*=\s*.+)?', text, re.MULTILINE):
        script.exports.append({
            "name": m.group(1),
            "type": (m.group(2) or "").strip(),
        })

    # Extract @onready var declarations
    for m in re.finditer(r'@onready\s+var\s+(\w+)(?:\s*:\s*(\w[\w\[\], ]*))?', text, re.MULTILINE):
        script.onready_vars.append({
            "name": m.group(1),
            "type": (m.group(2) or "").strip(),
        })

    # Extract method (func) declarations
    for m in re.finditer(r"^(?:static\s+)?func\s+(\w+)\s*(?:\(([^)]*)\))?(?:\s*->\s*(\w[\w\[\], ]*))?\s*:", text, re.MULTILINE):
        method = {"name": m.group(1), "is_virtual": m.group(1).startswith("_")}
        params_str = m.group(2)
        method["params"] = _parse_func_params(params_str) if params_str else []
        method["return_type"] = m.group(3) or ""
        script.methods.append(method)

    # Extract var declarations (not @export, not @onready, not inside func)
    for m in re.finditer(r'^var\s+(\w+)(?:\s*:\s*(\w[\w\[\], ]*))?(?:\s*=\s*.+)?', text, re.MULTILINE):
        script.vars.append({
            "name": m.group(1),
            "type": (m.group(2) or "").strip(),
        })

    # Extract const declarations
    for m in re.finditer(r'^const\s+(\w+)(?:\s*:\s*(\w[\w\[\], ]*))?\s*=\s*(.+)', text, re.MULTILINE):
        script.constants.append({
            "name": m.group(1),
            "type": (m.group(2) or "").strip(),
            "value": m.group(3).strip(),
        })

    # Collect engine type references
    engine_types = _find_engine_types(text)
    script.engine_refs = sorted(set(engine_types))

    return script


def _parse_signal_args(args_str: str) -> list[dict]:
    """Parse signal argument list: 'amount: int, direction: Vector2'"""
    args = []
    if not args_str.strip():
        return args
    for part in args_str.split(","):
        part = part.strip()
        if not part:
            continue
        # Handle typed args: 'name: Type' or untyped: 'name'
        m = re.match(r"(\w+)\s*:\s*(\w[\w\[\], ]*)", part)
        if m:
            args.append({"name": m.group(1), "type": m.group(2)})
        else:
            args.append({"name": part.strip(), "type": ""})
    return args


def _parse_func_params(params_str: str) -> list[dict]:
    """Parse function parameter list with types."""
    params = []
    if not params_str.strip():
        return params
    for part in params_str.split(","):
        part = part.strip()
        if not part:
            continue
        # Skip self references
        if part == "self":
            continue
        m = re.match(r"(\w+)\s*:\s*(\w[\w\[\], ]*)", part)
        if m:
            params.append({"name": m.group(1), "type": m.group(2)})
        else:
            params.append({"name": part.strip(), "type": ""})
    return params


# Known engine types that GDScript commonly references
_ENGINE_TYPES = frozenset({
    # Core
    "Node", "Node2D", "Node3D", "Control", "CanvasItem",
    # 2D
    "Sprite2D", "AnimatedSprite2D", "CharacterBody2D", "RigidBody2D",
    "StaticBody2D", "Area2D", "CollisionShape2D", "CollisionPolygon2D",
    "Camera2D", "AudioStreamPlayer2D", "TileMap", "TileMapLayer",
    "RichTextLabel", "Label", "Button", "LineEdit", "TextEdit",
    "Panel", "PanelContainer", "VBoxContainer", "HBoxContainer",
    "GridContainer", "MarginContainer", "ScrollContainer",
    "TextureRect", "ColorRect", "NinePatchRect", "ProgressBar",
    "OptionButton", "CheckBox", "CheckButton", "MenuButton",
    "FileDialog", "Timer", "Path2D", "PathFollow2D",
    "NavigationAgent2D", "RayCast2D", "VisibilityNotifier2D",
    "GPUParticles2D", "CPUParticles2D",
    # 3D
    "Node3D", "MeshInstance3D", "CharacterBody3D", "RigidBody3D",
    "StaticBody3D", "Area3D", "CollisionShape3D", "Camera3D",
    "DirectionalLight3D", "PointLight3D", "SpotLight3D",
    "WorldEnvironment", "Environment", "Sky", "MeshInstance3D",
    "AudioStreamPlayer3D", "NavigationAgent3D",
    # Resources
    "Resource", "Texture2D", "Texture", "Image", "Material",
    "ShaderMaterial", "StandardMaterial3D", "Font", "Theme",
    "AudioStream", "AudioStreamSample", "PackedScene",
    "ArrayMesh", "StyleBoxFlat",
    # Input
    "InputEvent", "InputEventKey", "InputEventMouseButton",
    "InputEventJoypadButton", "InputEventScreenTouch",
    # Data
    "Array", "Dictionary", "Vector2", "Vector3", "Vector4",
    "Vector2i", "Vector3i", "Color", "Rect2", "Rect2i",
    "Transform2D", "Transform3D", "Basis", "Quaternion",
    "AABB", "Plane", "Projection",
    "String", "StringName", "NodePath", "RID",
    "PackedByteArray", "PackedInt32Array", "PackedFloat32Array",
    "PackedStringArray", "PackedVector2Array", "PackedVector3Array",
    "PackedColorArray",
    "Callable", "Signal",
})


def _find_engine_types(text: str) -> list[str]:
    """Find engine type references in GDScript source.

    Matches type annotations, extends, class_name usages, preload types,
    and common engine API calls.
    """
    refs: set[str] = set()

    # Type annotations: var x: Type, func f() -> Type, @export var x: Type
    for m in re.finditer(r":\s*(\w+[\w\[\]]*)", text):
        t = m.group(1).rstrip("[]")
        if t in _ENGINE_TYPES or t.endswith("2D") or t.endswith("3D"):
            refs.add(t)

    # extends EngineType
    for m in re.finditer(r"extends\s+(\w+)", text):
        refs.add(m.group(1))

    # Common engine method calls on types: $Node.get_node, get_nodeOrNull, etc.
    for m in re.finditer(r"preload\(\"res://[^\"]+?/([\w]+)\.gd\"\)", text):
        name = m.group(1)
        # Convert snake_case to PascalCase (common Godot convention)
        pascal = "".join(w.capitalize() for w in name.split("_"))
        refs.add(pascal)

    return list(refs)


# =============================================================================
# Project.godot Parser
# =============================================================================


@dataclass
class ParsedProject:
    """A parsed project.godot file."""
    path: str
    name: str = ""
    version: str = ""
    main_scene: str = ""
    autoloads: list[dict] = field(default_factory=list)  # [{name, path}]
    input_actions: list[str] = field(default_factory=list)  # action names
    window_size: dict = field(default_factory=dict)  # {width, height}
    scenes: list[str] = field(default_factory=list)  # res:// paths


def parse_project_godot(path: str | Path) -> ParsedProject:
    """Parse a project.godot file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    project = ParsedProject(path=str(path))

    sections = parse_gd_sections(text)

    for section in sections:
        if section.name == "application":
            for key, val in section.properties:
                if key == "config/name":
                    project.name = val.strip('"')
                elif key == "run/main_scene":
                    project.main_scene = val.strip('"')
                elif key == "config/features":
                    m = re.search(r'"([\d.]+)"', val)
                    if m:
                        project.version = m.group(1)

        elif section.name == "autoload":
            for key, val in section.properties:
                # Autoload format: Name="*res://path/to/script.gd"
                autoload_path = val.strip().strip('"')
                autoload_path = autoload_path.lstrip("*")
                project.autoloads.append({"name": key, "path": autoload_path})

        elif section.name == "input":
            for key, _ in section.properties:
                project.input_actions.append(key)

        elif section.name == "display":
            for key, val in section.properties:
                if key == "window/size/viewport_width":
                    project.window_size["width"] = val
                elif key == "window/size/viewport_height":
                    project.window_size["height"] = val

    return project


# =============================================================================
# TRES Parser (reuses GD text format)
# =============================================================================


@dataclass
class ParsedResource:
    """A parsed .tres / .res resource file."""
    path: str
    type: str = ""  # resource type
    properties: list[tuple[str, str]] = field(default_factory=list)


def parse_tres(path: str | Path) -> ParsedResource:
    """Parse a .tres or .res resource file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    resource = ParsedResource(path=str(path))

    sections = parse_gd_sections(text)

    for section in sections:
        if section.name in ("gd_resource", "gd_scene"):
            resource.type = section.attrs.get("type", "")
            # Properties on the header section itself (e.g. bg_color in StyleBoxFlat)
            resource.properties.extend(section.properties)
        else:
            # Sub-resources or inline properties
            resource.properties.extend(section.properties)

    return resource


# =============================================================================
# Project Scanner — walk a project directory and parse everything
# =============================================================================


@dataclass
class ParsedProjectFiles:
    """All parsed files from a Godot project."""
    root: str
    project: ParsedProject | None = None
    scenes: list[ParsedScene] = field(default_factory=list)
    scripts: list[ParsedScript] = field(default_factory=list)
    resources: list[ParsedResource] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)  # all class_name declarations


def scan_project(project_root: str | Path) -> ParsedProjectFiles:
    """Scan a Godot project directory and parse all relevant files."""
    root = Path(project_root)
    result = ParsedProjectFiles(root=str(root))

    # Parse project.godot
    project_file = root / "project.godot"
    if project_file.exists():
        result.project = parse_project_godot(project_file)

    # Walk all directories for .tscn, .gd, .tres, .res
    for filepath in _walk_project_files(root):
        ext = filepath.suffix.lower()

        if ext == ".tscn":
            try:
                result.scenes.append(parse_tscn(filepath))
            except Exception:
                pass  # Skip unparseable scenes

        elif ext == ".gd":
            try:
                script = parse_gd(filepath)
                result.scripts.append(script)
                if script.class_name:
                    result.class_names.append(script.class_name)
            except Exception:
                pass  # Skip unparseable scripts

        elif ext in (".tres", ".res"):
            try:
                result.resources.append(parse_tres(filepath))
            except Exception:
                pass  # Skip unparseable resources

    return result


def _walk_project_files(root: Path) -> list[Path]:
    """Walk project directories, skipping common non-asset dirs."""
    skip_dirs = {".git", ".godot", "build", "__pycache__", ".import", "addons"}
    files: list[Path] = []

    for dirpath, dirnames, filenames in root.walk():
        # Prune skipped directories in-place
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]

        for filename in filenames:
            ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
            if ext in ("tscn", "gd", "tres", "res"):
                files.append(dirpath / filename)

    return files
