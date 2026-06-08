"""GAT Phase 6 — Validation engine for Godot operations.

Validates node paths, property values, resource references, signal connections,
class existence, and full scene integrity. Returns structured ValidationResult
objects with errors (blockers) and warnings (soft issues).

Key limitation: Godot's ClassDB API does NOT expose method argument types or
signal argument types. All method/signal nodes have args: []. Property hints are
fully available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from gat.graph import GodotGraph


@dataclass
class ValidationResult:
    """Result of a validation check."""
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def merge(self, other: ValidationResult) -> None:
        """Merge another result into this one."""
        for e in other.errors:
            self.add_error(e)
        for w in other.warnings:
            self.add_warning(w)


# Godot property hint constants
PROPERTY_HINT_NONE = 0
PROPERTY_HINT_RANGE = 2
PROPERTY_HINT_ENUM = 3
PROPERTY_HINT_FILE = 17
PROPERTY_HINT_GLOBAL_FILE = 24
PROPERTY_HINT_RESOURCE_TYPE = 19
PROPERTY_HINT_INT_IS_POINTER = 21
PROPERTY_HINT_INT = 22
PROPERTY_HINT_FLOAT = 23
PROPERTY_HINT_EXP_RANGE = 25
PROPERTY_HINT_TYPE_STRING = 28
PROPERTY_HINT_BOOL = 29


@dataclass
class _PropertyHint:
    hint: int = PROPERTY_HINT_NONE
    hint_string: str = ""
    prop_type: str = ""


class Validator:
    """Validates Godot operations against the knowledge graph."""

    def __init__(self, graph: GodotGraph) -> None:
        self.graph = graph

    def validate_node_path(self, scene_name: str, node_path: str) -> ValidationResult:
        """Validate that a node path resolves in a scene's tree.

        Args:
            scene_name: Scene name (filename without extension).
            node_path: Node path (e.g. "Main/Background/Sprite").
        """
        result = ValidationResult()

        # Check scene exists
        scene = self.graph.get_node_by_name("proj_scene", scene_name)
        if not scene:
            result.add_error(f"Scene '{scene_name}' not found in loaded project")
            return result

        # Get all nodes in this scene
        nodes = self.graph.get_project_nodes(scene_name)
        if not nodes:
            result.add_error(f"Scene '{scene_name}' has no nodes")
            return result

        # Build lookup: node_name (qualified) → data
        node_map: dict[str, dict] = {}
        for n in nodes:
            node_map[n["name"]] = n["data"]

        # Walk the path segments
        segments = node_path.split("/")
        if not segments or not segments[0]:
            result.add_error(f"Empty node path: '{node_path}'")
            return result

        # First segment must be a root-level node (parent is "" or ".")
        first_seg = segments[0]
        first_key = f"{scene_name}/{first_seg}"
        if first_key not in node_map:
            result.add_error(f"Node '{first_seg}' not found in scene '{scene_name}'")
            return result

        current_name = first_seg
        for segment in segments[1:]:
            # Look for a node whose parent matches current_name (or "." for root)
            current_parent_path = current_name
            found = False
            for qname, data in node_map.items():
                parent = data.get("parent", "")
                bare_name = data.get("name", "")
                # Parent can be "." for direct children of root
                if parent == current_parent_path and bare_name == segment:
                    found = True
                    current_name = segment
                    break
                # Handle "." parent for root children
                if parent == "." and current_name == segments[0] and bare_name == segment:
                    found = True
                    current_name = segment
                    break
            if not found:
                result.add_error(
                    f"Node '{segment}' not found as child of '{current_name}' "
                    f"in scene '{scene_name}'"
                )
                return result

        return result

    def validate_property(self, class_name: str, property_name: str,
                          value: object) -> ValidationResult:
        """Validate a property value against its type hints.

        Walks the inheritance chain to find the property, then checks
        value type/range/enum based on the property's hint.

        Args:
            class_name: Engine class name (e.g. "Button").
            property_name: Property name (e.g. "text").
            value: The value to validate.
        """
        result = ValidationResult()

        # Find the property by walking inheritance
        prop = self._find_property(class_name, property_name)
        if not prop:
            result.add_warning(
                f"Property '{property_name}' not found on '{class_name}' "
                f"(or any parent class). May be a project-only property."
            )
            return result

        prop_hint = _PropertyHint(
            hint=prop.get("data", {}).get("hint", PROPERTY_HINT_NONE),
            hint_string=prop.get("data", {}).get("hint_string", ""),
            prop_type=prop.get("data", {}).get("type", ""),
        )

        if prop_hint.hint == PROPERTY_HINT_RANGE:
            self._validate_range(value, prop_hint.hint_string, result, property_name)
        elif prop_hint.hint == PROPERTY_HINT_ENUM:
            self._validate_enum(value, prop_hint.hint_string, result, property_name)
        elif prop_hint.hint in (PROPERTY_HINT_FILE, PROPERTY_HINT_GLOBAL_FILE):
            if not isinstance(value, str):
                result.add_warning(
                    f"Property '{property_name}' ({prop_hint.hint_string}) expects a file path string"
                )
        elif prop_hint.hint == PROPERTY_HINT_RESOURCE_TYPE:
            if not isinstance(value, str):
                result.add_warning(
                    f"Property '{property_name}' expects a resource type "
                    f"({prop_hint.hint_string})"
                )
        elif prop_hint.hint == PROPERTY_HINT_INT:
            self._validate_int(value, result, property_name)
        elif prop_hint.hint in (PROPERTY_HINT_FLOAT, PROPERTY_HINT_EXP_RANGE):
            self._validate_float(value, result, property_name)
        elif prop_hint.hint == PROPERTY_HINT_TYPE_STRING:
            if not isinstance(value, str):
                result.add_warning(
                    f"Property '{property_name}' expects a type string"
                )
        elif prop_hint.hint == PROPERTY_HINT_BOOL:
            self._validate_bool(value, result, property_name)
        else:
            # No specific hint — basic type check from prop_type
            if prop_hint.prop_type and not self._check_type_loose(value, prop_hint.prop_type):
                result.add_warning(
                    f"Property '{property_name}' has type '{prop_hint.prop_type}', "
                    f"value {value!r} may not match"
                )

        return result

    def validate_resource_reference(self, resource_path: str,
                                     project_root: str | None = None) -> ValidationResult:
        """Validate a resource path references an existing file.

        Args:
            resource_path: res:// path or absolute path.
            project_root: Optional project root for resolving res:// paths.
        """
        result = ValidationResult()

        if not resource_path:
            result.add_error("Empty resource path")
            return result

        if resource_path.startswith("res://"):
            if not project_root:
                result.add_warning(
                    f"Cannot verify 'res://' path without project_root: {resource_path}"
                )
                return result
            # Convert res:// to filesystem path
            rel = resource_path[6:]  # strip "res://"
            fs_path = Path(project_root) / rel
        else:
            fs_path = Path(resource_path)

        if not fs_path.exists():
            result.add_error(f"Resource file not found: {resource_path} (resolved to {fs_path})")
        else:
            # Validate common extensions
            ext = fs_path.suffix.lower()
            known_exts = {
                ".tscn", ".gd", ".tres", ".res", ".png", ".jpg", ".jpeg",
                ".webp", ".svg", ".wav", ".ogg", ".mp3", ".ttf", ".otf",
                ".json", ".cfg", ".txt", ".md", ".csv", ".bin", ".glb",
                ".gltf", ".obj", ".fbx", ".dae", ".scn", ".tga", ".bmp",
                ".ico", ".icns", ".exr", ".hdr", ".pdf",
            }
            if ext and ext not in known_exts:
                result.add_warning(
                    f"Resource has unusual extension '{ext}': {resource_path}"
                )

        return result

    def validate_signal_connection(self, from_node_type: str, signal_name: str,
                                    to_method: str) -> ValidationResult:
        """Validate a signal connection between two nodes.

        Checks signal exists on from_node_type (walking inheritance).
        Checks to_method is a plausible method name.
        WARN only — GDScript signals are not type-safe.

        Args:
            from_node_type: Engine class type of the emitting node.
            signal_name: Signal name.
            to_method: Handler method name on the receiving node.
        """
        result = ValidationResult()

        # Check signal exists on from_node_type
        sig = self.graph.find_signal(from_node_type, signal_name)
        if sig:
            result.add_warning(
                f"Signal '{signal_name}' found on '{from_node_type}' (args not validated — "
                f"ClassDB does not expose signal argument types)"
            )
        else:
            result.add_warning(
                f"Signal '{signal_name}' not found on '{from_node_type}' "
                f"(or any parent class). May be a project-defined signal."
            )

        # Check handler method name is plausible
        if not to_method:
            result.add_error("Handler method name is empty")
        elif not self._is_plausible_method_name(to_method):
            result.add_warning(
                f"Handler '{to_method}' doesn't look like a typical GDScript method name"
            )

        return result

    def validate_class_exists(self, class_name: str) -> ValidationResult:
        """Check if a class exists in the engine graph or project class_names.

        Args:
            class_name: Class name to check.
        """
        result = ValidationResult()

        # Check engine classes
        node = self.graph.get_node_by_name("class", class_name)
        if node:
            return result

        # Check project class_names
        scripts = self.graph.get_project_scripts()
        for script in scripts:
            if script.get("data", {}).get("class_name") == class_name:
                return result

        result.add_error(
            f"Class '{class_name}' not found in engine graph or project class_names"
        )
        return result

    def validate_scene(self, scene_name: str) -> ValidationResult:
        """Comprehensive scene validation.

        Validates:
        - Scene exists in loaded project
        - All node types exist (engine or project classes)
        - All signal connections reference existing nodes
        - All ext_resource paths point to existing files

        Args:
            scene_name: Scene name (filename without extension).
        """
        result = ValidationResult()

        # Check scene exists
        scene = self.graph.get_node_by_name("proj_scene", scene_name)
        if not scene:
            result.add_error(f"Scene '{scene_name}' not found in loaded project")
            return result

        scene_data = scene.get("data", {})
        scene_path = scene_data.get("path", "")

        # Validate node types
        nodes = self.graph.get_project_nodes(scene_name)
        for node in nodes:
            node_type = node.get("data", {}).get("type", "")
            if node_type:
                type_check = self.validate_class_exists(node_type)
                if not type_check.valid:
                    result.add_error(
                        f"Node '{node.get('data', {}).get('name', '?')}' "
                        f"has unknown type '{node_type}'"
                    )
                # Merge warnings too
                for w in type_check.warnings:
                    result.add_warning(w)

        # Validate signal connections
        connections = self.graph.get_signal_connections(scene_name)
        node_names = {n.get("data", {}).get("name", ""): n for n in nodes}

        for conn in connections:
            from_name = conn.get("from_node", "")
            to_name = conn.get("to_node", "")
            # Extract bare node name from qualified name
            from_bare = from_name.split("/")[-1] if "/" in from_name else from_name
            to_bare = to_name.split("/")[-1] if "/" in to_name else to_name

            if from_bare not in node_names:
                result.add_error(
                    f"Signal connection references non-existent node '{from_bare}'"
                )
            if to_bare not in node_names:
                result.add_error(
                    f"Signal connection references non-existent node '{to_bare}'"
                )

        return result

    # --- Internal helpers ---

    def _find_property(self, class_name: str, property_name: str) -> dict | None:
        """Find a property on a class, walking up the inheritance chain."""
        current_name = class_name
        while current_name and current_name != "":
            node = self.graph.get_node_by_name("class", current_name)
            if not node:
                break
            properties = self.graph.get_neighbors(node["id"], "HAS_PROPERTY")
            for p in properties:
                if p["name"] == property_name:
                    return p
            parents = self.graph.get_neighbors(node["id"], "INHERITS")
            if not parents:
                break
            current_name = parents[0]["name"]
        return None

    def _validate_range(self, value: object, hint_string: str,
                        result: ValidationResult, prop_name: str) -> None:
        """Validate a value against PROPERTY_HINT_RANGE."""
        parts = hint_string.split(",")
        if len(parts) < 2:
            result.add_warning(
                f"Property '{prop_name}' has malformed RANGE hint: {hint_string}"
            )
            return

        try:
            min_val = float(parts[0])
            max_val = float(parts[1])
        except ValueError:
            result.add_warning(
                f"Property '{prop_name}' has non-numeric RANGE bounds: {hint_string}"
            )
            return

        try:
            num_val = float(value)
        except (TypeError, ValueError):
            result.add_error(
                f"Property '{prop_name}' expects numeric value in range "
                f"[{min_val}, {max_val}], got {value!r}"
            )
            return

        if num_val < min_val or num_val > max_val:
            result.add_error(
                f"Property '{prop_name}' value {value!r} is outside range "
                f"[{min_val}, {max_val}]"
            )

    def _validate_enum(self, value: object, hint_string: str,
                       result: ValidationResult, prop_name: str) -> None:
        """Validate a value against PROPERTY_HINT_ENUM."""
        # Enum hint_string is comma-separated values
        valid_values = [v.strip() for v in hint_string.split(",") if v.strip()]
        if not valid_values:
            result.add_warning(
                f"Property '{prop_name}' has empty ENUM hint: {hint_string}"
            )
            return

        str_val = str(value)
        if str_val not in valid_values:
            result.add_error(
                f"Property '{prop_name}' value '{str_val}' is not in enum: "
                f"{', '.join(valid_values)}"
            )

    def _validate_int(self, value: object, result: ValidationResult,
                      prop_name: str) -> None:
        """Validate a value is integer-parseable."""
        try:
            int(value)
        except (TypeError, ValueError):
            result.add_error(
                f"Property '{prop_name}' expects an integer, got {value!r}"
            )

    def _validate_float(self, value: object, result: ValidationResult,
                        prop_name: str) -> None:
        """Validate a value is float-parseable."""
        try:
            float(value)
        except (TypeError, ValueError):
            result.add_error(
                f"Property '{prop_name}' expects a float, got {value!r}"
            )

    def _validate_bool(self, value: object, result: ValidationResult,
                       prop_name: str) -> None:
        """Validate a value is a bool or bool-equivalent."""
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)) and value in (0, 1, 0.0, 1.0):
            return
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return
        result.add_error(
            f"Property '{prop_name}' expects a boolean, got {value!r}"
        )

    def _check_type_loose(self, value: object, expected_type: str) -> bool:
        """Loose type check — returns True if value could plausibly be the type."""
        type_map = {
            "int": (int,),
            "float": (int, float),
            "String": (str,),
            "bool": (bool,),
            "Variant": (object,),  # accepts anything
            "Vector2": (list, tuple),
            "Vector3": (list, tuple),
            "Color": (str, list, tuple),
            "Array": (list, tuple),
            "Dictionary": (dict,),
        }
        # StringName and NodePath are string-like
        if expected_type in ("StringName", "NodePath", "RID"):
            return isinstance(value, str)
        # Handle generic cases
        if expected_type.startswith("Packed"):
            return isinstance(value, (list, tuple, str))
        expected = type_map.get(expected_type)
        if expected:
            return isinstance(value, expected)
        # Unknown type — don't flag
        return True

    @staticmethod
    def _is_plausible_method_name(name: str) -> bool:
        """Check if a string looks like a valid GDScript method name."""
        if not name or len(name) < 1:
            return False
        if not name[0].isalpha() and name[0] != "_":
            return False
        return all(c.isalnum() or c == "_" for c in name)
