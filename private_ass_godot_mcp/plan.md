# GAT Implementation Plan

## Goal

Build a headless Godot IDE for AI agents — a complete machine-readable representation of Godot 4.6 and any Godot project, exposed through MCP tools.

Two interconnected systems:
1. **GAT** — deterministic tool-based interface (read/edit/run/export Godot projects)
2. **Knowledge Graph** — ClassDB extraction + project parser → unified queryable graph

They share the same foundation (ClassDB extraction) and converge at the MCP server layer.

---

## Phase 1: ClassDB Extraction

Extract the complete Godot 4.6 engine schema from ClassDB at runtime.

**Deliverable**: `engine_schema.json` — every class, method, property, signal, constant, enum, and inheritance chain.

**Approach**: Godot editor plugin script that calls `ClassDB.class_get_*()` APIs and dumps to JSON. Run once per Godot version, cache the result.

**Key APIs**:
- `ClassDB.get_class_list()` — all registered classes
- `ClassDB.class_get_property_list()` / `class_get_method_list()` / `class_get_signal_list()`
- `ClassDB.get_parent_class()` — inheritance
- `ClassDB.class_get_integer_constant_list()` / `class_get_enum_list()`

**Output schema per class**:
```json
{
  "class": "CharacterBody2D",
  "inherits": "PhysicsBody2D",
  "is_abstract": false,
  "is_instantiable": true,
  "properties": [{ "name": "velocity", "type": "Vector2", "hint": "...", "default": "..." }],
  "methods": [{ "name": "move_and_slide", "return_type": "bool", "args": [...], "is_virtual": false }],
  "signals": [{ "name": "body_entered", "args": [...] }],
  "constants": [{ "name": "MAX_SLIDES", "value": 4 }],
  "enums": [{ "name": "Mode", "values": [...] }]
}
```

---

## Phase 2: Engine Knowledge Graph

Convert the flat ClassDB schema into a graph database.

**Deliverable**: SQLite graph with relationship edges (INHERITS, HAS_METHOD, HAS_PROPERTY, HAS_SIGNAL, RETURNS, ACCEPTS, HAS_CONSTANT, HAS_ENUM).

**Storage**: SQLite with adjacency-list edges table. Lightweight, no external graph DB dependency.

**Schema**:
```sql
CREATE TABLE nodes (
  id INTEGER PRIMARY KEY,
  type TEXT,        -- 'class', 'method', 'property', 'signal', 'constant', 'enum'
  name TEXT NOT NULL,
  data JSON         -- full object data
);
CREATE TABLE edges (
  from_id INTEGER REFERENCES nodes(id),
  to_id INTEGER REFERENCES nodes(id),
  relation TEXT,   -- INHERITS, HAS_METHOD, HAS_PROPERTY, etc.
  data JSON         -- metadata (return types, arg types, hints)
);
CREATE INDEX idx_edges_from ON edges(from_id);
CREATE INDEX idx_edges_to ON edges(to_id);
CREATE INDEX idx_edges_rel ON edges(relation);
CREATE INDEX idx_nodes_type ON nodes(type);
CREATE INDEX idx_nodes_name ON nodes(name);
```

**Query examples**:
- "Find all classes derived from Node2D" → traverse INHERITS edges
- "What methods does CharacterBody2D have?" → traverse HAS_METHOD edges
- "What signal is called body_entered?" → look up signal node, traverse reverse edges

---

## Phase 3: Documentation Layer

Attach Godot docs to graph nodes.

**Approach**: Scrape/import Godot 4.6 docs (hosted HTML). Map doc pages to class names in the graph. Store as text blobs on node `data` JSON field.

**Source**: `https://docs.godotengine.org/en/stable/classes/`

**Not critical for MVP** — the graph is useful without docs. Docs are enrichment.

---

## Phase 4: Project Parser

Parse Godot project files into the same graph model.

**Sources**:
- `.tscn` — scene files (node tree hierarchy, properties, signal connections, resource references)
- `.gd` — GDScript files (class definitions, method signatures, signal declarations, references to engine types)
- `.tres` / `.res` — resource files
- `project.godot` — project settings, autoloads, input maps

**Node types for project graph**: Scene, NodeInstance, Script, Resource, SignalConnection, Asset.

**Relationship edges**: INSTANCE_OF, USES_METHOD, CONNECTS_SIGNAL, REFERENCES, CONTAINS_CHILD, HAS_SCRIPT, etc.

This allows the unified graph to answer: "What breaks if I remove this resource?" "Which scenes use CharacterBody2D?" "What scripts connect to the body_entered signal?"

---

## Phase 5: MCP Server

Expose graph + GAT tools to any MCP-capable agent.

**Implementation**: Python FastMCP server (matches existing MCP infrastructure: searchmcp, mnemosyne, fractal-graph all use FastMCP).

**Engine query tools**:
- `get_class(name)` — class + all methods, properties, signals, inheritance
- `find_inheritance(class_name)` — full ancestry chain
- `find_children(class_name)` — all derived classes
- `search_engine(query)` — find classes/methods/properties by name
- `get_method(class, method)` — method signature + argument types

**Project query tools**:
- `get_project_structure()` — scenes, scripts, resources overview
- `get_scene_tree(scene_path)` — full node hierarchy
- `get_node(scene, node_path)` — node details + properties
- `find_nodes_by_type(type)` — search across scenes
- `get_signal_connections(scene)` — all signal wiring

**Project editing tools** (Phase 6+):
- `create_node`, `delete_node`, `move_node`, `reparent_node`
- `set_property`, `get_property`, `list_properties`
- `connect_signal`, `disconnect_signal`

**Execution tools** (Phase 7+):
- `run_scene(scene, headless)` — `godot --headless`
- `build_project()`, `export_project(platform)`

---

## Phase 6: Validation Engine

Reject invalid operations before they corrupt project state.

**Checks**:
- Node paths exist before editing
- Property types match expected types
- Resource references point to existing files
- Signal signatures match (args align)
- Inheritance compatibility (can't assign RigidBody2D where Node2D expected)

---

## Phase 7: Headless Execution & Export

Run scenes and export builds without the editor.

**Leverage**: `godot --headless --script` for testing, `godot --export` for builds.

**Platforms**: Windows, Linux, Android, Web, macOS.

---

## Phase 8: Agent Memory Layer

Auto-generated summaries for large project understanding.

**Stored per session**: scene summaries, script summaries, asset relationship maps, build history, error history.

**Generated by**: traversing the project graph + reading script content, not by agent self-reflection.

---

## Implementation Order

| Step | What | Depends On | Est. Complexity |
|------|------|-----------|-----------------|
| 1 | Godot editor plugin: ClassDB dump script | Godot 4.6 | Low |
| 2 | SQLite graph loader (engine schema → graph DB) | Step 1 | Low |
| 3 | Python graph query library | Step 2 | Low |
| 4 | FastMCP server: engine query tools | Steps 2-3 | Low |
| 5 | Project parser (.tscn, .gd, project.godot) | None | Medium |
| 6 | Project graph merge (project → same graph DB) | Steps 2, 5 | Medium |
| 7 | FastMCP server: project query tools | Steps 4, 6 | Low |
| 8 | Documentation scraper + attachment | Steps 2-3 | Medium |
| 9 | GAT editing tools (node CRUD, properties) | Step 7 | Medium |
| 10 | Signal connection tools | Step 9 | Low |
| 11 | Validation engine | Steps 9-10 | Medium |
| 12 | Headless run + export tools | Step 7 | Medium |
| 13 | Agent memory / project summaries | Steps 6-7 | Low |

Steps 1-4 can ship immediately as a standalone engine reference tool. Steps 5-7 make it project-aware. Steps 8+ are iterative enhancements.

---

## GDScript & TSCN Issues for GAT Implementation

### Signals Are Not Type-Safe

GDScript signal parameter types are **informational only** — they are never enforced at parse time or runtime. You can `emit(true)` on a `signal damage_taken(amount: int)` and GDScript will happily pass a `bool`. You can connect a handler with zero parameters to a signal that emits three, and vice versa.

**Impact on GAT**: The validation engine CANNOT trust GDScript signal declarations for type checking. Signal parameter lists in `.gd` files are documentation, not contracts. GAT must either:
- Validate signal connections at runtime only (emit handler matching), not at parse time
- Or cross-reference handler function signatures against signal declarations and warn on mismatches, but never reject

**Recommendation**: Treat signal types as soft hints. Validate that the signal *exists* and the handler *exists*, but don't enforce parameter alignment statically.

### Untyped GDScript Is Slow and Fragile

Godot docs explicitly warn: "Untyped variables require the runtime to determine the type and dispatch the correct operation on every operation, while typed variables skip this resolution." The top GDScript performance bottleneck is missing type annotations (`var arr = []` vs `var arr: Array[int] = []`).

**Impact on GAT**: When generating or modifying GDScript, GAT should always emit typed code:
- `var hp: int = 100` not `var hp = 100`
- `func take_damage(amount: int) -> bool:` not `func take_damage(amount):`
- `var enemies: Array[Enemy] = []` not `var enemies = []`
- `const MOVE_SPEED: float = 50.0` not `const MOVE_SPEED = 50.0`

This gives agents typed autocomplete AND better runtime performance in the target project.

### Static Typing Is Optional (No Enforcement)

Godot has no project-level "require static typing" setting (proposal #3862 was closed without implementation). `var x = 5` and `var x: int = 5` are both valid forever. `:=` infers from RHS but can infer wrong types (e.g., `var x := some_function()` infers Variant if the function isn't typed).

**Impact on GAT**: Cannot assume scripts are typed. The project parser must handle both typed and untyped GDScript. The graph should annotate whether a script is fully typed or not.

### TSCN Format Has Changed Significantly (Godot 3 → 4)

Godot 4.x uses `format=3` and string-based UIDs (`uid://cexxx...`) instead of integer IDs. Scene files from Godot 3 (`format=2`) are NOT compatible. The TSCN spec is documented at `docs.godotengine.org/en/4.4/contributing/development/file_formats/tscn.html`.

**Key TSCN structure** (5 sections in order):
1. **File descriptor** — `[gd_scene load_steps=N format=3 uid="uid://..."]`
2. **External resources** — `[ext_resource path="..." type="..." id=N uid="uid://..."]`
3. **Internal resources** — `[sub_resource type="..." id=N]`
4. **Nodes** — `[node name="..." type="..." parent="Path/To/Node"]` + property key=value pairs
5. **Connections** — `[connection signal="..." from="..." to="..." method="..."]`

**Critical gotchas**:
- Properties equal to their default are **not stored** — reading a `.tscn` only shows overrides. GAT must merge with ClassDB defaults to get full state.
- Node parent paths are absolute but don't include the root node name. Direct children of root use `parent="."`.
- Scene must have exactly one root node with no `parent` field.
- `instance` keyword in nodes means the node is an instanced scene (PackedScene) — its children are `VolatileNode` type in overrides.
- Existing Python library: `godot_parser` (stevearc/godot_parser on GitHub, MIT) handles `.tscn`/`.tres` parsing with high-level and low-level APIs. Consider using this rather than writing a custom parser.

### _process vs _physics_process

`_process(delta)` runs per-frame (variable rate). `_physics_process(delta)` runs at fixed timestep (default 60/s). Physics code (movement, collision) must use `_physics_process`. Forgetting `* delta` in `_process` causes frame-rate-dependent behavior.

**Impact on GAT**: When generating movement code, always multiply by `delta` and use `_physics_process` for physics-related logic.

### Hardcoded Node Paths Are Fragile

`get_node("../UI/HealthBar")` breaks on rename/move. The idiomatic GDScript alternatives:
- `@onready var health_bar: ProgressBar = $UI/HealthBar` — `$` shorthand, relative to owner node
- `@export var health_bar: ProgressBar` — Inspector-assigned, survives tree changes

**Impact on GAT**: When writing GDScript that references other nodes, prefer `@export` for cross-scene references and `@onready` + `$` for same-scene references. Never generate `get_node()` with hardcoded string paths.

### Signal Connection Lifecycle Bug

Connecting signals in `_ready()` without disconnecting in `_exit_tree()` causes double-connections when nodes are freed and recreated (menus, level transitions, object pooling). `CONNECT_ONE_SHOT` is the fix for one-time connections.

**Impact on GAT**: Generated signal connection code must include disconnect logic or use `CONNECT_ONE_SHOT`. The validation engine should flag signal connections in `_ready()` that lack corresponding disconnects.

### @export Node References Must Be Set Manually

`@export var health_bar: ProgressBar` creates an Inspector field, but if the user doesn't drag-assign it in the editor, it's `null` at runtime. This is a common crash source.

**Impact on GAT**: When creating nodes with `@export` references, GAT should either:
- Use `@onready` + `$` instead (auto-resolves at runtime)
- Or set the export value in the `.tscn` file directly via the `__meta__` property or editor plugin

### class_name Registration Is Global

`class_name Rifle` registers `Rifle` globally in the editor. This means:
- Can be used as a type hint anywhere without `preload()`
- Name collisions between projects are possible
- `class_name` scripts appear in "Create New Script" dialog
- Inner classes (defined inside another class) do NOT need `class_name`

**Impact on GAT**: The project parser should collect all `class_name` declarations globally. The graph should treat `class_name` as a project-level type that can be used in type hints anywhere.

### Godot 4.6 New Features

Godot 4.6 added unified docking (Debugger as a regular dock), Tracy/Perfetto tracing profiler support, and Visual Profiler folding. Not directly relevant to GAT's parsing, but relevant to the execution/profiling phase.

### Property Hint System

ClassDB property lists include `hint` and `hint_string` fields that control how the Inspector renders them (e.g., `PROPERTY_HINT_RANGE`, `PROPERTY_HINT_FILE`, `PROPERTY_HINT_ENUM`). These are critical for the GAT validation engine — they define valid value ranges and types.

**Impact on GAT**: The graph must store property hints. The validation engine should use hints to validate property values (e.g., a `PROPERTY_HINT_RANGE` property should reject values outside the range).

### Inherited Methods and Virtual Functions

Godot has many virtual functions (`_ready`, `_process`, `_physics_process`, `_enter_tree`, `_exit_tree`, etc.) that are called by the engine lifecycle. GDScript overrides are NOT explicitly marked — any method starting with `_` could be a virtual override. `ClassDB` exposes which methods are virtual.

**Impact on GAT**: When parsing scripts, GAT should cross-reference method names against the engine graph's virtual method list to identify overrides. This is valuable for the agent understanding what lifecycle hooks a script uses.

### Covariance and Contravariance

When inheriting base class methods in GDScript:
- **Covariance**: Return type can be more specific (subtype) than parent
- **Contravariance**: Parameter types can be less specific (supertype) than parent

**Impact on GAT**: The validation engine should apply Liskov substitution when checking override compatibility.
