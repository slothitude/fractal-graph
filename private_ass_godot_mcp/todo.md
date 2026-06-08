# TODO

## Phase 1: ClassDB Extraction
- [x] Create Godot editor plugin scaffold (`addons/gat/`)
- [x] Write ClassDB dump script — iterate all classes, extract methods/properties/signals/constants/enums/inheritance
- [ ] Serialize to `engine_schema.json` (one JSON file, full Godot 4.6 schema)
- [ ] Test dump against Godot 4.6 editor — verify completeness (spot-check: CharacterBody2D, Control, ResourceLoader)
- [ ] Store generated schema in repo under `data/engine_schema.json`

## Phase 2: Engine Knowledge Graph
- [x] Design SQLite graph schema (nodes + edges tables + indexes)
- [x] Write graph loader: parse `engine_schema.json` → insert nodes + edges
- [x] Implement graph traversal: ancestry chain (INHERITS), derived classes (reverse INHERITS), class→methods/properties/signals
- [x] Write query library: `find_class`, `find_inheritance`, `find_children`, `find_by_type`, `find_method`, `find_signal`, `who_has_signal`, `who_has_method`
- [x] Verify queries: 20/20 tests passing

## Phase 3: Documentation Layer (tomb/godot/ vault — 1,078 classes + 557 tutorials)
- [x] Add `search_documentation(query)` tool to MCP server — delegates to `tomb_search(query)` (existing FTS5 index)
- [ ] Parse property tables from `tomb/godot/classes/*.md` → extract `{name, type, default}` → store in graph (fills `.tscn` default-value gap)
- [ ] Parse `tomb/godot/tutorials/scripting/overridable_functions.md` → VIRTUAL_METHOD flag on method nodes
- [ ] Parse YAML frontmatter `inherits` field → cross-reference against ClassDB extraction for verification
- NOTE: Descriptions, method signatures, signals, wikilinks, tutorials are all queryable via `tomb_search()` — no parsing needed
- NOTE: Dead links (1,570) are irrelevant for FTS5 search — dead wikilinks are just unresolvable text in the index

## Phase 4: Project Parser
- [ ] Evaluate `godot_parser` (stevearc/godot_parser, Python, MIT) — handles `.tscn`/`.tres` read+write, may skip custom parser entirely
- [ ] Write `.tscn` parser (or integrate godot_parser) — extract node tree, properties, signal connections, resource references
  - NOTE: `.tscn` only stores non-default properties. Must merge with ClassDB defaults for full state
  - NOTE: Godot 4.x uses `format=3` + string UIDs (`uid://...`). Godot 3 format=2 is incompatible
  - NOTE: Node parent paths are absolute but exclude root name. Direct children use `parent="."`
  - NOTE: `instance` nodes are PackedScene references; their editable children appear as `VolatileNode`
- [ ] Write `.gd` parser — extract class_name, extends, methods (typed or untyped), signals, onready vars, export vars, engine type references
  - NOTE: Signal parameter types are informational only — never enforced at parse/runtime. Cannot validate signal type safety
  - NOTE: `class_name` registers globally; collect all declarations as project-level types
  - NOTE: Virtual methods (`_ready`, `_process`, etc.) are not explicitly marked in GDScript. Cross-reference against ClassDB virtual method list
- [ ] Write `project.godot` parser — autoloads, input maps, project settings
- [ ] Write `.tres`/`.res` parser — resource properties and type info
- [ ] Insert project nodes into same graph DB with project-specific edges (INSTANCE_OF, USES_METHOD, CONNECTS_SIGNAL, REFERENCES, CONTAINS_CHILD)

## Phase 5: MCP Server
- [x] Set up FastMCP server scaffold (Python, `gat/server.py`)
- [x] Implement engine query tools: `get_class`, `find_inheritance`, `find_children`, `search_engine`, `get_method`, `get_signal`, `who_has_signal`, `who_has_method`, `graph_stats`
- [ ] Implement project query tools: `get_project_structure`, `get_scene_tree`, `get_node`, `find_nodes_by_type`, `get_signal_connections`
- [ ] Add to `.mcp.json` config for Claude Code integration

## Phase 6: Validation Engine
- [ ] Implement node path validation (paths must resolve in scene tree)
- [ ] Implement property type checking (value type matches property hint)
  - NOTE: ClassDB property hints (PROPERTY_HINT_RANGE, PROPERTY_HINT_FILE, etc.) define valid value ranges. Must store and use hints for validation
- [ ] Implement resource reference validation (referenced files exist)
- [ ] Implement signal signature compatibility — SOFT ONLY
  - NOTE: GDScript signals are NOT type-safe. `emit(true)` on `signal x(amount: int)` works. Cannot reject mismatches. Warn only.
  - Validate that signal and handler both exist, but don't enforce parameter alignment
- [ ] Implement Liskov substitution for override checks (covariance on returns, contravariance on params)
- [ ] Wire validation into all editing tools — reject before write

## Phase 7: GAT Editing Tools
- [ ] `create_node(parent, type, name)` — add node to scene
- [ ] `delete_node(scene, node_path)` — remove node
- [ ] `move_node`, `reparent_node`, `rename_node`
- [ ] `set_property(scene, node, property, value)` / `get_property`
- [ ] `connect_signal(from, signal, to, method)` / `disconnect_signal`
- [ ] All edits write back to `.tscn` files directly (no editor needed)

## Phase 8: Headless Execution & Export
- [ ] `run_scene(scene, headless)` — invoke `godot --headless --script`
- [ ] `build_project()` — trigger Godot build
- [ ] `export_project(platform)` — Windows, Linux, Android, Web, macOS
- [ ] `rebuild_imports()` — force re-import of assets

## Phase 9: Agent Memory
- [ ] Auto-generate scene summaries on parse (node count, types, key scripts)
- [ ] Auto-generate script summaries (extends, methods, signals, engine API usage)
- [ ] Store in `.gat/project_memory.json`
- [ ] Expose `get_project_summary()` tool
