# TODO

## Phase 1: ClassDB Extraction
- [x] Create Godot editor plugin scaffold (`addons/gat/`)
- [x] Write ClassDB dump script — iterate all classes, extract methods/properties/signals/constants/enums/inheritance
- [x] Serialize to `engine_schema.json` (one JSON file, full Godot 4.6 schema)
- [x] Test dump against Godot 4.6 editor — verify completeness (spot-check: CharacterBody2D, Control, ResourceLoader)
- [x] Store generated schema in repo under `data/engine_schema.json`
  - 1034 classes, 11701 methods, 4091 properties, 370 signals, 4327 constants, 572 enums
  - Generated via headless script (`headless_dump.gd` + `project.godot`)
  - 8.6MB JSON, loaded into graph DB in 1.2s
  - Fixed `class_name` reserved keyword issue in GDScript 4
  - Fixed `lastrowid` unreliable after ON CONFLICT DO UPDATE — use SELECT instead
  - Batch-loaded graph (single transaction) for performance

## Phase 2: Engine Knowledge Graph
- [x] Design SQLite graph schema (nodes + edges tables + indexes)
- [x] Write graph loader: parse `engine_schema.json` → insert nodes + edges
- [x] Implement graph traversal: ancestry chain (INHERITS), derived classes (reverse INHERITS), class→methods/properties/signals
- [x] Write query library: `find_class`, `find_inheritance`, `find_children`, `find_by_type`, `find_method`, `find_signal`, `who_has_signal`, `who_has_method`
- [x] Verify queries: 20/20 tests passing

## Phase 3: Documentation Layer (tomb/godot/ vault — 1,078 classes + 557 tutorials)
- [x] Add `search_documentation(query)` tool to MCP server — delegates to `tomb_search(query)` (existing FTS5 index)
- [x] Parse property tables from `tomb/godot/classes/*.md` → extract `{name, type, default}` → store in graph (fills `.tscn` default-value gap)
  - 6,897 property defaults parsed from 1,078 class docs
  - 4,182 matched to graph nodes (2,715 unmatched due to shared property names across classes with UNIQUE constraint)
  - Handles enum types as property type, empty defaults (read-only), complex defaults like Vector2(0, 0)
- [x] Parse `tomb/godot/tutorials/scripting/overridable_functions.md` → VIRTUAL_METHOD flag on method nodes
  - 8 base virtual methods found: _ready, _process, _physics_process, _input, _unhandled_input, _enter_tree, _exit_tree, _draw
  - 131 method nodes tagged across all classes that declare these virtuals
- [x] Parse YAML frontmatter `inherits` field → cross-reference against ClassDB extraction for verification
  - 951/1076 inheritance matches confirmed
  - 19 discrepancies — all classes in tomb docs but not in ClassDB (GDScript, CSharpScript, AreaLight3D, etc.)
  - `tomb_inherits` stored on class node data for reference
- [x] Wire enrichment into MCP server auto-load (runs on startup after engine schema load)
- NOTE: Descriptions, method signatures, signals, wikilinks, tutorials are all queryable via `tomb_search()` — no parsing needed
- NOTE: Dead links (1,570) are irrelevant for FTS5 search — dead wikilinks are just unresolvable text in the index

## Phase 4: Project Parser
- [x] Evaluate `godot_parser` (stevearc/godot_parser, Python, MIT) — handles `.tscn`/`.tres` read+write, may skip custom parser entirely
  - REJECTED: doesn't support format=3 uid:// references, unmaintained since Oct 2023
- [x] Write `.tscn` parser — extract node tree, properties, signal connections, resource references
  - NOTE: `.tscn` only stores non-default properties. Must merge with ClassDB defaults for full state
  - NOTE: Godot 4.x uses `format=3` + string UIDs (`uid://...`). Godot 3 format=2 is incompatible
  - NOTE: Node parent paths are absolute but exclude root name. Direct children use `parent="."`
  - NOTE: `instance` nodes are PackedScene references; their editable children appear as `VolatileNode`
- [x] Write `.gd` parser — extract class_name, extends, methods (typed or untyped), signals, onready vars, export vars, engine type references
  - NOTE: Signal parameter types are informational only — never enforced at parse/runtime. Cannot validate signal type safety
  - NOTE: `class_name` registers globally; collect all declarations as project-level types
  - NOTE: Virtual methods (`_ready`, `_process`, etc.) are not explicitly marked in GDScript. Cross-reference against ClassDB virtual method list
- [x] Write `project.godot` parser — autoloads, input maps, project settings
- [x] Write `.tres`/`.res` parser — resource properties and type info
- [x] Insert project nodes into same graph DB with project-specific edges (INSTANCE_OF, USES_METHOD, CONNECTS_SIGNAL, REFERENCES, CONTAINS_CHILD)
  - Node types: proj_project, proj_scene, proj_node, proj_script, proj_resource
  - Edge types: CONTAINS_CHILD, HAS_SCRIPT, INSTANCE_OF, CONNECTS_SIGNAL
- [x] Add MCP server project query tools: load_project, get_project_structure, get_scene_tree, get_node, find_nodes_by_type, get_signal_connections
- [x] 73 tests passing (37 engine graph + 36 project parser)

## Phase 5: MCP Server
- [x] Set up FastMCP server scaffold (Python, `gat/server.py`)
- [x] Implement engine query tools: `get_class`, `find_inheritance`, `find_children`, `search_engine`, `get_method`, `get_signal`, `who_has_signal`, `who_has_method`, `graph_stats`
- [ ] Implement project query tools: `get_project_structure`, `get_scene_tree`, `get_node`, `find_nodes_by_type`, `get_signal_connections`
- [x] Add to `.mcp.json` config for Claude Code integration

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
