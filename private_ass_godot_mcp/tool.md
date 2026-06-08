You could position this as a project similar to what Language Servers did for code editors: take a GUI-heavy environment and expose it as structured, machine-readable operations that an agent can understand.

# Godot Agent Toolkit (GAT)

## Vision

Godot is one of the most agent-friendly game engines from a source-code perspective, but much of the engine's functionality remains locked behind visual editor workflows.

Human developers interact with:

* Scene trees
* Inspector panels
* Signal connections
* Resource browsers
* Animation editors
* TileMap editors
* Import settings
* Export dialogs

LLM agents perform best when working with:

* Structured JSON
* Typed schemas
* Searchable indexes
* Tool calls
* Deterministic APIs

The goal of the Godot Agent Toolkit (GAT) is to bridge this gap.

GAT exposes the entire Godot project as a machine-readable and machine-editable interface that allows AI agents to build, modify, test, and export Godot games without interacting with the visual editor.

The toolkit functions as a headless Godot IDE for agents.

---

# Core Architecture

```text
Agent
    ↓
MCP Client
    ↓
Godot Agent Toolkit
    ↓
Godot Editor Plugin
    ↓
Godot Engine
    ↓
Project Files
```

The Godot plugin exposes structured tools.

The agent never manipulates editor state directly.

All actions flow through validated tool calls.

---

# Design Principles

## 1. Everything Is Structured

Never expose screenshots.

Never expose editor UI.

Expose state as JSON.

Example:

```json
{
  "path": "/root/World/Player",
  "type": "CharacterBody2D",
  "children": [
    "Sprite2D",
    "CollisionShape2D"
  ]
}
```

---

## 2. Full Round-Trip Editing

Agents must be able to:

* Read project state
* Modify project state
* Validate project state

Every readable object should be editable.

---

## 3. Deterministic Operations

Tools must produce repeatable outcomes.

Avoid ambiguous commands.

Good:

```json
{
  "tool": "set_property",
  "node": "/root/Player",
  "property": "speed",
  "value": 300
}
```

Bad:

```json
{
  "tool": "make player faster"
}
```

---

# Phase 1 — Project Discovery

## Goal

Allow agents to understand project structure.

## Tools

### list_scenes

Returns all scene files.

### list_scripts

Returns all scripts.

### list_resources

Returns all project resources.

### get_project_settings

Returns project configuration.

## Output Example

```json
{
  "scenes": [
    "res://scenes/main.tscn",
    "res://scenes/player.tscn"
  ]
}
```

---

# Phase 2 — Scene Tree Introspection

## Goal

Allow agents to inspect scene structure.

## Tools

### get_scene_tree

Returns complete hierarchy.

### get_node

Returns detailed node information.

### find_nodes_by_type

Searches for nodes.

### find_nodes_by_name

Searches by name.

## Example

```json
{
  "path": "/root/Main/Player",
  "type": "CharacterBody2D",
  "children": [
    "Sprite2D",
    "CollisionShape2D",
    "Camera2D"
  ]
}
```

---

# Phase 3 — Scene Editing

## Goal

Allow agents to construct and modify scenes.

## Tools

### create_node

### delete_node

### move_node

### duplicate_node

### rename_node

### reparent_node

## Example

```json
{
  "tool": "create_node",
  "parent": "/root/Main",
  "type": "Camera2D",
  "name": "MainCamera"
}
```

---

# Phase 4 — Inspector Access

## Goal

Expose every editable property.

## Tools

### list_properties

### get_property

### set_property

### reset_property

## Example

```json
{
  "node": "/root/Main/Player",
  "properties": {
    "speed": 250,
    "position": [100, 200]
  }
}
```

---

# Phase 5 — Signals

## Goal

Allow agents to reason about event flow.

## Tools

### list_signals

### list_connections

### connect_signal

### disconnect_signal

## Example

```json
{
  "from": "/root/Menu/Button",
  "signal": "pressed",
  "to": "/root/Menu",
  "method": "_on_button_pressed"
}
```

---

# Phase 6 — Resource System

## Goal

Expose assets and resources.

## Supported Resources

* Textures
* Audio
* Materials
* Meshes
* Animations
* TileSets
* Themes
* Fonts

## Tools

### get_resource

### create_resource

### modify_resource

### delete_resource

### find_resource_references

---

# Phase 7 — Script Analysis

## Goal

Allow deep code understanding.

## Tools

### list_classes

### get_script

### get_class_methods

### get_class_signals

### find_references

### rename_symbol

---

# Phase 8 — Engine Reflection

## Goal

Expose Godot itself.

Using ClassDB, generate a complete engine schema.

## Example

```json
{
  "CharacterBody2D": {
    "properties": [...],
    "methods": [...],
    "signals": [...]
  }
}
```

## Generated Data

* Classes
* Methods
* Signals
* Enums
* Constants
* Inheritance

This becomes a searchable engine database.

---

# Phase 9 — Project Index

## Goal

Create a searchable project knowledge graph.

Generated file:

```text
.gat/project_index.json
```

Contains:

* Scenes
* Scripts
* Nodes
* Signals
* Resources
* Dependencies

Allows rapid agent queries.

---

# Phase 10 — Headless Execution

## Goal

Allow testing without editor interaction.

Uses:

```bash
godot --headless
```

## Tools

### run_scene

### run_tests

### build_project

### export_project

### generate_navigation

### rebuild_imports

---

# Phase 11 — Build System

## Platforms

* Windows
* Linux
* Android
* Web
* macOS

## Tools

### export_windows

### export_linux

### export_android

### export_web

### export_macos

---

# Phase 12 — Agent Memory Layer

## Goal

Allow agents to maintain understanding of large projects.

Stores:

* Scene summaries
* Script summaries
* Asset relationships
* Build history
* Error history

Generated automatically.

---

# Phase 13 — MCP Server

Expose all functionality through MCP.

Example Tools

* get_scene_tree
* create_node
* set_property
* connect_signal
* run_scene
* export_android
* search_project
* find_resource

Any MCP-capable agent can immediately use the toolkit.

---

# Phase 14 — Validation Layer

Before modifications:

* Verify node paths exist
* Verify property types
* Verify resource references
* Verify signal compatibility

Reject invalid operations.

This prevents project corruption.

---

# Phase 15 — Long-Term Goal

A complete machine-readable representation of a Godot project.

The agent should be capable of:

* Creating scenes
* Editing scenes
* Managing resources
* Writing scripts
* Running tests
* Exporting builds
* Refactoring projects
* Exploring engine APIs

without opening the Godot editor.

The final result is effectively a headless Godot IDE designed specifically for AI agents.

The critical insight is that you don't actually need to automate the Godot GUI. You need to expose the *state behind the GUI*. Once scenes, nodes, properties, signals, resources, and ClassDB reflection are available as structured tools, agents can operate on Godot projects almost as effectively as they operate on source code.
