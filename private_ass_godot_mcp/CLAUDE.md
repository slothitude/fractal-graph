# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Godot Agent Toolkit (GAT)** — a headless Godot IDE for AI agents. Exposes the entire Godot project as structured, machine-readable JSON through MCP tools, allowing agents to build, modify, test, and export Godot games without the visual editor.

The companion **Godot Knowledge Graph** models the Godot engine itself (ClassDB extraction) and a specific project as a unified queryable graph. The key insight: documentation is metadata *on* the graph, not the graph itself. Answer relationship questions ("what inherits CharacterBody2D?", "what uses body_entered?") not text-search questions.

## Design Documents

- **`tool.md`** — GAT phased roadmap (15 phases): project discovery → scene introspection → scene editing → inspector access → signals → resources → script analysis → engine reflection → project index → headless execution → build system → agent memory → MCP server → validation
- **`stuctured_db.md`** — Knowledge graph architecture: ClassDB extraction → engine graph → documentation layer → project parser → unified graph → query engine → MCP interface → validation engine

## Architecture

```
Agent → MCP Client → GAT MCP Server → Godot Editor Plugin → Godot Engine → Project Files
```

All agent actions flow through validated tool calls. No direct editor state manipulation.

### Two Interconnected Systems

1. **GAT (tool.md)** — Tool-based approach: expose Godot as deterministic MCP tools (`get_scene_tree`, `create_node`, `set_property`, `connect_signal`, `run_scene`, `export_android`). 15 phases, each building on the last.

2. **Knowledge Graph (stuctured_db.md)** — Graph-based approach: extract Godot ClassDB into a graph (classes, methods, properties, signals, enums, constants, inheritance), parse project files (`.tscn`, `.gd`, `.tres`, `project.godot`) into the same graph model, merge into a unified engine+project graph queryable via MCP.

### Prioritization

Start with **Phase 1 (ClassDB extraction)** and **Phase 2 (engine graph)** from the knowledge graph design. A complete engine schema is valuable on its own before any project parsing or agent integration.

## Implementation Notes

- Target **Godot 4.6**
- MCP server will expose graph capabilities as tools to any MCP-capable agent
- Godot's `ClassDB` already contains classes, methods, properties, signals, constants, enums, and inheritance chains — no need to scrape docs
- Scene files (`.tscn`), scripts (`.gd`), resources (`.tres`/`.res`), and `project.godot` are the project data sources
- Validation layer should reject invalid operations before they corrupt project state

## Data Directory

`data/` contains:
- `searchv2.db` — SQLite database (pre-existing, likely template from searchv2 infrastructure)
- `workflows/` — empty, reserved for future workflow definitions
