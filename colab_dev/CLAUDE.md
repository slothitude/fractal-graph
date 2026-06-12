# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`colab_dev` is a planning and prototyping workspace for GATv2 (Godot Agent Toolkit v2) — a headless Godot IDE for AI agents. The actual GAT MCP server code lives at `../private_ass_godot_mcp/`.

This repo contains brainstorm docs, API credential notes, and a SQLite database (`data/searchv2.db`).

## Key Files

- `brainstorm.md` — GATv2 architecture roadmap, phase inventory, data flow description
- `info.md` — API credentials (Google Gemini, HuggingFace) — **never commit secrets**
- `data/searchv2.db` — SQLite database (searchv2 MCP data)

## Related Project

The GAT MCP server is at `C:\Users\aaron\private_ass_godot_mcp\` with these key files:
- `gat/server.py` — FastMCP server (20 tools, stdio transport)
- `gat/graph.py` — SQLite engine knowledge graph
- `gat/doc_parser.py` — Godot doc enrichment (property defaults, virtual methods)
- `gat/project_parser.py` — .tscn/.gd/.tres scene parsing
- `gat/validator.py` — Node paths, property values, signal connection validation
- `gat/tomb_search.py` — FTS5 index over 1,078 Godot class docs
- `data/engine_schema.json` — Godot 4.6 ClassDB dump (8.7 MB)
- `data/gat.db` — Pre-built SQLite graph (14 MB, auto-rebuildable)

## GAT Server Startup

```json
{
  "command": "C:/Python313/python.exe",
  "args": ["-m", "gat.server"],
  "cwd": "C:/Users/aaron/private_ass_godot_mcp",
  "env": {"PYTHONPATH": "C:/Users/aaron/private_ass_godot_mcp"}
}
```

Dependency: `fastmcp==3.2.0` (NOT 3.4.0 — broken import path). No other external deps.
