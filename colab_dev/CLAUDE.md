# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`colab_dev` is a planning and prototyping workspace for GATv2 (Godot Agent Toolkit v2) — a headless Godot IDE for AI agents. The actual GAT MCP server code lives at `../private_ass_godot_mcp/`.

This repo contains brainstorm docs, API credential notes, and a SQLite database (`data/searchv2.db`).

## Key Files

- `brainstorm.md` — GATv2 architecture roadmap, phase inventory, data flow description
- `info.md` — API credentials (Google Gemini, HuggingFace) — **never commit secrets**
- `data/searchv2.db` — SQLite database (searchv2 MCP data)
- `notebooks/grpo_train.ipynb` — Colab GRPO training notebook for RTS commander agent
- `setup.py` — Makes `rts_engine` pip-installable (`pip install -e .`)

## RTS Engine (`rts_engine/`)

Pure-Python Red Alert 2 text-state RTS engine for GRPO training. 90 tests passing.
- `engine.py` — GameEngine: tick loop, win conditions, action execution
- `state_encoder.py` — encode_state(): game state → compact token text
- `action_vocab.py` — 6 tool definitions + parse_action_text() + format_tools_prompt()
- `opponents/simple_ai.py` — SimpleAI: scripted baseline (build order → attack)
- `rewards.py` — combined_reward(): unit balance + structure health + outcome
- `adapters/commander_env.py` — CommanderEnv: gym-style wrapper + environment_factory()
- `setup.py` — Minimal setup.py so `pip install -e .` works

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
