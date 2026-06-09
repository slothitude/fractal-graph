# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

**Machine**: Rog (192.168.0.52) — Windows 11, AMD Ryzen 16c, CPU-only workstation
**Python**: 3.13 at `C:\Python313\python.exe` — pip 25.3
**Node**: v24.13.0 — global packages in `AppData\Roaming\npm`
**Rust**: 1.95.0
**Shell**: bash (Git Bash on Windows — use Unix paths, `/dev/null` not `NUL`, forward slashes)

### Windows Gotchas
- `\t` in Python string paths becomes TAB — always use forward slashes or raw strings
- `ollama.exe serve` from CLI (tray app ignores env vars)
- Docker credential helper fails via SSH to Lappy — use dummy `docker-credential-desktop.bat` in PATH
- IP Helper (`iphlpsvc`) holds ports 80/443 on Lappy — Tailscale depends on it, don't disable
- Shell commands use bash syntax — no `&&` chaining in PowerShell contexts
- `schtasks /run /tn TaskName` to trigger scheduled tasks
- **Use CLI commands, not manual edits** — when a CLI tool exists for config changes (e.g. `claude mcp add`), use it. Never manually edit `.mcp.json` or similar config files when there's an official command.

## Project Layout

### Main Workspace
`C:\Users\aaron\Desktop\dev\` — all active projects:

| Project | Description |
|---------|-------------|
| `the-rabbit-hole/` | Civilization narrative engine — Flask + SQLite, 31 news sources, game theory analysis |
| `the-guide/` | Encyclopaedia Terrana — Flask + SQLite FTS5, ARIA-7 alien anthropologist |
| `pi-pantheon/` | TypeScript extension for pi/coding-agent — council/court/mnemosyne tools |
| `geometric-cognition/` | Prime experiment — 131+ experiments, MVM series |
| `scumm_l/` | Godot 4.6 point-and-click adventure engine — LLM narration + AI art (Bonsai FLUX) |
| `monad_viz/` | Godot 4.6 interactive 3D visualizer for monad structures |
| `boss/` | SlothitudeGames website (slothitude.github.io) |
| `lan-dashboard/` | LAN monitoring dashboard |
| `plotter/` | Pen plotter web control — Flask, Hershey font, SVG→G-code, serial streaming |
| `telegram_ai_bot/` | Telegram bot |

### Knowledge Base
`C:\Users\aaron\Desktop\tomb\` — infrastructure docs, runbooks, scripts:
- `souls/` — Pantheon soul files (12 gods + mnemosyne)
- `scripts/` — reporter, generators, MCP servers, sync tools
- `services/` — service runbooks
- `conspiracy/` — conspiracy matching data
- `tomb-index.md` — manifest of all tomb docs
- `pantheon.db` — SQLite database for reports, memories, graph data

### Scripts
- `rabbit_reporter.py` — news pipeline (RSS → LLM analysis → report) — runs every 6h
- `tomorrow_generator.py` — prediction generator (24h/48h/72h forecasts) — runs 30min after reporter
- `mnemosyne_mcp.py` — MCP server for memory/search/research tools
- `tomb_sync.py` — sync tomb data across machines

### Scheduled Tasks (Windows Task Scheduler)
- `RabbitHole-Reporter-{0,6,12,18}` — 00:00/06:00/12:00/18:00
- `RabbitHole-Tomorrow-{0,6,12,18}` — 00:30/06:30/12:30/18:30

### Other Notable Directories
- `C:\Users\aaron\exploring\note\` — Loom (Godot visual node-graph notepad)
- `C:\Users\aaron\Desktop\GodotOS\` — GodotOS project
- `C:\Users\aaron\Desktop\hotswap\` — GPU swap scripts for Lappy
- `C:\docker\` — Docker compose files (rsshub, yt-transcriber, open-webui, the-guide)
- `C:\traefik\` — Traefik reverse proxy config on Lappy

## Remote Operations

### SSH
- **Lappy** (192.168.0.33): `aaron:T0b1@n7243` — GPU inference server
- **Pi** (192.168.0.237): `az:7243` — always-on server (use `sshpass -p '7243' ssh az@...`, installed via MSYS2 at `~/bin/sshpass`)
- **OpenWrt** (192.168.0.2): `root` — SSH key auth (no password). Use `ssh -o BatchMode=yes root@192.168.0.2`
- Use paramiko via `C:\Users\aaron\Desktop\lappy_ssh.py` for scripted SSH
- Hotswap GPU: `python _gpu_swap.py --to text|imggen` (paramiko to Lappy)

### MCP Servers
Config: `C:\Users\aaron\.claude\.mcp.json`
- **context7** — stdio (npx), API docs lookup
- **mnemosyne** — stdio (Python), memory/search/research briefs (17 tools)
- **web-reader** — SSE at Lappy:8003, SearXNG + Playwright
- **web-eyes** — SSE at Lappy:8004, web browsing/vision
- **lan-chat** — SSE at Lappy:8005, LAN messaging
- **chrome-vpn** — SSE at Lappy:8007, Chrome CDP in VPN Docker

### Lappy Services (192.168.0.33)
Key ports: 22 (SSH), 4000, 5000 (HelioAi), 5421 (Rabbit Hole), 5420 (The Guide), 8202 (ComfyUI), 8006 (TTS), 11434 (Ollama), 8888 (SearXNG), 9091 (Transmission), 3000 (Open WebUI)

## The Rabbit Hole — Architecture

**Public site**: https://slothitude.github.io/the-rabbit-hole/ (GitHub Pages, auto-published every 6h)

### Pipeline
1. **Fetch** — RSS (19 mainstream + 4 local + 2 conspiracy), Twitter/X via RSSHub (5 feeds), Polymarket via Gamma API (2 feeds)
2. **Process** — dedup → extract text → match conspiracies (title + description + full_text) → LLM game theory analysis per article
3. **Meta-analyze** — 5 layers: Timeline, Contradiction, Actor Simulation, Forecasting, Mythology Detection
4. **Publish** — article nodes (`art-{date}-{hash6}.md`) → pantheon.db → HTTP POST to Flask (port 5421) → GitHub Pages

### Prediction System (Tomorrow's Paper)
- `/today` — live articles, `/tomorrow` — 24h forecast, `/day-after-tomorrow` — 48h, `/future` — 72h+
- Predictions scored against real articles via `--score` flag
- DB: `predictions` table in `articles.db` (Lappy only, all access via Flask API)

### Flask API (port 5421)
- `POST /api/ingest-report` — ingest report (auth: `X-Ingest-Key: rabbit-hole-2026-secret`)
- `POST /api/ingest-prediction` — ingest prediction
- `GET /api/articles-with-forecasts` — articles with forecast data
- 21 route blueprints — see `the-rabbit-hole/routes/`

### LLM
- **Primary**: Z.ai GLM-5.1
- **Fallback**: Ollama qwen3 (local)
- Both reporter and tomorrow_generator use same LLM pattern

## Tool Router Gateway

**Gateway**: `C:\Users\aaron\Desktop\tool-router\gateway.py` — runs on Rog, port 8012
**Registry**: `C:\Users\aaron\Desktop\tool-router\registry.yaml` — servers, profiles, dedup rules

The gateway consolidates all MCP servers into one entry point. Claude Code connects
to `localhost:8012/sse` + alphabetty (stdio). Other servers load lazily by profile.

**Meta-tools** (always available via `mcp__gateway__` prefix):
- `router_switch(profile)` — switch profile (coding, research, media, image-gen, plotter, full)
- `router_status()` — show active servers, tools, profile
- `router_list_profiles()` — show all profiles
- `router_search(query)` — find tools across all servers
- `router_reload()` — hot-reload registry.yaml

**Adding a new server**: `python C:/Users/aaron/Desktop/tool-router/register.py --name <name> --url <url> --tags "tag1,tag2"`

**Start gateway**: `python C:/Users/aaron/Desktop/tool-router/gateway.py`
**Health check**: `curl http://localhost:8012/health`

## Common Commands

```bash
# Run reporter manually
python C:/Users/aaron/Desktop/tomb/scripts/rabbit_reporter.py --test    # 3 articles
python C:/Users/aaron/Desktop/tomb/scripts/rabbit_reporter.py --quick   # 2/feed
python C:/Users/aaron/Desktop/tomb/scripts/rabbit_reporter.py           # 3/feed (default)
python C:/Users/aaron/Desktop/tomb/scripts/rabbit_reporter.py --full    # 10/feed

# Run prediction generator
python C:/Users/aaron/Desktop/tomb/scripts/tomorrow_generator.py
python C:/Users/aaron/Desktop/tomb/scripts/tomorrow_generator.py --score  # score past predictions

# Start MCP servers on Lappy (via SSH or scheduled tasks)
schtasks /run /tn MCP-WebReader
schtasks /run /tn MCP-WebEyes

# Hotswap GPU on Lappy
python C:/Users/aaron/Desktop/hotswap/_gpu_swap.py --to text     # LLM inference mode
python C:/Users/aaron/Desktop/hotswap/_gpu_swap.py --to imggen   # ComfyUI/FLUX mode

# Ollama (use CLI, not tray)
ollama.exe serve
ollama list  # shared models on O: drive (\\LAPPY-SERVER\.ollama\models)

# MoG Olympics
ssh aaron@192.168.0.33 "python D:/mog/olympics_daily.py --model <name>"   # run one model
ssh aaron@192.168.0.33 "python D:/mog/olympics_daily.py --all"             # run all in roster

# Update scoreboard
scp aaron@192.168.0.33:D:/mog/olympics_history.json C:/Users/aaron/Desktop/dev/the-olympics/history.json
cd C:/Users/aaron/Desktop/dev/the-olympics && git add -A && git commit -m "Update results" && git push
```
