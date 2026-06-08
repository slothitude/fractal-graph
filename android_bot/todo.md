# Android Agent Bot — TODO

## Phase 1 — Skeleton

- [x] Scaffold Android project (Kotlin, Gradle, minSdk 28, targetSdk 34)
- [x] Add dependencies: kotlin-telegram-bot, okhttp, okhttp-sse, kotlinx-serialization, room, work-manager, onnxruntime-android
- [x] Create `AgentForegroundService.kt` — foreground service, persistent notification, `START_STICKY`
- [x] Create `BootReceiver.kt` — relaunch service on `BOOT_COMPLETED`
- [x] Create `TelegramBotService.kt` — long-polling loop
- [x] Create `AuthGate.kt` — whitelist user IDs + optional HMAC validation
- [x] Create `SessionManager.kt` — per-chat message history with configurable trim
- [x] Create `MainActivity.kt` — minimal UI: start/stop toggle + status indicator
- [x] Add permissions to AndroidManifest (INTERNET, FOREGROUND_SERVICE, POST_NOTIFICATIONS, BOOT_COMPLETED, BATTERY_OPTIMIZATIONS, READ/WRITE_EXTERNAL_STORAGE)
- [x] Implement echo bot — receive message, send it back
- [ ] **Test on device**: service survives screen-off and Doze for 30+ minutes

## Phase 2 — OpenAI-Compatible LLM

- [x] Create `LlmBackend.kt` — interface (complete, completeStreaming, StreamCallback, ToolDef)
- [x] Create `OpenAiCompatBackend.kt` — single client for any OpenAI-compatible provider
  - `POST /chat/completions` with tool_use support
  - Streaming via OkHttp SSE (EventSource listener)
  - Configurable baseUrl, apiKey, model
- [x] Create `LlmRouter.kt` — provider selection (WiFi check, user override, fallback chain)
- [x] Create provider persistence — SharedPreferences JSON with LlmProvider data class
- [x] Create provider config commands — `/providers`, `/provider add <name> <url> <key> <model>`, `/model <name>`
- [x] Wire prompt builder — system prompt + session history + user message + tool schemas
- [x] Stream LLM response back to Telegram via `editMessageText` (~800ms intervals, 4096 char cap)
- [x] Store API keys in SharedPreferences (encrypted prefs available for Phase 6 hardening)
- [x] Added `android:usesCleartextTraffic` for LAN HTTP providers (Ollama, LiteLLM)
- [ ] **Test**: connect to Ollama on Lappy (192.168.0.33:11434) and Anthropic API — both work

## Phase 3 — Tool Loop + MCP

- [x] Create `Tool.kt` interface — spec (name + JSON schema) + invoke(params)
- [x] Create `ToolRegistry.kt` — register/unregister tools, list specs for LLM
- [x] Create `ToolDispatcher.kt` — parse JSON tool-calls from LLM, dispatch to registry
- [x] Create `AgentLoopService.kt` — ReAct coroutine: plan → act → observe → reflect
- [x] Create `BashTool.kt` — ProcessBuilder, 30s timeout, 64KB stdout cap
- [x] Create `SearchTool.kt` — DuckDuckGo Instant Answer API wrapper
- [x] Add budget system — max loop steps per task, `/budget <n>` command
- [x] Add `/stop` interrupt — cancel running coroutine via Job
- [x] Build system prompt with tool descriptions injected
- [x] Create `mcp/McpClient.kt` — JSON-RPC 2.0 client (initialize, listTools, callTool)
- [x] Create `mcp/McpTransportStdio.kt` — subprocess stdio transport (Runtime.exec + stream pump)
- [x] Create `mcp/McpTransportSse.kt` — HTTP SSE transport (OkHttp)
- [x] Create `mcp/McpToolBridge.kt` — wraps MCP tools as `Tool` objects in the registry
- [x] Create `mcp/McpServerConfig.kt` — load/parse `mcp_servers.json`
- [x] Add `/mcp` commands — `/mcp list`, `/mcp add <name> <url|command>`, `/mcp remove <name>`
- [x] Auto-discover MCP tools on connection → inject into LLM tool list
- [ ] **Test**: ReAct task using local bash + search + one MCP server tool (3-5 steps)

## Phase 4 — Graph Memory

- [ ] Create `memory/GraphDatabase.kt` — SQLite tables for nodes (id, label, type, properties_json) and edges (id, from_id, to_id, relation, weight)
- [ ] Create `memory/GraphTraversal.kt` — BFS/DFS neighbors, shortest path, subgraph extraction
- [ ] Create `memory/EntityExtractor.kt` — prompt LLM to extract named entities + relationships from text after each agent turn
- [ ] Create `memory/EmbeddingStore.kt` — ONNX nomic-embed-text (or compatible), store float vectors in SQLite BLOB
- [ ] Create `memory/MemoryTool.kt` — `memory_search` (semantic), `memory_write` (store node/edge), `memory_graph` (query/traverse)
- [ ] Auto-extract entities after each tool result → write to graph
- [ ] **Test**: have a multi-step conversation, verify entities and relationships accumulate in graph

## Phase 5 — RAG

- [ ] Create `memory/RagStore.kt` — chunk storage + embedding vectors in SQLite
- [ ] Create `memory/RagIngester.kt` — text → chunk (~512 tokens, overlap) → embed → store
- [ ] Create `memory/RagTool.kt` — `rag_ingest` (load document), `rag_search` (semantic retrieval, top-K)
- [ ] Add ingestion sources — file paths, tool output, HTTP-fetched URLs
- [ ] Add `/rag` commands — `/rag ingest <path|url>`, `/rag search <query>`
- [ ] **Test**: ingest a text file, ask a question, verify relevant chunks are retrieved and injected into context

## Phase 6 — Git + Full Tool Set

- [ ] Create `GitTool.kt` — JGit wrapper (clone, diff, add, commit, push, log)
- [ ] Create `FileTool.kt` — read/write in app sandbox via Storage Access Framework
- [ ] Create `HttpTool.kt` — generic GET/POST with configurable headers
- [ ] Create `NotifyTool.kt` — fire notification on task completion
- [ ] Create `AndroidInfoTool.kt` — Build info, battery level, connectivity
- [ ] **Test**: end-to-end task — "clone a repo, find TODOs, commit a fix"

## Phase 7 — Local Model + Hardening

- [ ] Evaluate llama-android vs llama.cpp JNI for on-device inference
- [ ] Create `LocalLlmBackend.kt` — load 3-4B Q4 GGUF model, run inference
- [ ] Update `LlmRouter.kt` — auto-select local vs remote based on WiFi/task complexity
- [ ] Add model download via WorkManager (nightly, WiFi-only)
- [ ] Add fallback logic — if provider fails, try next in chain, then local model
- [ ] Battery exemption request flow — prompt user on first launch
- [ ] WorkManager tasks — nightly memory consolidation, embedding re-index, log rotation
- [ ] Bash sandbox hardening — stricter timeout/cap enforcement, allowed command whitelist
- [ ] AuthGate security audit — rate limiting, replay protection
- [ ] Error recovery — auto-restart Telegram polling on disconnect, reconnect MCP servers
- [ ] Session history compression — LLM-based summarization for long conversations
- [ ] `/tools` command — list all available tools (local + MCP) with descriptions
- [ ] `/status` command — show service state, active provider, MCP connections, memory stats
- [ ] Full Doze/App Standby testing on real device (overnight run)
