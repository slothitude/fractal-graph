# Android-Native Agent Runtime — Implementation Plan

## Overview

A persistent Android background agent controlled entirely through Telegram. Send a task, it reasons through a ReAct loop (Plan → Act → Observe → Reflect), invokes tools (local + MCP), and streams results back to chat. The phone is a headless compute node with graph-based memory and RAG.

**Target device**: Motorola Edge 50 Fusion (192.168.0.106), Android 14+, SSH port 8022.

---

## Architecture (5 layers)

```
Telegram → AuthGate → SessionManager → AgentLoop → ToolDispatcher → Local Tools
                      ↕                                  ↕              ↕
                   Graph DB                          LLM Router      MCP Client
                  + RAG Store                  (any OpenAI-compat)
                                                      + local model
```

### Layer 1 — Control Plane
- **TelegramBotService**: Long-polling via `kotlin-telegram-bot` SDK
- **AuthGate**: Whitelist sender user IDs + optional HMAC per-message
- **SessionManager**: Per-chat conversation history, sliding window with LLM compression
- Commands: `/run <task>`, `/stop`, `/status`, `/reset`, `/tools`, `/budget <n>`, `/mcp`, `/memory`

### Layer 2 — Agent Core
- **AgentLoopService**: Kotlin coroutine in foreground service, ReAct loop with budget limit
- **LLM Router**: Any OpenAI-compatible provider (Anthropic, OpenAI, Ollama, LiteLLM, vLLM, etc.) via a single unified client. Local model as fallback.
- **ToolDispatcher**: Deserialize JSON tool-calls from LLM → dispatch to local tools + MCP tools
- **MCP Client**: JSON-RPC over stdio or SSE. Connects to remote MCP servers. Tools appear in the same registry as local tools.

### Layer 3 — Tool Registry
| Tool | Source | Implementation |
|------|--------|---------------|
| `bash` | Local | ProcessBuilder, 30s timeout, 64KB stdout cap |
| `git` | Local | JGit (clone, commit, push, diff, log) |
| `search` | Local | DuckDuckGo Instant Answer API / SerpApi |
| `file_read/write` | Local | Storage Access Framework (app sandbox) |
| `http` | Local | OkHttp (GET/POST with auth) |
| `notify` | Local | NotificationManager |
| `android_info` | Local | Build, BatteryManager, ConnectivityManager |
| `*` | MCP | Any tool exposed by a connected MCP server |

MCP tools are discovered at connection time and injected into the LLM's tool list transparently.

### Layer 4 — Graph Memory + RAG
- **Graph Database**: Nodes (entities, concepts) + edges (relationships) stored in SQLite with a graph layer. Entities auto-extracted from conversations and tool results.
- **Embeddings**: ONNX Runtime with nomic-embed-text (or any compatible model) for semantic similarity.
- **RAG Store**: Document ingestion pipeline — text chunking → embedding → vector index (SQLite FTS5 + cosine similarity). Documents can be loaded via file, URL fetch, or tool output.
- **Memory Tool**: `memory_search` (semantic), `memory_write` (store fact/entity), `memory_graph` (traverse relationships), `rag_search` (retrieve relevant chunks).

### Layer 5 — Android Persistence
- **Foreground Service**: `START_STICKY` + persistent notification to survive Doze
- **Battery exemption**: `REQUEST_IGNORE_BATTERY_OPTIMIZATIONS`
- **WorkManager**: Nightly memory consolidation, embedding re-index, model downloads, log rotation
- **BootReceiver**: `BOOT_COMPLETED` → relaunch service

---

## Project Structure

```
app/
├── control/
│   ├── TelegramBotService.kt
│   ├── AuthGate.kt
│   └── SessionManager.kt
├── core/
│   ├── AgentLoopService.kt
│   ├── LlmRouter.kt
│   ├── OpenAiCompatClient.kt
│   └── ToolDispatcher.kt
├── tools/
│   ├── Tool.kt                    (interface)
│   ├── ToolRegistry.kt
│   ├── BashTool.kt
│   ├── GitTool.kt
│   ├── SearchTool.kt
│   ├── FileTool.kt
│   ├── HttpTool.kt
│   ├── NotifyTool.kt
│   └── AndroidInfoTool.kt
├── mcp/
│   ├── McpClient.kt               (JSON-RPC transport)
│   ├── McpTransportStdio.kt       (subprocess stdio)
│   ├── McpTransportSse.kt          (HTTP SSE)
│   ├── McpToolBridge.kt            (MCP tool → Tool adapter)
│   └── McpServerConfig.kt          (server list, auth)
├── memory/
│   ├── GraphDatabase.kt            (nodes + edges in SQLite)
│   ├── GraphTraversal.kt           (BFS/DFS, path finding)
│   ├── EntityExtractor.kt          (auto-extract entities from text)
│   ├── EmbeddingStore.kt           (ONNX embedding vectors)
│   ├── RagStore.kt                 (chunked documents + vector index)
│   ├── RagIngester.kt              (text → chunks → embed → store)
│   ├── MemoryTool.kt               (search, write, graph query)
│   └── RagTool.kt                  (RAG retrieval tool)
├── llm/
│   ├── LlmBackend.kt               (interface)
│   ├── OpenAiCompatBackend.kt      (unified client for any provider)
│   └── LocalLlmBackend.kt          (llama.cpp JNI)
├── service/
│   ├── AgentForegroundService.kt
│   └── BootReceiver.kt
└── ui/
    └── MainActivity.kt             (minimal: start/stop + config)
```

---

## Key Dependencies

```kotlin
// Telegram
implementation("io.github.kotlin-telegram-bot.kotlin-telegram-bot:telegram:6.1.0")

// HTTP / JSON
implementation("com.squareup.okhttp3:okhttp:4.12.0")
implementation("com.squareup.okhttp3:okhttp-sse:4.12.0")
implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.0")

// Git
implementation("org.eclipse.jgit:org.eclipse.jgit:6.9.0.202403050737-r")

// Persistence
implementation("androidx.room:room-runtime:2.6.1")
ksp("androidx.room:room-compiler:2.6.1")
implementation("androidx.room:room-ktx:2.6.1")

// Background work
implementation("androidx.work:work-runtime-ktx:2.9.0")

// Embeddings (RAG + graph)
implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")

// Local LLM (optional)
implementation("com.github.nicetomeetyou:llama-android:0.1.0")  // or llama.cpp JNI
```

---

## OpenAI-Compatible LLM Client

A single `OpenAiCompatBackend` handles **any** provider that speaks the OpenAI Chat Completions API:

```kotlin
data class LlmProvider(
    val name: String,          // e.g. "anthropic", "ollama", "litellm"
    val baseUrl: String,       // e.g. "https://api.anthropic.com/v1"
    val apiKey: String,
    val model: String,         // e.g. "claude-sonnet-4-6", "qwen3.5:4b"
    val supportsTools: Boolean,
    val supportsStreaming: Boolean,
)
```

Configure multiple providers in `providers.json` on device storage. LlmRouter picks based on:
1. User override (`/model <name>`)
2. WiFi available → remote preferred
3. Task complexity heuristics
4. Fallback chain if primary fails

Works out of the box with: Anthropic, OpenAI, Ollama, LiteLLM, vLLM, Groq, Together, etc.

---

## MCP Client

Implements the MCP protocol (JSON-RPC 2.0) so the agent can use remote tool servers:

```kotlin
interface McpTransport {
    suspend fun connect(server: McpServerConfig)
    suspend fun send(request: JsonRpcRequest): JsonRpcResponse
    suspend fun disconnect()
}

class McpClient(private val transport: McpTransport) {
    suspend fun initialize(): ServerCapabilities
    suspend fun listTools(): List<McpTool>
    suspend fun callTool(name: String, args: JsonObject): McpToolResult
}
```

**Transport modes**:
- **Stdio**: SSH/subprocess to local MCP server (e.g. on Pi or Lappy)
- **SSE**: HTTP SSE to remote MCP server (e.g. ComfyUI, searchmcp)

MCP tools are wrapped as `Tool` objects via `McpToolBridge` and registered in the same `ToolRegistry` as local tools. The LLM sees them identically.

Config stored in `mcp_servers.json`:
```json
[
  {
    "name": "searchmcp",
    "url": "http://192.168.0.33:8013/sse",
    "transport": "sse",
    "tools": ["search", "read_url"]
  },
  {
    "name": "comfyui",
    "command": "ssh",
    "args": ["az@192.168.0.237", "python", "/home/az/mcp_server.py"],
    "transport": "stdio"
  }
]
```

---

## Graph Memory

```kotlin
// SQLite-backed graph
data class Node(val id: Long, val label: String, val type: String, val properties: JsonObject)
data class Edge(val id: Long, val from: Long, val to: Long, val relation: String, val weight: Float)

class GraphDatabase {
    fun addNode(label, type, props): Long
    fun addEdge(from, to, relation, weight)
    fun neighbors(nodeId, maxDepth): List<Pair<Node, Edge>>
    fun shortestPath(from, to): List<Long>
    fun search(query: String, topK): List<Node>  // semantic via embeddings
}
```

Auto-extraction: after each agent turn, `EntityExtractor` pulls named entities and relationships from the LLM's text output and writes them to the graph. Over time the graph accumulates domain knowledge about whatever tasks the agent handles.

---

## RAG Store

```kotlin
class RagStore {
    fun ingest(document: String, metadata: Map<String, String>)
    // 1. Split into chunks (~512 tokens with overlap)
    // 2. Embed each chunk via ONNX nomic-embed-text
    // 3. Store chunk + embedding in SQLite

    fun search(query: String, topK: Int): List<RagChunk>
    // 1. Embed query
    // 2. Cosine similarity against stored embeddings
    // 3. Return top-K chunks with metadata
}
```

Ingestion sources: files on device, tool output (e.g. search results, code), URLs fetched via HttpTool.

---

## Phased Build (7 phases)

### Phase 1 — Skeleton
Foreground service + Telegram polling + echo bot. Confirm service survives screen-off and Doze.

### Phase 2 — OpenAI-Compatible LLM
Unified `OpenAiCompatClient`. Configure providers. Send message, get streamed response, stream back to Telegram.

### Phase 3 — Tool Loop + MCP
Tool interface + dispatcher. Bash + search tools. ReAct loop. MCP client (stdio + SSE). Budget + `/stop`.

### Phase 4 — Graph Memory
SQLite graph layer. Node/edge CRUD. Entity auto-extraction from agent output. Memory tools.

### Phase 5 — RAG
Chunking + embedding pipeline. ONNX nomic-embed-text. RAG search tool. Document ingestion from files and URLs.

### Phase 6 — Git + Full Tool Set
JGit, file, http, notify, android_info tools. End-to-end: "clone repo, find TODOs, commit fix."

### Phase 7 — Local Model + Hardening
llama.cpp JNI. LLM Router fallback. Battery exemption, BootReceiver, log rotation, sandbox hardening, security audit.

---

## Known Hard Parts

1. **Battery / process death** — Test on real hardware from day one. Emulator lies about Doze.
2. **MCP stdio on Android** — No native subprocess isolation. Use SSH to remote machines or in-process server implementations. Stdio transport may need `Runtime.exec` with careful stream management.
3. **ONNX on ARM** — nomic-embed-text runs fine on modern ARM (Snapdragon 7 Gen 3), but verify memory footprint. Consider quantized embedding models.
4. **Bash sandboxing** — No root → app sandbox only. Hard timeout (30s) + stdout cap (64KB).
5. **Context window management** — Sliding window drops early context; LLM summarization is better for long tasks.
6. **Streaming to Telegram** — `editMessageText` for live updates, rate-limit ~1 edit/sec.
7. **Provider config UX** — No UI means all config via Telegram commands or JSON files on device storage.
