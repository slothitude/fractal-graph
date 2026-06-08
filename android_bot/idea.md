Here's a comprehensive full plan for the Android-Native Agent Runtime.Here's the full plan, layer by layer.



\---



\## Project overview



A persistent Android background agent controlled entirely through Telegram. You send it a task, it reasons through a ReAct loop (Plan → Act → Observe → Reflect), invokes local tools, and streams results back to your chat. No UI required — the phone is a headless compute node.



\---



\## Layer 1 — Control plane



\*\*TelegramBotService\*\* runs the long-polling or webhook loop using the `TelegramBots` Java SDK (or `kotlin-telegram-bot`). Every incoming message passes through an `AuthGate` that checks the sender's user ID against a hardcoded whitelist and optionally validates a per-message HMAC signature. Rejected messages are silently dropped.



A `SessionManager` holds a per-chat conversation history (list of `Message` objects) trimmed to a configurable context window. The history feeds the LLM on every turn so multi-step tasks work naturally.



Useful commands to handle: `/run <task>`, `/stop`, `/status`, `/reset`, `/tools` (list available tools), `/budget <n>` (max loop steps for this task).



\---



\## Layer 2 — Agent core



The core is a `AgentLoopService` — a Kotlin coroutine that runs inside the foreground service.



```

suspend fun loop(task: String, session: Session): AgentResult {

&#x20;   var steps = 0

&#x20;   while (steps++ < session.budget) {

&#x20;       val plan     = llm.complete(session.toPrompt())   // think

&#x20;       val toolCall = plan.extractToolCall()              // parse JSON action

&#x20;       if (toolCall == null) return AgentResult.Done(plan.text)

&#x20;       val obs      = toolRegistry.invoke(toolCall)       // act

&#x20;       session.append(plan, obs)                          // observe

&#x20;       telegram.sendTyping()

&#x20;       telegram.send(obs.summary)                         // stream to chat

&#x20;   }

&#x20;   return AgentResult.BudgetExceeded

}

```



\*\*LLM Router\*\* selects between:

\- Local: `llama.cpp` via JNI or `llama-android`, targeting a 3–4B Q4 model on-device

\- Remote: Anthropic / OpenAI API over OkHttp when WiFi is available or the task demands it



\*\*Tool Dispatcher\*\* deserialises the model's JSON tool-call output against a registry of `ToolSpec` objects (name, description, JSON Schema params) and dispatches to the correct handler.



\---



\## Layer 3 — Tool registry



Each tool implements `Tool`:



```kotlin

interface Tool {

&#x20;   val spec: ToolSpec          // name + JSON schema shown to the LLM

&#x20;   suspend fun invoke(params: JsonObject): ToolResult

}

```



| Tool | Implementation | Notes |

|---|---|---|

| `bash` | `ProcessBuilder` into Termux shell or `/system/bin/sh` | Timeout + stdout/stderr cap |

| `git` | JGit or shell wrapper | clone, commit, push, diff, log |

| `search` | DuckDuckGo Instant Answer API or SerpApi | Returns top-N snippets |

| `memory\_read/write` | Room + optional on-device embeddings (ONNX) | Persists facts across sessions |

| `file\_read/write` | Storage Access Framework | Scoped to app sandbox or granted URIs |

| `http` | OkHttp | GET/POST with auth headers |

| `notify` | `NotificationManager` | Alerts you when a long task completes |

| `android\_info` | `Build`, `BatteryManager`, `ConnectivityManager` | System introspection |



All tools report a `ToolResult(ok: Boolean, text: String, artifacts: List<Uri>)`. File artifacts are sent back to Telegram as documents.



\---



\## Layer 4 — Android persistence



\*\*Why a Foreground Service?\*\* Android aggressively kills background processes. A foreground service with a persistent notification keeps the process alive. Pair it with `REQUEST\_IGNORE\_BATTERY\_OPTIMIZATIONS` so Doze doesn't freeze the coroutine.



```kotlin

class AgentService : Service() {

&#x20;   private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)



&#x20;   override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {

&#x20;       startForeground(NOTIF\_ID, buildNotification("Agent running"))

&#x20;       scope.launch { telegramBot.startPolling() }

&#x20;       return START\_STICKY   // restart if killed

&#x20;   }

}

```



\*\*WorkManager\*\* handles deferred work (nightly memory consolidation, model downloads, log rotation) using `CoroutineWorker` with exponential backoff.



A `BootReceiver` (`BOOT\_COMPLETED`) relaunch the service after device restart.



\---



\## Project structure



```

app/

├── control/

│   ├── TelegramBotService.kt

│   ├── AuthGate.kt

│   └── SessionManager.kt

├── core/

│   ├── AgentLoopService.kt

│   ├── LlmRouter.kt

│   └── ToolDispatcher.kt

├── tools/

│   ├── BashTool.kt

│   ├── GitTool.kt

│   ├── SearchTool.kt

│   ├── MemoryTool.kt

│   ├── FileTool.kt

│   ├── HttpTool.kt

│   └── NotifyTool.kt

├── memory/

│   ├── MemoryDatabase.kt   (Room)

│   └── EmbeddingStore.kt   (ONNX optional)

├── service/

│   ├── AgentForegroundService.kt

│   └── BootReceiver.kt

└── ui/

&#x20;   └── MainActivity.kt     (minimal: start/stop toggle only)

```



\---



\## Key dependencies (build.gradle)



```kotlin

// Telegram

implementation("io.github.kotlin-telegram-bot.kotlin-telegram-bot:telegram:6.1.0")



// LLM (remote)

implementation("com.squareup.okhttp3:okhttp:4.12.0")

implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.0")



// Git

implementation("org.eclipse.jgit:org.eclipse.jgit:6.9.0.202403050737-r")



// Room (memory)

implementation("androidx.room:room-runtime:2.6.1")

ksp("androidx.room:room-compiler:2.6.1")

implementation("androidx.room:room-ktx:2.6.1")



// WorkManager

implementation("androidx.work:work-runtime-ktx:2.9.0")



// ONNX (optional embeddings)

implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")

```



\---



\## Required permissions (AndroidManifest.xml)



```xml

<uses-permission android:name="android.permission.INTERNET"/>

<uses-permission android:name="android.permission.FOREGROUND\_SERVICE"/>

<uses-permission android:name="android.permission.FOREGROUND\_SERVICE\_DATA\_SYNC"/>

<uses-permission android:name="android.permission.RECEIVE\_BOOT\_COMPLETED"/>

<uses-permission android:name="android.permission.REQUEST\_IGNORE\_BATTERY\_OPTIMIZATIONS"/>

<uses-permission android:name="android.permission.POST\_NOTIFICATIONS"/>

<uses-permission android:name="android.permission.READ\_EXTERNAL\_STORAGE"/>

<uses-permission android:name="android.permission.WRITE\_EXTERNAL\_STORAGE"/>

```



\---



\## Phased build plan



\*\*Phase 1 — Skeleton (week 1).\*\* Foreground service + Telegram polling + echo bot. Confirm the service survives screen-off and Doze.



\*\*Phase 2 — Single-turn LLM (week 1–2).\*\* Wire in OkHttp → Anthropic API. Send a message, get a response, stream it back to Telegram.



\*\*Phase 3 — Tool loop (week 2–3).\*\* Implement `bash` and `search` tools. Test ReAct loop on 3–5 step tasks. Add budget / `/stop` interrupt.



\*\*Phase 4 — Memory + Git (week 3–4).\*\* Room-backed memory tool. JGit integration. Test a task that reads a repo, edits a file, and commits.



\*\*Phase 5 — Local model (week 4–5).\*\* Integrate `llama-android` or llama.cpp JNI. Add LLM Router that falls back to remote when on-device response quality is insufficient.



\*\*Phase 6 — Hardening.\*\* Battery exemption request flow, BootReceiver, log rotation via WorkManager, tool sandboxing (timeout + output cap), security audit of AuthGate.



\---



\## The hardest parts



\*\*Battery / process death\*\* — test on real hardware with battery saver enabled from day one. The emulator lies about Doze behaviour.



\*\*Bash sandboxing\*\* — without root, `ProcessBuilder` runs as the app user. Commands are scoped to the app sandbox unless you're on Termux with its own filesystem. Add a hard timeout (e.g. 30s) and stdout cap (e.g. 64KB) to prevent runaway processes.



\*\*Context window management\*\* — trimming session history is non-trivial. A sliding window drops early context; a summary-based approach (ask the LLM to compress older turns) is better for long tasks.



\*\*Streaming to Telegram\*\* — Telegram's `editMessageText` lets you stream partial output by periodically updating a message as the agent produces observations. Rate-limit to \~1 edit/second to stay within Telegram's API limits.

