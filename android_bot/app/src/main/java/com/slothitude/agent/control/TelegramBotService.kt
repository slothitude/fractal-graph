package com.slothitude.agent.control

import android.content.Context
import android.util.Log
import com.slothitude.agent.core.AgentLoopService
import com.slothitude.agent.llm.*
import com.slothitude.agent.mcp.*
import com.slothitude.agent.service.AgentForegroundService
import com.slothitude.agent.tools.*
import kotlinx.coroutines.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Telegram long-polling bot. Receives messages via getUpdates,
 * passes through AuthGate, manages sessions, dispatches commands,
 * and routes to the LLM / agent loop.
 *
 * Uses raw Telegram Bot API via OkHttp — no third-party bot library needed.
 */
class TelegramBotService(
    private val appContext: Context
) {
    companion object {
        private const val TAG = "TelegramBot"
        private const val API = "https://api.telegram.org/bot"

        private const val SYSTEM_PROMPT = """You are a helpful agent running on an Android phone.
You can access tools, search the web, read/write files, and execute commands.
Be concise. Respond directly."""

        private const val STREAM_INTERVAL_MS = 800L
        private const val MAX_EDIT_LENGTH = 4096
        private const val POLL_TIMEOUT_S = 60L
        private const val LONG_POLL_TIMEOUT_S = 90L
    }

    private val authGate = AuthGate(appContext)
    private val sessionManager = SessionManager()
    val llmRouter = LlmRouter(appContext)

    // Phase 3: tools + MCP
    val toolRegistry = ToolRegistry()
    private val toolDispatcher = ToolDispatcher(toolRegistry)
    private val mcpServerManager = McpServerManager(appContext)
    private val mcpClients = mutableMapOf<String, McpClient>()

    private var agentLoop: AgentLoopService? = null

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, java.util.concurrent.TimeUnit.SECONDS)
        .readTimeout(LONG_POLL_TIMEOUT_S + 15, java.util.concurrent.TimeUnit.SECONDS)
        .writeTimeout(30, java.util.concurrent.TimeUnit.SECONDS)
        .build()

    private val running = AtomicBoolean(false)
    private var pollJob: Job? = null
    private var offset = 0L  // getUpdates offset

    private val activeJobs = ConcurrentHashMap<Long, Job>()
    private var observationMessageId = ConcurrentHashMap<Long, Int>()

    /** Public accessors for UI config. */
    val authorizedUserIds: Set<Long> get() = authGate.authorizedUserIds
    val botToken: String get() = authGate.botToken

    fun setBotToken(token: String) { authGate.botToken = token }
    fun setWhitelist(ids: Set<Long>) { authGate.authorizedUserIds = ids }

    suspend fun startPolling() {
        authGate.seedDefaults()

        val token = authGate.botToken
        if (token.isBlank()) {
            Log.e(TAG, "No bot token configured")
            return
        }

        if (running.getAndSet(true)) {
            Log.w(TAG, "Already polling")
            return
        }

        registerLocalTools()
        connectMcpServers()

        agentLoop = AgentLoopService(
            sessionManager = sessionManager,
            toolRegistry = toolRegistry,
            dispatcher = toolDispatcher,
            onObservation = { text -> sendObservation(0L, text) },
            onToken = null
        )

        val scope = (appContext as? AgentForegroundService)?.scope ?: return
        pollJob = scope.launch(Dispatchers.IO) {
            Log.i(TAG, "Polling started")
            updateNotification("Agent running")
            try {
                while (running.get() && coroutineContext.isActive) {
                    pollOnce(token)
                }
            } catch (e: CancellationException) {
                Log.i(TAG, "Polling cancelled")
            } catch (e: Exception) {
                Log.e(TAG, "Polling error", e)
                delay(5000)
                if (running.get()) {
                    Log.i(TAG, "Restarting poll loop")
                }
            }
        }
    }

    suspend fun stopPolling() {
        if (!running.getAndSet(false)) return
        agentLoop?.stop()
        pollJob?.cancel()

        for ((_, client) in mcpClients) {
            try { client.disconnect() } catch (_: Exception) {}
        }
        mcpClients.clear()

        Log.i(TAG, "Polling stopped")
    }

    fun isRunning(): Boolean = running.get()

    private fun updateNotification(text: String) {
        (appContext as? AgentForegroundService)?.updateNotification(text)
    }

    // --- Telegram Bot API (raw HTTP) ---

    private fun apiPost(token: String, method: String, params: JSONObject): JSONObject? {
        val url = "$API$token/$method"
        val body = params.toString().toRequestBody("application/json".toMediaType())
        val request = Request.Builder().url(url).post(body).build()
        return try {
            val response = client.newCall(request).execute()
            val bodyStr = response.body?.string() ?: return null
            JSONObject(bodyStr)
        } catch (e: Exception) {
            Log.e(TAG, "API error: $method", e)
            null
        }
    }

    private fun tgRequest(token: String, method: String, params: JSONObject): JSONObject? {
        val url = "$API$token/$method"
        val body = params.toString().toRequestBody("application/json".toMediaType())
        val request = Request.Builder().url(url).post(body).build()
        return try {
            val response = client.newCall(request).execute()
            val bodyStr = response.body?.string() ?: return null
            val json = JSONObject(bodyStr)
            if (!json.optBoolean("ok", false)) {
                Log.e(TAG, "TG API error: $method - ${json.optString("description")}")
            }
            json
        } catch (e: Exception) {
            Log.e(TAG, "API error: $method", e)
            null
        }
    }

    private suspend fun sendMessage(chatId: Long, text: String): Int {
        val token = authGate.botToken
        if (token.isBlank()) return -1
        val params = JSONObject().apply {
            put("chat_id", chatId)
            put("text", text)
            put("disable_web_page_preview", true)
        }
        val result = withContext(Dispatchers.IO) {
            tgRequest(token, "sendMessage", params)
        }
        val msgId = result?.optJSONObject("result")?.optInt("message_id", -1) ?: -1
        Log.d(TAG, "sendMessage chatId=$chatId msgId=$msgId text=${text.take(50)}")
        return msgId
    }

    private suspend fun editMessage(chatId: Long, messageId: Int, text: String) {
        if (messageId < 0) return
        Log.d(TAG, "editMessage chatId=$chatId msgId=$messageId text=${text.take(80)}")
        val token = authGate.botToken
        if (token.isBlank()) return
        val params = JSONObject().apply {
            put("chat_id", chatId)
            put("message_id", messageId)
            put("text", text)
            put("disable_web_page_preview", true)
        }
        withContext(Dispatchers.IO) {
            tgRequest(token, "editMessageText", params)
        }
    }

    private suspend fun sendChatAction(chatId: Long, action: String) {
        val token = authGate.botToken
        if (token.isBlank()) return
        val params = JSONObject().apply {
            put("chat_id", chatId)
            put("action", action)
        }
        withContext(Dispatchers.IO) {
            tgRequest(token, "sendChatAction", params)
        }
    }

    private suspend fun sendMessageAndGetId(chatId: Long, text: String): Int {
        return sendMessage(chatId, text)
    }

    // --- Long polling ---

    private suspend fun pollOnce(token: String) {
        val params = JSONObject().apply {
            put("offset", offset)
            put("timeout", LONG_POLL_TIMEOUT_S)
            put("allowed_updates", JSONArray("[\"message\"]"))
        }

        val url = "$API$token/getUpdates"
        val body = params.toString().toRequestBody("application/json".toMediaType())
        val request = Request.Builder().url(url).post(body).build()

        val response = withContext(Dispatchers.IO) {
            client.newCall(request).execute()
        }

        val bodyStr = response.body?.string() ?: return
        val result = JSONObject(bodyStr)
        val ok = result.optBoolean("ok", false)
        if (!ok) {
            Log.w(TAG, "getUpdates failed: ${result.optString("description")}")
            return
        }

        val updates = result.optJSONArray("result") ?: JSONArray()
        for (i in 0 until updates.length()) {
            val update = updates.getJSONObject(i)
            val updateId = update.optLong("update_id", 0)
            if (updateId > 0) offset = updateId + 1
            try {
                handleUpdate(update)
            } catch (e: Exception) {
                Log.e(TAG, "Error handling update", e)
            }
        }
    }

    // --- Update handling ---

    private fun handleUpdate(update: JSONObject) {
        val message = update.optJSONObject("message") ?: return
        val chat = message.optJSONObject("chat")
        val chatId = chat?.optLong("id", 0) ?: 0
        val from = message.optJSONObject("from") ?: return
        val userId = from.optLong("id", 0)

        if (!authGate.isAuthorized(userId)) {
            Log.w(TAG, "Unauthorized: userId=$userId")
            return
        }

        val text = message.optString("text", "")
        if (text.isBlank()) return

        Log.d(TAG, "handleUpdate: chatId=$chatId userId=$userId text=${text.take(50)}")

        sessionManager.append(chatId, "user", text)

        if (text.startsWith("/")) {
            handleCommand(chatId, text)
            return
        }

        llmChat(chatId, text)
    }

    private fun handleCommand(chatId: Long, text: String) {
        val parts = text.split(" ", limit = 2)
        val cmd = parts[0].lowercase()
        val arg = parts.getOrElse(1) { "" }

        val scope = (appContext as? AgentForegroundService)?.scope ?: return

        when {
            cmd == "/reset" -> {
                activeJobs[chatId]?.cancel()
                activeJobs.remove(chatId)
                observationMessageId.remove(chatId)
                sessionManager.clear(chatId)
                scope.launch { sendMessage(chatId, "Session cleared.") }
            }

            cmd == "/run" -> {
                val task = arg.trim()
                if (task.isBlank()) {
                    scope.launch { sendMessage(chatId, "Usage: /run <task description>") }
                    return
                }
                observationMessageId.remove(chatId)
                runAgentLoop(chatId, task)
                scope.launch { sendMessage(chatId, "Started: $task") }
            }

            cmd == "/stop" -> {
                activeJobs[chatId]?.cancel()
                activeJobs.remove(chatId)
                agentLoop?.stop()
                observationMessageId.remove(chatId)
                scope.launch { sendMessage(chatId, "Stopped.") }
            }

            cmd == "/budget" -> {
                val n = arg.trim().toIntOrNull()
                if (n == null || n < 1) {
                    scope.launch { sendMessage(chatId, "Usage: /budget <n>. Default: 20") }
                    return
                }
                agentLoop?.setBudget(n)
                scope.launch { sendMessage(chatId, "Budget set to $n steps.") }
            }

            cmd == "/tools" -> {
                scope.launch {
                    val tools = toolRegistry.getSpecs()
                    if (tools.isEmpty()) {
                        sendMessage(chatId, "No tools registered.")
                    } else {
                        sendMessage(chatId, "Tools (${toolRegistry.size()}):\n${tools.map { "- ${it.name}: ${it.description}" }.joinToString("\n")}")
                    }
                }
            }

            cmd == "/status" -> scope.launch {
                val backend = llmRouter.selectBackend()
                val provider = if (backend != null) "${backend.name} (${backend.model})" else "none"
                val mcpInfo = mcpClients.entries.joinToString(", ") { "${it.key} (${it.value.isConnected()})" }.ifEmpty { "none" }
                val lines = mutableListOf<String>()
                lines.add("Status: ${if (isRunning()) "running" else "stopped"}")
                lines.add("Provider: $provider")
                lines.add("Tools: ${toolRegistry.size()} (local + MCP)")
                lines.add("MCP: $mcpInfo")
                lines.add("Session: ${sessionManager.size(chatId)} turns")
                lines.add("Auth: ${authGate.authorizedUserIds.joinToString(", ").ifEmpty { "open" }}")
                sendMessage(chatId, lines.joinToString("\n"))
            }

            cmd == "/help" -> scope.launch { sendMessage(chatId, HELP_TEXT) }

            cmd == "/providers" -> scope.launch {
                val providers = llmRouter.getProviders()
                if (providers.isEmpty()) {
                    sendMessage(chatId, "No providers. Use /provider add.")
                } else {
                    val active = llmRouter.getActiveProvider()
                    sendMessage(chatId, providers.map { p ->
                        val marker = if (p.name == active) ">>" else "  "
                        "$marker ${p.name} (${p.model}) @ ${p.baseUrl}"
                    }.joinToString("\n"))
                }
            }

            cmd == "/provider" && arg.startsWith("add ") -> {
                val addParts = arg.removePrefix("add ").trim().split(" ", limit = 4)
                if (addParts.size < 4) {
                    scope.launch { sendMessage(chatId, "Usage: /provider add <name> <url> <key> <model>") }
                    return
                }
                llmRouter.addProvider(LlmProvider(
                    name = addParts[0], baseUrl = addParts[1],
                    apiKey = addParts[2], model = addParts[3]
                ))
                scope.launch { sendMessage(chatId, "Added: ${addParts[0]} (${addParts[3]})") }
            }

            cmd == "/provider" && arg.startsWith("remove ") -> {
                llmRouter.removeProvider(arg.removePrefix("remove ").trim())
                scope.launch { sendMessage(chatId, "Removed.") }
            }

            cmd == "/model" -> {
                val name = arg.trim()
                if (name.isBlank()) {
                    scope.launch { sendMessage(chatId, "Current: ${llmRouter.getActiveProvider()}. Usage: /model <name>") }
                    return
                }
                llmRouter.setActiveProvider(name)
                scope.launch { sendMessage(chatId, "Switched to: $name") }
            }

            cmd == "/mcp" && arg.isBlank() -> scope.launch { handleMcpList(chatId) }
            cmd == "/mcp" && arg.startsWith("add ") -> scope.launch { handleMcpAdd(chatId, arg.removePrefix("add ").trim()) }
            cmd == "/mcp" && arg.startsWith("remove ") -> scope.launch { handleMcpRemove(chatId, arg.removePrefix("remove ").trim()) }
            cmd == "/mcp" && arg.startsWith("connect ") -> handleMcpConnect(chatId, arg.removePrefix("connect ").trim())

            else -> scope.launch { sendMessage(chatId, "Unknown: $cmd. Type /help.") }
        }
    }

    // --- MCP commands ---

    private suspend fun handleMcpList(chatId: Long) {
        val servers = mcpServerManager.getServers()
        if (servers.isEmpty()) {
            sendMessage(chatId, "No MCP servers. Use /mcp add.")
            return
        }
        val connected = mcpClients.map { it.key }.toSet()
        sendMessage(chatId, servers.map { s ->
            val status = if (s.name in connected) "connected" else "disconnected"
            "${s.name} (${s.transport}) [$status]"
        }.joinToString("\n"))
    }

    private suspend fun handleMcpAdd(chatId: Long, arg: String) {
        val addParts = arg.trim().split(" ", limit = 2)
        if (addParts.size < 2) {
            sendMessage(chatId, "Usage: /mcp add <name> <url>")
            return
        }
        val name = addParts[0]
        val endpoint = addParts[1]
        val config = if (endpoint.startsWith("http")) {
            McpServerConfig(name = name, transport = "sse", url = endpoint)
        } else {
            McpServerConfig(name = name, transport = "stdio", command = endpoint)
        }
        mcpServerManager.addServer(config)
        sendMessage(chatId, "Added MCP: $name (${config.transport}). Use /mcp connect $name.")
    }

    private suspend fun handleMcpRemove(chatId: Long, name: String) {
        mcpClients[name]?.let { try { it.disconnect() } catch (_: Exception) {} }
        mcpClients.remove(name)
        mcpServerManager.removeServer(name)
        sendMessage(chatId, "Removed MCP: $name")
    }

    private fun handleMcpConnect(chatId: Long, name: String) {
        mcpClients[name]?.let { runBlocking { try { it.disconnect() } catch (_: Exception) {} } }
        mcpClients.remove(name)

        val config = mcpServerManager.getServers().find { it.name == name }
        if (config == null) {
            val scope = (appContext as? AgentForegroundService)?.scope ?: return
            scope.launch { sendMessage(chatId, "Unknown MCP: $name") }
            return
        }

        val scope = (appContext as? AgentForegroundService)?.scope ?: return
        scope.launch {
            sendMessage(chatId, "Connecting to $name...")
            try {
                val client = McpClient(config)
                client.initialize()
                val tools = client.listTools()
                for (mcpTool in tools) {
                    toolRegistry.register(McpToolBridge(client, mcpTool))
                }
                mcpClients[name] = client
                sendMessage(chatId, "Connected: $name (${tools.size} tools)")
            } catch (e: Exception) {
                Log.e(TAG, "MCP connect failed: $name", e)
                sendMessage(chatId, "MCP connect failed: ${e.message}")
            }
        }
    }

    // --- Local tool registration ---

    private fun registerLocalTools() {
        toolRegistry.register(BashTool())
        toolRegistry.register(SearchTool())
        Log.i(TAG, "Registered ${toolRegistry.size()} local tools")
    }

    // --- MCP connection ---

    private suspend fun connectMcpServers() {
        val servers = mcpServerManager.getServers().filter { it.enabled }
        for (serverConfig in servers) {
            try {
                val client = McpClient(serverConfig)
                client.initialize()
                val tools = client.listTools()
                for (mcpTool in tools) {
                    toolRegistry.register(McpToolBridge(client, mcpTool))
                }
                mcpClients[serverConfig.name] = client
                Log.i(TAG, "Connected MCP: ${serverConfig.name} (${tools.size} tools)")
            } catch (e: Exception) {
                Log.e(TAG, "Failed MCP: ${serverConfig.name}", e)
            }
        }
    }

    // --- Single-turn LLM chat ---

    private fun llmChat(chatId: Long, userMessage: String) {
        val scope = (appContext as? AgentForegroundService)?.scope ?: return
        val job = scope.launch {
            val backend = llmRouter.selectBackend() ?: run {
                Log.w(TAG, "No LLM backend for chatId=$chatId")
                sendMessage(chatId, "No LLM provider configured. Use /provider add.")
                return@launch
            }
            Log.d(TAG, "llmChat: chatId=$chatId backend=${backend.name} model=${backend.model} streaming=${backend.supportsStreaming}")

            if (!backend.supportsStreaming) {
                llmChatBlocking(chatId, backend, userMessage)
                return@launch
            }

            sendChatAction(chatId, "typing")
            val placeholder = sendMessageAndGetId(chatId, "Thinking...")
            Log.d(TAG, "llmChat: placeholder=$placeholder for chatId=$chatId")

            val fullText = StringBuilder()

            backend.completeStreaming(
                messages = buildMessages(chatId, userMessage),
                callback = object : StreamCallback {
                    override fun onToken(token: String) { fullText.append(token) }
                    override fun onToolCall(toolCall: ToolCall) {}
                    override fun onComplete(fullResponse: String, toolCalls: List<ToolCall>, stopReason: String) {
                        val text = fullResponse.ifBlank { toolCalls.joinToString("\n") { "Call: ${it.name}(${it.arguments})" } }
                        sessionManager.append(chatId, "assistant", text)
                        runBlocking { editMessage(chatId, placeholder, text.take(MAX_EDIT_LENGTH)) }
                    }
                    override fun onError(error: Throwable) {
                        Log.e(TAG, "LLM streaming error", error)
                        val fallback = llmRouter.selectBackend()
                        if (fallback != null && fallback.name != backend.name) {
                            runBlocking { editMessage(chatId, placeholder, "Retrying with ${fallback.name}...") }
                            scope.launch(Dispatchers.IO) { llmChatBlocking(chatId, fallback, userMessage) }
                        } else {
                            runBlocking { editMessage(chatId, placeholder, "Error: ${error.message}") }
                        }
                    }
                }
            )

            streamLoop(placeholder, chatId, fullText)
        }
        activeJobs[chatId] = job
    }

    private suspend fun streamLoop(placeholder: Int, chatId: Long, fullText: StringBuilder) {
        var lastLength = 0
        val maxIterations = 120  // ~96 seconds max
        var iterations = 0
        while (iterations++ < maxIterations) {
            delay(STREAM_INTERVAL_MS)
            if (fullText.length > lastLength) {
                editMessage(chatId, placeholder, fullText.toString().take(MAX_EDIT_LENGTH))
                lastLength = fullText.length
            }
        }
    }

    private suspend fun llmChatBlocking(chatId: Long, backend: LlmBackend, userMessage: String) {
        sendChatAction(chatId, "typing")
        try {
            val response = backend.complete(buildMessages(chatId, userMessage))
            val text = response.content.ifBlank { response.toolCalls.joinToString("\n") { "Call: ${it.name}" } }
            sessionManager.append(chatId, "assistant", text)
            sendMessage(chatId, text.take(MAX_EDIT_LENGTH))
            response.usage?.let {
                Log.d(TAG, "Tokens: ${it.promptTokens}+${it.completionTokens}=${it.totalTokens}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "LLM error", e)
            sendMessage(chatId, "LLM error: ${e.message}")
        }
    }

    private fun buildMessages(chatId: Long, userMessage: String): List<LlmMessage> {
        val messages = mutableListOf<LlmMessage>()
        messages.add(LlmMessage(role = "system", content = SYSTEM_PROMPT))
        for (turn in sessionManager.getHistory(chatId)) {
            messages.add(LlmMessage(role = turn.role, content = turn.content))
        }
        return messages
    }

    // --- Agent loop ---

    private fun runAgentLoop(chatId: Long, task: String) {
        val scope = (appContext as? AgentForegroundService)?.scope ?: return
        val backend = llmRouter.selectBackend() ?: run {
            runBlocking { sendMessage(chatId, "No LLM provider configured.") }
            return
        }

        activeJobs[chatId]?.cancel()

        val loop = AgentLoopService(
            sessionManager = sessionManager,
            toolRegistry = toolRegistry,
            dispatcher = toolDispatcher,
            onObservation = { text ->
                runBlocking { sendObservation(chatId, text) }
            },
            onToken = null
        )

        activeJobs[chatId] = scope.launch {
            loop.startWithBackend(chatId, task, scope, backend)
            activeJobs.remove(chatId)
        }
    }

    private fun sendObservation(chatId: Long, text: String) {
        val msgId = observationMessageId.getOrPut(chatId) {
            runBlocking { sendMessageAndGetId(chatId, "") }
        }
        if (msgId < 0) {
            runBlocking { sendMessage(chatId, text.take(MAX_EDIT_LENGTH)) }
            return
        }
        runBlocking { editMessage(chatId, msgId, text.take(MAX_EDIT_LENGTH)) }
    }
}

private val HELP_TEXT = """
Commands:
/reset — clear session + stop running task
/status — service, provider, tools, MCP, session info
/help — this message

Agent:
/run <task> — start ReAct agent loop with tools
/stop — interrupt running agent loop
/budget <n> — set max loop steps (default 20)
/tools — list all available tools

LLM:
/providers — list configured providers
/provider add <name> <url> <key> <model>
/provider remove <name>
/model <name> — switch active provider

MCP:
/mcp — list connected MCP servers
/mcp add <name> <url> — add SSE MCP server
/mcp add <name> <command> — add stdio MCP server
/mcp remove <name> — remove MCP server
/mcp connect <name> — reconnect MCP server
""".trimIndent()
