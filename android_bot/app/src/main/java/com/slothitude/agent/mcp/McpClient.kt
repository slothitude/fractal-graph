package com.slothitude.agent.mcp

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

/**
 * MCP (Model Context Protocol) client.
 * Implements JSON-RPC 2.0 to communicate with MCP servers.
 */
class McpClient(private val config: McpServerConfig) {
    companion object {
        private const val TAG = "McpClient"
    }

    private val transport: McpTransport = when (config.transport) {
        "sse" -> McpTransportSse(config.url!!)
        "stdio" -> McpTransportStdio(config.command!!, config.args)
        else -> throw IllegalArgumentException("Unknown transport: ${config.transport}")
    }

    private var initialized = false
    private var serverCapabilities: JSONObject? = null

    /**
     * Initialize connection to the MCP server.
     */
    suspend fun initialize(): McpServerCapabilities {
        if (initialized) return McpServerCapabilities(serverCapabilities)

        val initRequest = JSONObject().apply {
            put("protocolVersion", "2024-11-05")
            put("capabilities", JSONObject())
            put("clientInfo", JSONObject().apply {
                put("name", "android-agent")
                put("version", "0.1.0")
            })
        }

        return withContext(Dispatchers.IO) {
            val result = sendRequest("initialize", initRequest)

            // Send initialized notification (no response expected)
            val initParams = JSONObject()
            sendNotification("notifications/initialized", initParams)

            initialized = true
            serverCapabilities = result.optJSONObject("capabilities")

            val caps = McpServerCapabilities(serverCapabilities)
            Log.i(TAG, "Connected to ${config.name}: tools=${caps.supportsTools}")

            // Auto-discover tools
            if (caps.supportsTools) {
                listTools()
            }

            caps
        }
    }

    /**
     * List all tools from the server.
     */
    suspend fun listTools(): List<McpToolDef> = withContext(Dispatchers.IO) {
        if (!initialized) throw IllegalStateException("Not initialized — call initialize() first")

        val result = sendRequest("tools/list", null)
        val tools = result.optJSONArray("tools") ?: JSONArray()

        (0 until tools.length()).map { i ->
            val toolObj = tools.getJSONObject(i)
            val name = toolObj.getString("name")
            val description = toolObj.optString("description", "")
            val inputSchema = toolObj.optJSONObject("inputSchema") ?: JSONObject()

            McpToolDef(name = name, description = description, inputSchema = inputSchema)
        }.also {
            Log.i(TAG, "${config.name}: ${it.size} tools available")
        }
    }

    /**
     * Call a tool on the server.
     */
    suspend fun callTool(name: String, arguments: JSONObject): McpToolResult = withContext(Dispatchers.IO) {
        if (!initialized) throw IllegalStateException("Not initialized")

        val result = sendRequest("tools/call", JSONObject(mapOf(
            "name" to name,
            "arguments" to arguments
        )))

        val content = result.optJSONArray("content") ?: JSONArray()
        val texts = (0 until content.length()).mapNotNull { i ->
            val item = content.getJSONObject(i)
            if (item.optString("type") == "text") item.optString("text") else null
        }

        val isError = result.optBoolean("isError", false)
        McpToolResult(texts = texts, isError = isError)
    }

    /**
     * Disconnect from the server.
     */
    suspend fun disconnect() {
        if (!initialized) return
        withContext(Dispatchers.IO) {
            try {
                transport.close()
                initialized = false
                Log.i(TAG, "Disconnected from ${config.name}")
            } catch (e: Exception) {
                Log.w(TAG, "Disconnect error", e)
            }
        }
    }

    fun isConnected(): Boolean = initialized

    // --- JSON-RPC ---

    private suspend fun sendRequest(method: String, params: JSONObject?): JSONObject {
        return transport.send(method, params)
    }

    private suspend fun sendNotification(method: String, params: JSONObject) {
        transport.sendNotification(method, params)
    }
}

/**
 * Server capabilities as reported by the MCP server.
 */
data class McpServerCapabilities(val raw: JSONObject?) {
    val supportsTools: Boolean = raw?.optJSONObject("tools") != null
    val supportsResources: Boolean = raw?.optJSONObject("resources") != null
    val supportsPrompts: Boolean = raw?.optJSONObject("prompts") != null
}

/**
 * Tool definition from an MCP server.
 */
data class McpToolDef(
    val name: String,
    val description: String,
    val inputSchema: JSONObject
)

/**
 * Result from a tool call.
 */
data class McpToolResult(
    val texts: List<String>,
    val isError: Boolean = false
) {
    fun combinedText(): String = texts.joinToString("\n")
}
