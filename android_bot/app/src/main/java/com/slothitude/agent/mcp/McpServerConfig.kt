package com.slothitude.agent.mcp

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject

/**
 * Configuration for a single MCP server connection.
 */
data class McpServerConfig(
    val name: String,
    val transport: String,          // "sse" or "stdio"
    val url: String? = null,         // for SSE transport
    val command: String? = null,     // for stdio transport
    val args: List<String> = emptyList(),
    val enabled: Boolean = true
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("name", name)
        put("transport", transport)
        if (url != null) put("url", url)
        if (command != null) put("command", command)
        if (args.isNotEmpty()) put("args", JSONArray(args))
        put("enabled", enabled)
    }

    companion object {
        fun fromJson(json: JSONObject): McpServerConfig = McpServerConfig(
            name = json.getString("name"),
            transport = json.getString("transport"),
            url = json.optString("url", null),
            command = json.optString("command", null),
            args = (0 until (json.optJSONArray("args")?.length() ?: 0)).map {
                json.getJSONArray("args").getString(it)
            },
            enabled = json.optBoolean("enabled", true)
        )
    }
}

/**
 * Manages MCP server configurations. Stores in SharedPreferences.
 */
class McpServerManager(context: Context) {
    companion object {
        private const val TAG = "McpServerManager"
        private const val PREFS_NAME = "agent_mcp_servers"
        private const val KEY_SERVERS = "mcp_servers_json"
    }

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun getServers(): List<McpServerConfig> {
        val raw = prefs.getString(KEY_SERVERS, null) ?: return emptyList()
        val arr = JSONArray(raw)
        return (0 until arr.length()).map { McpServerConfig.fromJson(arr.getJSONObject(it)) }
    }

    fun addServer(config: McpServerConfig) {
        val servers = getServers().toMutableList()
        servers.removeAll { it.name == config.name }
        servers.add(config)
        saveServers(servers)
        Log.i(TAG, "Added MCP server: ${config.name} (${config.transport})")
    }

    fun removeServer(name: String) {
        val servers = getServers().toMutableList()
        servers.removeAll { it.name == name }
        saveServers(servers)
        Log.i(TAG, "Removed MCP server: $name")
    }

    fun enableServer(name: String, enabled: Boolean) {
        val servers = getServers().map {
            if (it.name == name) it.copy(enabled = enabled) else it
        }
        saveServers(servers)
    }

    private fun saveServers(servers: List<McpServerConfig>) {
        val arr = JSONArray()
        for (s in servers) arr.put(s.toJson())
        prefs.edit().putString(KEY_SERVERS, arr.toString()).apply()
    }
}
