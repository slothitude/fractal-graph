package com.slothitude.agent.mcp

import android.util.Log
import com.slothitude.agent.llm.ToolDef
import com.slothitude.agent.tools.Tool
import com.slothitude.agent.tools.ToolResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject

/**
 * Bridges an MCP tool into the local Tool interface.
 * Wraps an McpClient + McpToolDef so it appears in the ToolRegistry
 * alongside local tools.
 */
class McpToolBridge(
    private val client: McpClient,
    private val mcpDef: McpToolDef
) : Tool {

    override val spec = ToolDef(
        name = mcpDef.name,
        description = "[MCP:${getServerName()}] ${mcpDef.description}",
        parameters = mcpDef.inputSchema
    )

    override suspend fun invoke(params: JSONObject): ToolResult = withContext(Dispatchers.IO) {
        try {
            val result = client.callTool(mcpDef.name, params)
            if (result.isError) {
                ToolResult(ok = false, text = result.combinedText(), error = "MCP tool error")
            } else {
                ToolResult(ok = true, text = result.combinedText())
            }
        } catch (e: Exception) {
            Log.e(TAG, "MCP tool call failed: ${mcpDef.name}", e)
            ToolResult(ok = false, text = "", error = e.message)
        }
    }

    private fun getServerName(): String {
        // Extract from client — we store the config name there
        return "unknown"
    }

    companion object {
        private const val TAG = "McpToolBridge"
    }
}
