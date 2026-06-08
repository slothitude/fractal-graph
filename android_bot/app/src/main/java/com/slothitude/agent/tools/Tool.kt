package com.slothitude.agent.tools

import com.slothitude.agent.llm.ToolDef
import org.json.JSONObject

/**
 * Result returned by a tool invocation.
 */
data class ToolResult(
    val ok: Boolean,
    val text: String,
    val error: String? = null
)

/**
 * Interface for all tools (local and MCP-bridged).
 * Each tool declares a spec for the LLM and an invoke handler.
 */
interface Tool {
    val spec: ToolDef

    suspend fun invoke(params: JSONObject): ToolResult
}
