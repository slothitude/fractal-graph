package com.slothitude.agent.tools

import com.slothitude.agent.llm.ToolDef
import android.util.Log
import org.json.JSONObject

/**
 * Central registry for all tools (local + MCP-bridged).
 * The LLM receives tool specs from here; the dispatcher calls invoke.
 */
class ToolRegistry {
    companion object {
        private const val TAG = "ToolRegistry"
    }

    private val tools = mutableMapOf<String, Tool>()

    fun register(tool: Tool) {
        tools[tool.spec.name] = tool
        Log.d(TAG, "Registered tool: ${tool.spec.name}")
    }

    fun unregister(name: String) {
        tools.remove(name)
        Log.d(TAG, "Unregistered tool: $name")
    }

    fun get(name: String): Tool? = tools[name]

    fun getAll(): Map<String, Tool> = tools.toMap()

    /**
     * Get tool specs as ToolDef list for the LLM request.
     */
    fun getSpecs(): List<ToolDef> = tools.values.map { it.spec }

    /**
     * Build tool descriptions string for the system prompt.
     */
    fun buildToolDescriptions(): String {
        if (tools.isEmpty()) return ""
        return buildString {
            append("## Available Tools\n\n")
            for ((_, tool) in tools) {
                append("- **${tool.spec.name}**: ${tool.spec.description}\n")
                append("  Parameters: ${tool.spec.parameters.toString(2)}\n\n")
            }
        }
    }

    fun size(): Int = tools.size

    fun clear() {
        tools.clear()
    }
}
