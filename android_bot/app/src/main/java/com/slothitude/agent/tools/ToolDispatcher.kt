package com.slothitude.agent.tools

import android.util.Log
import com.slothitude.agent.llm.ToolCall
import com.slothitude.agent.llm.LlmMessage

/**
 * Dispatches tool calls from the LLM to the appropriate registered tool.
 */
class ToolDispatcher(private val registry: ToolRegistry) {
    companion object {
        private const val TAG = "ToolDispatcher"
    }

    /**
     * Execute a single tool call and return the result as a ToolCallResult.
     */
    suspend fun dispatch(call: ToolCall): ToolCallResult {
        val tool = registry.get(call.name)
        if (tool == null) {
            Log.w(TAG, "Unknown tool: ${call.name}")
            return ToolCallResult(
                toolCallId = call.id,
                toolName = call.name,
                output = "Error: unknown tool '${call.name}'",
                isError = true
            )
        }

        Log.i(TAG, "Dispatching: ${call.name}(${call.arguments})")
        val result = tool.invoke(call.arguments)
        return ToolCallResult(
            toolCallId = call.id,
            toolName = call.name,
            output = if (result.ok) result.text else "Error: ${result.error ?: result.text}",
            isError = !result.ok
        )
    }

    /**
     * Convert a ToolCallResult into an LlmMessage for the conversation history.
     */
    fun resultToMessage(result: ToolCallResult): LlmMessage {
        return LlmMessage(
            role = "tool",
            content = result.output,
            toolCallId = result.toolCallId
        )
    }
}

/**
 * Result of a dispatched tool call.
 */
data class ToolCallResult(
    val toolCallId: String,
    val toolName: String,
    val output: String,
    val isError: Boolean = false
)
