package com.slothitude.agent.llm

import org.json.JSONObject

/**
 * A single message in the conversation.
 */
data class LlmMessage(
    val role: String,       // "system", "user", "assistant", "tool"
    val content: String,
    val toolCallId: String? = null,
    val toolCalls: List<ToolCall>? = null
)

/**
 * A tool call parsed from the LLM response.
 */
data class ToolCall(
    val id: String,
    val name: String,
    val arguments: JSONObject
)

/**
 * Tool definition sent to the LLM in the request.
 */
data class ToolDef(
    val name: String,
    val description: String,
    val parameters: JSONObject
)

/**
 * Result of a non-streaming completion.
 */
data class LlmResponse(
    val content: String,
    val toolCalls: List<ToolCall> = emptyList(),
    val stopReason: String,
    val model: String,
    val usage: TokenUsage? = null
)

data class TokenUsage(
    val promptTokens: Int,
    val completionTokens: Int,
    val totalTokens: Int
)

/**
 * Callback for streaming tokens.
 */
interface StreamCallback {
    fun onToken(token: String)
    fun onToolCall(toolCall: ToolCall)
    fun onComplete(fullText: String, toolCalls: List<ToolCall>, stopReason: String)
    fun onError(error: Throwable)
}

/**
 * Interface for any LLM backend (local or remote).
 */
interface LlmBackend {
    val name: String
    val model: String
    val supportsTools: Boolean
    val supportsStreaming: Boolean

    suspend fun complete(
        messages: List<LlmMessage>,
        tools: List<ToolDef>? = null,
        maxTokens: Int = 4096,
        temperature: Float = 0.7f
    ): LlmResponse

    suspend fun completeStreaming(
        messages: List<LlmMessage>,
        tools: List<ToolDef>? = null,
        maxTokens: Int = 4096,
        temperature: Float = 0.7f,
        callback: StreamCallback
    )
}
