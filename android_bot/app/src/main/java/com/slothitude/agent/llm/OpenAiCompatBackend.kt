package com.slothitude.agent.llm

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.sse.EventSource
import okhttp3.sse.EventSourceListener
import okhttp3.sse.EventSources
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/**
 * OpenAI-compatible LLM client. Works with any provider that speaks /v1/chat/completions:
 * OpenAI, Anthropic (OpenAI-compat endpoint), Ollama, LiteLLM, vLLM, Groq, Together, etc.
 */
class OpenAiCompatBackend(
    override val name: String,
    val baseUrl: String,          // e.g. "https://api.openai.com" (no /v1)
    val apiKey: String,
    override val model: String,
    override val supportsTools: Boolean = true,
    override val supportsStreaming: Boolean = true,
    private val timeoutSeconds: Long = 120
) : LlmBackend {

    companion object {
        private const val TAG = "OpenAiCompat"
        private val JSON = "application/json; charset=utf-8".toMediaType()
    }

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(timeoutSeconds, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    private val apiBase = run {
        val trimmed = baseUrl.trimEnd('/')
        // If baseUrl already ends with a version path like /v1, /v4, etc., use as-is
        if (Regex("/v\\d+$").containsMatchIn(trimmed)) trimmed
        else "$trimmed/v1"
    }

    override suspend fun complete(
        messages: List<LlmMessage>,
        tools: List<ToolDef>?,
        maxTokens: Int,
        temperature: Float
    ): LlmResponse = withContext(Dispatchers.IO) {
        val body = buildRequestBody(messages, tools, maxTokens, temperature, stream = false)
        val request = buildRequest(body)
        val response = client.newCall(request).execute()
        val responseBody = response.body?.string() ?: throw IOException("Empty response body")

        if (!response.isSuccessful) {
            Log.e(TAG, "HTTP ${response.code}: $responseBody")
            throw IOException("LLM error ${response.code}: $responseBody")
        }

        parseResponse(JSONObject(responseBody))
    }

    override suspend fun completeStreaming(
        messages: List<LlmMessage>,
        tools: List<ToolDef>?,
        maxTokens: Int,
        temperature: Float,
        callback: StreamCallback
    ) = withContext(Dispatchers.IO) {
        val body = buildRequestBody(messages, tools, maxTokens, temperature, stream = true)
        val request = buildRequest(body)

        val fullText = StringBuilder()
        val toolCalls = mutableMapOf<Int, MutableMap<String, Any?>>()
        val finishRef = AtomicReference<String>("stop")

        val eventSource = EventSources.createFactory(client).newEventSource(request, object : EventSourceListener() {

            override fun onEvent(eventSource: EventSource, id: String?, type: String?, data: String) {
                if (data == "[DONE]") {
                    callback.onComplete(fullText.toString(), collectToolCalls(toolCalls), finishRef.get())
                    eventSource.cancel()
                    return
                }
                try {
                    val json = JSONObject(data)
                    val choices = json.optJSONArray("choices") ?: return

                    for (i in 0 until choices.length()) {
                        val choice = choices.getJSONObject(i)
                        val delta = choice.optJSONObject("delta") ?: continue

                        // Content token
                        val token = delta.optString("content", null)
                        if (token != null) {
                            fullText.append(token)
                            callback.onToken(token)
                        }

                        // Tool call delta
                        val toolCallDelta = delta.optJSONArray("tool_calls")
                        if (toolCallDelta != null) {
                            for (j in 0 until toolCallDelta.length()) {
                                val tc = toolCallDelta.getJSONObject(j)
                                val idx = tc.getInt("index")
                                val entry = toolCalls.getOrPut(idx) { mutableMapOf("id" to null, "name" to null, "arguments" to StringBuilder()) }

                                if (tc.has("id")) entry["id"] = tc.getString("id")
                                if (tc.has("function")) {
                                    val fn = tc.getJSONObject("function")
                                    if (fn.has("name")) entry["name"] = fn.getString("name")
                                    if (fn.has("arguments")) {
                                        val sb = entry["arguments"] as StringBuilder
                                        sb.append(fn.getString("arguments"))
                                    }
                                }
                            }
                        }

                        // Finish reason
                        if (choice.has("finish_reason") && choice.getString("finish_reason") != "null") {
                            finishRef.set(choice.getString("finish_reason"))
                        }
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Error parsing SSE chunk", e)
                }
            }

            override fun onFailure(eventSource: EventSource, t: Throwable?, response: Response?) {
                val msg = t?.message ?: response?.body?.string() ?: "Unknown error"
                Log.e(TAG, "SSE failure: $msg")
                callback.onError(t ?: IOException(msg))
            }

            override fun onClosed(eventSource: EventSource) {
                // Stream ended normally
            }

            override fun onOpen(eventSource: EventSource, response: Response) {
                Log.d(TAG, "SSE stream opened from $apiBase")
            }
        })

        // Block until stream completes or errors
        // The callback pattern means we don't need to return anything here
        // The caller uses the callback to get results
    }

    // --- Internal helpers ---

    private fun buildRequest(body: RequestBody): Request {
        return Request.Builder()
            .url("$apiBase/chat/completions")
            .addHeader("Authorization", "Bearer $apiKey")
            .addHeader("Content-Type", "application/json")
            .post(body)
            .build()
    }

    private fun buildRequestBody(
        messages: List<LlmMessage>,
        tools: List<ToolDef>?,
        maxTokens: Int,
        temperature: Float,
        stream: Boolean
    ): RequestBody {
        val json = JSONObject().apply {
            put("model", model)
            put("max_tokens", maxTokens)
            put("temperature", temperature)
            put("stream", stream)

            // Messages
            val msgArray = JSONArray()
            for (msg in messages) {
                val msgObj = JSONObject().apply {
                    put("role", msg.role)
                    if (msg.role == "tool") {
                        put("tool_call_id", msg.toolCallId)
                    }
                    if (msg.role == "assistant" && msg.toolCalls != null && msg.toolCalls.isNotEmpty()) {
                        // Assistant message with tool calls — no content, just tool_calls array
                        val tcArray = JSONArray()
                        for (tc in msg.toolCalls) {
                            tcArray.put(JSONObject().apply {
                                put("id", tc.id)
                                put("type", "function")
                                put("function", JSONObject().apply {
                                    put("name", tc.name)
                                    put("arguments", tc.arguments.toString())
                                })
                            })
                        }
                        put("tool_calls", tcArray)
                    } else {
                        put("content", msg.content)
                    }
                }
                msgArray.put(msgObj)
            }
            put("messages", msgArray)

            // Tools
            if (tools != null && supportsTools) {
                val toolsArray = JSONArray()
                for (tool in tools) {
                    toolsArray.put(JSONObject().apply {
                        put("type", "function")
                        put("function", JSONObject().apply {
                            put("name", tool.name)
                            put("description", tool.description)
                            put("parameters", tool.parameters)
                        })
                    })
                }
                put("tools", toolsArray)
            }
        }
        return json.toString().toRequestBody(JSON)
    }

    private fun parseResponse(json: JSONObject): LlmResponse {
        val choice = json.getJSONArray("choices").getJSONObject(0)
        val message = choice.getJSONObject("message")
        val content = message.optString("content", "")
        val stopReason = choice.optString("finish_reason", "stop")

        val toolCalls = mutableListOf<ToolCall>()
        if (message.has("tool_calls")) {
            val tcArray = message.getJSONArray("tool_calls")
            for (i in 0 until tcArray.length()) {
                val tc = tcArray.getJSONObject(i)
                val fn = tc.getJSONObject("function")
                toolCalls.add(ToolCall(
                    id = tc.getString("id"),
                    name = fn.getString("name"),
                    arguments = JSONObject(fn.getString("arguments"))
                ))
            }
        }

        var usage: TokenUsage? = null
        if (json.has("usage")) {
            val u = json.getJSONObject("usage")
            usage = TokenUsage(
                promptTokens = u.getInt("prompt_tokens"),
                completionTokens = u.optInt("completion_tokens", 0),
                totalTokens = u.optInt("total_tokens", 0)
            )
        }

        return LlmResponse(
            content = content,
            toolCalls = toolCalls,
            stopReason = stopReason,
            model = json.optString("model", model),
            usage = usage
        )
    }

    private fun collectToolCalls(toolCalls: Map<Int, MutableMap<String, Any?>>): List<ToolCall> {
        return toolCalls.values.mapNotNull { entry ->
            val id = entry["id"] as? String ?: return@mapNotNull null
            val name = entry["name"] as? String ?: return@mapNotNull null
            val argsRaw = entry["arguments"] as? StringBuilder
            val argsJson = try {
                JSONObject(argsRaw?.toString() ?: "{}")
            } catch (e: Exception) {
                JSONObject()  // partial args, will be retried
            }
            ToolCall(id, name, argsJson)
        }.sortedBy { it.id }
    }
}
