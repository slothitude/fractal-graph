package com.slothitude.agent.mcp

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.sse.EventSource
import okhttp3.sse.EventSourceListener
import okhttp3.sse.EventSources
import org.json.JSONObject
import java.io.*
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/**
 * Transport interface for MCP communication.
 */
interface McpTransport {
    suspend fun send(method: String, params: JSONObject?): JSONObject
    suspend fun sendNotification(method: String, params: JSONObject)
    suspend fun close()
}

/**
 * HTTP SSE transport for MCP servers (e.g. remote MCP servers).
 * POST for requests, SSE for server-to-client notifications.
 */
class McpTransportSse(private val url: String) : McpTransport {
    companion object {
        private const val TAG = "McpTransportSSE"
        private val JSON = "application/json".toMediaType()
    }

    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .build()

    private var sessionId: String? = null
    private var eventSource: EventSource? = null

    override suspend fun send(method: String, params: JSONObject?): JSONObject = withContext(Dispatchers.IO) {
        val id = System.currentTimeMillis().toInt()
        val requestJson = JSONObject().apply {
            put("jsonrpc", "2.0")
            put("id", id)
            put("method", method)
            if (params != null) put("params", params)
        }

        val builder = Request.Builder()
            .url(url)
            .post(requestJson.toString().toRequestBody(JSON))
            .addHeader("Content-Type", "application/json")
            .addHeader("Accept", "application/json, text/event-stream")

        sessionId?.let { builder.addHeader("Mcp-Session-Id", it) }

        val request = builder.build()
        val response = client.newCall(request).execute()

        // Extract session ID
        val sid = response.header("Mcp-Session-Id")
        if (sid != null) sessionId = sid

        val contentType = response.header("Content-Type") ?: ""

        if (contentType.contains("text/event-stream")) {
            readSseResponse(response)
        } else {
            val body = response.body?.string() ?: throw IOException("Empty response")
            parseJsonRpcResponse(body)
        }
    }

    override suspend fun sendNotification(method: String, params: JSONObject) {
        withContext(Dispatchers.IO) {
            val requestJson = JSONObject().apply {
                put("jsonrpc", "2.0")
                put("method", method)
                if (params != null) put("params", params)
            }

            val builder = Request.Builder()
                .url(url)
                .post(requestJson.toString().toRequestBody(JSON))
                .addHeader("Content-Type", "application/json")

            sessionId?.let { builder.addHeader("Mcp-Session-Id", it) }

            val request = builder.build()
            client.newCall(request).execute().close()
        }
    }

    override suspend fun close() {
        eventSource?.cancel()
        sessionId = null
    }

    private fun readSseResponse(response: Response): JSONObject {
        val resultRef = AtomicReference<JSONObject>()
        val latch = CountDownLatch(1)

        eventSource = EventSources.createFactory(client).newEventSource(
            Request.Builder().url(url).build(),
            object : EventSourceListener() {
                override fun onEvent(eventSource: EventSource, id: String?, type: String?, data: String) {
                    try {
                        if (data == "[DONE]") {
                            latch.countDown()
                            return
                        }
                        val json = JSONObject(data)
                        if (json.has("result")) {
                            resultRef.set(json.getJSONObject("result"))
                        } else if (json.has("result") == false && json.has("error")) {
                            resultRef.set(json)  // error response
                        } else if (json.has("id") && json.has("result") == false) {
                            // This is our response
                            resultRef.set(json)
                        }
                        latch.countDown()
                    } catch (e: Exception) {
                        Log.w(TAG, "Error parsing SSE", e)
                        latch.countDown()
                    }
                }
                override fun onFailure(eventSource: EventSource, t: Throwable?, response: Response?) {
                    latch.countDown()
                }
            }
        )

        latch.await(30, TimeUnit.SECONDS)
        val result = resultRef.get()
        if (result != null) return result
        throw IOException("Timeout waiting for SSE response")
    }

    private fun parseJsonRpcResponse(body: String): JSONObject {
        val json = JSONObject(body)
        if (json.has("error")) {
            throw IOException("JSON-RPC error: ${json.getJSONObject("error")}")
        }
        return json.getJSONObject("result")
    }
}

/**
 * Stdio transport for local MCP servers (subprocess).
 * Writes JSON-RPC messages to stdin, reads responses from stdout.
 */
class McpTransportStdio(
    private val command: String,
    private val args: List<String> = emptyList()
) : McpTransport {
    companion object {
        private const val TAG = "McpTransportStdio"
    }

    private var process: Process? = null
    private var stdinWriter: BufferedWriter? = null
    private var stdoutReader: BufferedReader? = null

    override suspend fun send(method: String, params: JSONObject?): JSONObject = withContext(Dispatchers.IO) {
        ensureProcess()

        val id = System.currentTimeMillis().toInt()
        val requestJson = JSONObject().apply {
            put("jsonrpc", "2.0")
            put("id", id)
            put("method", method)
            if (params != null) put("params", params)
        }

        stdinWriter?.write(requestJson.toString())
        stdinWriter?.write("\n")
        stdinWriter?.flush()

        // Read response (JSON-RPC responses have "id" field)
        readResponse(id)
    }

    override suspend fun sendNotification(method: String, params: JSONObject) {
        withContext(Dispatchers.IO) {
            ensureProcess()

            val requestJson = JSONObject().apply {
                put("jsonrpc", "2.0")
                put("method", method)
                if (params != null) put("params", params)
            }

            stdinWriter?.write(requestJson.toString())
            stdinWriter?.write("\n")
            stdinWriter?.flush()
        }
    }

    override suspend fun close() {
        withContext(Dispatchers.IO) {
            try {
                stdinWriter?.close()
                stdoutReader?.close()
                process?.destroyForcibly()
            } catch (e: Exception) {
                Log.w(TAG, "Error closing process", e)
            }
            process = null
        }
    }

    private fun ensureProcess() {
        if (process?.isAlive == true) return

        Log.i(TAG, "Starting process: $command ${args.joinToString(" ")}")
        val pb = ProcessBuilder(command, *args.toTypedArray())
        pb.redirectErrorStream(true)
        process = pb.start()
        stdinWriter = BufferedWriter(OutputStreamWriter(process!!.outputStream))
        stdoutReader = BufferedReader(InputStreamReader(process!!.inputStream))
    }

    private fun readResponse(expectedId: Int): JSONObject {
        val reader = stdoutReader ?: throw IOException("Process not running")

        while (true) {
            val line = reader.readLine() ?: throw IOException("Process exited unexpectedly")
            if (line.isBlank()) continue

            try {
                val json = JSONObject(line)
                // Skip notifications (no "id")
                if (!json.has("id")) continue
                // Match our request ID
                if (json.optInt("id") == expectedId) {
                    if (json.has("error")) {
                        throw IOException("JSON-RPC error: ${json.getJSONObject("error")}")
                    }
                    return json.getJSONObject("result")
                }
            } catch (e: Exception) {
                // Not JSON or not our response — skip
            }
        }
    }
}
