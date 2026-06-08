package com.slothitude.agent.tools

import android.util.Log
import com.slothitude.agent.llm.ToolDef
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * Execute shell commands via ProcessBuilder.
 * Scoped to app sandbox unless running in Termux.
 */
class BashTool : Tool {
    companion object {
        private const val TAG = "BashTool"
        private const val TIMEOUT_MS = 30_000L
        private const val MAX_OUTPUT_BYTES = 64 * 1024  // 64KB
    }

    override val spec = ToolDef(
        name = "bash",
        description = "Execute a shell command and return stdout+stderr. Commands run as the app user with a 30s timeout and 64KB output cap.",
        parameters = JSONObject("""{
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory (optional)"
                }
            },
            "required": ["command"]
        }""")
    )

    override suspend fun invoke(params: JSONObject): ToolResult = withContext(Dispatchers.IO) {
        val command = params.getString("command")
        if (command == null) return@withContext ToolResult(ok = false, text = "", error = "Missing 'command' param")
        val cwd = params.optString("cwd", null)

        Log.i(TAG, "Executing: $command")
        val startTime = System.currentTimeMillis()

        try {
            val pb = if (cwd != null) {
                ProcessBuilder("/system/bin/sh", "-c", command).directory(java.io.File(cwd))
            } else {
                ProcessBuilder("/system/bin/sh", "-c", command)
            }

            pb.redirectErrorStream(true)
            val process = pb.start()

            val output = StringBuilder()
            val reader = BufferedReader(InputStreamReader(process.inputStream))
            var totalBytes = 0
            var timedOut = false

            while (true) {
                val elapsed = System.currentTimeMillis() - startTime
                if (elapsed > TIMEOUT_MS) {
                    process.destroyForcibly()
                    timedOut = true
                    break
                }

                val line = reader.readLine() ?: break
                val lineBytes = line.toByteArray(Charsets.UTF_8).size
                if (totalBytes + lineBytes > MAX_OUTPUT_BYTES) {
                    output.append("\n... [truncated at ${MAX_OUTPUT_BYTES/1024}KB]")
                    break
                }
                output.append(line).append("\n")
                totalBytes += lineBytes
            }

            if (timedOut) {
                ToolResult(ok = false, text = output.toString(), error = "Timeout after ${TIMEOUT_MS/1000}s")
            } else {
                val exitCode = process.waitFor()
                val text = output.toString().trim()
                val elapsed = System.currentTimeMillis() - startTime

                if (exitCode != 0) {
                    Log.w(TAG, "Exit code $exitCode (${elapsed}ms)")
                    ToolResult(ok = true, text = text, error = "Exit code: $exitCode")
                } else {
                    Log.d(TAG, "Completed in ${elapsed}ms, ${text.length} chars")
                    ToolResult(ok = true, text = text)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Execution error", e)
            ToolResult(ok = false, text = "", error = e.message)
        }
    }
}
