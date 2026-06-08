package com.slothitude.agent.core

import android.util.Log
import com.slothitude.agent.control.SessionManager
import com.slothitude.agent.llm.*
import com.slothitude.agent.tools.*
import kotlinx.coroutines.*
import java.util.concurrent.atomic.AtomicBoolean

/**
 * ReAct agent loop: Plan → Act → Observe → Reflect.
 * Runs as a coroutine inside the foreground service.
 * Budget-limited, cancellable, streams observations to Telegram.
 */
class AgentLoopService(
    private val sessionManager: SessionManager,
    private val toolRegistry: ToolRegistry,
    private val dispatcher: ToolDispatcher,
    private val onObservation: ((String) -> Unit)? = null,
    private val onToken: ((String) -> Unit)? = null
) {
    companion object {
        private const val TAG = "AgentLoop"
        private const val DEFAULT_BUDGET = 20
        private const val STREAM_INTERVAL_MS = 800L
        private const val MAX_EDIT_LENGTH = 4096

        private const val SYSTEM_PROMPT = """You are an autonomous agent running on an Android phone.
You have access to tools. Think step by step.

On each turn:
1. Decide what to do next
2. If you need to use a tool, call it with the right parameters
3. Observe the result
4. Decide the next step or respond to the user

When done, respond directly without calling a tool.
Be concise — no filler text."""
    }

    private val running = AtomicBoolean(false)
    private var job: Job? = null
    private var budget = DEFAULT_BUDGET

    val isRunning: Boolean get() = running.get()

    fun setBudget(n: Int) { budget = if (n > 0) n else DEFAULT_BUDGET }

    fun stop() {
        job?.cancel()
        running.set(false)
        Log.i(TAG, "Agent loop stopped")
    }

    /**
     * Start the agent loop for a task.
     */
    fun start(chatId: Long, userMessage: String, scope: CoroutineScope) {
        if (running.getAndSet(true)) {
            Log.w(TAG, "Already running")
            return
        }

        job = scope.launch {
            try {
                runLoop(chatId, userMessage)
            } catch (e: CancellationException) {
                Log.i(TAG, "Loop cancelled")
                onObservation?.invoke("Stopped.")
            } catch (e: Exception) {
                Log.e(TAG, "Loop error", e)
                onObservation?.invoke("Error: ${e.message}")
            } finally {
                running.set(false)
            }
        }
    }

    private suspend fun runLoop(chatId: Long, userMessage: String) {
        val messages = mutableListOf<LlmMessage>()
        messages.add(LlmMessage(role = "system", content = buildSystemPrompt()))

        // Append session history
        for (turn in sessionManager.getHistory(chatId)) {
            messages.add(LlmMessage(role = turn.role, content = turn.content))
        }

        var steps = 0

        while (steps++ < budget) {
            if (!running.get()) break

            Log.d(TAG, "Step $steps/$budget")
            val backend = selectBackend() ?: break

            // Non-streaming call — rely on onObservation for output
            val response = backend.complete(
                messages = messages,
                tools = toolRegistry.getSpecs(),
                maxTokens = 4096,
                temperature = 0.7f
            )

            // Store assistant response
            if (response.content.isNotBlank()) {
                messages.add(LlmMessage(
                    role = "assistant",
                    content = response.content,
                    toolCalls = response.toolCalls.ifEmpty { null }
                ))
                sessionManager.append(chatId, "assistant", response.content)
            } else if (response.toolCalls.isNotEmpty()) {
                messages.add(LlmMessage(
                    role = "assistant",
                    content = "",
                    toolCalls = response.toolCalls
                ))
                sessionManager.append(chatId, "assistant", response.toolCalls.joinToString(", ") { "Call: ${it.name}" })
            }

            // If no tool calls, we're done — send final response
            if (response.toolCalls.isEmpty()) {
                onObservation?.invoke(response.content.take(MAX_EDIT_LENGTH))
                Log.i(TAG, "Loop complete after $steps steps (no tool calls)")
                return
            }

            // Dispatch each tool call
            for (toolCall in response.toolCalls) {
                if (!running.get()) break
                Log.i(TAG, "Tool call: ${toolCall.name}(${toolCall.arguments})")

                val result = dispatcher.dispatch(toolCall)
                val obsMsg = "[${result.toolName}] ${result.output.take(1024)}"
                onObservation?.invoke(obsMsg)

                // Add tool result to messages
                messages.add(dispatcher.resultToMessage(result))
                sessionManager.append(chatId, "tool", obsMsg)
            }

            if (response.stopReason == "end_turn" || response.stopReason == "stop") {
                // Check if LLM responded after tool results
            }

            response.usage?.let {
                Log.d(TAG, "Tokens: ${it.promptTokens}+${it.completionTokens}=${it.totalTokens}")
            }
        }

        if (steps > budget) {
            onObservation?.invoke("Budget exceeded ($budget steps). Use /budget to increase.")
            Log.w(TAG, "Budget exceeded at $budget steps")
        }
    }

    private fun buildSystemPrompt(): String {
        return buildString {
            append(SYSTEM_PROMPT)
            append("\n\n")
            append(toolRegistry.buildToolDescriptions())
        }
    }

    private suspend fun selectBackend(): LlmBackend? {
        // The router is injected by the caller — this is a placeholder
        // TelegramBotService will pass the router's selectBackend result
        return null
    }

    /**
     * Start the loop with an explicit backend (from LlmRouter).
     */
    fun startWithBackend(chatId: Long, userMessage: String, scope: CoroutineScope, backend: LlmBackend) {
        if (running.getAndSet(true)) {
            Log.w(TAG, "Already running")
            return
        }

        job = scope.launch {
            try {
                runLoopWithBackend(chatId, userMessage, backend)
            } catch (e: CancellationException) {
                Log.i(TAG, "Loop cancelled")
                onObservation?.invoke("Stopped.")
            } catch (e: Exception) {
                Log.e(TAG, "Loop error", e)
                onObservation?.invoke("Error: ${e.message}")
            } finally {
                running.set(false)
            }
        }
    }

    private suspend fun runLoopWithBackend(chatId: Long, userMessage: String, backend: LlmBackend) {
        val messages = mutableListOf<LlmMessage>()
        messages.add(LlmMessage(role = "system", content = buildSystemPrompt()))

        for (turn in sessionManager.getHistory(chatId)) {
            messages.add(LlmMessage(role = turn.role, content = turn.content))
        }

        var steps = 0

        while (steps++ < budget) {
            if (!running.get()) break

            Log.d(TAG, "Step $steps/$budget")

            val response = try {
                backend.complete(
                    messages = messages,
                    tools = toolRegistry.getSpecs(),
                    maxTokens = 4096,
                    temperature = 0.7f
                )
            } catch (e: Exception) {
                Log.e(TAG, "LLM error at step $steps", e)
                onObservation?.invoke("LLM error: ${e.message}")
                break
            }

            if (response.content.isNotBlank()) {
                messages.add(LlmMessage(
                    role = "assistant",
                    content = response.content,
                    toolCalls = response.toolCalls.ifEmpty { null }
                ))
                sessionManager.append(chatId, "assistant", response.content)
            } else if (response.toolCalls.isNotEmpty()) {
                messages.add(LlmMessage(
                    role = "assistant",
                    content = "",
                    toolCalls = response.toolCalls
                ))
                sessionManager.append(chatId, "assistant", response.toolCalls.joinToString(", ") { "Call: ${it.name}" })
            }

            if (response.toolCalls.isEmpty()) {
                onObservation?.invoke(response.content.take(MAX_EDIT_LENGTH))
                Log.i(TAG, "Loop complete after $steps steps")
                return
            }

            for (toolCall in response.toolCalls) {
                if (!running.get()) break
                val result = dispatcher.dispatch(toolCall)
                val obsMsg = "[${result.toolName}] ${result.output.take(1024)}"
                onObservation?.invoke(obsMsg)
                messages.add(dispatcher.resultToMessage(result))
                sessionManager.append(chatId, "tool", obsMsg)
            }

            response.usage?.let {
                Log.d(TAG, "Tokens: ${it.promptTokens}+${it.completionTokens}=${it.totalTokens}")
            }
        }

        if (steps > budget) {
            onObservation?.invoke("Budget exceeded ($budget steps).")
        }
    }
}
