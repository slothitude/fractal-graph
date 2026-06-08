package com.slothitude.agent.control

import android.util.Log

/**
 * Per-chat conversation history. Sliding window with configurable max messages.
 * Later phases will add LLM-based summarization for long conversations.
 */
class SessionManager(
    private val maxMessages: Int = 50
) {
    companion object {
        private const val TAG = "SessionManager"
    }

    data class Turn(
        val role: String,       // "user", "assistant", "system", "tool"
        val content: String
    )

    private val sessions = mutableMapOf<Long, MutableList<Turn>>()

    fun append(chatId: Long, role: String, content: String) {
        val history = sessions.getOrPut(chatId) { mutableListOf() }
        history.add(Turn(role, content))
        trim(chatId)
    }

    fun getHistory(chatId: Long): List<Turn> {
        return sessions[chatId] ?: emptyList()
    }

    fun clear(chatId: Long) {
        sessions.remove(chatId)
        Log.i(TAG, "Cleared session for chat $chatId")
    }

    fun clearAll() {
        sessions.clear()
        Log.i(TAG, "Cleared all sessions")
    }

    fun size(chatId: Long): Int = sessions[chatId]?.size ?: 0

    /**
     * Build prompt text from session history for the LLM.
     * Formats as numbered turns for clarity.
     */
    fun buildPromptText(chatId: Long): String {
        val history = getHistory(chatId)
        return buildString {
            for (turn in history) {
                append("[${turn.role}]: ${turn.content}\n")
            }
        }
    }

    private fun trim(chatId: Long) {
        val history = sessions[chatId] ?: return
        if (history.size > maxMessages) {
            val trimCount = history.size - maxMessages
            repeat(trimCount) { history.removeAt(0) }
            Log.d(TAG, "Trimmed $trimCount turns from chat $chatId")
        }
    }
}
