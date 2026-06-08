package com.slothitude.agent.control

import android.content.Context
import android.content.SharedPreferences
import android.util.Base64
import android.util.Log
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
import kotlin.math.min

/**
 * Validates incoming Telegram messages against an authorized user whitelist
 * and optional per-message HMAC signature.
 */
class AuthGate(context: Context) {

    companion object {
        private const val TAG = "AuthGate"
        private const val PREFS_NAME = "agent_auth"
        private const val KEY_WHITELIST = "whitelist_user_ids"
        private const val KEY_HMAC_SECRET = "hmac_secret"
        private const val KEY_BOT_TOKEN = "bot_token"
    }

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    /** Telegram bot token (stored in plain SharedPreferences for bot init). */
    var botToken: String
        get() = prefs.getString(KEY_BOT_TOKEN, "") ?: ""
        set(value) = prefs.edit().putString(KEY_BOT_TOKEN, value).apply()

    /** Seed defaults if none configured. */
    fun seedDefaults() {
        if (botToken.isBlank()) {
            botToken = "8735369358:AAGS42LeK97HlNFz3TA5SEe6YZdk-BhYLpY"
            Log.i(TAG, "Seeded default bot token (GhostKV)")
        }
        if (authorizedUserIds.isEmpty()) {
            authorizedUserIds = setOf(5597932516L)  // @slothitudegames
            Log.i(TAG, "Seeded authorized user: @slothitudegames")
        }
    }

    /** Set of authorized Telegram user IDs. */
    var authorizedUserIds: Set<Long>
        get() {
            val raw = prefs.getString(KEY_WHITELIST, "") ?: ""
            if (raw.isBlank()) return emptySet()
            return raw.split(",").mapNotNull { it.trim().toLongOrNull() }.toSet()
        }
        set(value) = prefs.edit().putString(KEY_WHITELIST, value.joinToString(",")).apply()

    /** Optional HMAC secret for message verification (hex string). */
    var hmacSecret: ByteArray?
        get() {
            val hex = prefs.getString(KEY_HMAC_SECRET, "") ?: ""
            if (hex.isBlank()) return null
            return try {
                hex.chunked(2).map { it.toInt(16).toByte() }.toByteArray()
            } catch (e: Exception) {
                Log.w(TAG, "Invalid HMAC hex", e)
                null
            }
        }
        set(value) {
            val hex = value?.joinToString("") { "%02x".format(it) } ?: ""
            prefs.edit().putString(KEY_HMAC_SECRET, hex).apply()
        }

    /**
     * Check if a message from the given user ID is authorized.
     * Returns true if the user is in the whitelist (or whitelist is empty — open mode).
     */
    fun isAuthorized(userId: Long): Boolean {
        if (authorizedUserIds.isEmpty()) {
            Log.w(TAG, "No whitelist configured — allowing all users (open mode)")
            return true
        }
        val ok = userId in authorizedUserIds
        if (!ok) Log.w(TAG, "Rejected message from unauthorized user $userId")
        return ok
    }

    /**
     * Verify HMAC signature on a message payload.
     * Returns true if no HMAC secret is configured (verification disabled).
     */
    fun verifySignature(text: String, signature: String?): Boolean {
        val secret = hmacSecret ?: return true  // no HMAC configured
        if (signature == null) {
            Log.w(TAG, "Message has no HMAC signature but HMAC is configured")
            return false
        }
        return try {
            val mac = Mac.getInstance("HmacSHA256").apply {
                init(SecretKeySpec(secret, "HmacSHA256"))
            }
            val expected = mac.doFinal(text.toByteArray(Charsets.UTF_8))
            val provided = Base64.decode(signature, Base64.DEFAULT)
            constantTimeEquals(expected, provided)
        } catch (e: Exception) {
            Log.e(TAG, "HMAC verification failed", e)
            false
        }
    }

    private fun constantTimeEquals(a: ByteArray, b: ByteArray): Boolean {
        if (a.size != b.size) return false
        var result = 0
        for (i in a.indices) result = result or ((a[i].toInt() xor b[i].toInt()))
        return result == 0
    }
}
