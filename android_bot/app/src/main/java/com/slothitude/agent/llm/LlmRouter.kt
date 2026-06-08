package com.slothitude.agent.llm

import android.content.Context
import android.content.SharedPreferences
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject

/**
 * Provider configuration stored in SharedPreferences.
 */
data class LlmProvider(
    val name: String,
    val baseUrl: String,
    val apiKey: String,
    val model: String,
    val supportsTools: Boolean = true,
    val supportsStreaming: Boolean = true,
    val localOnly: Boolean = false,    // never use when on metered connection
    val priority: Int = 0             // higher = preferred
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("name", name)
        put("baseUrl", baseUrl)
        put("apiKey", apiKey)
        put("model", model)
        put("supportsTools", supportsTools)
        put("supportsStreaming", supportsStreaming)
        put("localOnly", localOnly)
        put("priority", priority)
    }

    companion object {
        fun fromJson(json: JSONObject): LlmProvider = LlmProvider(
            name = json.getString("name"),
            baseUrl = json.getString("baseUrl"),
            apiKey = json.getString("apiKey"),
            model = json.getString("model"),
            supportsTools = json.optBoolean("supportsTools", true),
            supportsStreaming = json.optBoolean("supportsStreaming", true),
            localOnly = json.optBoolean("localOnly", false),
            priority = json.optInt("priority", 0)
        )
    }
}

/**
 * Routes LLM requests to the best available provider.
 * Selection logic:
 *   1. User override (/model <name>)
 *   2. WiFi available? → prefer remote (higher priority)
 *   3. No WiFi → prefer local providers
 *   4. Fallback chain on failure
 */
class LlmRouter(private val context: Context) {

    companion object {
        private const val TAG = "LlmRouter"
        private const val PREFS_NAME = "agent_providers"
        private const val KEY_PROVIDERS = "providers_json"
        private const val KEY_ACTIVE = "active_provider"
    }

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager

    private val backends = mutableMapOf<String, LlmBackend>()
    private var userOverride: String? = null

    // --- Provider management ---

    fun addProvider(provider: LlmProvider) {
        val providers = getProviders().toMutableList()
        providers.removeAll { it.name == provider.name }
        providers.add(provider)
        saveProviders(providers)
        rebuildBackends()
        Log.i(TAG, "Added provider: ${provider.name} (${provider.baseUrl})")
    }

    fun removeProvider(name: String) {
        val providers = getProviders().toMutableList()
        providers.removeAll { it.name == name }
        saveProviders(providers)
        backends.remove(name)
        if (userOverride == name) userOverride = null
        if (getActiveProvider() == name) {
            val first = providers.firstOrNull()
            prefs.edit().putString(KEY_ACTIVE, first?.name ?: "").apply()
        }
        Log.i(TAG, "Removed provider: $name")
    }

    fun getProviders(): List<LlmProvider> {
        val raw = prefs.getString(KEY_PROVIDERS, null) ?: return emptyList()
        val arr = JSONArray(raw)
        return (0 until arr.length()).map { LlmProvider.fromJson(arr.getJSONObject(it)) }
    }

    fun getProvider(name: String): LlmProvider? {
        return getProviders().find { it.name == name }
    }

    fun setActiveProvider(name: String) {
        if (getProviders().none { it.name == name }) {
            Log.w(TAG, "Cannot set active: provider '$name' not found")
            return
        }
        prefs.edit().putString(KEY_ACTIVE, name).apply()
        userOverride = name
        Log.i(TAG, "Active provider: $name")
    }

    fun getActiveProvider(): String {
        return userOverride
            ?: prefs.getString(KEY_ACTIVE, "") ?: ""
    }

    fun clearOverride() {
        userOverride = null
    }

    // --- Backend selection ---

    /**
     * Select the best backend for the current network conditions.
     */
    fun selectBackend(needsTools: Boolean = false): LlmBackend? {
        val providers = getProviders()
        if (providers.isEmpty()) {
            Log.w(TAG, "No providers configured")
            return null
        }

        // User override takes priority
        val override = userOverride
        if (override != null) {
            val backend = backends[override]
            if (backend != null) {
                if (needsTools && !backend.supportsTools) {
                    Log.w(TAG, "Override $override doesn't support tools, falling back")
                } else {
                    return backend
                }
            }
        }

        val isWifi = isWifiConnected()

        // Sort by priority, filter by network
        val sorted = providers
            .filter { if (!isWifi) !it.localOnly else true }
            .sortedByDescending { it.priority }

        for (provider in sorted) {
            val backend = backends[provider.name]
            if (backend == null) continue
            if (needsTools && !backend.supportsTools) continue
            return backend
        }

        // If nothing matches network, try local anyway
        for (provider in sorted) {
            return backends[provider.name] ?: continue
        }

        return null
    }

    /**
     * Get a backend by name.
     */
    fun getBackend(name: String): LlmBackend? = backends[name]

    /**
     * Get all available backends.
     */
    fun getBackends(): Map<String, LlmBackend> = backends.toMap()

    // --- Private ---

    private fun rebuildBackends() {
        backends.clear()
        for (provider in getProviders()) {
            try {
                backends[provider.name] = OpenAiCompatBackend(
                    name = provider.name,
                    baseUrl = provider.baseUrl,
                    apiKey = provider.apiKey,
                    model = provider.model,
                    supportsTools = provider.supportsTools,
                    supportsStreaming = provider.supportsStreaming
                )
            } catch (e: Exception) {
                Log.e(TAG, "Failed to create backend for ${provider.name}", e)
            }
        }
    }

    private fun saveProviders(providers: List<LlmProvider>) {
        val arr = JSONArray()
        for (p in providers) arr.put(p.toJson())
        prefs.edit().putString(KEY_PROVIDERS, arr.toString()).apply()
    }

    private fun isWifiConnected(): Boolean {
        val network = cm.activeNetwork ?: return false
        val caps = cm.getNetworkCapabilities(network) ?: return false
        return caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI)
    }

    /**
     * Seed default providers if none configured.
     * Called once on first init.
     */
    fun seedDefaults() {
        if (getProviders().isNotEmpty()) return
        Log.i(TAG, "No providers configured — seeding defaults")
        addProvider(LlmProvider(
            name = "zai",
            baseUrl = "https://api.z.ai/api/coding/paas/v4",
            apiKey = "b7143b5694e443eaa6858550fd2bcf2e.Jl2Q0bsQWWKrBdbd",
            model = "glm-5.1",
            supportsTools = true,
            supportsStreaming = true,
            priority = 10
        ))
        setActiveProvider("zai")
    }

    init {
        seedDefaults()
        rebuildBackends()
    }
}
