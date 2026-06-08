package com.slothitude.agent.tools

import android.util.Log
import com.slothitude.agent.llm.ToolDef
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * Web search via DuckDuckGo Instant Answer API.
 * Lightweight, no API key needed, returns top-N snippets.
 */
class SearchTool : Tool {
    companion object {
        private const val TAG = "SearchTool"
        private const val DDG_URL = "https://api.duckduckgo.com/"
        private const val MAX_RESULTS = 5
    }

    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .build()

    override val spec = ToolDef(
        name = "search",
        description = "Search the web using DuckDuckGo. Returns top result titles, snippets, and URLs.",
        parameters = JSONObject("""{
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return (default 5)"
                }
            },
            "required": ["query"]
        }""")
    )

    override suspend fun invoke(params: JSONObject): ToolResult = withContext(Dispatchers.IO) {
        val query = params.getString("query")
        if (query == null) return@withContext ToolResult(ok = false, text = "", error = "Missing 'query'")
        val maxResults = params.optInt("max_results", MAX_RESULTS)

        Log.i(TAG, "Searching: $query")
        try {
            val url = "${DDG_URL}?q=${java.net.URLEncoder.encode(query, "UTF-8")}&format=json&no_html=1"
            val request = Request.Builder().url(url).build()
            val response = client.newCall(request).execute()
            val body = response.body?.string()
            if (body == null) return@withContext ToolResult(ok = false, text = "", error = "Empty response")

            if (!response.isSuccessful) {
                return@withContext ToolResult(ok = false, text = "", error = "HTTP ${response.code}")
            }

            val json = JSONObject(body)
            val results = StringBuilder()

            val abstract = json.optString("Abstract", "")
            if (abstract.isNotBlank()) {
                results.append(abstract).append("\n\n")
            }

            val abstractUrl = json.optString("AbstractURL", "")
            if (abstractUrl.isNotBlank()) {
                results.append("Source: $abstractUrl\n\n")
            }

            val topics = json.optJSONArray("RelatedTopics")
            if (topics == null) {
                return@withContext ToolResult(ok = true, text = results.toString().ifBlank { "No results found for: $query" })
            }

            var count = 0
            var i = 0
            while (i < topics.length() && count < maxResults) {
                val topic = topics.getJSONObject(i)
                val text = topic.optString("Text", "")
                val firstUrl = topic.optString("FirstURL", "")

                if (text.isNotBlank() && !firstUrl.contains("duckduckgo.com")) {
                    results.append("- $text")
                    if (firstUrl.isNotBlank()) results.append("\n  $firstUrl")
                    results.append("\n\n")
                    count++
                }
                i++
            }

            val result = results.toString().trim()
            if (result.isBlank()) {
                ToolResult(ok = true, text = "No results found for: $query")
            } else {
                Log.d(TAG, "Found $count results")
                ToolResult(ok = true, text = result)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Search error", e)
            ToolResult(ok = false, text = "", error = e.message)
        }
    }
}
