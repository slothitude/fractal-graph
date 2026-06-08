package com.slothitude.agent

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import android.provider.Settings
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.material.textfield.TextInputEditText
import com.slothitude.agent.control.TelegramBotService
import com.slothitude.agent.service.AgentForegroundService

class MainActivity : AppCompatActivity() {

    private lateinit var statusText: TextView
    private lateinit var tokenInput: TextInputEditText
    private lateinit var whitelistInput: TextInputEditText
    private lateinit var saveButton: Button
    private lateinit var toggleButton: Button

    private lateinit var botConfig: BotConfig

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        botConfig = BotConfig(this)

        statusText = findViewById(R.id.statusText)
        tokenInput = findViewById(R.id.tokenInput)
        whitelistInput = findViewById(R.id.whitelistInput)
        saveButton = findViewById(R.id.saveButton)
        toggleButton = findViewById(R.id.toggleButton)

        loadConfig()
        requestNotifications()
        requestBatteryExemption()

        saveButton.setOnClickListener {
            saveConfig()
            Toast.makeText(this, "Config saved", Toast.LENGTH_SHORT).show()
        }

        toggleButton.setOnClickListener {
            toggleService()
        }
    }

    private fun loadConfig() {
        tokenInput.setText(botConfig.botToken, TextView.BufferType.EDITABLE)
        whitelistInput.setText(botConfig.whitelist, TextView.BufferType.EDITABLE)
    }

    private fun saveConfig() {
        botConfig.botToken = tokenInput.text?.toString() ?: ""
        botConfig.whitelist = whitelistInput.text?.toString() ?: ""
    }

    private fun toggleService() {
        val intent = Intent(this, AgentForegroundService::class.java)
        if (isServiceRunning()) {
            stopService(intent)
            toggleButton.text = getString(R.string.btn_start)
            toggleButton.setBackgroundColor(getColor(R.color.accent))
            statusText.text = getString(R.string.status_stopped)
        } else {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                startForegroundService(intent)
            } else {
                startService(intent)
            }
            toggleButton.text = getString(R.string.btn_stop)
            toggleButton.setBackgroundColor(getColor(R.color.red))
            statusText.text = getString(R.string.status_running)
        }
    }

    private fun isServiceRunning(): Boolean {
        // Simple heuristic — button state
        return toggleButton.text == getString(R.string.btn_stop)
    }

    private fun requestNotifications() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED
            ) {
                requestPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
            }
        }
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (!isGranted) {
            Toast.makeText(this, "Notification permission required for foreground service", Toast.LENGTH_LONG).show()
        }
    }

    private fun requestBatteryExemption() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
            if (!pm.isIgnoringBatteryOptimizations(packageName)) {
                val intent = Intent(
                    Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
                    Uri.parse("package:$packageName")
                )
                startActivity(intent)
            }
        }
    }

    /**
     * Persisted config stored in SharedPreferences.
     */
    private class BotConfig(context: Context) {
        private val prefs = context.getSharedPreferences("agent_config", Context.MODE_PRIVATE)

        var botToken: String
            get() = prefs.getString("bot_token", "") ?: ""
            set(value) = prefs.edit().putString("bot_token", value).apply()

        var whitelist: String
            get() = prefs.getString("whitelist", "") ?: ""
            set(value) = prefs.edit().putString("whitelist", value).apply()
    }
}
