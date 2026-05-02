package com.example.testnudge.llm

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File

/**
 * Loads the FunctionGemma .litertlm bundle from assets and runs text
 * generation through Google AI Edge's LiteRT-LM runtime.
 *
 * Engine init is heavy (loads ~280 MB and warms up), so the Engine is held
 * as a process-wide singleton. Each `generateResponse` call uses a fresh
 * Conversation so prior turns don't leak between requests.
 */
class LlmInferenceService private constructor(
    private val engine: Engine
) {

    suspend fun generateResponse(userInput: String, systemInstruction: String): String {
        val config = ConversationConfig(
            systemInstruction = Contents.of(systemInstruction)
        )
        return engine.createConversation(config).use { conversation ->
            val collected = StringBuilder()
            conversation.sendMessageAsync(userInput).collect { message ->
                collected.append(message.toString())
            }
            collected.toString()
        }
    }

    fun close() {
        engine.close()
    }

    companion object {
        private const val TAG = "LlmInferenceService"
        private const val ASSET_NAME = "tiny_garden.litertlm"
        private const val LOCAL_NAME = "tiny_garden.litertlm"

        private val initMutex = Mutex()

        @Volatile
        private var instance: LlmInferenceService? = null

        suspend fun get(context: Context): LlmInferenceService {
            instance?.let { return it }
            return initMutex.withLock {
                instance ?: build(context.applicationContext).also { instance = it }
            }
        }

        private fun build(context: Context): LlmInferenceService {
            val modelFile = ensureModelOnDisk(context)
            Log.i(TAG, "Loading LLM from ${modelFile.absolutePath} (${modelFile.length()} bytes)")
            val config = EngineConfig(
                modelPath = modelFile.absolutePath,
                backend = Backend.CPU(),
                cacheDir = context.cacheDir.path
            )
            val engine = Engine(config)
            engine.initialize()
            return LlmInferenceService(engine)
        }

        private fun ensureModelOnDisk(context: Context): File {
            val target = File(context.filesDir, LOCAL_NAME)
            if (target.exists() && target.length() > 0) return target
            context.assets.open(ASSET_NAME).use { input ->
                target.outputStream().use { output -> input.copyTo(output) }
            }
            return target
        }
    }
}
