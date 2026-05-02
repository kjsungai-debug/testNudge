package com.example.testnudge.llm

import org.json.JSONArray
import org.json.JSONObject

/**
 * Builds the developer-turn content (system instruction + tool declarations)
 * that gets passed to LiteRT-LM via [com.google.ai.edge.litertlm.ConversationConfig.systemInstruction].
 *
 * The .litertlm bundle's embedded chat template already wraps everything in
 * `<start_of_turn>developer ... <end_of_turn>` and adds the user/model turns,
 * so we only need to emit the inner body. The user's free-form text is sent
 * separately as the message — it becomes the user-turn content.
 *
 * Tool declarations are emitted using FunctionGemma's raw template syntax
 * (`<start_function_declaration>...<end_function_declaration>`) which the
 * tokenizer recognizes as special tokens.
 */
object FunctionGemmaPrompt {

    private const val SYSTEM_PROMPT =
        "You are a model that can do function calling with the following functions"

    /**
     * Arbitrary tool schema. Edit this JSON to expose different functions to
     * the model — the schema follows the OpenAI-style function declaration
     * shape that the chat template expects.
     */
    val TOOLS_JSON: String = """
        [
          {
            "type": "function",
            "function": {
              "name": "get_current_temperature",
              "description": "Gets the current temperature for a given location.",
              "parameters": {
                "type": "object",
                "properties": {
                  "location": {
                    "type": "string",
                    "description": "The city name, e.g. San Francisco"
                  }
                },
                "required": ["location"]
              }
            }
          },
          {
            "type": "function",
            "function": {
              "name": "send_text_message",
              "description": "Sends a text message to the given recipient.",
              "parameters": {
                "type": "object",
                "properties": {
                  "recipient": {
                    "type": "string",
                    "description": "Phone number or contact name of the recipient"
                  },
                  "body": {
                    "type": "string",
                    "description": "The message body"
                  }
                },
                "required": ["recipient", "body"]
              }
            }
          },
          {
            "type": "function",
            "function": {
              "name": "set_alarm",
              "description": "Sets an alarm for the specified time on the user's device.",
              "parameters": {
                "type": "object",
                "properties": {
                  "time": {
                    "type": "string",
                    "description": "Alarm time in HH:MM 24-hour format"
                  },
                  "label": {
                    "type": "string",
                    "description": "Optional label for the alarm"
                  }
                },
                "required": ["time"]
              }
            }
          }
        ]
    """.trimIndent()

    fun buildSystemInstruction(toolsJson: String = TOOLS_JSON): String {
        val tools = JSONArray(toolsJson)
        return buildString {
            append(SYSTEM_PROMPT)
            for (i in 0 until tools.length()) {
                append("<start_function_declaration>")
                append(formatFunctionDeclaration(tools.getJSONObject(i)))
                append("<end_function_declaration>")
            }
        }
    }

    /**
     * Returns the FunctionGemma chat-template-rendered prompt as a single
     * string for display/debugging. Mirrors the structure produced by
     * `chat_template.jinja` but reconstructed in Kotlin — the actual prompt
     * the LiteRT-LM runtime feeds to the model is generated internally and
     * may differ slightly in whitespace.
     */
    fun buildDisplayPrompt(userInput: String, toolsJson: String = TOOLS_JSON): String {
        val tools = JSONArray(toolsJson)
        return buildString {
            append("<bos>")
            append("<start_of_turn>developer\n")
            append(SYSTEM_PROMPT)
            for (i in 0 until tools.length()) {
                append("<start_function_declaration>")
                append(formatFunctionDeclaration(tools.getJSONObject(i)))
                append("<end_function_declaration>")
            }
            append("<end_of_turn>\n")
            append("<start_of_turn>user\n")
            append(userInput.trim())
            append("<end_of_turn>\n")
            append("<start_of_turn>model\n")
        }
    }

    data class ParamSpec(
        val name: String,
        val type: String,
        val description: String,
        val required: Boolean
    )

    /** Returns the parameter spec map for the given function name, or null if no such tool. */
    fun lookupTool(functionName: String, toolsJson: String = TOOLS_JSON): Map<String, ParamSpec>? {
        val tools = JSONArray(toolsJson)
        for (i in 0 until tools.length()) {
            val tool = tools.getJSONObject(i)
            val function = tool.getJSONObject("function")
            if (function.optString("name") != functionName) continue
            val params = function.optJSONObject("parameters") ?: return emptyMap()
            val properties = params.optJSONObject("properties") ?: return emptyMap()
            val requiredArr = params.optJSONArray("required")
            val requiredSet = mutableSetOf<String>()
            if (requiredArr != null) {
                for (j in 0 until requiredArr.length()) requiredSet.add(requiredArr.getString(j))
            }
            val result = linkedMapOf<String, ParamSpec>()
            properties.keys().forEach { key ->
                val prop = properties.getJSONObject(key)
                result[key] = ParamSpec(
                    name = key,
                    type = prop.optString("type", "string"),
                    description = prop.optString("description", ""),
                    required = key in requiredSet
                )
            }
            return result
        }
        return null
    }

    private fun formatFunctionDeclaration(tool: JSONObject): String {
        val function = tool.getJSONObject("function")
        val name = function.getString("name")
        val description = function.optString("description", "")
        val parameters = function.optJSONObject("parameters")
        return buildString {
            append("declaration:").append(name).append("{")
            append("description:<escape>").append(description).append("<escape>")
            if (parameters != null) {
                append(",parameters:{")
                val properties = parameters.optJSONObject("properties")
                if (properties != null && properties.length() > 0) {
                    append("properties:{")
                    append(formatProperties(properties))
                    append("},")
                }
                val required = parameters.optJSONArray("required")
                if (required != null && required.length() > 0) {
                    append("required:[")
                    for (i in 0 until required.length()) {
                        if (i > 0) append(",")
                        append("<escape>").append(required.getString(i)).append("<escape>")
                    }
                    append("],")
                }
                val type = parameters.optString("type", "")
                if (type.isNotEmpty()) {
                    append("type:<escape>").append(type.uppercase()).append("<escape>")
                }
                append("}")
            }
            append("}")
        }
    }

    private fun formatProperties(properties: JSONObject): String {
        val keys = properties.keys().asSequence().toList().sorted()
        return keys.joinToString(",") { key ->
            val value = properties.getJSONObject(key)
            val description = value.optString("description", "")
            val type = value.optString("type", "string").uppercase()
            "$key:{description:<escape>$description<escape>,type:<escape>$type<escape>}"
        }
    }
}
