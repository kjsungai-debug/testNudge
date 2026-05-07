package com.example.testnudge.llm

import org.json.JSONArray
import org.json.JSONObject

/**
 * Builds the developer-turn content (system instruction + tool declarations)
 * that gets passed to LiteRT-LM via [com.google.ai.edge.litertlm.ConversationConfig.systemInstruction],
 * plus a per-mode user-turn prefix that the caller prepends to the user input.
 *
 * The prompts here mirror the two training prompts (`text_input_action` and
 * `text_input_recall`) shipped with the FunctionGemma fine-tune's `prompt.py`
 * — same system text, same tool declarations (with empty inner descriptions /
 * types where the training prompt has them), same user-turn prefix.
 */
object FunctionGemmaPrompt {

    // ─── System prompts (developer-turn body, before tool declarations) ──────────

    /** System text for Action Param. Source: `text_input_action` in prompt.py. */
    const val ACTION_SYSTEM_PROMPT: String =
        "You are a virtual assistant to proactively suggest helpful actions. " +
            "Analyze the given conversation and call the appropriate tool. " +
            "If no tool applies, call `unknown`."

    /** System text for Query Rewrite. Source: `text_input_recall` in prompt.py. */
    const val RECALL_SYSTEM_PROMPT: String = """You are a conversation history search assistant. Given the conversation, generate a search query to find relevant information from past conversations or device memory. ALWAYS call the `search` tool.

Available tool:
- search: Search device memory or conversation history for information. Params: query (search query), intent (type of information: NO_RECALL_INTENT, ACCOUNT_ID, ACCOUNT_NAME, ADDRESS, BIRTHDAY, BRAND, BUSINESS_NAME, CARD_CVV, CARD_NUMBER, CONFIRMATION_NUMBER, COUPON_CODE, DRIVERS_LICENSE_NUMBER, DURATION, EMAIL, END_DATETIME, EVENT_NAME, FLIGHT_NUMBER, ID_NUMBER, LOCATION_NAME, LOCATION_REGION, MEDIA_ALBUM, MEDIA_TITLE, MEMBERSHIP_NUMBER, ORDER_CARRIER, ORDER_ITEM_NAME_QUANTITY, PASSPORT_NUMBER, PERSON_NAME, PHONE_NUMBER, PLATFORM, PRICE, PRODUCT_CATEGORY, PRODUCT_NAME, PURCHASED, QUANTITY, SEAT_NUMBER, START_DATETIME, TRACKING_NUMBER, TRANSPORTATION_ARRIVAL_GATE, TRANSPORTATION_ARRIVAL_LOCATION, TRANSPORTATION_ARRIVAL_TERMINAL, TRANSPORTATION_DEPARTURE_GATE, TRANSPORTATION_DEPARTURE_LOCATION, TRANSPORTATION_DEPARTURE_TERMINAL, TRANSPORTATION_TYPE, URL, VEHICLE_LICENSE_PLATE, WIFI_PASSWORD, WIFI_SSID), entity (entity category: PERSON, BUSINESS, EVENT, RESERVATION, LOCATION, TRANSPORTATION, HOTEL, MOVIE, MUSIC, BOOK, VIDEO, ARTICLE, TV, PRODUCT, ORDER, SOCIAL_MEDIA, CREDENTIALS, PAYMENT, PERSONAL_INFO, DOCUMENT)"""

    // ─── User-turn prefixes (prepended to inputText before sendMessageAsync) ─────

    /** User-turn prefix for Action Param — matches the framing in text_input_action. */
    const val ACTION_USER_PREFIX: String =
        "Refer to the following conversation to generate action.\n<user>: "

    /** User-turn prefix for Query Rewrite — matches the framing in text_input_recall. */
    const val RECALL_USER_PREFIX: String =
        "Refer to the following conversation.\nuser: "

    // ─── Tool schemas (rendered into <start_function_declaration>...<end_function_declaration>) ──

    /**
     * Action Param tool set. Nine functions (call, create_calendar, location,
     * payment, photo, reminder, search, unknown, view_calendar). Inner
     * description/type fields are intentionally left empty — that's how the
     * training prompt was formatted.
     */
    val ACTION_PARAM_TOOL_JSON: String = """
        [
          {"type":"function","function":{"name":"call","description":"Make a call or send a text to a contact.","parameters":{"type":"object","properties":{"name":{"description":"","type":""},"number":{"description":"","type":""}}}}},
          {"type":"function","function":{"name":"create_calendar","description":"Create a calendar event with the given details.","parameters":{"type":"object","properties":{"end":{"description":"","type":""},"location":{"description":"","type":""},"start":{"description":"","type":""},"title":{"description":"","type":""}}}}},
          {"type":"function","function":{"name":"location","description":"Open a location in maps or navigate.","parameters":{"type":"object","properties":{"location":{"description":"","type":""}}}}},
          {"type":"function","function":{"name":"payment","description":"Send a payment via a banking app.","parameters":{"type":"object","properties":{"account":{"description":"","type":""},"application":{"description":"","type":""}}}}},
          {"type":"function","function":{"name":"photo","description":"Share or capture a photo.","parameters":{"type":"object","properties":{"photo":{"description":"","type":""}}}}},
          {"type":"function","function":{"name":"reminder","description":"Set a reminder.","parameters":{"type":"object","properties":{"start":{"description":"","type":""},"title":{"description":"","type":""}}}}},
          {"type":"function","function":{"name":"search","description":"Search on the device for information.","parameters":{"type":"object","properties":{"entity":{"description":"","type":""},"intent":{"description":"","type":""},"query":{"description":"","type":""}}}}},
          {"type":"function","function":{"name":"unknown","description":"No actionable tool call applies.","parameters":{"type":"object"}}},
          {"type":"function","function":{"name":"view_calendar","description":"View calendar events for a given time.","parameters":{"type":"object","properties":{"start":{"description":"","type":""}}}}}
        ]
    """.trimIndent()

    /**
     * Query Rewrite tool set. A single `search` function with all intent and
     * entity enums listed in the function description (matching text_input_recall
     * verbatim).
     */
    val QUERY_REWRITE_TOOL_JSON: String = """
        [
          {"type":"function","function":{"name":"search","description":"Search device memory or conversation history for information. Intent types: NO_RECALL_INTENT, ACCOUNT_ID, ACCOUNT_NAME, ADDRESS, BIRTHDAY, BRAND, BUSINESS_NAME, CARD_CVV, CARD_NUMBER, CONFIRMATION_NUMBER, COUPON_CODE, DRIVERS_LICENSE_NUMBER, DURATION, EMAIL, END_DATETIME, EVENT_NAME, FLIGHT_NUMBER, ID_NUMBER, LOCATION_NAME, LOCATION_REGION, MEDIA_ALBUM, MEDIA_TITLE, MEMBERSHIP_NUMBER, ORDER_CARRIER, ORDER_ITEM_NAME_QUANTITY, PASSPORT_NUMBER, PERSON_NAME, PHONE_NUMBER, PLATFORM, PRICE, PRODUCT_CATEGORY, PRODUCT_NAME, PURCHASED, QUANTITY, SEAT_NUMBER, START_DATETIME, TRACKING_NUMBER, TRANSPORTATION_ARRIVAL_GATE, TRANSPORTATION_ARRIVAL_LOCATION, TRANSPORTATION_ARRIVAL_TERMINAL, TRANSPORTATION_DEPARTURE_GATE, TRANSPORTATION_DEPARTURE_LOCATION, TRANSPORTATION_DEPARTURE_TERMINAL, TRANSPORTATION_TYPE, URL, VEHICLE_LICENSE_PLATE, WIFI_PASSWORD, WIFI_SSID. Entity types: PERSON, BUSINESS, EVENT, RESERVATION, LOCATION, TRANSPORTATION, HOTEL, MOVIE, MUSIC, BOOK, VIDEO, ARTICLE, TV, PRODUCT, ORDER, SOCIAL_MEDIA, CREDENTIALS, PAYMENT, PERSONAL_INFO, DOCUMENT.","parameters":{"type":"object","properties":{"entity":{"description":"","type":""},"intent":{"description":"","type":""},"query":{"description":"","type":""}}}}}
        ]
    """.trimIndent()

    // ─── Builders ─────────────────────────────────────────────────────────────────

    /**
     * Returns the developer-turn body — `systemPrompt` directly followed by
     * one `<start_function_declaration>...<end_function_declaration>` per
     * tool, with no separator. LiteRT-LM wraps the result in
     * `<start_of_turn>developer ... <end_of_turn>`.
     */
    fun buildSystemInstruction(systemPrompt: String, toolsJson: String): String {
        val tools = JSONArray(toolsJson)
        return buildString {
            append(systemPrompt)
            for (i in 0 until tools.length()) {
                append("<start_function_declaration>")
                append(formatFunctionDeclaration(tools.getJSONObject(i)))
                append("<end_function_declaration>")
            }
        }
    }

    /**
     * Returns the full chat-template-rendered prompt as a single string for
     * display/debugging. Mirrors the structure of `text_input_action` /
     * `text_input_recall` in prompt.py — the runtime's actual prompt may
     * differ in whitespace.
     *
     * `userTurnContent` is the full user-turn body (already includes the
     * mode-specific user prefix + the user's input).
     */
    fun buildDisplayPrompt(
        systemPrompt: String,
        userTurnContent: String,
        toolsJson: String
    ): String {
        return buildString {
            append("<bos>")
            append("<start_of_turn>developer\n")
            append(buildSystemInstruction(systemPrompt, toolsJson))
            append("<end_of_turn>\n")
            append("<start_of_turn>user\n")
            append(userTurnContent.trimEnd())
            append("<end_of_turn>\n")
            append("<start_of_turn>model\n")
        }
    }

    // ─── Schema lookup (used by the Output breakdown in Raw Data) ────────────────

    data class ParamSpec(
        val name: String,
        val type: String,
        val description: String,
        val required: Boolean
    )

    /** Returns the parameter spec map for the given function name, or null if no such tool. */
    fun lookupTool(functionName: String, toolsJson: String): Map<String, ParamSpec>? {
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
                    type = prop.optString("type", ""),
                    description = prop.optString("description", ""),
                    required = key in requiredSet
                )
            }
            return result
        }
        return null
    }

    // ─── Rendering helpers ───────────────────────────────────────────────────────

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
            val type = value.optString("type", "")
            val typeRendered = if (type.isEmpty()) "" else type.uppercase()
            "$key:{description:<escape>$description<escape>,type:<escape>$typeRendered<escape>}"
        }
    }
}
