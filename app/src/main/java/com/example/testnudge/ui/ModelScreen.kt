package com.example.testnudge.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.example.testnudge.data.SampleText
import com.example.testnudge.data.SampleTextRepository
import com.example.testnudge.llm.FunctionGemmaPrompt
import com.example.testnudge.llm.LlmInferenceService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private enum class ModelSubMode(
    val assetFile: String,
    val label: String,
    val toolsJson: String
) {
    ActionParam(
        assetFile = "action_param_samples.json",
        label = "Action Param",
        toolsJson = FunctionGemmaPrompt.ACTION_PARAM_TOOL_JSON
    ),
    QueryRewrite(
        assetFile = "query_rewrite_samples.json",
        label = "Query Rewrite",
        toolsJson = FunctionGemmaPrompt.QUERY_REWRITE_TOOL_JSON
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val scrollState = rememberScrollState()

    var subMode by remember { mutableStateOf(ModelSubMode.ActionParam) }
    var samples by remember { mutableStateOf<List<SampleText>>(emptyList()) }
    var selectedSample by remember { mutableStateOf<SampleText?>(null) }
    var inputText by remember { mutableStateOf("") }
    var resultText by remember { mutableStateOf("") }
    var rawText by remember { mutableStateOf("") }
    var dropdownExpanded by remember { mutableStateOf(false) }
    var running by remember { mutableStateOf(false) }

    LaunchedEffect(subMode) {
        samples = SampleTextRepository.load(context, subMode.assetFile)
        selectedSample = null
        inputText = ""
        resultText = ""
        rawText = ""
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            ModelSubMode.entries.forEach { mode ->
                if (subMode == mode) {
                    Button(
                        onClick = { subMode = mode },
                        modifier = Modifier.weight(1f)
                    ) {
                        Text(mode.label)
                    }
                } else {
                    OutlinedButton(
                        onClick = { subMode = mode },
                        modifier = Modifier.weight(1f)
                    ) {
                        Text(mode.label)
                    }
                }
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            ExposedDropdownMenuBox(
                expanded = dropdownExpanded,
                onExpandedChange = { dropdownExpanded = it },
                modifier = Modifier.weight(0.75f)
            ) {
                OutlinedTextField(
                    value = selectedSample?.title.orEmpty(),
                    onValueChange = {},
                    readOnly = true,
                    label = { Text("Sample Text") },
                    placeholder = { Text("Select a sample") },
                    trailingIcon = {
                        ExposedDropdownMenuDefaults.TrailingIcon(expanded = dropdownExpanded)
                    },
                    modifier = Modifier
                        .menuAnchor()
                        .fillMaxWidth()
                )
                ExposedDropdownMenu(
                    expanded = dropdownExpanded,
                    onDismissRequest = { dropdownExpanded = false }
                ) {
                    samples.forEach { sample ->
                        DropdownMenuItem(
                            text = { Text(sample.title) },
                            onClick = {
                                selectedSample = sample
                                inputText = sample.text
                                dropdownExpanded = false
                            }
                        )
                    }
                }
            }
            Button(
                enabled = !running,
                onClick = {
                    val capturedInput = inputText
                    val capturedToolsJson = subMode.toolsJson
                    runInference(
                        inputText = capturedInput,
                        toolsJson = capturedToolsJson,
                        onStart = {
                            running = true
                            rawText = "Running inference..."
                            resultText = ""
                        },
                        onResult = { elapsedMs, raw ->
                            rawText = buildRawDataReport(
                                elapsedMs = elapsedMs,
                                userInput = capturedInput,
                                toolsJson = capturedToolsJson,
                                raw = raw
                            )
                            resultText = parseFunctionGemmaOutput(raw)
                            running = false
                        },
                        onError = { msg ->
                            rawText = "Error: $msg"
                            resultText = ""
                            running = false
                        },
                        scope = scope,
                        contextProvider = { context }
                    )
                },
                modifier = Modifier
                    .weight(0.25f)
                    .height(56.dp)
            ) {
                Text(if (running) "..." else "Run")
            }
        }

        OutlinedTextField(
            value = inputText,
            onValueChange = { inputText = it },
            label = { Text("Input") },
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 120.dp)
        )

        OutlinedTextField(
            value = resultText,
            onValueChange = {},
            readOnly = true,
            label = { Text("Result") },
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 100.dp)
        )

        OutlinedTextField(
            value = rawText,
            onValueChange = {},
            readOnly = true,
            label = { Text("Raw Data") },
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 320.dp)
        )
    }
}

private fun runInference(
    inputText: String,
    toolsJson: String,
    onStart: () -> Unit,
    onResult: (elapsedMs: Long, raw: String) -> Unit,
    onError: (String) -> Unit,
    scope: kotlinx.coroutines.CoroutineScope,
    contextProvider: () -> android.content.Context
) {
    if (inputText.isBlank()) {
        onError("Input is empty.")
        return
    }
    onStart()
    val systemInstruction = FunctionGemmaPrompt.buildSystemInstruction(toolsJson)
    val context = contextProvider()
    scope.launch {
        val outcome = withContext(Dispatchers.IO) {
            runCatching {
                val service = LlmInferenceService.get(context)
                val start = System.currentTimeMillis()
                val response = service.generateResponse(inputText, systemInstruction)
                val elapsedMs = System.currentTimeMillis() - start
                elapsedMs to response
            }
        }
        outcome.fold(
            onSuccess = { (elapsedMs, raw) -> onResult(elapsedMs, raw) },
            onFailure = { e -> onError(e.message ?: e::class.simpleName.orEmpty()) }
        )
    }
}

/**
 * Extract function calls from FunctionGemma's raw output and pretty-print them
 * as `name(arg=value, ...)`. Falls back to the trimmed raw text when no
 * function-call tokens are present (e.g. plain text answers).
 */
private fun parseFunctionGemmaOutput(raw: String): String {
    val matches = FUNCTION_CALL_REGEX.findAll(raw).toList()
    if (matches.isEmpty()) return raw.trim()
    return matches.joinToString("\n") { match ->
        val name = match.groupValues[1]
        val args = parseArgsCompact(match.groupValues[2])
        "$name($args)"
    }
}

private fun parseArgsCompact(body: String): String {
    if (body.isBlank()) return ""
    val normalized = body.replace("<escape>", "\"")
    return normalized
        .split(",")
        .mapNotNull { kv ->
            val sep = kv.indexOf(':')
            if (sep <= 0) null else {
                val key = kv.substring(0, sep).trim()
                val value = kv.substring(sep + 1).trim()
                "$key=$value"
            }
        }
        .joinToString(", ")
}

/**
 * Builds the full Raw Data report shown in the box, with three sections:
 *
 *   Elapsed: <ms>
 *
 *   Prompt:
 *     <full chat-template-rendered prompt>
 *
 *   Output:
 *     - Function name + per-parameter extraction breakdown
 *     - Trailing raw model response for reference
 */
private fun buildRawDataReport(
    elapsedMs: Long,
    userInput: String,
    toolsJson: String,
    raw: String
): String {
    val displayPrompt = FunctionGemmaPrompt.buildDisplayPrompt(userInput, toolsJson)
    return buildString {
        append("Elapsed: ").append(elapsedMs).append(" ms")
        append("\n\n")
        append("Prompt:\n")
        append(displayPrompt)
        append("\n\n")
        append("Output:\n")
        append(buildOutputBreakdown(raw, toolsJson))
    }
}

private data class ParsedArg(
    val key: String,
    val rawValue: String,
    val cleanValue: String,
    val isQuoted: Boolean
)

private fun parseArgsDetailed(body: String): List<ParsedArg> {
    if (body.isBlank()) return emptyList()
    return body.split(",").mapNotNull { kv ->
        val trimmed = kv.trim()
        val sep = trimmed.indexOf(':')
        if (sep <= 0) return@mapNotNull null
        val key = trimmed.substring(0, sep).trim()
        val rawValue = trimmed.substring(sep + 1).trim()
        val isQuoted = rawValue.startsWith("<escape>") && rawValue.endsWith("<escape>")
        val cleanValue = if (isQuoted) {
            rawValue.removePrefix("<escape>").removeSuffix("<escape>")
        } else {
            rawValue
        }
        ParsedArg(key = key, rawValue = rawValue, cleanValue = cleanValue, isQuoted = isQuoted)
    }
}

private fun buildOutputBreakdown(raw: String, toolsJson: String): String {
    val matches = FUNCTION_CALL_REGEX.findAll(raw).toList()
    if (matches.isEmpty()) {
        return "No function call detected — model returned plain text.\n\n" +
            "Raw model response:\n" + raw.trim()
    }
    return buildString {
        matches.forEachIndexed { idx, match ->
            if (idx > 0) append("\n")
            val name = match.groupValues[1]
            val argsBody = match.groupValues[2]
            val toolSchema = FunctionGemmaPrompt.lookupTool(name, toolsJson)
            append("Function #").append(idx + 1).append(": ").append(name).append('\n')
            if (toolSchema == null) {
                append("  (function not declared in tools schema)\n")
            }
            val parsed = parseArgsDetailed(argsBody)
            if (parsed.isEmpty()) {
                append("  Parameters: (none)\n")
            } else {
                append("  Parameters:\n")
                parsed.forEach { arg ->
                    val spec = toolSchema?.get(arg.key)
                    val declaredType = spec?.type ?: "?"
                    val required = if (spec?.required == true) " required" else ""
                    val inferredType = if (arg.isQuoted) "string" else "literal"
                    append("    - ")
                        .append(arg.key)
                        .append(" (declared: ")
                        .append(declaredType)
                        .append(required)
                        .append(", parsed as: ")
                        .append(inferredType)
                        .append(")\n")
                    append("        value:           ").append('"').append(arg.cleanValue).append('"').append('\n')
                    append("        extracted from:  ").append(arg.key).append(':').append(arg.rawValue).append('\n')
                    if (!spec?.description.isNullOrEmpty()) {
                        append("        description:     ").append(spec!!.description).append('\n')
                    }
                }
            }
            // Flag schema mismatches
            if (toolSchema != null) {
                val parsedKeys = parsed.map { it.key }.toSet()
                val missingRequired = toolSchema.values
                    .filter { it.required && it.name !in parsedKeys }
                    .map { it.name }
                if (missingRequired.isNotEmpty()) {
                    append("  Missing required: ").append(missingRequired.joinToString(", ")).append('\n')
                }
                val unknownKeys = parsedKeys.filter { it !in toolSchema.keys }
                if (unknownKeys.isNotEmpty()) {
                    append("  Unknown keys: ").append(unknownKeys.joinToString(", ")).append('\n')
                }
            }
        }
        append("\nRaw model response:\n")
        append(raw.trim())
    }
}

private val FUNCTION_CALL_REGEX = Regex(
    "<start_function_call>call:(\\w+)\\{(.*?)\\}<end_function_call>",
    RegexOption.DOT_MATCHES_ALL
)
