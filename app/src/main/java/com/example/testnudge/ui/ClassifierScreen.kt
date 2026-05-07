package com.example.testnudge.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
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
import com.example.testnudge.llm.TinyBertGatingService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

enum class ClassifierMode(val assetFile: String, val label: String) {
    Gating("gating_samples.json", "Gating")
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ClassifierScreen(mode: ClassifierMode) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var samples by remember(mode) { mutableStateOf<List<SampleText>>(emptyList()) }
    var selectedSample by remember(mode) { mutableStateOf<SampleText?>(null) }
    var inputText by remember(mode) { mutableStateOf("") }
    var resultText by remember(mode) { mutableStateOf("") }
    var dropdownExpanded by remember { mutableStateOf(false) }
    var running by remember { mutableStateOf(false) }

    LaunchedEffect(mode) {
        samples = SampleTextRepository.load(context, mode.assetFile)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        ExposedDropdownMenuBox(
            expanded = dropdownExpanded,
            onExpandedChange = { dropdownExpanded = it }
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

        OutlinedTextField(
            value = inputText,
            onValueChange = { inputText = it },
            label = { Text("Input") },
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 140.dp)
        )

        OutlinedTextField(
            value = resultText,
            onValueChange = {},
            readOnly = true,
            label = { Text("Result") },
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 160.dp)
        )

        Button(
            enabled = !running,
            onClick = {
                val capturedInput = inputText
                if (capturedInput.isBlank()) {
                    resultText = "Input is empty."
                    return@Button
                }
                running = true
                resultText = "Running gating..."
                scope.launch {
                    val outcome = withContext(Dispatchers.IO) {
                        runCatching {
                            val service = TinyBertGatingService.get(context)
                            val start = System.currentTimeMillis()
                            val scores = service.classify(capturedInput)
                            val elapsedMs = System.currentTimeMillis() - start
                            elapsedMs to scores
                        }
                    }
                    resultText = outcome.fold(
                        onSuccess = { (elapsedMs, scores) -> formatScores(elapsedMs, scores) },
                        onFailure = { e -> "Error: ${e.message ?: e::class.simpleName.orEmpty()}" }
                    )
                    running = false
                }
            },
            modifier = Modifier.align(Alignment.End)
        ) {
            Text(if (running) "Running..." else "Run")
        }
    }
}

private const val DECISION_THRESHOLD = 0.5f

private fun formatScores(elapsedMs: Long, scores: List<TinyBertGatingService.Score>): String {
    val top = scores.first()
    val passed = scores.filter { it.probability >= DECISION_THRESHOLD }
    val passedSummary = if (passed.isEmpty()) "(none above $DECISION_THRESHOLD)" else
        passed.joinToString(", ") { it.label }
    val maxLabelWidth = scores.maxOf { it.label.length }
    return buildString {
        append("Elapsed: ").append(elapsedMs).append(" ms\n\n")
        append("Top label: ").append(top.label)
            .append(" (").append("%.3f".format(top.probability)).append(")\n")
        append("Above threshold (>= ").append(DECISION_THRESHOLD).append("): ")
            .append(passedSummary).append("\n\n")
        append("All scores:\n")
        scores.forEach { score ->
            append("  ")
                .append(score.label.padEnd(maxLabelWidth))
                .append("  ")
                .append("%.3f".format(score.probability))
                .append('\n')
        }
    }.trimEnd()
}
