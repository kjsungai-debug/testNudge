package com.example.testnudge.llm

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.text.tokenizers.BertTokenizer
import org.tensorflow.lite.support.text.tokenizers.BertTokenizer.BertTokenizerOptions
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp

/**
 * Loads the tinyBERT gating classifier (BertForSequenceClassification, 4
 * multi-label heads) and exposes per-label sigmoid scores.
 *
 *  - Model file:  app/src/main/assets/tinybert_gating.tflite
 *  - Vocab file:  app/src/main/assets/tinybert_vocab.txt
 *
 * Both files are produced by `tools/convert_tinybert.py`. The model is mmap'd
 * directly out of the APK via `AssetFileDescriptor`, so we don't pay the
 * disk-copy cost we use for the larger LiteRT-LM bundle.
 */
class TinyBertGatingService private constructor(
    private val interpreter: Interpreter,
    private val tokenizer: BertTokenizer
) {

    /** Single label score, in [0, 1]. */
    data class Score(val label: String, val probability: Float)

    /**
     * Tokenizes [text] (truncating/padding to [MAX_SEQ_LEN]), runs the
     * classifier, and returns scores sorted descending.
     */
    fun classify(text: String): List<Score> {
        val ids = encode(text)
        val inputIds = IntArray(MAX_SEQ_LEN) { if (it < ids.size) ids[it] else PAD_ID }
        val attentionMask = IntArray(MAX_SEQ_LEN) { if (it < ids.size) 1 else 0 }
        val tokenTypeIds = IntArray(MAX_SEQ_LEN) // single-sentence → all zeros

        val inputs = arrayOf<Any>(
            arrayOf(inputIds),
            arrayOf(attentionMask),
            arrayOf(tokenTypeIds)
        )
        val logits = Array(1) { FloatArray(LABELS.size) }
        val outputs = mutableMapOf<Int, Any>(0 to logits)

        interpreter.runForMultipleInputsOutputs(inputs, outputs)

        return LABELS.indices
            .map { i -> Score(LABELS[i], sigmoid(logits[0][i])) }
            .sortedByDescending { it.probability }
    }

    private fun encode(text: String): IntArray {
        val pieces = tokenizer.tokenize(text)
        val capped = if (pieces.size > MAX_SEQ_LEN - 2) pieces.subList(0, MAX_SEQ_LEN - 2) else pieces
        val withSpecials = mutableListOf<String>().apply {
            add(CLS_TOKEN)
            addAll(capped)
            add(SEP_TOKEN)
        }
        return tokenizer.convertTokensToIds(withSpecials).toIntArray()
    }

    fun close() {
        interpreter.close()
    }

    companion object {
        private const val TAG = "TinyBertGatingService"
        private const val MODEL_ASSET = "tinybert_gating.tflite"
        private const val VOCAB_ASSET = "tinybert_vocab.txt"
        private const val MAX_SEQ_LEN = 128
        private const val CLS_TOKEN = "[CLS]"
        private const val SEP_TOKEN = "[SEP]"
        private const val PAD_ID = 0

        /** Output label order (matches `id2label` in the HF config). */
        val LABELS: List<String> = listOf("action", "recall", "safety", "unknown")

        @Volatile
        private var instance: TinyBertGatingService? = null

        fun get(context: Context): TinyBertGatingService {
            return instance ?: synchronized(this) {
                instance ?: build(context.applicationContext).also { instance = it }
            }
        }

        private fun build(context: Context): TinyBertGatingService {
            val modelBuffer = mmapAsset(context, MODEL_ASSET)
            Log.i(TAG, "Loaded $MODEL_ASSET (${modelBuffer.capacity()} bytes)")
            val interpreter = Interpreter(
                modelBuffer,
                Interpreter.Options().apply { setNumThreads(2) }
            )
            val vocab = readVocab(context, VOCAB_ASSET)
            // Tokenizer was trained as a multilingual cased model — keep case.
            val tokenizer = BertTokenizer(
                vocab,
                BertTokenizerOptions.builder().setDoLowerCase(false).build()
            )
            return TinyBertGatingService(interpreter, tokenizer)
        }

        private fun readVocab(context: Context, assetName: String): Map<String, Int> {
            val map = LinkedHashMap<String, Int>()
            context.assets.open(assetName).bufferedReader().useLines { lines ->
                lines.forEachIndexed { index, line -> map[line] = index }
            }
            return map
        }

        private fun mmapAsset(context: Context, name: String): MappedByteBuffer {
            val fd: AssetFileDescriptor = context.assets.openFd(name)
            FileInputStream(fd.fileDescriptor).use { input ->
                return input.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    fd.startOffset,
                    fd.declaredLength
                )
            }
        }

        private fun sigmoid(x: Float): Float = (1.0 / (1.0 + exp(-x.toDouble()))).toFloat()
    }
}
