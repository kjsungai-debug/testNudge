package com.example.testnudge.data

import android.content.Context
import android.util.Log
import org.json.JSONArray

object SampleTextRepository {

    private const val TAG = "SampleTextRepository"

    fun load(context: Context, assetFileName: String): List<SampleText> {
        return try {
            val raw = context.assets.open(assetFileName).bufferedReader().use { it.readText() }
            val array = JSONArray(raw)
            List(array.length()) { i ->
                val obj = array.getJSONObject(i)
                SampleText(
                    title = obj.getString("title"),
                    text = obj.getString("text")
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load $assetFileName", e)
            emptyList()
        }
    }
}
