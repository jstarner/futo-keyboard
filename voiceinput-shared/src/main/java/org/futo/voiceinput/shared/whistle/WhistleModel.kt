package org.futo.voiceinput.shared.whistle

import android.util.Log
import androidx.annotation.Keep
import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.newSingleThreadContext
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.futo.voiceinput.shared.ggml.InferenceCancelledException
import org.futo.voiceinput.shared.ggml.InvalidModelException
import org.futo.voiceinput.shared.types.VoiceInputModel
import org.json.JSONObject
import java.nio.Buffer

private const val TAG = "WhistleModel"

// The Cactus engine holds one process-global speech model and is not thread-safe,
// so every native call runs on this thread
@OptIn(DelicateCoroutinesApi::class)
private val whistleContext = newSingleThreadContext("whistle-inference")

private val SUPPORTED_LANGUAGES = setOf("en", "de", "fr", "es", "it", "nl", "pl")

private const val SAMPLE_RATE = 16000

// The engine transcribes at most 30 seconds per call
private const val MAX_WINDOW_SAMPLES = 30 * SAMPLE_RATE

// Longer clips are cut at the quietest 100ms frame within the last 2 seconds of each window
private const val SPLIT_SEARCH_SAMPLES = 2 * SAMPLE_RATE
private const val SPLIT_FRAME_SAMPLES = SAMPLE_RATE / 10

@Keep
internal object WhistleNative {
    @JvmStatic external fun isAvailableNative(): Boolean
    @JvmStatic external fun loadFromBufferNative(buffer: Buffer): String?
    @JvmStatic external fun transcribeNative(samples: FloatArray, language: String, keywords: String): String
}

class WhistleModel(
    private val modelBuffer: Buffer
) : VoiceInputModel {
    @Volatile private var cancelled = false
    @Volatile private var closed = false

    init {
        if(!WhistleNative.isAvailableNative()) {
            throw InvalidModelException("Whistle is not supported on this CPU architecture")
        }

        runBlocking(whistleContext) { loadIntoEngine() }
    }

    private fun loadIntoEngine() {
        // Forget the previous model first, a failed load may have replaced it
        engineBuffer = null

        val error = WhistleNative.loadFromBufferNative(modelBuffer)
        if(error != null) {
            Log.e(TAG, "Failed to load Whistle model: $error")
            throw InvalidModelException(error)
        }

        engineBuffer = modelBuffer
    }

    @Throws(InferenceCancelledException::class)
    override suspend fun infer(
        samples: FloatArray,
        glossary: List<String>,
        languages: Array<String>,
        bailLanguages: Array<String>,
        suppressNonSpeechTokens: Boolean,
        partialResultCallback: (String) -> Unit
    ): String = withContext(whistleContext) {
        if(closed) {
            throw IllegalStateException("WhistleModel has already been closed, cannot infer")
        }
        cancelled = false

        // Another Whistle model may have been loaded into the engine since
        if(engineBuffer !== modelBuffer) {
            loadIntoEngine()
        }

        // Empty string = let the engine detect the language
        val language = languages.singleOrNull()?.takeIf { it in SUPPORTED_LANGUAGES } ?: ""
        val keywords = glossary.joinToString(separator = "\n")

        val windows = splitIntoWindows(samples)
        val texts = mutableListOf<String>()
        for(window in windows) {
            if(cancelled) throw InferenceCancelledException()

            val text = try {
                val json = WhistleNative.transcribeNative(
                    samples.copyOfRange(window.first, window.last + 1),
                    language,
                    keywords
                )

                JSONObject(json).optString("text", "").trim()
            } catch(e: Exception) {
                // Callers only expect cancellation, so keep whatever was transcribed so far
                Log.e(TAG, "Whistle transcription failed", e)
                break
            }

            if(text.isNotEmpty()) texts.add(text)

            if(windows.size > 1) {
                partialResultCallback(texts.joinToString(separator = " "))
            }
        }

        if(cancelled) throw InferenceCancelledException()

        return@withContext texts.joinToString(separator = " ")
    }

    override fun cancel() {
        // Takes effect between 30-second windows, the engine cannot interrupt a call
        cancelled = true
    }

    override suspend fun close() = withContext(whistleContext) {
        // The engine has no unload call and may still reference the mapped buffer,
        // so engineBuffer keeps it alive until another model replaces it
        closed = true
    }

    companion object {
        // Buffer of the model currently loaded in the engine. Only accessed on whistleContext
        private var engineBuffer: Buffer? = null
    }
}

private fun splitIntoWindows(samples: FloatArray): List<IntRange> {
    val windows = mutableListOf<IntRange>()

    var start = 0
    while(samples.size - start > MAX_WINDOW_SAMPLES) {
        val limit = start + MAX_WINDOW_SAMPLES
        val end = quietestCut(samples, limit - SPLIT_SEARCH_SAMPLES, limit)
        windows.add(start until end)
        start = end
    }

    if(start < samples.size) {
        windows.add(start until samples.size)
    }

    return windows
}

// Returns the middle of the lowest-energy frame in [from, to)
private fun quietestCut(samples: FloatArray, from: Int, to: Int): Int {
    var best = to
    var bestEnergy = Double.MAX_VALUE

    var frameStart = from
    while(frameStart + SPLIT_FRAME_SAMPLES <= to) {
        var energy = 0.0
        for(i in frameStart until frameStart + SPLIT_FRAME_SAMPLES) {
            energy += samples[i] * samples[i]
        }

        if(energy < bestEnergy) {
            bestEnergy = energy
            best = frameStart + SPLIT_FRAME_SAMPLES / 2
        }

        frameStart += SPLIT_FRAME_SAMPLES / 2
    }

    return best
}
