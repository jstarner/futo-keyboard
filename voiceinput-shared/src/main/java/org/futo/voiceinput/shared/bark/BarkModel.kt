package org.futo.voiceinput.shared.bark

import android.content.Context
import android.util.Log
import androidx.annotation.Keep
import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.newSingleThreadContext
import kotlinx.coroutines.withContext
import org.futo.voiceinput.shared.ggml.InferenceCancelledException
import org.futo.voiceinput.shared.ggml.InvalidModelException
import org.futo.voiceinput.shared.types.VoiceInputModel
import org.futo.voiceinput.shared.util.splitIntoWindows
import java.nio.Buffer

private const val TAG = "BarkModel"

@OptIn(DelicateCoroutinesApi::class)
private val barkContext = newSingleThreadContext("bark-inference")

private const val SAMPLE_RATE = 16000

// Longer clips are transcribed in pieces of at most this length
private const val MAX_WINDOW_SAMPLES = 30 * SAMPLE_RATE

// Tokenizer of nvidia/parakeet-tdt-0.6b-v3 (CC-BY-4.0), one piece per line,
// special tokens left empty. .bark files don't include it.
private const val VOCAB_ASSET = "bark_vocab.txt"

private const val WORD_BOUNDARY = '▁'

@Keep
internal object BarkNative {
    @JvmStatic external fun openFromBufferNative(buffer: Buffer): Long
    @JvmStatic external fun transcribeNative(handle: Long, samples: FloatArray): IntArray?
    @JvmStatic external fun cancelNative(handle: Long)
    @JvmStatic external fun closeNative(handle: Long)
}

/**
 * Bark (.bark) speech model: a compressed Parakeet TDT, English only.
 * The glossary, language and symbol suppression options have no equivalent and are ignored.
 */
class BarkModel(
    context: Context,
    modelBuffer: Buffer
) : VoiceInputModel {
    private var handle: Long = 0L
    private val vocab = loadVocab(context)
    @Volatile private var cancelled = false

    init {
        handle = try {
            BarkNative.openFromBufferNative(modelBuffer)
        } catch(e: IllegalArgumentException) {
            Log.e(TAG, "Failed to load Bark model: ${e.message}")
            throw InvalidModelException(e.message ?: "The Bark model could not be loaded")
        }
    }

    @Throws(InferenceCancelledException::class)
    override suspend fun infer(
        samples: FloatArray,
        glossary: List<String>,
        languages: Array<String>,
        bailLanguages: Array<String>,
        suppressNonSpeechTokens: Boolean,
        partialResultCallback: (String) -> Unit
    ): String = withContext(barkContext) {
        if(handle == 0L) {
            throw IllegalStateException("BarkModel has already been closed, cannot infer")
        }
        cancelled = false

        val windows = splitIntoWindows(samples, MAX_WINDOW_SAMPLES)
        val texts = mutableListOf<String>()
        for(window in windows) {
            if(cancelled) throw InferenceCancelledException()

            val tokens = try {
                BarkNative.transcribeNative(handle, samples.copyOfRange(window.first, window.last + 1))
                    ?: throw InferenceCancelledException()
            } catch(e: RuntimeException) {
                // Callers only expect cancellation, so keep whatever was transcribed so far
                Log.e(TAG, "Bark transcription failed", e)
                break
            }

            val text = decodeTokens(tokens)
            if(text.isNotEmpty()) texts.add(text)

            if(windows.size > 1) {
                partialResultCallback(texts.joinToString(separator = " "))
            }
        }

        if(cancelled) throw InferenceCancelledException()

        return@withContext texts.joinToString(separator = " ")
    }

    private fun decodeTokens(tokens: IntArray): String {
        val text = StringBuilder()
        for(token in tokens) {
            if(token in vocab.indices) text.append(vocab[token])
        }
        return text.toString().replace(WORD_BOUNDARY, ' ').trim()
    }

    override fun cancel() {
        cancelled = true
        if(handle != 0L) BarkNative.cancelNative(handle)
    }

    override suspend fun close() = withContext(barkContext) {
        if(handle != 0L) {
            BarkNative.closeNative(handle)
        }
        handle = 0L
    }

    companion object {
        @Volatile private var cachedVocab: List<String>? = null

        private fun loadVocab(context: Context): List<String> {
            cachedVocab?.let { return it }
            return context.assets.open(VOCAB_ASSET).bufferedReader(Charsets.UTF_8).use { it.readLines() }.also {
                cachedVocab = it
            }
        }
    }
}
