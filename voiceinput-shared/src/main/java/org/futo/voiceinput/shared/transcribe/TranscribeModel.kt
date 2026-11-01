package org.futo.voiceinput.shared.transcribe

import android.os.SystemClock
import android.util.Log
import androidx.annotation.Keep
import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.newSingleThreadContext
import kotlinx.coroutines.withContext
import org.futo.voiceinput.shared.ggml.InferenceCancelledException
import org.futo.voiceinput.shared.ggml.InvalidModelException
import org.futo.voiceinput.shared.types.VoiceInputModel

private const val TAG = "TranscribeModel"

private const val SAMPLE_RATE = 16000
private const val NUM_THREADS = 4

@OptIn(DelicateCoroutinesApi::class)
private val transcribeContext = newSingleThreadContext("transcribe-inference")

@Keep
internal object TranscribeNative {
    init {
        System.loadLibrary("jni_transcribe")
    }

    @JvmStatic external fun openNative(path: String, threads: Int): Long
    @JvmStatic external fun transcribeNative(handle: Long, samples: FloatArray): String?
    @JvmStatic external fun resetCancelNative(handle: Long)
    @JvmStatic external fun cancelNative(handle: Long)
    @JvmStatic external fun closeNative(handle: Long)
}

/**
 * GGUF speech model running on transcribe.cpp. Weights are memory mapped from [path].
 * The glossary, language and symbol suppression options are ignored, and there are no partial results.
 */
class TranscribeModel(
    path: String
) : VoiceInputModel {
    private val handleLock = Any()
    private var handle: Long = 0L

    init {
        val start = SystemClock.elapsedRealtime()
        handle = try {
            TranscribeNative.openNative(path, NUM_THREADS)
        } catch(e: IllegalArgumentException) {
            Log.e(TAG, "Failed to load model: ${e.message}")
            throw InvalidModelException(e.message ?: "The model could not be loaded")
        } catch(e: LinkageError) {
            // libjni_transcribe.so is only built for arm64-v8a
            Log.e(TAG, "Failed to load native library", e)
            throw InvalidModelException("This device is not supported")
        }
        Log.i(TAG, "Loaded model in ${SystemClock.elapsedRealtime() - start} ms")
    }

    @Throws(InferenceCancelledException::class)
    override suspend fun infer(
        samples: FloatArray,
        glossary: List<String>,
        languages: Array<String>,
        bailLanguages: Array<String>,
        suppressNonSpeechTokens: Boolean,
        partialResultCallback: (String) -> Unit
    ): String = withContext(transcribeContext) {
        val currentHandle = handle
        if(currentHandle == 0L) {
            throw IllegalStateException("TranscribeModel has already been closed, cannot infer")
        }

        // AudioRecognizer cancels the job before calling cancel(), so a cancel that comes
        // before this reset still stops here
        TranscribeNative.resetCancelNative(currentHandle)
        ensureActive()

        val start = SystemClock.elapsedRealtime()
        val text = try {
            TranscribeNative.transcribeNative(currentHandle, samples) ?: throw InferenceCancelledException()
        } catch(e: RuntimeException) {
            // Callers only expect cancellation
            Log.e(TAG, "Transcription failed", e)
            ""
        }

        val elapsed = SystemClock.elapsedRealtime() - start
        val audioMs = samples.size * 1000L / SAMPLE_RATE
        Log.i(TAG, "Transcribed $audioMs ms of audio in $elapsed ms")

        return@withContext text.trim()
    }

    override fun cancel() {
        synchronized(handleLock) {
            if(handle != 0L) TranscribeNative.cancelNative(handle)
        }
    }

    override suspend fun close() = withContext(transcribeContext) {
        synchronized(handleLock) {
            if(handle != 0L) {
                TranscribeNative.closeNative(handle)
            }
            handle = 0L
        }
    }
}
