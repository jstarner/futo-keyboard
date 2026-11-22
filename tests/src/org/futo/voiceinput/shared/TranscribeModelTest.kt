package org.futo.voiceinput.shared

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.futo.voiceinput.shared.transcribe.TranscribeModel
import org.futo.voiceinput.shared.types.ModelFormat
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

private const val TAG = "TranscribeModelTest"

/**
 * The transcription tests need a model and a 16 kHz mono 16-bit sample.wav, pushed to
 * /sdcard/Android/data/<app id>/files/, and are skipped without them. The vocabulary test
 * also needs vocabulary.txt (one term per line) and transcribes every vocab*.wav.
 */
@RunWith(AndroidJUnit4::class)
class TranscribeModelTest {
    private fun ggufHeader(vararg stringKeys: Pair<String, String>): ByteArrayInputStream {
        val out = ByteArrayOutputStream()

        fun write(size: Int, block: ByteBuffer.() -> Unit) {
            out.write(ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN).apply(block).array())
        }
        fun writeString(value: String) {
            val bytes = value.toByteArray()
            write(8) { putLong(bytes.size.toLong()) }
            out.write(bytes)
        }

        write(20) {
            putInt(3)
            putLong(0)
            putLong(stringKeys.size + 2L)
        }

        // Values that have to be skipped
        writeString("some.number")
        write(8) { putInt(4); putInt(1234) }
        writeString("some.strings")
        write(16) { putInt(9); putInt(8); putLong(2) }
        writeString("a")
        writeString("bc")

        for((key, value) in stringKeys) {
            writeString(key)
            write(4) { putInt(8) }
            writeString(value)
        }

        return ByteArrayInputStream(out.toByteArray())
    }

    @Test
    fun readsSpeechModelHeader() {
        val info = ModelFormat.readGGUFInfo(
            ggufHeader("general.architecture" to "parakeet", "general.name" to "Parakeet TDT", "stt.variant" to "tdt-0.6b-v3")
        )
        assertEquals("parakeet", info.architecture)
        assertEquals("Parakeet TDT", info.name)
        assertTrue(info.isSpeechModel)
    }

    @Test
    fun parakeetCppLayoutNeedsConversion() {
        val info = ModelFormat.readGGUFInfo(
            ggufHeader("general.architecture" to "parakeet", "general.name" to "fluid-speech-mini", "parakeet.arch" to "tdt")
        )
        assertFalse(info.isSpeechModel)
        assertTrue(info.isUnconvertedSpeechModel)
    }

    @Test
    fun languageModelIsNotASpeechModel() {
        val info = ModelFormat.readGGUFInfo(ggufHeader("general.architecture" to "llama"))
        assertFalse(info.isSpeechModel)
        assertFalse(info.isUnconvertedSpeechModel)
    }

    @Test
    fun truncatedHeaderIsNotASpeechModel() {
        val bytes = ggufHeader("general.architecture" to "parakeet").readBytes()
        val info = ModelFormat.readGGUFInfo(ByteArrayInputStream(bytes, 0, 30))
        assertNull(info.architecture)
        assertFalse(info.isSpeechModel)
    }

    private fun readSamples(file: File): FloatArray {
        val bytes = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)

        var position = 12
        while(position + 8 <= bytes.capacity()) {
            val id = String(bytes.array(), position, 4, Charsets.US_ASCII)
            val size = bytes.getInt(position + 4)
            if(id == "data") {
                return FloatArray(size / 2) { bytes.getShort(position + 8 + it * 2) / 32768.0f }
            }
            position += 8 + size + (size and 1)
        }

        throw IllegalArgumentException("No data chunk in ${file.name}")
    }

    private fun logProcessMemory(label: String) {
        val status = File("/proc/self/status").readLines()
            .filter { it.startsWith("VmHWM") || it.startsWith("VmRSS") || it.startsWith("RssAnon") }
        Log.i(TAG, "$label: ${status.joinToString { it.replace(Regex("\\s+"), " ") }}")
    }

    @Test
    fun transcribesSample() = runBlocking {
        val dir = InstrumentationRegistry.getInstrumentation().targetContext.getExternalFilesDir(null)!!
        val wav = File(dir, "sample.wav")
        val modelFiles = dir.listFiles { file -> file.name.endsWith(".gguf") }.orEmpty()
        assumeTrue(wav.exists() && modelFiles.isNotEmpty())

        val samples = readSamples(wav)

        for(modelFile in modelFiles.sortedBy { it.length() }) {
            logProcessMemory("Before ${modelFile.name}")

            val model = TranscribeModel(modelFile.absolutePath)
            logProcessMemory("Loaded ${modelFile.name}")

            // The first run has a cold cache, the second is the one that repeats in use
            repeat(2) {
                val text = model.infer(samples, listOf(), arrayOf(), arrayOf(), false) { }
                Log.i(TAG, "${modelFile.name}: $text")
                assertTrue(text.isNotBlank())
            }

            logProcessMemory("Finished ${modelFile.name}")
            model.close()
        }
    }

    @Test
    fun transcribesWithVocabulary() = runBlocking {
        val dir = InstrumentationRegistry.getInstrumentation().targetContext.getExternalFilesDir(null)!!
        val vocabularyFile = File(dir, "vocabulary.txt")
        val clips = dir.listFiles { file -> file.name.startsWith("vocab") && file.name.endsWith(".wav") }.orEmpty()
        val modelFile = dir.listFiles { file -> file.name.endsWith(".gguf") }.orEmpty().minByOrNull { it.length() }
        assumeTrue(vocabularyFile.exists() && clips.isNotEmpty() && modelFile != null)

        val vocabulary = vocabularyFile.readLines().filter { it.isNotBlank() }
        val model = TranscribeModel(modelFile!!.absolutePath)
        for(clip in clips.sortedBy { it.name }) {
            val samples = readSamples(clip)
            val plain = model.infer(samples, listOf(), arrayOf(), arrayOf(), false) { }
            val withVocabulary = model.infer(samples, vocabulary, arrayOf(), arrayOf(), false) { }
            Log.i(TAG, "${clip.name} without: $plain")
            Log.i(TAG, "${clip.name} with:    $withVocabulary")
            assertTrue(withVocabulary.isNotBlank())
        }
        model.close()
    }
}
