package org.futo.voiceinput.shared.types

import java.io.DataInputStream
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Identifies voice input model files. Magics are the first 4 bytes, read big-endian */
object ModelFormat {
    /** Legacy ggml file used by whisper.cpp ("lmgg") */
    val WHISPER_GGML_MAGIC = 0x6c6d6767.toUInt()

    /** Cactus .cact container used by Whistle */
    val WHISTLE_CACT_MAGIC = 0x842ae105.toUInt()

    /** Bytes needed by isBark */
    const val BARK_PROBE_SIZE = 64

    /** Bark .bark file: 8-byte little-endian header length, then a JSON header naming the format */
    fun isBark(header: ByteArray): Boolean {
        if(header.size < 16) return false
        if((4 until 8).any { header[it] != 0.toByte() }) return false
        if(header[8] != '{'.code.toByte()) return false
        return String(header, 8, header.size - 8, Charsets.ISO_8859_1).contains("\"bark-levels-v")
    }

    fun isBark(buffer: ByteBuffer): Boolean {
        val header = ByteArray(minOf(BARK_PROBE_SIZE, buffer.capacity()))
        buffer.duplicate().apply { position(0) }.get(header)
        return isBark(header)
    }

    /** GGUF container ("GGUF"), used by transformer language models and transcribe.cpp speech models */
    val GGUF_MAGIC = 0x47475546.toUInt()

    /** Values of general.architecture that run on transcribe.cpp */
    private val SPEECH_GGUF_ARCHITECTURES = setOf("parakeet")

    private const val GGUF_TYPE_STRING = 8
    private const val GGUF_TYPE_ARRAY = 9
    private val GGUF_TYPE_SIZES = mapOf(
        0 to 1L, 1 to 1L, 2 to 2L, 3 to 2L, 4 to 4L, 5 to 4L, 6 to 4L, 7 to 1L, 10 to 8L, 11 to 8L, 12 to 8L
    )

    private const val GGUF_MAX_KEYS_SCANNED = 64
    private const val GGUF_MAX_STRING_LENGTH = 4096L

    data class GGUFInfo(val architecture: String?, val name: String?, val sttVariant: String? = null) {
        /** transcribe.cpp files carry stt.variant; parakeet.cpp files share the architecture but not the layout */
        val isSpeechModel: Boolean get() = architecture in SPEECH_GGUF_ARCHITECTURES && sttVariant != null
        val isUnconvertedSpeechModel: Boolean get() = architecture in SPEECH_GGUF_ARCHITECTURES && sttVariant == null
    }

    private fun DataInputStream.readUInt32LE(): Long =
        Integer.reverseBytes(readInt()).toLong() and 0xffffffffL

    private fun DataInputStream.readUInt64LE(): Long =
        java.lang.Long.reverseBytes(readLong())

    private fun DataInputStream.skipFully(count: Long) {
        if(count < 0) throw IOException("Invalid GGUF length")
        var remaining = count
        while(remaining > 0) {
            val skipped = skip(remaining)
            if(skipped > 0) {
                remaining -= skipped
            } else {
                readByte()
                remaining--
            }
        }
    }

    private fun DataInputStream.readGGUFString(): String {
        val length = readUInt64LE()
        if(length < 0 || length > GGUF_MAX_STRING_LENGTH) throw IOException("Invalid GGUF string length")
        val bytes = ByteArray(length.toInt())
        readFully(bytes)
        return String(bytes, Charsets.UTF_8)
    }

    private fun DataInputStream.skipGGUFValue(type: Int) {
        when(type) {
            GGUF_TYPE_STRING -> skipFully(readUInt64LE())
            GGUF_TYPE_ARRAY -> {
                val elementType = readUInt32LE().toInt()
                val count = readUInt64LE()
                if(count < 0) throw IOException("Invalid GGUF array length")
                if(elementType == GGUF_TYPE_STRING) {
                    for(i in 0 until count) skipFully(readUInt64LE())
                } else {
                    val size = GGUF_TYPE_SIZES[elementType] ?: throw IOException("Invalid GGUF array type")
                    skipFully(Math.multiplyExact(count, size))
                }
            }
            else -> skipFully(GGUF_TYPE_SIZES[type] ?: throw IOException("Invalid GGUF type"))
        }
    }

    /** Reads the GGUF header, with [stream] positioned right after the 4 magic bytes */
    fun readGGUFInfo(stream: InputStream): GGUFInfo {
        var architecture: String? = null
        var name: String? = null
        var sttVariant: String? = null

        try {
            val input = DataInputStream(stream)
            val version = input.readUInt32LE()
            if(version < 2) return GGUFInfo(null, null)

            input.readUInt64LE() // Tensor count
            val keyCount = input.readUInt64LE()

            for(i in 0 until minOf(keyCount, GGUF_MAX_KEYS_SCANNED.toLong())) {
                val key = input.readGGUFString()
                val type = input.readUInt32LE().toInt()

                if(type == GGUF_TYPE_STRING && key == "general.architecture") {
                    architecture = input.readGGUFString()
                } else if(type == GGUF_TYPE_STRING && key == "general.name") {
                    name = input.readGGUFString()
                } else if(type == GGUF_TYPE_STRING && key == "stt.variant") {
                    sttVariant = input.readGGUFString()
                } else {
                    input.skipGGUFValue(type)
                }

                if(architecture != null && name != null && sttVariant != null) break
            }
        } catch(_: IOException) {
            // Truncated or malformed, keep what was found so far
        } catch(_: ArithmeticException) {
        }

        return GGUFInfo(architecture, name, sttVariant)
    }

    fun magicOf(buffer: ByteBuffer): UInt? {
        if(buffer.capacity() < 4) return null
        return buffer.duplicate().order(ByteOrder.BIG_ENDIAN).getInt(0).toUInt()
    }

    fun magicOf(file: File): UInt? {
        val bytes = ByteArray(4)
        val read = file.inputStream().use { it.read(bytes) }
        return if(read == 4) ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).getInt().toUInt() else null
    }
}
