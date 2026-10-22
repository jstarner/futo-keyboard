package org.futo.voiceinput.shared.types

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

    fun magicOf(buffer: ByteBuffer): UInt? {
        if(buffer.capacity() < 4) return null
        return buffer.duplicate().order(ByteOrder.BIG_ENDIAN).getInt(0).toUInt()
    }
}
