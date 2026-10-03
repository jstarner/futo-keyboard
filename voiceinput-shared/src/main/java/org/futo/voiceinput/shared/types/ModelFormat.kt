package org.futo.voiceinput.shared.types

import java.nio.ByteBuffer
import java.nio.ByteOrder

/** First 4 bytes of a voice input model file, read big-endian */
object ModelFormat {
    /** Legacy ggml file used by whisper.cpp ("lmgg") */
    val WHISPER_GGML_MAGIC = 0x6c6d6767.toUInt()

    /** Cactus .cact container used by Whistle */
    val WHISTLE_CACT_MAGIC = 0x842ae105.toUInt()

    fun magicOf(buffer: ByteBuffer): UInt? {
        if(buffer.capacity() < 4) return null
        return buffer.duplicate().order(ByteOrder.BIG_ENDIAN).getInt(0).toUInt()
    }
}
