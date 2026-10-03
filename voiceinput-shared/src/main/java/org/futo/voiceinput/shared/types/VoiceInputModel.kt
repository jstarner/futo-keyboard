package org.futo.voiceinput.shared.types

import org.futo.voiceinput.shared.ggml.BailLanguageException
import org.futo.voiceinput.shared.ggml.InferenceCancelledException

/** A loaded speech recognition model (Whisper or Whistle) */
interface VoiceInputModel {
    // empty languages = autodetect any language
    // 1 language = will force that language
    // 2 or more languages = autodetect between those languages
    @Throws(BailLanguageException::class, InferenceCancelledException::class)
    suspend fun infer(
        samples: FloatArray,
        glossary: List<String>,
        languages: Array<String>,
        bailLanguages: Array<String>,
        suppressNonSpeechTokens: Boolean,
        partialResultCallback: (String) -> Unit
    ): String

    fun cancel()

    suspend fun close()
}
