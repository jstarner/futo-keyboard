package org.futo.voiceinput.shared.whisper

import android.content.Context
import org.futo.voiceinput.shared.types.ModelLoader
import org.futo.voiceinput.shared.types.VoiceInputModel


class ModelManager(
    val context: Context
) {
    private val loadedModels: HashMap<Any, VoiceInputModel> = hashMapOf()

    fun obtainModel(model: ModelLoader): VoiceInputModel {
        val key = model.key(context)
        if (!loadedModels.contains(key)) {
            loadedModels[key] = model.load(context)
        }

        return loadedModels[key]!!
    }

    fun cancelAll() {
        loadedModels.forEach {
            it.value.cancel()
        }
    }

    suspend fun cleanUp() {
        for (model in loadedModels.entries) {
            model.value.cancel()
            model.value.close()
        }

        loadedModels.clear()
    }
}
