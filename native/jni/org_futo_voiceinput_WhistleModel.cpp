#include <string>
#include <vector>
#include <jni.h>
#include "defines.h"
#include "org_futo_voiceinput_WhistleModel.h"
#include "jni_common.h"
#include "jni_utils.h"

#ifdef HAVE_NEEDLE
#include "needle.h"

// The prebuilt armeabi-v7a engine was built against libc++ 21, which exports
// std::__hash_memory. Older NDK libc++ only has the inline hash it wraps.
#if _LIBCPP_VERSION < 210000
_LIBCPP_BEGIN_NAMESPACE_STD
size_t __hash_memory(const void *ptr, size_t size) noexcept {
    return __murmur2_or_cityhash<size_t>()(ptr, size);
}
_LIBCPP_END_NAMESPACE_STD
#endif
#endif

// The Cactus engine keeps one process-global speech model and is not thread-safe.
// All calls are serialized on a single thread by WhistleModel.kt.

static jboolean WhistleModel_isAvailable(JNIEnv *env, jclass clazz) {
#ifdef HAVE_NEEDLE
    return JNI_TRUE;
#else
    return JNI_FALSE;
#endif
}

// Returns null on success, otherwise an error message
static jstring WhistleModel_loadFromBuffer(JNIEnv *env, jclass clazz, jobject buffer) {
#ifdef HAVE_NEEDLE
    auto *buffer_address = reinterpret_cast<const unsigned char *>(env->GetDirectBufferAddress(buffer));
    jlong buffer_capacity = env->GetDirectBufferCapacity(buffer);
    if(buffer_address == nullptr || buffer_capacity <= 0) {
        return string2jstring(env, "Model buffer is not a direct buffer");
    }

    AKLOGI("Attempting to load Whistle model from buffer...");
    int res = needle_load(buffer_address, (unsigned long long)buffer_capacity);
    if(res < 0) {
        const char *err = needle_last_error();
        AKLOGE("needle_load failed (%d): %s", res, err ? err : "");
        return string2jstring(env, err ? err : "needle_load failed");
    }

    if((needle_models() & NEEDLE_SPEECH) == 0) {
        AKLOGE("Loaded .cact file does not contain a speech model");
        return string2jstring(env, "The .cact file does not contain a speech model");
    }

    return nullptr;
#else
    return string2jstring(env, "Whistle is not supported on this CPU architecture");
#endif
}

// Returns the engine's JSON output. Empty language or keywords mean none.
static jstring WhistleModel_transcribe(JNIEnv *env, jclass clazz, jfloatArray samples_array, jstring language, jstring keywords) {
#ifdef HAVE_NEEDLE
    std::string language_str = jstring2string(env, language);
    std::string keywords_str = jstring2string(env, keywords);

    int num_samples = env->GetArrayLength(samples_array);
    jfloat *samples = env->GetFloatArrayElements(samples_array, nullptr);

    std::vector<char> out(64 * 1024, '\0');

    AKLOGI("Calling needle_transcribe with %d samples", num_samples);
    int res = needle_transcribe(
            samples,
            num_samples,
            language_str.empty() ? nullptr : language_str.c_str(),
            keywords_str.empty() ? nullptr : keywords_str.c_str(),
            0,
            out.data(),
            (int)out.size());

    env->ReleaseFloatArrayElements(samples_array, samples, JNI_ABORT);

    if(res < 0) {
        const char *err = needle_last_error();
        AKLOGE("needle_transcribe failed (%d): %s", res, err ? err : "");
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), err ? err : "needle_transcribe failed");
        return nullptr;
    }

    out[out.size() - 1] = '\0';
    return string2jstring(env, out.data());
#else
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "Whistle is not supported on this CPU architecture");
    return nullptr;
#endif
}

static const JNINativeMethod sMethods[] = {
        {
                const_cast<char *>("isAvailableNative"),
                const_cast<char *>("()Z"),
                reinterpret_cast<void *>(WhistleModel_isAvailable)
        },
        {
                const_cast<char *>("loadFromBufferNative"),
                const_cast<char *>("(Ljava/nio/Buffer;)Ljava/lang/String;"),
                reinterpret_cast<void *>(WhistleModel_loadFromBuffer)
        },
        {
                const_cast<char *>("transcribeNative"),
                const_cast<char *>("([FLjava/lang/String;Ljava/lang/String;)Ljava/lang/String;"),
                reinterpret_cast<void *>(WhistleModel_transcribe)
        }
};

namespace voiceinput {
    int register_WhistleModel(JNIEnv *env) {
        const char *const kClassPathName = "org/futo/voiceinput/shared/whistle/WhistleNative";
        return latinime::registerNativeMethods(env, kClassPathName, sMethods, NELEMS(sMethods));
    }
}
