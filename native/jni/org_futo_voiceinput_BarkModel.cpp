#include <string>
#include <vector>
#include <jni.h>
#include <unistd.h>
#include "bark/bark.h"
#include "defines.h"
#include "org_futo_voiceinput_BarkModel.h"
#include "jni_common.h"
#include "jni_utils.h"

struct BarkModelState {
    bark::Model *model = nullptr;
    volatile int cancel_flag = 0;
};

// Returns a handle, or throws IllegalArgumentException with the reason
static jlong BarkModel_openFromBuffer(JNIEnv *env, jclass clazz, jobject buffer) {
    auto *buffer_address = reinterpret_cast<const uint8_t *>(env->GetDirectBufferAddress(buffer));
    jlong buffer_capacity = env->GetDirectBufferCapacity(buffer);
    if(buffer_address == nullptr || buffer_capacity <= 0) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "Model buffer is not a direct buffer");
        return 0L;
    }

    long num_procs = sysconf(_SC_NPROCESSORS_ONLN);
    if(num_procs < 2 || num_procs > 16) num_procs = 6; // Make sure the number is sane

    AKLOGI("Attempting to load Bark model from buffer...");
    std::string error;
    bark::Model *model = bark::load(buffer_address, (size_t)buffer_capacity, (int)num_procs, error);
    if(model == nullptr) {
        AKLOGE("Failed to load Bark model: %s", error.c_str());
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), error.c_str());
        return 0L;
    }

    auto *state = new BarkModelState();
    state->model = model;
    return reinterpret_cast<jlong>(state);
}

// Returns token ids, or null if cancelled
static jintArray BarkModel_transcribe(JNIEnv *env, jclass clazz, jlong handle, jfloatArray samples_array) {
    auto *state = reinterpret_cast<BarkModelState *>(handle);
    state->cancel_flag = 0;

    size_t num_samples = env->GetArrayLength(samples_array);
    if(num_samples > bark::max_samples()) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "Audio is too long for one Bark call");
        return nullptr;
    }

    std::vector<float> samples(num_samples);
    env->GetFloatArrayRegion(samples_array, 0, (jsize)num_samples, samples.data());

    AKLOGI("Transcribing %d samples with Bark", (int)num_samples);
    std::vector<int> tokens;
    std::string error;
    if(!bark::transcribe(state->model, samples.data(), samples.size(), &state->cancel_flag, tokens, error)) {
        if(state->cancel_flag) {
            AKLOGI("Bark transcription cancelled");
            return nullptr;
        }
        AKLOGE("Bark transcription failed: %s", error.c_str());
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), error.c_str());
        return nullptr;
    }

    jintArray result = env->NewIntArray((jsize)tokens.size());
    if(result != nullptr && !tokens.empty()) {
        env->SetIntArrayRegion(result, 0, (jsize)tokens.size(), reinterpret_cast<const jint *>(tokens.data()));
    }
    return result;
}

static void BarkModel_cancel(JNIEnv *env, jclass clazz, jlong handle) {
    auto *state = reinterpret_cast<BarkModelState *>(handle);
    state->cancel_flag = 1;
}

static void BarkModel_close(JNIEnv *env, jclass clazz, jlong handle) {
    auto *state = reinterpret_cast<BarkModelState *>(handle);
    bark::release(state->model);
    delete state;
}

static const JNINativeMethod sMethods[] = {
        {
                const_cast<char *>("openFromBufferNative"),
                const_cast<char *>("(Ljava/nio/Buffer;)J"),
                reinterpret_cast<void *>(BarkModel_openFromBuffer)
        },
        {
                const_cast<char *>("transcribeNative"),
                const_cast<char *>("(J[F)[I"),
                reinterpret_cast<void *>(BarkModel_transcribe)
        },
        {
                const_cast<char *>("cancelNative"),
                const_cast<char *>("(J)V"),
                reinterpret_cast<void *>(BarkModel_cancel)
        },
        {
                const_cast<char *>("closeNative"),
                const_cast<char *>("(J)V"),
                reinterpret_cast<void *>(BarkModel_close)
        }
};

namespace voiceinput {
    int register_BarkModel(JNIEnv *env) {
        const char *const kClassPathName = "org/futo/voiceinput/shared/bark/BarkNative";
        return latinime::registerNativeMethods(env, kClassPathName, sMethods, NELEMS(sMethods));
    }
}
