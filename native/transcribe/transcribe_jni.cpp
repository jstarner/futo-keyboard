#include <atomic>
#include <string>

#include <android/log.h>
#include <asm/hwcap.h>
#include <jni.h>
#include <sys/auxv.h>

#include "transcribe.h"

#define TAG "TranscribeJni"

struct TranscribeState {
    transcribe_model *model = nullptr;
    transcribe_session *session = nullptr;
    std::atomic<bool> cancelled{false};
};

static void throwException(JNIEnv *env, const char *clazz, const std::string &message) {
    jclass exception = env->FindClass(clazz);
    if(exception != nullptr) env->ThrowNew(exception, message.c_str());
}

// Must match GGML_CPU_ARM_ARCH in CMakeLists.txt
static bool cpuSupported() {
    const unsigned long hwcap = getauxval(AT_HWCAP);
    return (hwcap & HWCAP_ASIMDDP) != 0 && (hwcap & HWCAP_ASIMDHP) != 0;
}

static bool shouldAbort(void *user_data) {
    return static_cast<TranscribeState *>(user_data)->cancelled.load();
}

static void closeState(TranscribeState *state) {
    transcribe_session_free(state->session);
    transcribe_model_free(state->model);
    delete state;
}

extern "C" JNIEXPORT jlong JNICALL
Java_org_futo_voiceinput_shared_transcribe_TranscribeNative_openNative(JNIEnv *env, jclass, jstring path, jint threads) {
    if(!cpuSupported()) {
        throwException(env, "java/lang/IllegalArgumentException", "This CPU is not supported");
        return 0;
    }

    const char *path_chars = env->GetStringUTFChars(path, nullptr);
    if(path_chars == nullptr) return 0;

    auto *state = new TranscribeState();

    transcribe_model_load_params load_params;
    transcribe_model_load_params_init(&load_params);
    load_params.backend = TRANSCRIBE_BACKEND_CPU;

    transcribe_status status = transcribe_model_load_file(path_chars, &load_params, &state->model);
    env->ReleaseStringUTFChars(path, path_chars);

    if(status == TRANSCRIBE_OK) {
        transcribe_session_params session_params;
        transcribe_session_params_init(&session_params);
        session_params.n_threads = threads;
        status = transcribe_session_init(state->model, &session_params, &state->session);
    }

    if(status != TRANSCRIBE_OK) {
        closeState(state);
        throwException(env, "java/lang/IllegalArgumentException", transcribe_status_string(status));
        return 0;
    }

    transcribe_set_abort_callback(state->session, shouldAbort, state);
    return reinterpret_cast<jlong>(state);
}

// Returns null if cancelled
extern "C" JNIEXPORT jstring JNICALL
Java_org_futo_voiceinput_shared_transcribe_TranscribeNative_transcribeNative(JNIEnv *env, jclass, jlong handle, jfloatArray samples) {
    auto *state = reinterpret_cast<TranscribeState *>(handle);

    const jsize n_samples = env->GetArrayLength(samples);
    jfloat *pcm = env->GetFloatArrayElements(samples, nullptr);
    if(pcm == nullptr) return nullptr;

    transcribe_run_params run_params;
    transcribe_run_params_init(&run_params);
    run_params.timestamps = TRANSCRIBE_TIMESTAMPS_NONE;

    const transcribe_status status = transcribe_run(state->session, pcm, n_samples, &run_params);
    env->ReleaseFloatArrayElements(samples, pcm, JNI_ABORT);

    if(status == TRANSCRIBE_ERR_ABORTED) return nullptr;

    if(status != TRANSCRIBE_OK && status != TRANSCRIBE_ERR_OUTPUT_TRUNCATED) {
        throwException(env, "java/lang/RuntimeException", transcribe_status_string(status));
        return nullptr;
    }

    if(status == TRANSCRIBE_ERR_OUTPUT_TRUNCATED) {
        __android_log_print(ANDROID_LOG_WARN, TAG, "Transcript was truncated");
    }

    return env->NewStringUTF(transcribe_full_text(state->session));
}

extern "C" JNIEXPORT void JNICALL
Java_org_futo_voiceinput_shared_transcribe_TranscribeNative_resetCancelNative(JNIEnv *, jclass, jlong handle) {
    reinterpret_cast<TranscribeState *>(handle)->cancelled.store(false);
}

extern "C" JNIEXPORT void JNICALL
Java_org_futo_voiceinput_shared_transcribe_TranscribeNative_cancelNative(JNIEnv *, jclass, jlong handle) {
    reinterpret_cast<TranscribeState *>(handle)->cancelled.store(true);
}

extern "C" JNIEXPORT void JNICALL
Java_org_futo_voiceinput_shared_transcribe_TranscribeNative_closeNative(JNIEnv *, jclass, jlong handle) {
    closeState(reinterpret_cast<TranscribeState *>(handle));
}
