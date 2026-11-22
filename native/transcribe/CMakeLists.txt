# libjni_transcribe.so: transcribe.cpp (GGUF speech models) behind a small JNI layer.
#
# This is separate from libjni_latinime because transcribe.cpp brings its own, much
# newer ggml with the same symbol names as the one in native/jni/src/ggml. Everything
# except the JNI entry points stays hidden, see transcribe_jni.map.

# Debug builds included: ggml is unusably slow without optimization
set(CMAKE_C_FLAGS_DEBUG   "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -fvisibility=hidden -ffunction-sections -fdata-sections")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -ffunction-sections -fdata-sections")

set(TRANSCRIBE_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(TRANSCRIBE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TRANSCRIBE_BUILD_SHARED   OFF CACHE BOOL "" FORCE)
set(TRANSCRIBE_INSTALL        OFF CACHE BOOL "" FORCE)
set(TRANSCRIBE_USE_SYSTEM_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
# transcribe_jni.cpp refuses to open a model on CPUs without these features
set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)

# Local changes to transcribe.cpp, applied to the submodule checkout once (see patches/)
find_package(Git REQUIRED)
file(GLOB TRANSCRIBE_PATCHES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch)
# Configure again when a patch changes, so it gets applied
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${TRANSCRIBE_PATCHES})
list(SORT TRANSCRIBE_PATCHES)
foreach(patch ${TRANSCRIBE_PATCHES})
    execute_process(COMMAND ${GIT_EXECUTABLE} apply --reverse --check ${patch}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/transcribe.cpp
            RESULT_VARIABLE already_applied OUTPUT_QUIET ERROR_QUIET)
    if(NOT already_applied EQUAL 0)
        execute_process(COMMAND ${GIT_EXECUTABLE} apply ${patch}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/transcribe.cpp
                RESULT_VARIABLE apply_result)
        if(NOT apply_result EQUAL 0)
            message(FATAL_ERROR "Could not apply ${patch} to transcribe.cpp")
        endif()
    endif()
endforeach()

# transcribe.cpp's public include path uses CMAKE_SOURCE_DIR, which assumes it is the top-level project
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/transcribe.cpp/include)

add_subdirectory(transcribe.cpp EXCLUDE_FROM_ALL)

add_library(jni_transcribe SHARED transcribe_jni.cpp)

target_link_libraries(jni_transcribe PRIVATE transcribe log)
target_link_options(jni_transcribe PRIVATE
        -Wl,--exclude-libs,ALL
        -Wl,--gc-sections
        -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/transcribe_jni.map)
set_target_properties(jni_transcribe PROPERTIES
        LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/transcribe_jni.map)
