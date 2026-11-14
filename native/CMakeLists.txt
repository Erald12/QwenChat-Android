# Builds libchatbridge.so: chat_bridge.cpp + llama.cpp + ggml, all linked statically into ONE shared
# library. The app loads it with ctypes. One self-contained .so (static libc++ too) means no
# dependency clashes with the libraries python-for-android ships.
#
# Configure with the Android NDK toolchain - see build_android.sh.

cmake_minimum_required(VERSION 3.18)
project(chatbridge C CXX)

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp" CACHE PATH "Path to the llama.cpp source tree")

# llama.cpp / ggml as static libraries, without tools, tests, examples, server or networking
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL        OFF CACHE BOOL "" FORCE)   # not part of the NDK stable API
set(GGML_NATIVE          OFF CACHE BOOL "" FORCE)   # cross-compiling: the build machine is not the phone
set(GGML_OPENMP          OFF CACHE BOOL "" FORCE)   # ggml's own thread pool instead of an OpenMP runtime
set(GGML_LLAMAFILE       OFF CACHE BOOL "" FORCE)   # not supported on Android

# Static libraries that end up inside a shared library must be position independent
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(${LLAMA_CPP_DIR} llama.cpp EXCLUDE_FROM_ALL)

add_library(chatbridge SHARED chat_bridge.cpp)
target_compile_features(chatbridge PRIVATE cxx_std_17)
target_link_libraries(chatbridge PRIVATE llama)

# Export only the chat_* functions (marked with visibility("default")); keep llama/ggml symbols private
set_target_properties(chatbridge PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

if (ANDROID)
    target_link_libraries(chatbridge PRIVATE log)    # __android_log_write -> adb logcat -s llama
    target_link_options(chatbridge PRIVATE
        -Wl,--exclude-libs,ALL                       # don't re-export symbols of the static libs
        -Wl,--gc-sections
        -Wl,-z,max-page-size=16384)                  # 16 KB page alignment (required on Android 15+ devices)
endif()
