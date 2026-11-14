#!/usr/bin/env bash
# Cross-compiles llama.cpp + chat_bridge.cpp for Android arm64-v8a with the Android NDK and copies
# libchatbridge.so to ../libs/arm64-v8a/, where buildozer.spec (android.add_libs_arm64_v8a) picks it up.
#
#   ANDROID_NDK=/path/to/android-ndk-r28c ./native/build_android.sh
#
# Optional:
#   LLAMA_TAG=b11963                          llama.cpp release to build (the bridge is written against it)
#   ARM_ARCH=armv8.2-a+dotprod+fp16           CPU baseline - Helio G85 (Cortex-A75/A55) supports
#                                             dotprod + fp16, which make quantized math much faster.
#                                             Use "armv8-a" for a build that runs on any arm64 phone.
set -euo pipefail

: "${ANDROID_NDK:?Set ANDROID_NDK to the Android NDK root (e.g. android-ndk-r28c)}"
LLAMA_TAG="${LLAMA_TAG:-b11963}"
ARM_ARCH="${ARM_ARCH:-armv8.2-a+dotprod+fp16}"
ANDROID_PLATFORM="${ANDROID_PLATFORM:-android-28}"   # must match android.ndk_api in buildozer.spec

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/../libs/arm64-v8a"
BUILD="$HERE/build-android"

if [ ! -d "$HERE/llama.cpp" ]; then
    git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp "$HERE/llama.cpp"
fi

cmake -S "$HERE" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
    -DANDROID_STL=c++_static \
    -DGGML_CPU_ARM_ARCH="$ARM_ARCH"

cmake --build "$BUILD" --config Release --target chatbridge --parallel "$(nproc)"

mkdir -p "$OUT"
cp "$BUILD/libchatbridge.so" "$OUT/"
"$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" --strip-unneeded "$OUT/libchatbridge.so"

echo "Built $OUT/libchatbridge.so ($(du -h "$OUT/libchatbridge.so" | cut -f1))"
echo "Exported symbols:"
"$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-nm" -D --defined-only "$OUT/libchatbridge.so" | grep " T chat_"
