[app]

title = Qwen Chat
package.name = qwenchat
package.domain = org.offlineai
version = 1.0

source.dir = .
# gguf: the model the GitHub workflow downloads into models/ is packed into the APK
source.include_exts = py,png,jpg,kv,json,gguf
# Build inputs that must NOT end up inside the APK's Python files
source.exclude_dirs = native,libs,model-cache,.github,.git,bin,.buildozer,tests,__pycache__

# python-for-android recipes. "android" + "pyjnius" give access to Android APIs (storage folders,
# native library dir). llama.cpp is NOT a requirement here: it is built separately with the NDK
# (native/build_android.sh) and added below as a prebuilt .so.
requirements = python3,kivy,pyjnius,android

orientation = portrait
fullscreen = 0
android.presplash_color = #075E54

# No permissions at all - the app never touches the network. A model you copy yourself goes into
# Android/data/org.offlineai.qwenchat/files, which the app can read without any permission.
# android.permissions = (none)

# ---- Android SDK / NDK ---------------------------------------------------------------------------
android.api = 35
# Android 9+. Must match ANDROID_PLATFORM (android-28) used to compile libchatbridge.so
android.minapi = 28
android.ndk_api = 28
# NDK used by python-for-android. The GitHub workflow downloads NDK r28c once, compiles llama.cpp with
# it, and points buildozer at the same folder through the APP_ANDROID_NDK_PATH environment variable.
android.ndk = 28c
android.accept_sdk_license = True

# 64-bit ARM only: the Helio G85 is arm64, and llama.cpp needs 64-bit for models this size
android.archs = arm64-v8a

# ---- llama.cpp --------------------------------------------------------------------------------------
# libchatbridge.so = llama.cpp + native/chat_bridge.cpp, cross-compiled for arm64-v8a with the NDK by
# native/build_android.sh (run it BEFORE buildozer; the workflow does). It is copied into the APK's
# native library folder (lib/arm64-v8a/), where main.py loads it with ctypes.
android.add_libs_arm64_v8a = libs/arm64-v8a/*.so

android.allow_backup = False
android.debug_artifact = apk
android.release_artifact = apk

# Pinned python-for-android release (Python 3.14, Kivy 2.3.1, recommends NDK r28c)
p4a.branch = v2026.05.09


[buildozer]
log_level = 2
warn_on_root = 1
