# Qwen Chat — offline AI chat for Android

A WhatsApp-style chat app that runs **Qwen2.5-Instruct** entirely on the phone's CPU. No internet, no
account, no server. The internet is used only to download a model from the in-app Models menu.

Built for a **Vivo Y27** (MediaTek Helio G85, 6 GB RAM): Kivy UI, llama.cpp inference, and a GitHub
Actions workflow that builds the APK on every push.

## How it works

```
main.py  (Kivy UI + ChatEngine)            runs on Python 3.14 inside the APK (python-for-android)
   │  ctypes
   ▼
libchatbridge.so  (native/chat_bridge.cpp + llama.cpp, statically linked)
   │                                        cross-compiled for arm64-v8a with the Android NDK
   ▼
models/qwen2.5-1.5b-instruct-q4_k_m.gguf   memory-mapped from the app's storage
```

**Why not the `llama-cpp-python` package?** It has no python-for-android recipe. pip would compile its
C++ for the build machine (x86) instead of the phone. So this project compiles llama.cpp itself with the
NDK, plus a ~300-line C bridge (`native/chat_bridge.cpp`) that exposes six plain functions to `ctypes`.
The bridge is compiled against a pinned llama.cpp release, so the compiler checks every API call. That
matters because llama.cpp's C structs change between releases, and a wrong struct layout in Python
would crash silently.

### Memory and speed on the Vivo Y27

| Model (Q4_K_M) | File | RAM in use | Speed on Helio G85 (estimate) |
|---|---|---|---|
| **Qwen2.5-1.5B-Instruct** (default) | 1.1 GB | ~1.4 GB | ~4–6 tokens/s |
| Qwen2.5-0.5B-Instruct | 0.5 GB | ~0.7 GB | ~10–15 tokens/s |

The speeds are estimates, not measurements. The status line in the app shows the real tokens/s after
each reply.

How the app stays clear of out-of-memory kills (budget: **under 2 GB**, leaving ~4 GB to Android):

| Part (Qwen2.5-1.5B Q4_K_M) | Memory |
|---|---|
| Weights, **memory-mapped** (file-backed, so Android can drop and re-read them) | 1066 MB |
| KV cache, n_ctx 2048, f16 | 56 MB |
| Compute buffer, n_batch 256, `n_outputs_max = 1` | ~60–90 MB |
| Python + Kivy + text textures | ~150 MB |
| **Total** | **~1.4 GB** |

These are estimates. The app measures itself on the phone, as below.

- **Measured, not guessed:** after loading, the app evaluates the system prompt once. That also makes
  the first reply faster. It then reads its own memory from `/proc/self/status`. If that plus the model
  file is over 2 GB (for example a bigger model copied to the phone), it reloads with half the context,
  down to a minimum of 512 tokens.
- **`n_outputs_max = 1`:** by default this llama.cpp version reserves room for 256 outputs × 151,936
  vocabulary logits in its compute buffer (~148 MB). The app only ever reads the last token's, so the
  bridge reserves one.
- **Long conversations:** llama.cpp allocates the KV cache and buffers once, so the native memory
  doesn't grow during a chat. When the chat outgrows the context, the oldest messages are forgotten
  down to 60% of it, so that happens only every ~10 messages. The bridge then **shifts the KV cache**
  (deletes the forgotten tokens and moves the rest back) instead of re-reading the whole chat. On the
  G85 that saves up to a minute per trim. Forgotten messages are also dropped from the Python history.
- **What does grow is the UI:** each bubble holds a rendered text texture, several MB for a long reply
  at the Y27's 2.75 density. The screen keeps the newest 80 bubbles. Every 30 s and after each reply,
  the app checks its memory and Android's free memory. When either is tight (app over 2 GB, or under
  400 MB free on the phone), it cuts the screen to the newest 20 bubbles and runs the garbage collector.
  **New** clears the KV cache and the history.
- **Threads:** 4 for generation and 6 for prompt evaluation, explained below.
- llama.cpp is compiled for `armv8.2-a+dotprod+fp16`. The G85's Cortex-A75/A55 cores support these
  instructions, which speed up quantized math considerably.

**Threads on the Helio G85 (2x Cortex-A75 + 6x Cortex-A55):** generating a token reads all ~1 GB of
weights, so speed is limited by memory bandwidth. Around 4 threads saturate the G85's LPDDR4X; more
threads only add heat (the 12 nm chip throttles) and busy-wait on cores Android needs for the UI.
So generation uses **4 threads**, leaving 4 cores free. Prompt evaluation is compute-bound and comes in
bursts of a few seconds, so it uses **6**, leaving 2 cores free. These are reasoned defaults, not
measurements. The status line shows tokens/s after every reply, so you can try `N_THREADS` 3 or 5 on
your phone.

### Screen fit on the Vivo Y27 (1080 x 2388, 20:9, teardrop notch)
- **Safe area:** the app asks Android which part of the screen is covered by the status bar and
  notch (top) and the navigation bar (bottom), and pads the whole layout by that much. The bands are
  painted header-green and chat-beige, so they look like part of the app.
- **Keyboard:** when the keyboard opens, the message list shrinks and the input bar sits directly above
  the keyboard. Kivy's own `Window.softinput_mode` can't do this here: under SDL2 on Android, Kivy
  2.3.1 always reports a keyboard height of 0. So `SafeArea` in `main.py` measures it through Android's
  `getWindowVisibleDisplayFrame`.
- **Sizing:** every position comes from the layouts (`size_hint`); nothing has absolute coordinates.
  Touch targets and text use `dp`/`sp`, which Android scales to the screen density (2.75 on the Y27).
- **Bubbles** wrap their text (long URLs are broken too) and are never wider than 80% of the screen.
- **Scrolling:** sending a message and every streamed update scroll the chat to the newest text.

Settings are at the top of `main.py`: `N_CTX`, `N_THREADS` (try 2–6; the G85 has 2 fast and 6 slow cores),
`N_BATCH`, `MAX_REPLY_TOKENS`, sampling parameters and the system prompt.

## Repository layout

```
QwenChat-Android/
├── .github/workflows/build-apk.yml   CI: NDK -> llama.cpp -> model download -> buildozer -> APK
├── native/
│   ├── chat_bridge.cpp               C API used by main.py (load / start / next / abort / reset / free)
│   ├── CMakeLists.txt                links llama.cpp + ggml + bridge into ONE libchatbridge.so
│   └── build_android.sh              cross-compiles it with the NDK into libs/arm64-v8a/
├── models/                           the workflow downloads the .gguf here (don't commit it)
├── main.py                           chat UI + inference engine
├── buildozer.spec                    APK config (adds libs/arm64-v8a/*.so to the APK)
├── requirements.txt                  for the PC: desktop UI preview, buildozer
└── .gitignore
```

## Build the APK with GitHub Actions

1. **Create a GitHub repository** (private is fine) and push this folder to it, keeping the layout
   above. `.github/workflows/build-apk.yml` must be at exactly that path.
   ```
   cd QwenChat-Android
   git init
   git add .
   git commit -m "Offline Qwen chat"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
2. **The push starts the build.** Open the repo's **Actions** tab → *Build APK*. The first run takes about
   45–75 minutes (Android SDK, NDK, Python and Kivy for Android, llama.cpp). Later runs reuse caches and
   are much faster.
3. **Download the APK** when the run is green: open the run → *Artifacts* → `qwen-chat-apk`. You get a
   zip containing `qwenchat-1.0-arm64-v8a-debug.apk`.

To bundle a different model, use **Actions → Build APK → Run workflow** and pick `qwen2.5-0.5b`
(smaller and faster) or `none` (no model inside; see below).

### Where the .gguf model goes

**You don't need to place it anywhere.** The workflow downloads the official Qwen GGUF from Hugging
Face during the build and packs it into the APK. Don't commit `.gguf` files to the repo: GitHub rejects
files over 100 MB, and Git LFS's free quota is too small for a 1 GB model.

To use another model without rebuilding (or with an APK built with `none`), copy any `.gguf` into the
app's own folder on the phone. The app checks this folder first:
```
Internal storage/Android/data/org.offlineai.qwenchat/files/
```
1. Install and open the app once, so Android creates the folder.
2. Connect the phone to a PC with USB, choose *File transfer*, and copy the `.gguf` into that folder.
   Android 13 file managers often can't open `Android/data`, but a PC over USB can.
3. Restart the app.

## Install on the phone

1. Copy the APK to the phone (or download the artifact on the phone itself) and open it. Allow
   *Install unknown apps* for your file manager or browser when asked.
2. The first start unpacks the model from the APK, which takes about a minute with the 1.5B model.
   After that it starts in seconds.
3. Storage: about 2.3 GB with the 1.5B model (the APK plus the unpacked copy), about 1 GB with 0.5B.

Logs, if something goes wrong: `adb logcat -s python llama`.

## Work on the UI on your PC

```
pip install kivy==2.3.1
python main.py
```
Without `libchatbridge.so` the app runs in **demo mode** and echoes your messages, so you can try out the
UI. To chat with the real model on Linux, build the bridge for your PC with plain CMake (without the NDK
toolchain) and point `CHATBRIDGE_LIB` at the resulting `.so`.

## Build locally instead (Linux or WSL)

```
export ANDROID_NDK=~/android-ndk-r28c                      # NDK r28c from developer.android.com/ndk
bash native/build_android.sh                               # -> libs/arm64-v8a/libchatbridge.so
curl -L -o models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
pip install buildozer==1.6.0
APP_ANDROID_NDK_PATH=$ANDROID_NDK buildozer android debug  # -> bin/*.apk
```

## Notes

- Versions are pinned so the build doesn't break when upstream changes: llama.cpp `b11963`,
  python-for-android `v2026.05.09`, buildozer `1.6.0`, NDK `r28c`. To move to a newer llama.cpp, change
  `LLAMA_TAG` in the workflow. If its C API changed, the compiler will point at the lines in
  `chat_bridge.cpp` that need updating.
- The library is built for ARMv8.2 with dotprod (any arm64 phone from roughly 2018 onwards). For older
  phones, build with `ARM_ARCH=armv8-a`.
- Chinese, Japanese and Korean replies use the phone's system CJK font. Kivy's default font lacks
  those characters.
- Small models make mistakes. Don't rely on the answers for anything important.
