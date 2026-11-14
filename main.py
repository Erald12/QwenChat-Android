"""
Qwen Chat - a fully offline chatbot for Android (Kivy UI + llama.cpp).

The model runs on the phone's CPU through libchatbridge.so, a small C library built from llama.cpp
(see native/). Nothing is sent over the network - the app doesn't even request the INTERNET permission.

Tuned for a Vivo Y27 (MediaTek Helio G85, 6 GB RAM): settings below.
"""
import codecs
import ctypes
import gc
import glob
import os
import queue
import threading
import time

# ---- settings ---------------------------------------------------------------------------------

# Tuned for the MediaTek Helio G85 (2x Cortex-A75 @ 2.0 GHz + 6x Cortex-A55 @ 1.8 GHz, 12 nm, LPDDR4X)
# with 6 GB RAM.
#
# Threads. Generating a token reads every weight once (~1 GB for the 1.5B model), so it is limited by
# memory bandwidth, not by CPU. About 4 threads already saturate the G85's memory bus. More threads
# only add heat (the 12 nm chip throttles) and take the cores Android needs for the UI. llama.cpp's
# threads also spin while waiting, so each one keeps a core at 100%. 4 threads leave 4 of the 8 cores
# to Android. Prompt evaluation is compute-bound and comes in short bursts (a few seconds per message,
# thanks to the KV cache reuse), so it uses 6 threads and still leaves 2 cores to Android.
N_THREADS = 4            # generation
N_THREADS_BATCH = 6      # prompt evaluation

# Memory - the app must stay under MEMORY_BUDGET_MB. Qwen2.5-1.5B-Instruct Q4_K_M:
#   weights           1066 MB  memory-mapped: file-backed pages Android can drop and re-read
#   KV cache            56 MB  n_ctx 2048 x 28 layers x 2 KV heads x 128 dims x 2 (K, V) x 2 bytes (f16)
#   compute buffer  ~60-90 MB  scales with N_BATCH (llama.cpp prints the exact size to logcat)
#   Python + Kivy  ~150 MB     interpreter, UI, text textures
#   total          ~1.4 GB     -> ~0.6 GB headroom under the 2 GB budget
# The native part (KV cache + compute buffer) is allocated once at load and does not grow during a
# conversation. After loading, the app measures itself and halves N_CTX if it is over the budget
# (e.g. with a bigger model copied to the phone).
N_CTX = 2048             # context window. 2048 tokens = ~25 messages. 4096 would fit too, but every
                         # generated token reads the whole KV cache, so a fuller context is slower
N_BATCH = 256            # prompt tokens per step: big enough for full speed on the A55s, small buffer
MIN_CTX = 512            # never shrink the context below this
MEMORY_BUDGET_MB = 2000  # hard ceiling for the app (model + context + UI)
LOW_SYSTEM_MEMORY_MB = 400   # Android is close to killing apps below this much free memory

MAX_REPLY_TOKENS = 512   # longest reply; also reserved in the context when trimming old messages
TRIM_TARGET = 0.6        # when the chat outgrows the context, forget old messages down to 60% of it,
                         # so this happens every ~10 messages instead of on every message
MAX_BUBBLES = 80         # messages kept on screen (each bubble holds a text texture of up to several MB)
BUBBLES_UNDER_PRESSURE = 20  # kept when memory runs low

# Qwen2.5 recommended sampling settings
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
REPEAT_PENALTY = 1.05

SYSTEM_PROMPT = ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant. "
                 "Keep answers short and clear unless the user asks for detail.")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLED_MODEL_DIR = os.path.join(APP_DIR, "models")   # the GitHub workflow downloads the model here
LIBRARY_NAME = "libchatbridge.so"

try:
    from kivy.utils import platform as PLATFORM
except ImportError:  # engine used without Kivy (tests)
    PLATFORM = "android" if "ANDROID_ARGUMENT" in os.environ else "desktop"
IS_ANDROID = PLATFORM == "android"


# ==== inference engine (no Kivy below until the UI section) ===================================

def android_context():
    from jnius import autoclass
    return autoclass("org.kivy.android.PythonActivity").mActivity


def external_model_dir():
    """
    Folder where you can drop your own .gguf without rebuilding the APK:
    Android: /storage/emulated/0/Android/data/<package>/files  (copy it there over USB)
    Desktop: ./models
    """
    if IS_ANDROID:
        try:
            files_dir = android_context().getExternalFilesDir(None)
            if files_dir is not None:
                return files_dir.getAbsolutePath()
        except Exception:
            pass
        return None
    return BUNDLED_MODEL_DIR


def find_model():
    """
    Returns the path of the model to load: a .gguf in the external folder wins over the bundled one.
    """
    for folder in (external_model_dir(), BUNDLED_MODEL_DIR):
        if folder and os.path.isdir(folder):
            models = sorted(glob.glob(os.path.join(folder, "*.gguf")))
            if models:
                return models[0]
    return None


def process_memory_mb():
    """
    This app's memory in MB from /proc/self/status: {"anon": heap and native buffers, "file": mapped
    files (the model weights), "rss": total resident}. None where /proc doesn't exist (Windows).
    """
    fields = {"RssAnon:": "anon", "RssFile:": "file", "VmRSS:": "rss"}
    result = {}
    try:
        with open("/proc/self/status") as file:
            for line in file:
                key = line.split(":")[0] + ":"
                if key in fields:
                    result[fields[key]] = int(line.split()[1]) // 1024
    except (OSError, ValueError):
        return None
    return result if "anon" in result else None


def available_memory_mb():
    try:
        with open("/proc/meminfo") as file:
            for line in file:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError):
        pass
    return None


def load_library():
    """
    Loads libchatbridge.so. On Android it is packaged in the APK's native library folder
    (buildozer.spec: android.add_libs_arm64_v8a), where the linker finds it by name.
    """
    candidates = [os.environ.get("CHATBRIDGE_LIB"), LIBRARY_NAME]
    if IS_ANDROID:
        try:
            native_dir = android_context().getApplicationInfo().nativeLibraryDir
            candidates.append(os.path.join(native_dir, LIBRARY_NAME))
        except Exception:
            pass
    candidates.append(os.path.join(APP_DIR, "libs", LIBRARY_NAME))
    errors = []
    for candidate in filter(None, candidates):
        try:
            return ctypes.CDLL(candidate)
        except OSError as e:
            errors.append(f"{candidate}: {e}")
    raise OSError("could not load " + LIBRARY_NAME + "\n" + "\n".join(errors))


class LlamaBridge:
    """
    ctypes wrapper around native/chat_bridge.cpp. Not thread-safe: ChatEngine calls it from one thread
    (except abort(), which is safe from any thread).
    """

    def __init__(self, lib=None):
        self.lib = lib or load_library()
        lib = self.lib
        lib.chat_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.chat_load.restype = ctypes.c_int
        lib.chat_start.argtypes = [ctypes.c_char_p, ctypes.c_float, ctypes.c_float, ctypes.c_int,
                                   ctypes.c_float, ctypes.c_uint]
        lib.chat_start.restype = ctypes.c_int
        lib.chat_next.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.chat_next.restype = ctypes.c_int
        lib.chat_count_tokens.argtypes = [ctypes.c_char_p]
        lib.chat_count_tokens.restype = ctypes.c_int
        lib.chat_cached_tokens.argtypes = []
        lib.chat_cached_tokens.restype = ctypes.c_int
        for name in ("chat_abort", "chat_reset", "chat_free"):
            getattr(lib, name).argtypes = []
            getattr(lib, name).restype = None
        for name in ("chat_error", "chat_system_info"):
            getattr(lib, name).argtypes = []
            getattr(lib, name).restype = ctypes.c_char_p
        self.buffer = ctypes.create_string_buffer(1024)

    def error(self):
        return (self.lib.chat_error() or b"").decode("utf-8", "replace")

    def system_info(self):
        return (self.lib.chat_system_info() or b"").decode("utf-8", "replace")

    def load(self, path, n_ctx, n_threads, n_threads_batch, n_batch):
        if self.lib.chat_load(path.encode("utf-8"), n_ctx, n_threads, n_threads_batch, n_batch) != 0:
            raise RuntimeError(self.error())

    def count_tokens(self, text):
        return self.lib.chat_count_tokens(text.encode("utf-8"))

    def cached_tokens(self):
        return self.lib.chat_cached_tokens()

    def start(self, prompt, temp, top_p, top_k, repeat_penalty, seed):
        """Evaluates the prompt (reusing the cached part). Returns the number of new prompt tokens."""
        result = self.lib.chat_start(prompt.encode("utf-8"), temp, top_p, top_k, repeat_penalty, seed)
        if result < 0:
            raise RuntimeError(self.error())
        return result

    def next(self):
        """Returns the raw bytes of the next text piece, or None at the end of the reply."""
        n = self.lib.chat_next(self.buffer, len(self.buffer))
        if n < 0:
            raise RuntimeError(self.error())
        return self.buffer.raw[:n] if n else None

    def abort(self):
        self.lib.chat_abort()

    def reset(self):
        self.lib.chat_reset()

    def free(self):
        self.lib.chat_free()


class DemoBridge:
    """
    Stand-in for LlamaBridge when libchatbridge.so isn't available (running main.py on a PC to work on
    the UI). Streams a canned reply.
    """

    def load(self, path, n_ctx, n_threads, n_threads_batch, n_batch):
        time.sleep(0.5)

    def cached_tokens(self):
        return 0

    def system_info(self):
        return "demo mode"

    def count_tokens(self, text):
        return len(text.encode("utf-8")) // 3 + 1

    def start(self, prompt, *args):
        question = prompt.rsplit("<|im_start|>user\n", 1)[-1].split("<|im_end|>")[0]
        self.reply = iter(f"(Demo mode - no model loaded.) You said: **{question}**".encode("utf-8"))
        return self.count_tokens(prompt)

    def next(self):
        time.sleep(0.03)
        byte = next(self.reply, None)
        return None if byte is None else bytes([byte])

    def abort(self):
        pass

    def reset(self):
        pass

    def free(self):
        pass


def chatml(messages):
    """
    Qwen's chat format. messages: [{"role": "user" | "assistant", "content": str}].
    """
    def clean(text):
        # A user typing "<|im_end|>" must not be able to inject control tokens
        return text.replace("<|", "<​|")

    parts = [f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"]
    for message in messages:
        parts.append(f"<|im_start|>{message['role']}\n{clean(message['content'])}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


class ChatEngine:
    """
    Runs the model on one background thread so the UI never freezes.
    Callbacks are called from that thread - the UI wraps them to run on its own thread.
    """

    def __init__(self, bridge_factory=None):
        self.bridge_factory = bridge_factory or LlamaBridge
        self.bridge = None
        self.jobs = queue.Queue()
        self.stop_requested = threading.Event()
        self.busy = False
        self.ready = False
        self.model_path = None
        self.n_ctx = N_CTX
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            job()

    def load(self, path, on_ready, on_error):
        def job():
            try:
                try:
                    self.bridge = self.bridge_factory()
                except OSError:
                    if IS_ANDROID:
                        raise
                    self.bridge = DemoBridge()  # desktop without the native library: UI demo
                memory = self._load_within_budget(path or "")
                self.model_path = path
                self.ready = True
                on_ready(self.bridge.system_info(), memory)
            except Exception as e:
                on_error(str(e))
        self.jobs.put(job)

    def _load_within_budget(self, path):
        """
        Loads the model, measures the app's real memory use and, if it is over MEMORY_BUDGET_MB,
        reloads with half the context. Returns the measurement ({"estimate_mb", "n_ctx", ...}).
        """
        model_mb = os.path.getsize(path) // (1024 * 1024) if path and os.path.exists(path) else 0
        n_ctx = N_CTX
        while True:
            self.bridge.load(path, n_ctx, N_THREADS, N_THREADS_BATCH, N_BATCH)
            self.n_ctx = n_ctx
            # Warm-up: evaluating the system prompt now makes the first reply faster (it is cached),
            # and touches the compute buffers so the measurement below is real
            self.bridge.start(chatml([]), TEMPERATURE, TOP_P, TOP_K, REPEAT_PENALTY, 0)
            memory = process_memory_mb()
            if memory is None:  # no /proc (desktop): nothing to measure
                return {"n_ctx": n_ctx}
            # Worst case: every page of the memory-mapped weights resident at once
            estimate = memory["anon"] + model_mb
            memory.update(n_ctx=n_ctx, model_mb=model_mb, estimate_mb=estimate)
            if estimate <= MEMORY_BUDGET_MB or n_ctx <= MIN_CTX:
                return memory
            n_ctx //= 2

    def fit_history(self, messages):
        """
        Keeps the prompt inside the context, leaving room for a full reply. When it doesn't fit,
        forgets the oldest messages down to TRIM_TARGET of the space, so this happens only every few
        messages. The bridge then shifts the KV cache instead of re-reading the whole chat.
        Returns (prompt, messages that were kept).
        """
        budget = self.n_ctx - MAX_REPLY_TOKENS - 16
        kept = list(messages)
        prompt = chatml(kept)
        if self.bridge.count_tokens(prompt) <= budget:
            return prompt, kept
        target = int(budget * TRIM_TARGET)
        while len(kept) > 1:
            kept = kept[2:] if len(kept) > 2 else kept[1:]  # drop the oldest question + answer
            prompt = chatml(kept)
            if self.bridge.count_tokens(prompt) <= target:
                break
        return prompt, kept

    def send(self, messages, on_text, on_done, on_error):
        """
        Generates the assistant's reply to `messages` (the last one is the user's new message).
        on_text(text so far) is called while streaming, on_done(text, stats) at the end.
        """
        self.stop_requested.clear()

        def job():
            self.busy = True
            text = ""
            try:
                prompt, kept = self.fit_history(messages)
                if self.bridge.count_tokens(prompt) > self.n_ctx - MAX_REPLY_TOKENS - 16:
                    raise RuntimeError("Your message is too long for the model's context.")
                started = time.time()
                prompt_tokens = self.bridge.start(prompt, TEMPERATURE, TOP_P, TOP_K, REPEAT_PENALTY,
                                                  int(time.time()) & 0xFFFFFFFF)
                prompt_seconds = time.time() - started

                # Tokens can end in the middle of a multi-byte character (emoji, Chinese...):
                # the incremental decoder holds those bytes back until the character is complete
                decoder = codecs.getincrementaldecoder("utf-8")("replace")
                generated = 0
                gen_started = time.time()
                last_push = 0.0
                while generated < MAX_REPLY_TOKENS and not self.stop_requested.is_set():
                    piece = self.bridge.next()
                    if piece is None:
                        break
                    generated += 1
                    text += decoder.decode(piece)
                    now = time.time()
                    if now - last_push > 0.08:  # ~12 UI updates per second is plenty
                        on_text(text)
                        last_push = now
                text += decoder.decode(b"", final=True)
                gen_seconds = max(time.time() - gen_started, 1e-6)
                stats = {
                    "prompt_tokens": prompt_tokens, "prompt_seconds": prompt_seconds,
                    "tokens": generated, "tokens_per_second": generated / gen_seconds,
                    "dropped_messages": len(messages) - len(kept),
                    "stopped": self.stop_requested.is_set(),
                    "context_used": self.bridge.cached_tokens() / self.n_ctx,
                }
                on_done(text, stats)
            except Exception as e:
                on_error(str(e), text)
            finally:
                self.busy = False
        self.jobs.put(job)

    def stop(self):
        self.stop_requested.set()
        if self.bridge is not None:
            self.bridge.abort()  # also interrupts a long prompt evaluation

    def reset(self):
        def job():
            if self.bridge is not None:
                self.bridge.reset()
        self.jobs.put(job)

    def shutdown(self):
        self.stop()

        def job():
            if self.bridge is not None:
                self.bridge.free()
        self.jobs.put(job)
        self.jobs.put(None)


# ==== UI (Kivy) =================================================================================

from kivy.app import App  # noqa: E402
from kivy.clock import Clock, mainthread  # noqa: E402
from kivy.core.text import Label as CoreLabel  # noqa: E402
from kivy.core.window import Window  # noqa: E402
from kivy.graphics import Color, Ellipse, Rectangle, RoundedRectangle  # noqa: E402
from kivy.metrics import dp, sp  # noqa: E402
from kivy.uix.behaviors import ButtonBehavior  # noqa: E402
from kivy.uix.boxlayout import BoxLayout  # noqa: E402
from kivy.uix.gridlayout import GridLayout  # noqa: E402
from kivy.uix.label import Label  # noqa: E402
from kivy.uix.scrollview import ScrollView  # noqa: E402
from kivy.uix.textinput import TextInput  # noqa: E402
from kivy.uix.widget import Widget  # noqa: E402
from kivy.utils import escape_markup, get_color_from_hex as hex_color  # noqa: E402

# WhatsApp-like colors
HEADER_GREEN = hex_color("#075E54")
SEND_GREEN = hex_color("#00A884")
CHAT_BACKGROUND = hex_color("#EFEAE2")
USER_BUBBLE = hex_color("#D9FDD3")
BOT_BUBBLE = hex_color("#FFFFFF")
TEXT_COLOR = hex_color("#111B21")
TIME_COLOR = hex_color("#667781")
NOTICE_BUBBLE = hex_color("#FFF3C4")

BUBBLE_MAX_WIDTH = 0.80          # fraction of the screen width
BUBBLE_PADDING_X = dp(10)


def pick_font():
    """
    Kivy's default font (Roboto) has no Chinese / Japanese / Korean glyphs, and Qwen often answers in
    Chinese when asked. Use the phone's CJK system font when there is one.
    """
    for path in ("/system/fonts/NotoSansCJK-Regular.ttc", "/system/fonts/NotoSansSC-Regular.otf",
                 "/system/fonts/DroidSansFallback.ttf", "/system/fonts/DroidSansFallbackFull.ttf",
                 "C:/Windows/Fonts/msyh.ttc"):  # last one: desktop preview on Windows
        if os.path.exists(path):
            return path
    return "Roboto"


CJK_FONT = pick_font()
CJK_RANGES = ((0x2E80, 0x9FFF), (0xAC00, 0xD7AF), (0xF900, 0xFAFF), (0xFF00, 0xFFEF))


def font_for(text):
    # Roboto (which has a real bold face) unless the text contains Chinese / Japanese / Korean
    for char in text:
        code = ord(char)
        if code >= 0x2E80 and any(low <= code <= high for low, high in CJK_RANGES):
            return CJK_FONT
    return "Roboto"


def to_markup(text):
    """Escapes Kivy markup and renders **bold** (complete pairs only, so streaming text stays valid)."""
    parts = escape_markup(text).split("**")
    out = parts[0]
    for i, part in enumerate(parts[1:], 1):
        if i % 2 == 1 and i < len(parts) - 1:
            out += "[b]" + part
        elif i % 2 == 0:
            out += "[/b]" + part
        else:
            out += "**" + part  # unmatched trailing ** (still streaming)
    return out


class SafeArea:
    """
    Measures what covers the app's window on Android, in Kivy pixels: the status bar and the camera
    notch at the top, the navigation bar and the on-screen keyboard at the bottom.

    Kivy's own keyboard handling can't be used: under SDL2 on Android, Kivy 2.3.1's
    Window.keyboard_height is a stub that always returns 0, so Window.softinput_mode has no effect.
    Instead this compares where the app's content view is on screen with the part of the screen
    Android says is really visible (getWindowVisibleDisplayFrame). That works both when Android shrinks
    the window for the keyboard (adjustResize - then nothing is covered and the padding is 0) and when
    the window stays full size (edge-to-edge on Android 15+) - so the padding is never applied twice.
    """

    def __init__(self):
        self.available = False
        # Desktop testing: "left,top,right,bottom" in pixels, e.g. "0,32,0,48"
        self.simulated = os.environ.get("QWENCHAT_TEST_INSETS")
        if not IS_ANDROID:
            return
        try:
            from jnius import autoclass
            self.Rect = autoclass("android.graphics.Rect")
            activity = android_context()
            self.decor = activity.getWindow().getDecorView()
            self.content = self.decor.findViewById(0x01020002)  # android.R.id.content
            self.available = self.content is not None
            self.use_adjust_resize()
        except Exception as e:
            print(f"[QwenChat] safe area detection unavailable: {e}")

    @staticmethod
    def use_adjust_resize():
        """
        Ask Android to shrink the window above the keyboard instead of sliding the whole window up
        (which would push the header off the screen).
        """
        from android.runnable import run_on_ui_thread

        @run_on_ui_thread
        def apply():
            android_context().getWindow().setSoftInputMode(0x10)  # SOFT_INPUT_ADJUST_RESIZE
        apply()

    def insets(self):
        if self.simulated:
            return tuple(int(value) for value in self.simulated.split(","))
        if not self.available:
            return 0, 0, 0, 0
        try:
            view = self.Rect()
            self.content.getGlobalVisibleRect(view)       # where the app is drawn (screen pixels)
            visible = self.Rect()
            self.decor.getWindowVisibleDisplayFrame(visible)  # not covered by bars or keyboard
            left = max(0, visible.left - view.left)
            top = max(0, visible.top - view.top)
            right = max(0, view.right - visible.right)
            bottom = max(0, view.bottom - visible.bottom)

            # The teardrop notch: normally inside the status bar, but count it in case the status
            # bar is hidden
            root_insets = self.decor.getRootWindowInsets()
            cutout = root_insets.getDisplayCutout() if root_insets is not None else None
            if cutout is not None:
                top = max(top, cutout.getSafeInsetTop() - view.top)
                left = max(left, cutout.getSafeInsetLeft() - view.left)

            # Screen pixels -> Kivy window pixels (identical unless the surface is scaled)
            width, height = view.width(), view.height()
            sx = Window.system_size[0] / width if width > 0 else 1
            sy = Window.system_size[1] / height if height > 0 else 1
            return (round(left * sx), round(top * sy), round(right * sx), round(bottom * sy))
        except Exception as e:
            print(f"[QwenChat] reading safe area failed: {e}")
            return 0, 0, 0, 0


class RoundButton(ButtonBehavior, Label):
    def __init__(self, color, **kwargs):
        super().__init__(**kwargs)
        self.fill = color
        with self.canvas.before:
            self.fill_color = Color(*color)
            self.circle = Ellipse()
        self.bind(pos=self._redraw, size=self._redraw)

    def _redraw(self, *_):
        self.circle.pos, self.circle.size = self.pos, self.size

    def set_fill(self, color):
        self.fill_color.rgba = color


class Bubble(BoxLayout):
    """
    One chat message: a rounded bubble, right-aligned for the user, left-aligned for the bot,
    with the time in the corner.
    """

    def __init__(self, text, mine, notice=False, **kwargs):
        super().__init__(size_hint_y=None, padding=(dp(10), dp(2)), **kwargs)
        self.mine = mine
        self.raw_text = ""
        self.box = BoxLayout(orientation="vertical", size_hint=(None, None),
                             padding=(BUBBLE_PADDING_X, dp(6), BUBBLE_PADDING_X, dp(4)), spacing=dp(1))
        with self.box.canvas.before:
            Color(*(NOTICE_BUBBLE if notice else USER_BUBBLE if mine else BOT_BUBBLE))
            self.background = RoundedRectangle(radius=[dp(10)])
        self.box.bind(pos=self._redraw, size=self._redraw)

        self.label = Label(markup=True, color=TEXT_COLOR, font_size=sp(15),
                           size_hint=(None, None), halign="left", valign="top")
        self.time = Label(text=time.strftime("%H:%M"), color=TIME_COLOR, font_size=sp(10),
                          size_hint=(None, None), halign="right", valign="bottom")
        self.time.texture_update()
        self.time_width = self.time.texture_size[0]  # measured once - text_size changes it later
        self.box.add_widget(self.label)
        self.box.add_widget(self.time)
        if mine:
            self.add_widget(Widget())
            self.add_widget(self.box)
        else:
            self.add_widget(self.box)
            self.add_widget(Widget())
        self.bind(width=lambda *_: self.relayout())
        self.set_text(text)

    def _redraw(self, *_):
        self.background.pos, self.background.size = self.box.pos, self.box.size

    def set_text(self, text):
        self.raw_text = text
        self.label.font_name = font_for(text)
        self.label.text = to_markup(text) if text else "..."
        self.relayout()

    def natural_width(self, text):
        # Width of the longest line, measured without rendering (cheap even for long replies)
        measure = CoreLabel(font_size=self.label.font_size, font_name=self.label.font_name)
        widest = 0
        for line in text.split("\n")[:40]:
            widest = max(widest, measure.get_extents(line[:300])[0])
        return widest

    def relayout(self):
        # The bubble is at most 80% of the screen width, and always fits the row it sits in
        row_width = self.width if self.width > 100 else Window.width
        bubble_max = min(Window.width * BUBBLE_MAX_WIDTH, row_width - self.padding[0] - self.padding[2])
        inner_max = bubble_max - BUBBLE_PADDING_X * 2
        text_width = min(self.natural_width(self.raw_text or "..."), inner_max)
        # text_size width makes the label wrap (long words are broken too)
        self.label.text_size = (text_width, None)
        self.label.texture_update()
        self.label.size = self.label.texture_size
        bubble_width = max(self.label.width, self.time_width + dp(4))
        self.time.size = (bubble_width, self.time.texture_size[1])
        self.time.text_size = self.time.size
        self.box.size = (bubble_width + BUBBLE_PADDING_X * 2, self.label.height + self.time.height + dp(11))
        self.height = self.box.height + dp(4)


class ChatScreen(BoxLayout):
    """
    Header / messages / input bar stacked vertically. Everything is placed by the layouts
    (size_hint) - no absolute positions. The root's padding is the safe area: the space under the
    status bar + notch, the navigation bar and the keyboard. Padding the root (instead of scrolling)
    shrinks the message list, so the input bar always sits directly above the keyboard.
    """

    def __init__(self, app, **kwargs):
        super().__init__(orientation="vertical", **kwargs)
        self.app = app
        self.insets = (0, 0, 0, 0)

        # The safe-area bands are painted so they don't show as empty stripes:
        # header green behind the status bar / notch, chat background behind the navigation bar
        with self.canvas.before:
            Color(*HEADER_GREEN)
            self.top_band = Rectangle()
            Color(*CHAT_BACKGROUND)
            self.bottom_band = Rectangle()
        self.bind(pos=self._paint_bands, size=self._paint_bands)

        # Header: avatar, name + status, "New chat"
        header = BoxLayout(size_hint_y=None, height=dp(58), padding=(dp(10), dp(8)), spacing=dp(10))
        with header.canvas.before:
            Color(*HEADER_GREEN)
            header_bg = Rectangle()
        header.bind(pos=lambda w, _: setattr(header_bg, "pos", w.pos),
                    size=lambda w, _: setattr(header_bg, "size", w.size))
        avatar = RoundButton(hex_color("#25D366"), text="Q", bold=True, font_size=sp(20),
                             size_hint=(None, None), size=(dp(42), dp(42)))
        titles = BoxLayout(orientation="vertical")
        self.title = Label(text="Qwen", bold=True, font_size=sp(17), halign="left", valign="bottom")
        self.status = Label(text="starting...", font_size=sp(12), color=(1, 1, 1, 0.8),
                            halign="left", valign="top")
        for label in (self.title, self.status):
            label.bind(size=lambda w, _: setattr(w, "text_size", w.size))
            titles.add_widget(label)
        new_chat = RoundButton(hex_color("#128C7E"), text="New", font_size=sp(13),
                               size_hint=(None, None), size=(dp(42), dp(42)))
        new_chat.bind(on_release=lambda *_: self.app.new_chat())
        header.add_widget(avatar)
        header.add_widget(titles)
        header.add_widget(new_chat)

        # Messages
        self.scroll = ScrollView(do_scroll_x=False, bar_width=dp(3))
        with self.scroll.canvas.before:
            Color(*CHAT_BACKGROUND)
            chat_bg = Rectangle()
        self.scroll.bind(pos=lambda w, _: setattr(chat_bg, "pos", w.pos),
                         size=lambda w, _: setattr(chat_bg, "size", w.size))
        self.messages = GridLayout(cols=1, size_hint_y=None, spacing=dp(2), padding=(0, dp(8)))
        self.messages.bind(minimum_height=self.messages.setter("height"))
        # Column = empty space + messages, at least as tall as the view: a short chat sits at the
        # bottom, right above the input bar (like WhatsApp), and scroll_y = 0 is always the newest message
        self.column = BoxLayout(orientation="vertical", size_hint_y=None)
        self.column.add_widget(Widget())
        self.column.add_widget(self.messages)
        self.messages.bind(height=self._fit_column)
        self.scroll.bind(height=self._fit_column)
        self.scroll.add_widget(self.column)

        # Input bar
        bar = BoxLayout(size_hint_y=None, height=dp(60), padding=(dp(8), dp(8)), spacing=dp(8))
        with bar.canvas.before:
            Color(*CHAT_BACKGROUND)
            bar_bg = Rectangle()
        bar.bind(pos=lambda w, _: setattr(bar_bg, "pos", w.pos),
                 size=lambda w, _: setattr(bar_bg, "size", w.size))
        self.input = TextInput(hint_text="Message", multiline=True, font_size=sp(15), font_name=CJK_FONT,
                               background_normal="", background_active="", background_color=(1, 1, 1, 1),
                               foreground_color=TEXT_COLOR, cursor_color=SEND_GREEN,
                               padding=(dp(14), dp(11)))
        self.input.bind(minimum_height=self._grow_input)
        self.send_button = RoundButton(SEND_GREEN, text="Send", bold=True, font_size=sp(13),
                                       size_hint=(None, None), size=(dp(44), dp(44)))
        self.send_button.bind(on_release=lambda *_: self.app.send_or_stop())
        bar.add_widget(self.input)
        bar.add_widget(self.send_button)
        self.bar = bar

        self.add_widget(header)
        self.add_widget(self.scroll)
        self.add_widget(bar)

    def _fit_column(self, *_):
        self.column.height = max(self.messages.height, self.scroll.height)

    def _paint_bands(self, *_):
        left, top, right, bottom = self.insets
        self.top_band.pos = (self.x, self.top - top)
        self.top_band.size = (self.width, top)
        self.bottom_band.pos = (self.x, self.y)
        self.bottom_band.size = (self.width, bottom)

    def apply_insets(self, insets):
        """Pads the whole screen by the safe area. Returns True if it changed."""
        if insets == self.insets:
            return False
        self.insets = insets
        left, top, right, bottom = insets
        self.padding = [left, top, right, bottom]
        self._paint_bands()
        return True

    def _grow_input(self, *_):
        # Grow with the text up to ~5 lines
        height = min(max(dp(44), self.input.minimum_height), dp(130))
        self.bar.height = height + dp(16)

    def add_bubble(self, text, mine, notice=False):
        bubble = Bubble(text, mine, notice=notice, width=self.messages.width)
        self.messages.add_widget(bubble)
        self.prune_bubbles(MAX_BUBBLES)
        self.scroll_to_end()
        return bubble

    def prune_bubbles(self, keep):
        """
        Removes the oldest bubbles from the screen. Each one holds a rendered text texture (several MB
        for a long reply at the Y27's 2.75x density), which is what grows in a long conversation.
        """
        removed = 0
        while len(self.messages.children) > keep:
            self.messages.remove_widget(self.messages.children[-1])  # children[-1] is the oldest
            removed += 1
        return removed

    def scroll_to_end(self, *_):
        """Shows the newest message. Runs again after the next layout pass, once new heights are known."""
        def scroll(dt):
            self.scroll.scroll_y = 0
        Clock.schedule_once(scroll, 0)
        Clock.schedule_once(scroll, 0.05)

    def set_generating(self, generating):
        self.send_button.text = "Stop" if generating else "Send"
        self.send_button.set_fill(hex_color("#D9534F") if generating else SEND_GREEN)


class QwenChatApp(App):
    title = "Qwen Chat"

    def build(self):
        Window.clearcolor = CHAT_BACKGROUND
        Window.softinput_mode = ""  # keyboard handled by SafeArea (Kivy's modes do nothing on Android SDL2)
        self.engine = ChatEngine()
        self.history = []        # [{"role", "content"}] sent to the model
        self.reply_bubble = None
        self.generating = False
        self.clear_after_reply = False
        self.screen = ChatScreen(self)
        self.safe_area = SafeArea()
        if self.safe_area.available or self.safe_area.simulated:
            # Android sends no event Kivy can see when the keyboard opens, so check 5x per second
            # (a few cheap Java calls) and right after the input gains or loses focus
            Clock.schedule_interval(self.update_safe_area, 0.2)
            self.screen.input.bind(focus=lambda *_: Clock.schedule_once(self.update_safe_area, 0.35))
        Window.bind(on_resize=lambda *_: Clock.schedule_once(self.update_safe_area, 0))
        Clock.schedule_once(self.update_safe_area, 0)
        Clock.schedule_once(lambda dt: self.load_model(), 0.3)
        Clock.schedule_interval(lambda dt: self.check_memory(), 30)
        return self.screen

    def update_safe_area(self, *_):
        if self.screen.apply_insets(self.safe_area.insets()):
            self.screen.scroll_to_end()  # keyboard opened / closed: keep the newest message in view

    # ---- model -------------------------------------------------------------------------------

    def load_model(self):
        path = find_model()
        if path is None and IS_ANDROID:
            folder = external_model_dir() or "Android/data/<package>/files"
            self.screen.status.text = "no model found"
            self.screen.add_bubble(
                "No model file found.\n\nCopy a .gguf model (e.g. qwen2.5-1.5b-instruct-q4_k_m.gguf) "
                f"into this folder with a USB cable, then restart the app:\n{folder}", False, notice=True)
            return
        size_mb = os.path.getsize(path) // (1024 * 1024) if path else 0
        free_mb = available_memory_mb()
        if free_mb is not None and size_mb and free_mb < size_mb + 600:
            self.screen.add_bubble(
                f"Low memory: {free_mb} MB free, the model needs about {size_mb + 400} MB. "
                "Close other apps if the chat crashes.", False, notice=True)
        self.screen.status.text = "loading model..."
        self.engine.load(path, self.on_model_ready, self.on_model_error)

    @mainthread
    def on_model_ready(self, system_info, memory):
        name = os.path.basename(self.engine.model_path or "demo")
        self.screen.status.text = "online - offline model"
        print(f"[QwenChat] model {name} loaded - context {self.engine.n_ctx} - memory {memory} - {system_info}")
        if self.engine.n_ctx < N_CTX:
            self.screen.add_bubble(
                f"To stay under {MEMORY_BUDGET_MB} MB of memory, the context was reduced to "
                f"{self.engine.n_ctx} tokens, so the chat remembers fewer messages.", False, notice=True)
        elif memory.get("estimate_mb", 0) > MEMORY_BUDGET_MB:
            self.screen.add_bubble(
                f"This model needs about {memory['estimate_mb']} MB, more than the {MEMORY_BUDGET_MB} MB "
                "budget. The app may be closed by Android - a smaller model is recommended.",
                False, notice=True)

    def check_memory(self):
        """
        Frees what the app can free when memory gets tight: old bubbles (text textures) and Python
        garbage. The model's own memory is fixed and memory-mapped, so it isn't the part that grows.
        """
        memory = process_memory_mb()
        free = available_memory_mb()
        own = memory["anon"] + memory.get("file", 0) if memory else 0
        if (memory and own > MEMORY_BUDGET_MB) or (free is not None and free < LOW_SYSTEM_MEMORY_MB):
            removed = self.screen.prune_bubbles(BUBBLES_UNDER_PRESSURE)
            gc.collect()
            print(f"[QwenChat] low memory (app {own} MB, system free {free} MB) - "
                  f"removed {removed} old bubbles")

    @mainthread
    def on_model_error(self, error):
        self.screen.status.text = "model failed to load"
        self.screen.add_bubble(f"Could not load the model:\n{error}", False, notice=True)

    # ---- chat --------------------------------------------------------------------------------

    def send_or_stop(self):
        if self.generating:
            self.engine.stop()
            return
        text = self.screen.input.text.strip()
        if not text:
            return
        if not self.engine.ready:
            self.screen.add_bubble("The model is still loading - try again in a moment.", False, notice=True)
            return
        self.screen.input.text = ""
        self.screen.add_bubble(text, True)
        self.history.append({"role": "user", "content": text})
        self.reply_bubble = self.screen.add_bubble("", False)
        self.generating = True
        self.screen.set_generating(True)
        self.screen.status.text = "typing..."
        self.engine.send(list(self.history), self.on_text, self.on_done, self.on_error)

    @mainthread
    def on_text(self, text):
        if self.reply_bubble is not None:
            self.reply_bubble.set_text(text.strip())
            self.screen.scroll_to_end()

    @mainthread
    def on_done(self, text, stats):
        # Messages the model could no longer fit are forgotten for good, so the history (and the
        # prompt built from it) never grows past the context
        del self.history[:stats["dropped_messages"]]
        self.finish_reply(text)
        status = f"{stats['tokens_per_second']:.1f} tokens/s - context {stats['context_used']:.0%}"
        if stats["dropped_messages"]:
            status += " - oldest messages forgotten"
        self.screen.status.text = status
        print(f"[QwenChat] {stats}")
        self.check_memory()

    @mainthread
    def on_error(self, error, partial_text):
        self.finish_reply(partial_text)
        if not error.startswith("stopped"):
            self.screen.add_bubble(f"Error: {error}", False, notice=True)
        self.screen.status.text = "online - offline model"

    def finish_reply(self, text):
        if text.strip():
            # Keep the exact text the model produced, so the next prompt matches the cached tokens
            self.history.append({"role": "assistant", "content": text})
            self.reply_bubble.set_text(text.strip())
        else:
            self.history.pop()  # nothing came back - forget the question too
            self.screen.messages.remove_widget(self.reply_bubble)
        self.reply_bubble = None
        self.generating = False
        self.screen.set_generating(False)
        self.screen.scroll_to_end()
        if self.clear_after_reply:
            self.new_chat()

    def new_chat(self):
        if self.generating:
            self.clear_after_reply = True  # finish_reply() clears once the model has stopped
            self.engine.stop()
            return
        self.clear_after_reply = False
        self.history = []
        self.screen.messages.clear_widgets()
        self.engine.reset()  # clears the KV cache
        gc.collect()         # and releases the old bubbles' textures
        self.screen.status.text = "online - offline model" if self.engine.ready else self.screen.status.text

    # ---- lifecycle ---------------------------------------------------------------------------

    def on_pause(self):
        return True  # keep the model loaded while the app is in the background

    def on_stop(self):
        self.engine.shutdown()


if __name__ == "__main__":
    QwenChatApp().run()
