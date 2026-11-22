# Mumble: Implementation Guide (live captions for video calls)

**Audience:** an engineer or coding agent with no prior context. This file is self-contained: it explains what to
build, why each design choice exists, the exact platform APIs, every pitfall we already hit, and includes complete,
tested reference source code (Appendix A–C). Following it top to bottom should produce a working app.

**Status of the reference:** built and used in real Zoom calls on macOS 14 (Apple Silicon), Python 3.12, Nov 2026.

---

## 1. What we are building

A desktop app that shows **live subtitles of what the user says**, burned into the bottom of **their own webcam
video**, so everyone in a Zoom / Google Meet / Teams call can read them, with no plugin or meeting-app integration.

How it works: the app opens the real webcam, draws a caption bar on each frame, and publishes the result as a
**virtual camera**. In the meeting app, the user selects that virtual camera instead of the real one.

```
 microphone ──► speech recognizer (Apple Speech, on-device when available) ──► "current caption text"
                                                                                        │
 webcam ──► frame (BGR, 1280×720, 30 fps) ──► draw caption bar on frame ◄───────────────┘
                                                   │
                                                   ├──► virtual camera ──► Zoom / Meet / Teams (un-mirrored)
                                                   └──► local preview window (mirrored, keyboard control)
```

### 1.1 User-facing behaviour (acceptance spec)

| # | Behaviour |
|---|---|
| U1 | Press **`t`** in the preview window: captions on/off. **`q`** or **Esc** quits. `--captions` starts with them on. |
| U2 | While the user speaks, the bottom of the video shows a rounded, translucent black bar with white text, horizontally centred, ~6% of frame height above the bottom edge. |
| U3 | Text updates **word by word** while speaking (partial results), not only at sentence end. |
| U4 | At most **2 lines**; the newest words win (older lines scroll off). Bar width ≤ 82% of frame width. |
| U5 | After **2.5 s of silence** the current caption is "finished"; the next utterance starts a fresh caption. A finished caption stays visible for **3 s**, then disappears. |
| U6 | Captions are punctuated and sentence-style ("I think we should ship it."), and biased toward words in an editable `vocab.txt` (e.g. "eye contact" not "I contact"). Edits to `vocab.txt` apply to the **next** caption without restarting. |
| U7 | Viewers in the call see readable (non-mirrored) text. The local preview is mirrored like a selfie, and the caption in the preview is also readable (drawn after mirroring). |
| U8 | Audio stays on the Mac when the on-device model exists; otherwise Apple's server recognition is used and the app logs which mode is active. |
| U9 | Runs indefinitely (hour-long calls) without the recognizer stopping. |
| U10 | If permissions are missing, the app says exactly which one and where to enable it. If no virtual camera is available, it falls back to preview only. |
| U11 | Language selectable with `--locale` (e.g. `en-US`, `en-IN`, `hi-IN`). |

### 1.2 Out of scope (for v1)
Captioning other participants (we only hear the local mic), translation, transcript export, global hotkeys (keys
only work while the preview window is focused). See §9 for how to add them.

---

## 2. Platform & dependencies

**OS:** macOS 13+ (uses Apple's Speech framework through PyObjC). Tested on macOS 14.5, Apple Silicon.
Linux/Windows would need a different recognizer (see §9) and virtual-camera backend; the rest of the design holds.

**Language:** Python **3.12** (pin it: some ML wheels lag new Python versions; 3.14 failed for us).

**Python packages** (`requirements.txt`):
```
numpy
opencv-contrib-python<5        # 5.x had no prebuilt macOS 14 wheel and failed to compile
pillow                         # text rendering with real fonts (OpenCV's Hershey fonts look bad)
pyvirtualcam                   # publishes frames to the OBS virtual camera device
pyobjc-framework-AVFoundation  # camera/mic permission + device listing + AVAudioEngine (mic)
pyobjc-framework-Speech        # SFSpeechRecognizer
```
Install: `uv venv --python 3.12 .venv && uv pip install --python .venv -r requirements.txt`
(or `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`).

**Virtual camera driver: OBS Studio (one time).** `pyvirtualcam` on macOS writes into **OBS Virtual Camera**, a
system camera extension that ships with OBS:
1. `brew install --cask obs`
2. Open OBS once → **Start Virtual Camera** → approve the system extension when macOS asks (System Settings ›
   General › Login Items & Extensions › Camera Extensions). A reboot may be needed.
3. **Stop Virtual Camera** and **quit OBS**. OBS itself must NOT be running while our app runs (if OBS holds the
   virtual camera, Zoom shows OBS's output/black instead of ours).
4. In Zoom: Settings › Video › Camera › **OBS Virtual Camera**.

---

## 3. macOS permissions (the #1 source of "it doesn't work")

macOS grants Camera, Microphone and Speech Recognition access to the **app that launched the process**, not
to the Python script. Consequences:

- **Run the app from Terminal.app** (or iTerm). Do not launch it from an IDE helper, an AI-agent sandbox, or a
  background service: those either have no permission or can't show the prompt. A double-clickable launcher
  (`run.command`, Appendix C) opens it in Terminal.
- Three permissions are needed, all for the terminal app: **Camera**, **Microphone**, **Speech Recognition**.
  The app requests each one programmatically on first run (macOS shows a dialog). If a dialog was denied earlier,
  macOS won't ask again: the user must enable it in System Settings › Privacy & Security › (Camera | Microphone |
  Speech Recognition), then **quit and reopen the terminal**.
- Status codes: `AVCaptureDevice.authorizationStatusForMediaType_(...)` and
  `SFSpeechRecognizer.authorizationStatus()` both return **3 = authorized**, 0 = not determined (we may ask),
  1 = restricted, 2 = denied.
- Request permissions **before opening the camera**. Prompts can block for many seconds; if the camera loop is
  already running, video stalls.

**On-device recognition:** `SFSpeechRecognizer.supportsOnDeviceRecognition()` is true when the Mac has the
dictation model for that locale. To get it: System Settings › Keyboard › **Dictation: On** (downloads the model),
then restart the app. Without it, Apple's server recognition is used (audio goes to Apple; see the 1-minute
limit in §5.3).

---

## 4. Architecture

Two independent components plus a host loop. Keep them separate: the recognizer must never block the video.

| Component | Responsibility | Threading |
|---|---|---|
| **`CaptionEngine`** (`captions.py`) | Mic → Apple Speech → thread-safe "current caption text" | Audio tap on an AVAudioEngine thread; results on a background `NSOperationQueue`; one watchdog thread |
| **`CaptionRenderer`** (`captions.py`) | Text → BGRA "subtitle bar" sprite, cached until the text changes | Called from the video loop (cheap when cached) |
| **Host loop** (`mumble.py`) | Webcam → draw sprite → virtual camera + preview; keyboard | Main thread (OpenCV `imshow`/`waitKey` must be on the main thread on macOS) |

The only shared state is the caption string, guarded by a lock. The video loop calls `engine.current_text()` each
frame (non-blocking) and `renderer.sprite(text)` (returns the cached image unless the text changed).

No main run loop (NSRunLoop) is required: the recognizer delivers callbacks on its own `NSOperationQueue` because
we call `recognizer.setQueue_(NSOperationQueue.alloc().init())`. Without that, callbacks are scheduled on the main
run loop, which an OpenCV loop never spins, and **no results ever arrive**.

---

## 5. CaptionEngine in detail

### 5.1 Audio capture
- `AVAudioEngine.alloc().init()`; `node = engine.inputNode()`; `fmt = node.outputFormatForBus_(0)` (use the
  hardware's native format, don't force 16 kHz: the recognizer accepts it).
- `node.installTapOnBus_bufferSize_format_block_(0, 1024, fmt, tap)` where `tap(buffer, when)` appends the
  buffer to the **current** recognition request: `request.appendAudioPCMBuffer_(buffer)`.
  The tap reads `self._request` each time, so swapping requests (§5.3) needs no tap reinstall.
- `engine.prepare()`; `ok, err = engine.startAndReturnError_(None)` (PyObjC returns a tuple for NSError** out
  params).
- Stop: end task, `inputNode().removeTapOnBus_(0)`, `engine.stop()`.

### 5.2 Recognition request configuration (the "smart captions" part)
For every new `SFSpeechAudioBufferRecognitionRequest`:
| Setting | Value | Why |
|---|---|---|
| `setShouldReportPartialResults_` | `True` | word-by-word updates (U3) |
| `setTaskHint_` | `SFSpeechRecognitionTaskHintDictation` | tells the model it's full sentences, not short commands; clearly better homophone choices ("I" vs "eye") |
| `setContextualStrings_` | list from `vocab.txt` (≤ 100 phrases) | biases toward names/jargon; Apple recommends ≤ 100 |
| `setRequiresOnDeviceRecognition_` | `True` **only if** `supportsOnDeviceRecognition()` | keeps audio local; setting it without the model makes every request fail |
| `setAddsPunctuation_` | `True` (guard with `hasattr`; macOS 13+) | punctuation and capitalisation |

`vocab.txt` format: one phrase per line, `#` starts a comment, blank lines ignored, first 100 kept. Re-read it when
its mtime changes, at the start of each new request (that's what makes edits apply to the next caption, U6).

### 5.3 Utterance lifecycle (why requests are recycled)
One recognition request should represent **one utterance** (one caption). The engine ends the current request and
starts a new one when:
1. **Silence:** the current request has produced text and no new words arrived for `SILENCE_SPLIT = 2.5 s` (U5).
   Without this, a single request accumulates the whole conversation into one ever-growing caption.
2. **Server time limit:** in server mode, Apple stops a request after ~60 s. Recycle at `SERVER_MAX = 50 s`
   (on-device has no such limit) (U9).
3. **Final result or error** from the recognizer (e.g. `isFinal()`, or an error such as "No speech detected",
   which is benign and shouldn't be logged): restart after 50 ms via `threading.Timer`.

A background **watchdog** thread checks conditions 1–2 every 200 ms.

**Generation counter (important):** every new request increments `self._gen`; each result handler captures its
own `gen` and ignores callbacks when `gen != self._gen`. Cancelled tasks keep delivering late callbacks for a
while; without this guard, old text overwrites the new caption or triggers duplicate restarts.

### 5.4 Displayed text
- Handler: `text = str(result.bestTranscription().formattedString())`; store it with a timestamp under the lock.
- `current_text()`: returns the stored text, or `""` once it's older than `SHOW_AFTER = 3.0 s` (U5).
- When the watchdog recycles on silence, the last caption is **not** cleared immediately: it stays until
  `SHOW_AFTER` expires, so viewers can finish reading it.

### 5.5 Permissions helper
`ensure_permissions()` asks for Microphone (`AVCaptureDevice.requestAccessForMediaType_completionHandler_`) and
Speech (`SFSpeechRecognizer.requestAuthorization_`) if not yet authorized, waiting on a `threading.Event` (60 s
timeout) because the callbacks are asynchronous. **PyObjC completion blocks must return `None`**: a lambda
whose expression returns a value (e.g. `lambda ok: done.set() or result.append(ok)`) can crash. Use a `def` or make
sure the expression's value is None.

---

## 6. CaptionRenderer in detail
- Font: Pillow `ImageFont.truetype` with the first available of `/System/Library/Fonts/SFNS.ttf`,
  `HelveticaNeue.ttc`, `Helvetica.ttc`; fallback `ImageFont.load_default()`. Size = `max(18, 4.8% of frame
  height)` (≈ 34 px at 720p).
- Word-wrap greedily to `82%` of frame width minus padding; keep the **last 2 lines** (U4).
- Draw on a transparent RGBA image: rounded rectangle (radius = padding, fill black alpha 170/255), white text.
  Padding = 0.5 × font size; line height = 1.3 × font size.
- Convert RGBA → **BGRA** (OpenCV channel order): `rgba[:, :, [2, 1, 0, 3]]`.
- **Cache** the sprite keyed by the text: rendering with Pillow is ~ms, but the text changes only a few times a
  second while frames arrive at 30 fps.
- Position: centre-x = frame centre; centre-y = `h − 6%·h − sprite_h/2`.

**Compositing** (`overlay_bgra`): integer alpha blend of the BGRA sprite onto the BGR frame in place, clipped to
the frame. Using `uint16` math with alpha in 0..256 and `>> 8` is several times faster than float blending, which
matters at 30 fps.

---

## 7. Host loop in detail (`mumble.py`)
1. Parse args; create `CaptionEngine`, call `ensure_permissions()` (**before** the camera); start it if
   `--captions`.
2. Camera permission check (AVFoundation, as in §3).
3. **Pick the camera correctly.** OpenCV's `cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)` index order is the
   AVFoundation devices **sorted by `uniqueID`**, not the order AVFoundation returns them (which can change between
   runs). List names with that sort, then choose the first device whose name does not contain any of `obs`,
   `virtual`, `camo`, `mmhmm`, `snap camera`. Opening the virtual camera as input feeds our own output back in;
   the symptom is one 1920×1080 frame, then nothing.
4. Request 1280×720 @ 30 fps; read up to 30 times with 100 ms sleeps (cameras warm up). Many Mac cameras ignore
   the requested size and deliver 1080p: resize every frame to the working size (`INTER_AREA`).
5. Open `pyvirtualcam.Camera(width=w, height=h, fps=fps, fmt=PixelFormat.BGR)`; on failure continue preview-only.
6. Per frame:
   - **Output to the call:** copy the frame only when captions are on, draw the caption, `vcam.send(out)`. Do NOT
     mirror this: Zoom mirrors the self-view locally but sends the original to others.
   - **Preview:** `cv2.flip(frame, 1)` (selfie mirror), then draw the caption on the flipped image so it reads
     correctly; add a small green "CC" indicator; `imshow`; `waitKey(1)` handles `t`/`q`/Esc.
   - With `--no-preview`, pace with `vcam.sleep_until_next_frame()`.
7. On exit (including Ctrl+C): stop captions, release camera, close virtual camera, destroy windows.

---

## 8. Pitfalls we hit (and the fix)

| Symptom | Cause | Fix |
|---|---|---|
| "Camera not authorized" even after allowing | Script launched from an IDE/agent helper, not Terminal | Run from Terminal.app (`run.command`) |
| Zoom shows black / OBS logo | OBS app running and owning the virtual camera, or our app not running | Quit OBS; keep the app running |
| One frame then frozen; 1920×1080 input | Opened the virtual camera as input (wrong index order) | Sort devices by `uniqueID`; skip virtual names (§7.3) |
| Captions never appear, no errors | Recognizer callbacks scheduled on the main run loop, which never runs | `recognizer.setQueue_(NSOperationQueue.alloc().init())` |
| Captions stop after ~1 minute | Server-mode request limit | Recycle requests every 50 s (§5.3) |
| One endless, growing caption | Single request for the whole session | Split on 2.5 s silence (§5.3) |
| Old words reappear / flicker after a split | Late callbacks from cancelled tasks | Generation counter (§5.3) |
| Every request fails instantly | `requiresOnDeviceRecognition=True` on a Mac without the model | Set it only if `supportsOnDeviceRecognition()` |
| "I contact" instead of "eye contact" | Default command-style recognition, no context | Dictation task hint + `contextualStrings` from `vocab.txt` |
| Video stalls at start-up | Permission prompts while the camera loop runs | Speech setup before opening the camera |
| Mirrored, unreadable captions for viewers | Drawing after mirroring the output | Mirror only the preview; draw on each image after any flip |
| Crash in a permission callback | PyObjC block returned a non-None value | Completion handlers return None |
| `pip install` fails on Python 3.14 / OpenCV 5 | No wheels | Python 3.12, `opencv-contrib-python<5` |

---

## 9. Extensions (not implemented)
1. **Global hotkey** (toggle without focusing the preview): `AppKit.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_`
   (needs Accessibility permission for the terminal) or a menu-bar item (`rumps`).
2. **Better accuracy (AI captions):** stream 16 kHz mono chunks to faster-whisper (`base.en`, `compute_type="int8"`,
   CPU) with a sliding window and VAD, or post-correct Apple's partials with an LLM given recent context + vocab.
   Expect 0.5–1.5 s more latency.
3. **Translation:** recognize in the source language, translate each finished utterance (LLM or MT API), show the
   translation (optionally with the original in smaller text).
4. **Transcript export:** append every finished utterance with timestamps to a `.txt`/`.srt` on stop.
5. **Other platforms:** Windows/Linux need a different recognizer (Vosk, whisper.cpp, Azure/Google streaming STT)
   and virtual-camera backend (`pyvirtualcam` supports OBS on Windows and v4l2loopback on Linux).
6. **Captions for other participants:** capture system/app audio (macOS ScreenCaptureKit with `capturesAudio`) and
   show them in a separate overlay window, since you can't put them on your own camera meaningfully.

---

## 10. Test plan

**Automated (no hardware needed):**
- Renderer: `sprite("")` is `None`; same text returns the identical cached object; a very long text yields ≤ 2
  lines and width ≤ 82% of the frame; after `overlay_bgra` on a mid-grey frame, the bar region is darker, text pixels
  are near-white, and pixels outside the bar are unchanged.
- `load_vocab`: comments, blank lines, the 100-item cap.
- `overlay_bgra`: sprite partly outside the frame is clipped without errors; opacity 0 leaves the frame unchanged.
- Engine logic can be tested by faking time: the watchdog recycles after `SILENCE_SPLIT` only when text was produced,
  and after `SERVER_MAX` only in server mode; stale-generation callbacks are ignored.

**Manual (with hardware):**
1. `python mumble.py --no-vcam` from Terminal: preview opens, permission prompts appear once, `t` shows "CC".
2. Speak: words appear within ~0.5 s and update as you talk; stop for 3 s: the caption finishes and disappears ~3 s
   later; speak again: a fresh caption.
3. Speak for 2+ minutes continuously: captions keep flowing (recycling works).
4. Add a made-up product name to `vocab.txt` while running; say it: recognized on the next caption.
5. With OBS installed: run without `--no-vcam`, select OBS Virtual Camera in Zoom, verify the other participant (or
   a second device) sees readable, non-mirrored captions.
6. Startup log states "on-device" or "Apple server" mode correctly.

**Performance:** at 1280×720 @ 30 fps the loop uses well under one CPU core; recognition runs in Apple's speech
service process. Caption rendering is cached and is not a measurable cost.

---
## Appendix A: `captions.py` (engine + renderer, complete)

```python
"""Live captions from the Mac's microphone using Apple's Speech framework.

Uses on-device recognition when the Mac has the model (turn on Dictation in
System Settings > Keyboard to download it); otherwise Apple's server recognition.

CaptionEngine  - mic -> SFSpeechRecognizer -> latest caption text (thread-safe)
CaptionRenderer - caption text -> cached BGRA "subtitle bar" sprite
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np

SILENCE_SPLIT = 2.5   # s without new words -> end this utterance (longer = more context per caption)
SHOW_AFTER = 3.0      # s a finished caption stays on screen
SERVER_MAX = 50.0     # s; Apple's server mode stops a request at ~60 s, so recycle before that


MAX_VOCAB = 100  # Apple recommends keeping contextual strings to ~100 phrases


def load_vocab(path: Path | None) -> list[str]:
    """One word/phrase per line; '#' starts a comment. Bias recognition toward these."""
    if path is None or not path.exists():
        return []
    words = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            words.append(line)
    return words[:MAX_VOCAB]


def configure_request(req, Speech, vocab: list[str], on_device: bool) -> None:
    """Settings that make recognition smarter: sentence-style hint, vocabulary, punctuation."""
    req.setTaskHint_(Speech.SFSpeechRecognitionTaskHintDictation)  # full sentences, not commands
    if vocab:
        req.setContextualStrings_(vocab)
    if on_device:
        req.setRequiresOnDeviceRecognition_(True)
    if hasattr(req, "setAddsPunctuation_"):
        req.setAddsPunctuation_(True)


class CaptionEngine:
    def __init__(self, locale: str = "en-US", vocab_path: Path | None = None,
                 silence_split: float = SILENCE_SPLIT):
        import AVFoundation as AV
        import Foundation
        import Speech

        self.AV, self.Speech = AV, Speech
        self._lock = threading.Lock()
        self._text = ""
        self._updated = 0.0
        self._task_started = 0.0
        self._task_has_text = False
        self._task = None
        self._request = None
        self._engine = None
        self._running = False
        self._gen = 0  # ignores callbacks from old recognition tasks
        self.silence_split = silence_split
        self.vocab_path = vocab_path
        self._vocab: list[str] = []
        self._vocab_mtime = None
        self._reload_vocab()

        self.recognizer = Speech.SFSpeechRecognizer.alloc().initWithLocale_(
            Foundation.NSLocale.localeWithLocaleIdentifier_(locale))
        if self.recognizer is None:
            raise RuntimeError(f"Speech recognition does not support locale {locale}")
        # deliver results on a background queue (no main run loop needed)
        self.recognizer.setQueue_(Foundation.NSOperationQueue.alloc().init())
        self.on_device = bool(self.recognizer.supportsOnDeviceRecognition())
        self._watchdog = threading.Thread(target=self._watch, daemon=True)
        self._watchdog.start()

    # ------------------------------------------------------------ permissions
    def ensure_permissions(self) -> bool:
        AV, Speech = self.AV, self.Speech
        ok = True
        if AV.AVCaptureDevice.authorizationStatusForMediaType_(AV.AVMediaTypeAudio) != 3:
            print("Asking macOS for microphone permission - click 'Allow'...")
            ok &= _wait_for(lambda cb: AV.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
                AV.AVMediaTypeAudio, lambda granted: cb(bool(granted))))
        if Speech.SFSpeechRecognizer.authorizationStatus() != 3:
            print("Asking macOS for speech recognition permission - click 'Allow'...")
            ok &= _wait_for(lambda cb: Speech.SFSpeechRecognizer.requestAuthorization_(
                lambda status: cb(status == 3)))
        if not ok:
            print("\nCaptions need Microphone + Speech Recognition permission for your terminal app:\n"
                  "  System Settings > Privacy & Security > Microphone / Speech Recognition")
        return ok

    # ------------------------------------------------------------ control
    def start(self) -> None:
        if self._running:
            return
        AV = self.AV
        self._engine = AV.AVAudioEngine.alloc().init()
        node = self._engine.inputNode()
        fmt = node.outputFormatForBus_(0)

        def tap(buffer, when):
            req = self._request
            if req is not None:
                req.appendAudioPCMBuffer_(buffer)

        node.installTapOnBus_bufferSize_format_block_(0, 1024, fmt, tap)
        self._engine.prepare()
        ok, err = self._engine.startAndReturnError_(None)
        if not ok:
            raise RuntimeError(f"Could not start microphone: {err}")
        self._running = True
        self._new_task()
        mode = "on-device" if self.on_device else "Apple server (enable Dictation for on-device)"
        print(f"  captions ON ({mode})")

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._end_task()
        self._engine.inputNode().removeTapOnBus_(0)
        self._engine.stop()
        self._engine = None
        with self._lock:
            self._text = ""
        print("  captions OFF")

    @property
    def running(self) -> bool:
        return self._running

    def current_text(self) -> str:
        """Caption to show now ('' when nothing recent)."""
        with self._lock:
            if self._text and time.monotonic() - self._updated > SHOW_AFTER:
                self._text = ""
            return self._text

    # ------------------------------------------------------------ internals
    def _reload_vocab(self) -> None:
        """Re-read the vocab file if it changed, so you can edit it while the app runs."""
        if self.vocab_path is None:
            return
        try:
            mtime = os.path.getmtime(self.vocab_path)
        except OSError:
            return
        if mtime != self._vocab_mtime:
            self._vocab_mtime = mtime
            self._vocab = load_vocab(self.vocab_path)
            print(f"  [captions] vocabulary: {len(self._vocab)} phrases from {self.vocab_path.name}")

    def _new_task(self) -> None:
        Speech = self.Speech
        self._gen += 1
        gen = self._gen
        req = Speech.SFSpeechAudioBufferRecognitionRequest.alloc().init()
        req.setShouldReportPartialResults_(True)
        self._reload_vocab()
        configure_request(req, Speech, self._vocab, self.on_device)

        def handler(result, error):
            if gen != self._gen:
                return
            if result is not None:
                text = str(result.bestTranscription().formattedString())
                if text:
                    with self._lock:
                        self._text = text
                        self._updated = time.monotonic()
                        self._task_has_text = True
            if error is not None or (result is not None and result.isFinal()):
                if error is not None and "No speech detected" not in str(error.localizedDescription()):
                    print(f"  [captions] {error.localizedDescription()}")
                self._restart_soon(gen)

        self._request = req
        self._task_started = time.monotonic()
        with self._lock:
            self._task_has_text = False
        self._task = self.recognizer.recognitionTaskWithRequest_resultHandler_(req, handler)

    def _end_task(self) -> None:
        if self._request is not None:
            self._request.endAudio()
        if self._task is not None:
            self._task.cancel()
        self._request = None
        self._task = None

    def _restart_soon(self, gen: int) -> None:
        def go():
            if self._running and gen == self._gen:
                self._end_task()
                self._new_task()
        threading.Timer(0.05, go).start()

    def _watch(self) -> None:
        """Split utterances on silence, and recycle server requests before Apple's 1-minute cap."""
        while True:
            time.sleep(0.2)
            if not self._running or self._task is None:
                continue
            now = time.monotonic()
            with self._lock:
                silent = self._task_has_text and now - self._updated > self.silence_split
                too_long = not self.on_device and now - self._task_started > SERVER_MAX
            if silent or too_long:
                gen = self._gen
                self._gen += 1  # drop late results from the old task; caption stays until SHOW_AFTER
                self._end_task()
                if self._running and gen + 1 == self._gen:
                    self._new_task()


def _wait_for(start, timeout: float = 60.0) -> bool:
    done, out = threading.Event(), {}

    def cb(value):
        out["v"] = value
        done.set()

    start(cb)
    done.wait(timeout)
    return bool(out.get("v"))


# ---------------------------------------------------------------- rendering

FONT_CANDIDATES = [
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
]


class CaptionRenderer:
    """Turns caption text into a BGRA subtitle bar (last 2 lines), cached until the text changes."""

    def __init__(self, frame_w: int, frame_h: int, max_lines: int = 2):
        from PIL import ImageFont
        self.w, self.h = frame_w, frame_h
        self.max_lines = max_lines
        size = max(18, int(frame_h * 0.048))
        self.font = None
        for path in FONT_CANDIDATES:
            try:
                self.font = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        if self.font is None:
            self.font = ImageFont.load_default()
        self.pad = int(size * 0.5)
        self.line_h = int(size * 1.3)
        self.max_w = int(frame_w * 0.82)
        self._cache_text = None
        self._cache_sprite = None

    def _wrap(self, text: str) -> list[str]:
        lines, cur = [], ""
        for word in text.split():
            trial = f"{cur} {word}".strip()
            if self.font.getlength(trial) <= self.max_w - 2 * self.pad or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines[-self.max_lines:]  # newest words only

    def sprite(self, text: str) -> np.ndarray | None:
        if not text:
            return None
        if text == self._cache_text:
            return self._cache_sprite
        from PIL import Image, ImageDraw
        lines = self._wrap(text)
        text_w = max(int(self.font.getlength(line)) for line in lines)
        bw = text_w + 2 * self.pad
        bh = len(lines) * self.line_h + self.pad
        img = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, bw - 1, bh - 1), radius=self.pad, fill=(0, 0, 0, 170))
        for i, line in enumerate(lines):
            d.text((self.pad, self.pad // 2 + i * self.line_h), line, font=self.font,
                   fill=(255, 255, 255, 255))
        rgba = np.array(img)
        self._cache_text = text
        self._cache_sprite = rgba[:, :, [2, 1, 0, 3]].copy()  # RGBA -> BGRA
        return self._cache_sprite

    def position(self, sprite: np.ndarray) -> tuple[int, int]:
        """Centre point for the bar: bottom-centre with a margin."""
        sh = sprite.shape[0]
        return self.w // 2, self.h - int(self.h * 0.06) - sh // 2
```

## Appendix B: `mumble.py` (host app, complete)

```python
"""Mumble: live subtitles of what you say, burned into your webcam video, sent to Zoom as a virtual camera.

    webcam --> frame --> draw caption bar --> virtual camera (OBS Virtual Camera) --> Zoom / Meet / Teams
    mic ----> Apple Speech (SFSpeechRecognizer) --> current caption text --^

Run from Terminal.app (macOS ties Camera / Microphone / Speech permissions to the launching app):
    python mumble.py                 # press t in the preview window to toggle captions, q to quit
    python mumble.py --captions      # start with captions on
    python mumble.py --no-vcam       # preview only (no OBS needed)
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from captions import SILENCE_SPLIT, CaptionEngine, CaptionRenderer

ROOT = Path(__file__).resolve().parent
VIRTUAL_CAM_WORDS = ("obs", "virtual", "camo", "mmhmm", "snap camera")


# ----------------------------------------------------------------------------- drawing

def overlay_bgra(frame: np.ndarray, sprite: np.ndarray, cx: int, cy: int, opacity: float = 1.0) -> None:
    """Alpha-blend a BGRA sprite onto a BGR frame in place, centred at (cx, cy), clipped to the frame."""
    sh, sw = sprite.shape[:2]
    fh, fw = frame.shape[:2]
    x0, y0 = cx - sw // 2, cy - sh // 2
    x1, y1 = x0 + sw, y0 + sh
    fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x1, fw), min(y1, fh)
    if fx0 >= fx1 or fy0 >= fy1:
        return
    sp = sprite[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    alpha = (sp[:, :, 3:4].astype(np.uint16) * int(opacity * 256)) >> 8      # integer blend: fast
    if not alpha.any():
        return
    roi = frame[fy0:fy1, fx0:fx1]
    roi[:] = ((sp[:, :, :3].astype(np.uint16) * alpha + roi.astype(np.uint16) * (256 - alpha)) >> 8).astype(np.uint8)


# ----------------------------------------------------------------------------- camera (macOS)

def list_cameras() -> list[str]:
    """Camera names in the order OpenCV's AVFoundation backend indexes them.
    OpenCV sorts devices by uniqueID, NOT the order AVFoundation lists them in. Getting this wrong opens the
    OBS Virtual Camera as the input (our own output fed back in: one 1080p frame, then nothing)."""
    try:
        import AVFoundation as AV
        devices = AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo)
        return [str(d.localizedName()) for d in sorted(devices, key=lambda d: str(d.uniqueID()))]
    except Exception:  # noqa: BLE001
        return []


def pick_camera(requested: int | None, name_hint: str | None) -> int:
    """Webcam index. Never auto-picks a virtual camera."""
    names = list_cameras()
    if names:
        print("Cameras: " + ", ".join(f"[{i}] {n}" for i, n in enumerate(names)))
    if requested is not None:
        return requested
    if name_hint:
        for i, n in enumerate(names):
            if name_hint.lower() in n.lower():
                return i
        print(f"  ! no camera matching '{name_hint}', auto-picking")
    for i, n in enumerate(names):
        if not any(w in n.lower() for w in VIRTUAL_CAM_WORDS):
            return i
    return 0


def ensure_camera_permission() -> bool:
    try:
        import AVFoundation as AV
    except Exception:  # noqa: BLE001
        return True
    status = AV.AVCaptureDevice.authorizationStatusForMediaType_(AV.AVMediaTypeVideo)
    if status == 3:                                   # authorized
        return True
    if status == 0:                                   # never asked: show the macOS prompt
        print("Asking macOS for camera permission - click 'Allow'...")
        done, result = threading.Event(), {}

        def handler(granted):                         # ObjC completion block: must return None
            result["granted"] = bool(granted)
            done.set()

        AV.AVCaptureDevice.requestAccessForMediaType_completionHandler_(AV.AVMediaTypeVideo, handler)
        done.wait(60)
        if result.get("granted"):
            return True
    print("\nCamera permission is not granted to the app that launched this script.\n"
          "  System Settings > Privacy & Security > Camera > turn on your terminal app, then quit and reopen it.")
    return False


def open_virtual_cam(w: int, h: int, fps: int):
    try:
        import pyvirtualcam
        cam = pyvirtualcam.Camera(width=w, height=h, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR)
        print(f"Virtual camera ON: '{cam.device}' -> choose it in Zoom > Settings > Video > Camera")
        return cam
    except Exception as e:  # noqa: BLE001
        print(f"Virtual camera unavailable ({e}). Preview only. Install OBS and start its Virtual Camera once.")
        return None


# ----------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=None, help="webcam index (default: first non-virtual camera)")
    ap.add_argument("--camera-name", default=None, help="pick the webcam by name, e.g. 'FaceTime'")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--captions", action="store_true", help="start with captions on")
    ap.add_argument("--locale", default="en-US", help="speech language, e.g. en-US, en-IN, hi-IN")
    ap.add_argument("--vocab", type=Path, default=ROOT / "vocab.txt", help="phrases to bias toward (live reload)")
    ap.add_argument("--caption-pause", type=float, default=SILENCE_SPLIT,
                    help="seconds of silence that end one caption and start the next")
    ap.add_argument("--no-vcam", action="store_true", help="preview only, no virtual camera")
    ap.add_argument("--no-preview", action="store_true", help="no preview window (no keyboard control)")
    args = ap.parse_args()

    # Speech setup BEFORE the camera: permission prompts can take a while and must not stall frames.
    captions: CaptionEngine | None = None
    try:
        captions = CaptionEngine(args.locale, args.vocab, args.caption_pause)
        if not captions.ensure_permissions():
            captions = None
        elif args.captions:
            captions.start()
    except Exception as e:  # noqa: BLE001
        print(f"Captions unavailable: {e}")
        captions = None

    if not ensure_camera_permission():
        return 1
    idx = pick_camera(args.camera, args.camera_name)
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    ok, frame = False, None
    for _ in range(30):                               # cameras take a moment to warm up
        ok, frame = cap.read()
        if ok:
            break
        time.sleep(0.1)
    if not ok:
        print("Could not read from the webcam. Try --camera-name FaceTime or --camera 1.")
        return 1
    cam_h, cam_w = frame.shape[:2]
    if cam_w > args.width:                            # many Mac cameras ignore the requested size (send 1080p)
        w, h = args.width, int(round(cam_h * args.width / cam_w / 2) * 2)
    else:
        w, h = cam_w, cam_h
    print(f"Webcam {cam_w}x{cam_h} -> output {w}x{h}.  Keys (click the preview window): t = captions, q = quit")

    renderer = CaptionRenderer(w, h)
    vcam = None if args.no_vcam else open_virtual_cam(w, h, args.fps)

    def draw_captions(img: np.ndarray) -> None:
        if captions is None or not captions.running:
            return
        sprite = renderer.sprite(captions.current_text())
        if sprite is not None:
            cx, cy = renderer.position(sprite)
            overlay_bgra(img, sprite, cx, cy)

    misses = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                misses += 1
                if misses > 150:
                    print("Webcam stopped sending frames (another app using it exclusively?)")
                    break
                time.sleep(0.02)
                continue
            misses = 0
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

            if vcam is not None:
                out = frame.copy() if captions is not None and captions.running else frame
                draw_captions(out)                    # un-mirrored, so people in the call can read it
                vcam.send(out)

            if not args.no_preview:
                preview = cv2.flip(frame, 1)          # mirrored like a selfie view...
                draw_captions(preview)                # ...captions drawn AFTER the flip so they read correctly
                if captions is not None and captions.running:
                    cv2.putText(preview, "CC", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Mumble (preview - not sent to Zoom)", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("t"):
                    if captions is None:
                        print("  captions unavailable (see messages above)")
                    else:
                        try:
                            captions.stop() if captions.running else captions.start()
                        except Exception as e:  # noqa: BLE001
                            print(f"  captions error: {e}")
            elif vcam is not None:
                vcam.sleep_until_next_frame()
    except KeyboardInterrupt:
        pass
    finally:
        if captions is not None:
            captions.stop()
        cap.release()
        if vcam is not None:
            vcam.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## Appendix C: supporting files

### `requirements.txt`
```
# Python 3.12. OpenCV < 5: 5.x has no prebuilt macOS 14 wheel.
numpy
opencv-contrib-python<5
pillow
pyvirtualcam
pyobjc-framework-AVFoundation
pyobjc-framework-Speech
```

### `vocab.txt` (example)
```
# Words and phrases the captions should prefer (one per line, max 100).
# Edit while the app runs: changes apply to the next caption.
# Add your product names, people's names and jargon here.

eye contact
interview
candidate
Zoom
MediaPipe
```

### `run.command` (launcher; `chmod +x` it)
```zsh
#!/bin/zsh
# Double-click in Finder (or: open -a Terminal run.command) so macOS grants Camera/Microphone/Speech
# permissions to Terminal. Extra arguments are passed through, e.g. ./run.command --captions
cd "$(dirname "$0")"
.venv/bin/python -u mumble.py "$@"
echo; echo "Mumble stopped. Press Enter to close."; read
```

### File layout
```
Mumble/
  mumble.py              # Appendix B
  captions.py            # Appendix A
  vocab.txt
  requirements.txt
  run.command
  .venv/                 # created at install
```
