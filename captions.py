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
