"""Spike: capture a Zoom window's video + Zoom's audio via ScreenCaptureKit from Python.

Run from Terminal.app (needs Screen Recording permission):
    .venv/bin/python spikes/sck_capture_spike.py [seconds]
Writes spikes/out/frame.png and spikes/out/audio.wav and prints frame/audio stats.
"""
import sys, threading, time, wave
from pathlib import Path

import numpy as np
import cv2
import objc
import Quartz
import CoreMedia as CM
import ScreenCaptureKit as SCK
from Foundation import NSObject

OUT = Path(__file__).parent / "out"; OUT.mkdir(exist_ok=True)
SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 6

def p(*a): print(*a, flush=True)

import AppKit
AppKit.NSApplication.sharedApplication()  # initialise the CoreGraphics connection (CGS_REQUIRE_INIT)
p("screen capture access:", Quartz.CGPreflightScreenCaptureAccess())
if not Quartz.CGPreflightScreenCaptureAccess():
    Quartz.CGRequestScreenCaptureAccess()
    p("REQUESTED permission. Enable Terminal in System Settings > Privacy & Security > Screen Recording, quit & reopen Terminal, rerun.")
    sys.exit(2)

got = threading.Event(); box = {}
def on_content(content, error):
    box["c"], box["e"] = content, error; got.set()
SCK.SCShareableContent.getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(True, True, on_content)
got.wait(10)
content = box.get("c")
if content is None:
    p("no shareable content:", box.get("e")); sys.exit(1)
wins = [w for w in content.windows() if w.owningApplication() and "zoom" in str(w.owningApplication().applicationName()).lower()]
for w in wins:
    f = w.frame(); p(f"zoom window id={w.windowID()} title={w.title()!r} size={int(f.size.width)}x{int(f.size.height)} onscreen={w.isOnScreen()}")
if not wins:
    p("no Zoom windows"); sys.exit(1)
win = max(wins, key=lambda w: w.frame().size.width * w.frame().size.height)
p("using window", win.windowID(), repr(win.title()))

flt = SCK.SCContentFilter.alloc().initWithDesktopIndependentWindow_(win)
cfg = SCK.SCStreamConfiguration.alloc().init()
scale = 2
cfg.setWidth_(int(win.frame().size.width * scale)); cfg.setHeight_(int(win.frame().size.height * scale))
cfg.setMinimumFrameInterval_(CM.CMTimeMake(1, 15))
cfg.setPixelFormat_(1111970369)  # kCVPixelFormatType_32BGRA 'BGRA'
cfg.setShowsCursor_(False)
cfg.setCapturesAudio_(True)
cfg.setSampleRate_(48000)
cfg.setChannelCount_(1)
cfg.setExcludesCurrentProcessAudio_(True)

stats = {"frames": 0, "idle": 0, "audio_bufs": 0}
audio_chunks = []
last_frame = {}

SCStreamOutput = objc.protocolNamed("SCStreamOutput")

class Out(NSObject, protocols=[SCStreamOutput]):
    def stream_didOutputSampleBuffer_ofType_(self, stream, sb, kind):
        try:
            if kind == SCK.SCStreamOutputTypeScreen:
                pb = CM.CMSampleBufferGetImageBuffer(sb)
                if pb is None:
                    stats["idle"] += 1; return
                Quartz.CVPixelBufferLockBaseAddress(pb, 1)
                try:
                    h = Quartz.CVPixelBufferGetHeight(pb); w = Quartz.CVPixelBufferGetWidth(pb)
                    bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
                    base = Quartz.CVPixelBufferGetBaseAddress(pb)
                    arr = np.frombuffer(base.as_buffer(bpr * h), dtype=np.uint8).reshape(h, bpr // 4, 4)[:, :w, :3].copy()
                finally:
                    Quartz.CVPixelBufferUnlockBaseAddress(pb, 1)
                last_frame["f"] = arr; stats["frames"] += 1
            else:
                bb = CM.CMSampleBufferGetDataBuffer(sb)
                if bb is None: return
                n = CM.CMBlockBufferGetDataLength(bb)
                st, data = CM.CMBlockBufferCopyDataBytes(bb, 0, n, None)
                if st == 0:
                    audio_chunks.append(np.frombuffer(bytes(data), dtype=np.float32).copy())
                    stats["audio_bufs"] += 1
                    if stats["audio_bufs"] == 1:
                        fd = CM.CMSampleBufferGetFormatDescription(sb)
                        asbd = CM.CMAudioFormatDescriptionGetStreamBasicDescription(fd)
                        p("audio format:", asbd)
        except Exception as e:  # never raise into ObjC
            p("handler error:", repr(e))

out = Out.alloc().init()
stream = SCK.SCStream.alloc().initWithFilter_configuration_delegate_(flt, cfg, None)
ok1 = stream.addStreamOutput_type_sampleHandlerQueue_error_(out, SCK.SCStreamOutputTypeScreen, None, None)
ok2 = stream.addStreamOutput_type_sampleHandlerQueue_error_(out, SCK.SCStreamOutputTypeAudio, None, None)
p("add outputs:", ok1, ok2)
started = threading.Event(); err = {}
def on_started(e):  # ObjC completion blocks must return None
    err["e"] = e
    started.set()
stream.startCaptureWithCompletionHandler_(on_started)
started.wait(10); p("start error:", err.get("e"))
time.sleep(SECS)
stopped = threading.Event()
stream.stopCaptureWithCompletionHandler_(lambda e: stopped.set()); stopped.wait(5)

p("stats:", stats, f"-> {stats['frames']/SECS:.1f} fps")
if "f" in last_frame:
    cv2.imwrite(str(OUT / "frame.png"), last_frame["f"]); p("frame", last_frame["f"].shape, "-> spikes/out/frame.png")
if audio_chunks:
    a = np.concatenate(audio_chunks)
    p(f"audio samples {a.size} ({a.size/48000:.1f}s), rms {np.sqrt(np.mean(a**2)):.4f}, peak {np.abs(a).max():.3f}")
    with wave.open(str(OUT / "audio.wav"), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(48000)
        wf.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())
p("done")
