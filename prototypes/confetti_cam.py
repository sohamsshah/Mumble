"""Gesture-triggered branded logo confetti + live captions for Zoom (or any video app).

Webcam -> MediaPipe gesture recognizer -> logo confetti drawn on frames -> virtual camera.

Run:  .venv/bin/python confetti_cam.py
Keys in the preview window:  c = fire confetti manually,  q / Esc = quit
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("GLOG_minloglevel", "2")       # quiet MediaPipe/TFLite startup logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))   # captions.py lives at the repo root
DEFAULT_MODEL = ROOT / "models" / "gesture_recognizer.task"
DEFAULT_LOGOS = ROOT / "logos"

# MediaPipe's built-in gesture names:
# None, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou


# --------------------------------------------------------------------------- logos

def load_logos(folder: Path, size: int) -> list[np.ndarray]:
    """Load PNG/JPG logos as BGRA, scaled so the longest side == size."""
    logos = []
    for path in sorted(folder.glob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  ! could not read {path.name}, skipping")
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        h, w = img.shape[:2]
        scale = size / max(h, w)
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
        logos.append(img)
        print(f"  + logo {path.name}")
    return logos


def placeholder_logos(size: int) -> list[np.ndarray]:
    """Colourful stand-in logos so the app works before you add your own."""
    colours = [(66, 133, 244), (52, 168, 83), (251, 188, 5), (234, 67, 53), (171, 71, 188)]
    logos = []
    for i, rgb in enumerate(colours):
        img = np.zeros((size, size, 4), np.uint8)
        bgr = (rgb[2], rgb[1], rgb[0], 255)
        c = size // 2
        if i % 2 == 0:
            cv2.circle(img, (c, c), c - 2, bgr, -1, cv2.LINE_AA)
        else:  # star
            pts = []
            for k in range(10):
                r = (c - 2) if k % 2 == 0 else (c - 2) * 0.45
                a = -math.pi / 2 + k * math.pi / 5
                pts.append((c + r * math.cos(a), c + r * math.sin(a)))
            cv2.fillPoly(img, [np.array(pts, np.int32)], bgr, cv2.LINE_AA)
        fs = size / (130 if i % 2 == 0 else 220)  # smaller text inside stars
        (tw, th), _ = cv2.getTextSize("LOGO", cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, size // 40))
        cv2.putText(img, "LOGO", (c - tw // 2, c + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255, 255),
                    max(1, size // 40), cv2.LINE_AA)
        logos.append(img)
    return logos


def overlay_bgra(frame: np.ndarray, sprite: np.ndarray, cx: int, cy: int, opacity: float) -> None:
    """Alpha-blend a BGRA sprite onto a BGR frame, centred at (cx, cy), clipped to frame."""
    sh, sw = sprite.shape[:2]
    fh, fw = frame.shape[:2]
    x0, y0 = cx - sw // 2, cy - sh // 2
    x1, y1 = x0 + sw, y0 + sh
    fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x1, fw), min(y1, fh)
    if fx0 >= fx1 or fy0 >= fy1:
        return
    sp = sprite[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    # integer blend (much faster than float): alpha in 0..256
    alpha = (sp[:, :, 3:4].astype(np.uint16) * int(opacity * 256)) >> 8
    if not alpha.any():
        return
    roi = frame[fy0:fy1, fx0:fx1]
    roi[:] = ((sp[:, :, :3].astype(np.uint16) * alpha
               + roi.astype(np.uint16) * (256 - alpha)) >> 8).astype(np.uint8)


# --------------------------------------------------------------------------- confetti

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    spin: float
    flip_phase: float
    flip_speed: float
    scale: float
    logo: int
    born: float
    life: float


class Confetti:
    GRAVITY = 900.0   # px / s^2 (at 720p)
    DRAG = 0.985

    def __init__(self, logos: list[np.ndarray], count: int, duration: float):
        self.logos = logos
        self.count = count
        self.duration = duration
        self.particles: list[Particle] = []

    def burst(self, w: int, h: int, now: float) -> None:
        """Two confetti cannons firing up and inwards from the bottom corners."""
        k = h / 720.0
        for i in range(self.count):
            left = i % 2 == 0
            x = w * (0.05 if left else 0.95)
            direction = 1 if left else -1
            ang = math.radians(random.uniform(55, 80))
            speed = random.uniform(900, 1500) * k
            self.particles.append(Particle(
                x=x + random.uniform(-20, 20), y=h + 30,
                vx=direction * math.cos(ang) * speed,
                vy=-math.sin(ang) * speed,
                angle=random.uniform(0, 360), spin=random.uniform(-360, 360),
                flip_phase=random.uniform(0, math.tau), flip_speed=random.uniform(4, 10),
                scale=random.uniform(0.55, 1.1) * k,
                logo=random.randrange(len(self.logos)),
                born=now + random.uniform(0, 0.25),  # stagger so it feels like a stream
                life=self.duration + random.uniform(-0.4, 0.4),
            ))

    @property
    def active(self) -> bool:
        return bool(self.particles)

    def update_and_draw(self, frame: np.ndarray, now: float, dt: float) -> None:
        h, w = frame.shape[:2]
        alive = []
        for p in self.particles:
            if now < p.born:
                alive.append(p)
                continue
            age = now - p.born
            if age > p.life or p.y > h + 200:
                continue
            p.vy += self.GRAVITY * (h / 720.0) * dt
            p.vx *= self.DRAG
            p.vy *= self.DRAG
            # flutter: once falling, cap fall speed and sway sideways like paper
            if p.vy > 0:
                p.vy = min(p.vy, 260 * h / 720.0)
                p.vx += math.sin(age * 3 + p.flip_phase) * 25 * dt * 60
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.angle += p.spin * dt
            p.flip_phase += p.flip_speed * dt
            alive.append(p)

            fade = min(1.0, (p.life - age) / 0.6)  # fade out over the last 0.6 s
            self._draw(frame, p, fade)
        self.particles = alive

    def _draw(self, frame: np.ndarray, p: Particle, opacity: float) -> None:
        src = self.logos[p.logo]
        sh, sw = src.shape[:2]
        # one affine warp = scale * 3D "flip" (squash x by |cos|) * rotation
        sx = p.scale * max(abs(math.cos(p.flip_phase)), 0.12)
        sy = p.scale
        d = max(4, int(math.hypot(sw * sx, sh * sy)) + 2)
        a = math.radians(p.angle)
        ca, sa = math.cos(a), math.sin(a)
        m = np.array([[ca * sx, -sa * sy, 0.0],
                      [sa * sx, ca * sy, 0.0]], np.float32)
        # map source centre to canvas centre
        m[:, 2] = np.array([d / 2, d / 2]) - m[:, :2] @ np.array([sw / 2, sh / 2])
        sprite = cv2.warpAffine(src, m, (d, d), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        overlay_bgra(frame, sprite, int(p.x), int(p.y), opacity)


# --------------------------------------------------------------------------- gestures

class GestureTrigger:
    """Fires once when `gesture` is held for `hold` seconds, then waits `cooldown`."""

    def __init__(self, model: Path, gesture: str, hold: float, cooldown: float, min_score: float):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (GestureRecognizer, GestureRecognizerOptions,
                                                   RunningMode)

        self.mp = mp
        self.recognizer = GestureRecognizer.create_from_options(GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
        ))
        self.gestures = [gesture] if isinstance(gesture, str) else list(gesture)
        self.hold = hold
        self.cooldown = cooldown
        self.min_score = min_score
        self.seen_since: dict[str, float | None] = {g: None for g in self.gestures}
        self.last_match = {g: -1e9 for g in self.gestures}
        self.last_fire = {g: -1e9 for g in self.gestures}
        self.last_label = "no hand"
        self.last_ts_ms = -1
        self._last_logged = ""

    GRACE = 0.3  # seconds a gesture may flicker out without resetting the hold timer

    def process(self, frame_bgr: np.ndarray, now: float) -> list[str]:
        """Returns the gestures that fired on this frame (usually empty)."""
        small = cv2.resize(frame_bgr, (640, int(640 * frame_bgr.shape[0] / frame_bgr.shape[1])))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(now * 1000), self.last_ts_ms + 1)  # must be strictly increasing
        self.last_ts_ms = ts
        result = self.recognizer.recognize_for_video(image, ts)

        label, matched = "no hand", set()
        for hand in result.gestures:
            top = hand[0]
            label = f"{top.category_name} {top.score:.2f}"
            if top.category_name in self.seen_since and top.score >= self.min_score:
                matched.add(top.category_name)
        self.last_label = label
        name = label.split(" ")[0]
        if name != self._last_logged:  # log changes so the terminal shows what the model sees
            print(f"  [seen] {label}")
            self._last_logged = name

        fired = []
        for g in self.gestures:
            if g in matched:
                self.last_match[g] = now
                if self.seen_since[g] is None:
                    self.seen_since[g] = now
            elif now - self.last_match[g] > self.GRACE:
                self.seen_since[g] = None  # gone for real (short dropouts are forgiven)
            since = self.seen_since[g]
            if (since is not None and now - since >= self.hold
                    and now - self.last_fire[g] >= self.cooldown):
                self.last_fire[g] = now
                self.seen_since[g] = None
                fired.append(g)
        return fired


class GestureWorker:
    """Runs GestureTrigger on a background thread so the video never waits on the model.

    The main loop hands over the newest frame; older unprocessed frames are dropped.
    """

    def __init__(self, trigger: GestureTrigger):
        import threading
        self.trigger = trigger
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._fired: list[str] = []
        self._stop = False
        self._event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame_bgr: np.ndarray) -> None:
        small = cv2.resize(frame_bgr, (640, int(640 * frame_bgr.shape[0] / frame_bgr.shape[1])))
        with self._lock:
            self._frame = small
        self._event.set()

    def fired(self) -> list[str]:
        with self._lock:
            f, self._fired = self._fired, []
        return f

    @property
    def label(self) -> str:
        return self.trigger.last_label

    def _run(self) -> None:
        while not self._stop:
            self._event.wait(0.5)
            self._event.clear()
            with self._lock:
                frame, self._frame = self._frame, None
            if frame is None:
                continue
            fired = self.trigger.process(frame, time.monotonic())
            if fired:
                with self._lock:
                    self._fired.extend(fired)

    def close(self) -> None:
        self._stop = True
        self._event.set()
        self._thread.join(timeout=2)


# --------------------------------------------------------------------------- camera setup (macOS)

VIRTUAL_CAM_WORDS = ("obs", "virtual", "camo", "mmhmm", "snap camera")


def list_cameras() -> list[str]:
    """Camera names in the order OpenCV's AVFoundation backend indexes them.

    OpenCV sorts devices by uniqueID, which is NOT the order AVFoundation lists them in
    (and AVFoundation's own order changes between runs). Getting this wrong opens the
    OBS Virtual Camera as input: 1920x1080, one frame, then nothing.
    """
    try:
        import AVFoundation as AV
        devices = AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo)
        devices = sorted(devices, key=lambda d: str(d.uniqueID()))
        return [str(d.localizedName()) for d in devices]
    except Exception:  # noqa: BLE001  (pyobjc missing / not macOS)
        return []


def pick_camera(requested: int | None, name_hint: str | None) -> int:
    """Return the webcam index. Never auto-picks a virtual camera (that would feed our own output back in)."""
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
    """Check macOS camera permission; ask for it if it was never asked. Returns True if allowed."""
    try:
        import threading
        import AVFoundation as AV
    except Exception:  # noqa: BLE001
        return True  # can't check; let OpenCV try
    status = AV.AVCaptureDevice.authorizationStatusForMediaType_(AV.AVMediaTypeVideo)
    if status == 3:  # authorized
        return True
    if status == 0:  # not determined -> show the macOS prompt
        print("Asking macOS for camera permission - click 'Allow' in the popup...")
        done, result = threading.Event(), {}

        def handler(granted):
            result["granted"] = bool(granted)
            done.set()

        AV.AVCaptureDevice.requestAccessForMediaType_completionHandler_(AV.AVMediaTypeVideo, handler)
        done.wait(60)
        if result.get("granted"):
            return True
    print("\nCamera permission is NOT granted to the app that launched this script.\n"
          "  Fix: System Settings > Privacy & Security > Camera > turn on your terminal app\n"
          "       (Terminal / iTerm), then quit and reopen that terminal.\n"
          "  If no popup appeared, run it from Terminal.app: double-click 'run.command' in Finder.")
    return False


# --------------------------------------------------------------------------- main loop

def open_virtual_cam(w: int, h: int, fps: int):
    try:
        import pyvirtualcam
        cam = pyvirtualcam.Camera(width=w, height=h, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR)
        print(f"Virtual camera ON: '{cam.device}'  -> pick it in Zoom > Settings > Video")
        return cam
    except Exception as e:  # noqa: BLE001
        print(f"Virtual camera unavailable ({e}).\n"
              "  Preview-only mode. Install OBS and start its Virtual Camera once (see README).")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=None,
                    help="webcam index (default: first non-virtual camera)")
    ap.add_argument("--camera-name", default=None, help="pick webcam by name, e.g. 'FaceTime'")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--logos", type=Path, default=DEFAULT_LOGOS, help="folder of logo PNGs")
    ap.add_argument("--logo-size", type=int, default=90, help="logo size in px at 720p")
    ap.add_argument("--count", type=int, default=70, help="logos per burst")
    ap.add_argument("--duration", type=float, default=3.5, help="seconds each logo lives")
    ap.add_argument("--gesture", default="Thumb_Up",
                    help="Thumb_Up, Victory, Open_Palm, Pointing_Up, ILoveYou, Closed_Fist, Thumb_Down")
    ap.add_argument("--captions", action="store_true", help="start with live captions on")
    ap.add_argument("--no-captions", action="store_true", help="disable the captions feature")
    ap.add_argument("--locale", default="en-US", help="speech language, e.g. en-US, en-IN, hi-IN")
    ap.add_argument("--vocab", type=Path, default=REPO / "vocab.txt",
                    help="words/phrases to bias captions toward (edit live)")
    ap.add_argument("--caption-pause", type=float, default=2.5,
                    help="seconds of silence before a new caption starts")
    ap.add_argument("--timer-label", default="", help="small text under the timer, e.g. 'Q&A'")
    ap.add_argument("--hold", type=float, default=0.3, help="seconds the gesture must be held")
    ap.add_argument("--cooldown", type=float, default=3.0)
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--no-vcam", action="store_true", help="preview only, no virtual camera")
    ap.add_argument("--no-preview", action="store_true", help="no preview window")
    args = ap.parse_args()

    print("Loading logos...")
    logos = load_logos(args.logos, args.logo_size) if args.logos.exists() else []
    if not logos:
        print(f"  no logos in {args.logos}/ - using placeholders (drop PNGs there)")
        logos = placeholder_logos(args.logo_size)

    trigger = GestureWorker(
        GestureTrigger(DEFAULT_MODEL, [args.gesture],
                       args.hold, args.cooldown, args.min_score))
    confetti = Confetti(logos, args.count, args.duration)

    captions, renderer = None, None
    if not args.no_captions:
        try:
            from captions import CaptionEngine, CaptionRenderer
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
    cam_index = pick_camera(args.camera, args.camera_name)
    names = list_cameras()
    cam_name = names[cam_index] if cam_index < len(names) else f"#{cam_index}"
    cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    ok, frame = False, None
    for _ in range(30):  # camera can take a moment to warm up
        ok, frame = cap.read()
        if ok:
            break
        time.sleep(0.1)
    if not ok:
        print(f"Could not read from webcam '{cam_name}'. Try --camera-name FaceTime or --camera 1.")
        return 1
    cam_h, cam_w = frame.shape[:2]
    # Many Mac cameras ignore the requested size and send 1080p; work at the requested size.
    if cam_w > args.width:
        w, h = args.width, int(round(cam_h * args.width / cam_w / 2) * 2)
    else:
        w, h = cam_w, cam_h
    print(f"Webcam '{cam_name}' {cam_w}x{cam_h} -> output {w}x{h}.")
    print(f"Gesture: '{args.gesture}' = confetti.")
    print("Keys (click the preview window first):\n"
          "  t = captions on/off   1-9 = countdown N min (0 = 10)   w = stopwatch / pause / resume\n"
          "  + / - = one minute more / less   x = clear timer   g = gaze tracking (preview only)\n"
          "  c = confetti   q = quit")

    def toggle_captions() -> None:
        if captions is None:
            print("  captions unavailable (see messages above)")
            return
        try:
            captions.stop() if captions.running else captions.start()
        except Exception as e:  # noqa: BLE001
            print(f"  captions error: {e}")

    from widgets import TimerWidget
    timer = TimerWidget(w, h, label=args.timer_label)

    from gaze import GazeWorker
    gaze = GazeWorker()

    def draw_widgets(img: np.ndarray, now: float) -> None:
        sprite = timer.sprite(now)
        if sprite is not None:
            cx, cy = timer.position(sprite)
            overlay_bgra(img, sprite, cx, cy, 1.0)

    def draw_captions(img: np.ndarray) -> None:
        if captions is None or not captions.running:
            return
        sprite = renderer.sprite(captions.current_text())
        if sprite is not None:
            cx, cy = renderer.position(sprite)
            overlay_bgra(img, sprite, cx, cy, 1.0)

    if captions is not None:
        from captions import CaptionRenderer
        renderer = CaptionRenderer(w, h)

    vcam = None if args.no_vcam else open_virtual_cam(w, h, args.fps)

    last = time.monotonic()
    fps_avg = 0.0
    misses = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                misses += 1
                if misses == 1 or misses % 30 == 0:
                    print(f"  ! webcam returned no frame ({misses}x), retrying...")
                if misses > 150:
                    print("Webcam stopped sending frames - is another app using it exclusively?")
                    break
                time.sleep(0.02)
                continue
            misses = 0
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            now = time.monotonic()
            dt = min(now - last, 0.1)
            last = now
            fps_avg = fps_avg * 0.9 + (1.0 / max(dt, 1e-3)) * 0.1

            trigger.submit(frame)  # before drawing, so the model never sees confetti
            gaze.submit(frame)
            for g in trigger.fired():
                if g == args.gesture:
                    print(f"  {g}!  confetti")
                    confetti.burst(w, h, now)
            confetti.update_and_draw(frame, now, dt)

            if vcam is not None:
                overlays = (captions is not None and captions.running) or timer.state != "hidden"
                out = frame.copy() if overlays else frame
                draw_widgets(out, now)
                draw_captions(out)  # text drawn un-mirrored so Zoom viewers can read it
                vcam.send(out)  # what Zoom sees: no debug text

            if not args.no_preview:
                preview = cv2.flip(frame, 1)  # mirror like a selfie view
                draw_widgets(preview, now)  # after the flip, so text reads correctly here too
                gaze.draw_hud(preview)  # preview only - never sent to Zoom
                draw_captions(preview)
                if captions is not None and captions.running:
                    cv2.putText(preview, "CC", (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(preview, f"{trigger.label}   {fps_avg:4.1f} fps",
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Confetti Cam (preview - not sent to Zoom)", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    confetti.burst(w, h, now)
                if key == ord("t"):
                    toggle_captions()
                msg = None
                if ord("1") <= key <= ord("9"):
                    msg = timer.start_countdown(now, key - ord("0"))
                elif key == ord("0"):
                    msg = timer.start_countdown(now, 10)
                elif key == ord("w"):
                    msg = timer.toggle(now)
                elif key in (ord("+"), ord("=")):
                    msg = timer.adjust(now, +1)
                elif key in (ord("-"), ord("_")):
                    msg = timer.adjust(now, -1)
                elif key == ord("x"):
                    msg = timer.clear()
                elif key == ord("g"):
                    msg = gaze.toggle()
                if msg:
                    print(f"  {msg}")
            elif vcam is not None:
                vcam.sleep_until_next_frame()
    except KeyboardInterrupt:
        pass
    finally:
        trigger.close()
        try:
            gaze.close()
        except NameError:
            pass
        if captions is not None:
            captions.stop()
        cap.release()
        if vcam is not None:
            vcam.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
