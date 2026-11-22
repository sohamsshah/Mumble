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
