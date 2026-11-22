"""Party Cam: branded logo confetti + a cat that sits beside you and reacts to your face.

Webcam -> MediaPipe (hand gestures + face) -> confetti & cat drawn on frames -> virtual camera -> Zoom

  👍  hold Thumb_Up      -> logo confetti
  ✌️  hold Victory       -> cat on / off

Run:  .venv/bin/python party_cam.py
Preview keys:  c = confetti,  k = cat on/off,  q / Esc = quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from cat_companion import FACE_MODEL, Cat, FaceTracker
from confetti_cam import (DEFAULT_LOGOS, DEFAULT_MODEL, Confetti, load_logos, open_virtual_cam,
                          placeholder_logos)


class Gestures:
    """Runs the gesture recognizer once per frame; reports gestures held for `hold` seconds."""

    def __init__(self, model: Path, watch: list[str], hold: float, cooldown: float,
                 min_score: float):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (GestureRecognizer, GestureRecognizerOptions,
                                                   RunningMode)

        self.recognizer = GestureRecognizer.create_from_options(GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
        ))
        self.watch, self.hold, self.cooldown, self.min_score = watch, hold, cooldown, min_score
        self.since: dict[str, float] = {}
        self.last_fire: dict[str, float] = {}
        self.label = "None"
        self.last_ts_ms = -1

    def process(self, mp_image, now: float) -> list[str]:
        ts = max(int(now * 1000), self.last_ts_ms + 1)
        self.last_ts_ms = ts
        result = self.recognizer.recognize_for_video(mp_image, ts)
        seen = set()
        self.label = "None"
        for hand in result.gestures:
            top = hand[0]
            if top.category_name != "None" and top.score >= self.min_score:
                seen.add(top.category_name)
                self.label = f"{top.category_name} {top.score:.2f}"

        fired = []
        for g in self.watch:
            if g not in seen:
                self.since.pop(g, None)
                continue
            start = self.since.setdefault(g, now)
            if now - start >= self.hold and now - self.last_fire.get(g, -1e9) >= self.cooldown:
                self.last_fire[g] = now
                self.since.pop(g, None)
                fired.append(g)
        return fired


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--logos", type=Path, default=DEFAULT_LOGOS)
    ap.add_argument("--logo-size", type=int, default=90)
    ap.add_argument("--count", type=int, default=70)
    ap.add_argument("--duration", type=float, default=3.5)
    ap.add_argument("--confetti-gesture", default="Thumb_Up")
    ap.add_argument("--cat-gesture", default="Victory")
    ap.add_argument("--cat-side", choices=["auto", "left", "right"], default="auto",
                    help="side of the frame others see (auto = away from your face)")
    ap.add_argument("--cat", action="store_true", help="start with the cat on")
    ap.add_argument("--hold", type=float, default=0.3)
    ap.add_argument("--cooldown", type=float, default=2.0)
    ap.add_argument("--min-score", type=float, default=0.6)
    ap.add_argument("--no-vcam", action="store_true")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    import mediapipe as mp

    print("Loading logos...")
    logos = load_logos(args.logos, args.logo_size) if args.logos.exists() else []
    if not logos:
        print(f"  no logos in {args.logos}/ - using placeholders")
        logos = placeholder_logos(args.logo_size)
    confetti = Confetti(logos, args.count, args.duration)
    gestures = Gestures(DEFAULT_MODEL, [args.confetti_gesture, args.cat_gesture], args.hold,
                        args.cooldown, args.min_score)
    tracker = FaceTracker(FACE_MODEL)
    cat = Cat(args.cat_side)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    ok, frame = cap.read()
    if not ok:
        print("Could not read from webcam. Check macOS camera permission for your terminal app\n"
              "(System Settings > Privacy & Security > Camera).")
        return 1
    h, w = frame.shape[:2]
    print(f"Webcam {w}x{h}.  {args.confetti_gesture} = confetti,  {args.cat_gesture} = cat on/off.\n"
          "Preview keys: c = confetti, k = cat, q = quit")
    vcam = None if args.no_vcam else open_virtual_cam(w, h, args.fps)

    start_cat = args.cat
    last = time.monotonic()
    fps_avg = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            now = time.monotonic()
            dt = min(now - last, 0.1)
            last = now
            fps_avg = fps_avg * 0.9 + (1.0 / max(dt, 1e-3)) * 0.1

            # one small RGB image shared by both models
            small = cv2.resize(frame, (640, int(640 * h / w)))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
            fired = gestures.process(mp_image, now)
            face = tracker.process(mp_image, now)

            if start_cat:
                cat.toggle(now, face)
                start_cat = False
            if args.confetti_gesture in fired:
                print("  confetti!")
                confetti.burst(w, h, now)
            if args.cat_gesture in fired:
                cat.toggle(now, face)
                print(f"  cat {'on' if cat.on else 'off'}")

            cat.update(face, now, dt, w, h)
            cat.draw(frame, now, dt)
            confetti.update_and_draw(frame, now, dt)

            if vcam is not None:
                vcam.send(frame)  # what Zoom sees

            if not args.no_preview:
                preview = cv2.flip(frame, 1)
                info = (f"{gestures.label}  |  face {'yes' if face.found else 'no'}  "
                        f"tilt {face.roll:+.0f}  turn {face.yaw:+.2f}  "
                        f"smile {face.smile:.2f}  mouth {face.jaw:.2f}  |  {fps_avg:4.1f} fps")
                cv2.putText(preview, info, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                            cv2.LINE_AA)
                cv2.imshow("Party Cam (preview - not sent to Zoom)", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    confetti.burst(w, h, now)
                if key == ord("k"):
                    cat.toggle(now, face)
            elif vcam is not None:
                vcam.sleep_until_next_frame()
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if vcam is not None:
            vcam.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
