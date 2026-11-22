"""Gaze / attention tracking from the webcam (MediaPipe Face Landmarker).

Per frame:  eyes (iris position between eye corners) + head turn (nose vs face centre)
            -> ON SCREEN / LEFT / RIGHT / UP / DOWN / NO FACE
Directions are the PERSON's own left/right.

The first ~2 s after starting are calibration: look at your screen normally. That sets your
personal "centre" (a webcam sits above the screen, so looking at the screen reads as slightly down).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
FACE_MODEL = ROOT / "models" / "face_landmarker.task"

# landmark indices (MediaPipe face mesh with irises)
IRIS_A, IRIS_B = 468, 473
EYE_A = (33, 133)       # corners of one eye
EYE_B = (362, 263)      # corners of the other eye
NOSE, CHEEK_A, CHEEK_B = 1, 234, 454
EYE_TOP, MOUTH = 168, 13  # between the eyes, centre of the mouth

# thresholds relative to the calibrated centre (tune with the numbers shown in the HUD)
HEAD_WEIGHT = 0.5  # how much head turn adds to eye direction (eyes often counter-rotate)
GAZE_H_T = 0.12    # combined eyes + head horizontal threshold
HEAD_EXTREME = 0.35  # head turned this far counts as away whatever the eyes do
EYE_V_UP_T = 0.25
EYE_V_DOWN_T = 0.30
HEAD_V_T = 0.12
CALIBRATE_S = 2.0
DEBOUNCE_S = 0.35      # a new state must hold this long before it counts
LOOKAWAY_MIN_S = 1.0   # shorter glances are not logged as look-aways


@dataclass
class GazeStats:
    started: float = 0.0
    on_screen_s: float = 0.0
    tracked_s: float = 0.0
    lookaways: int = 0
    longest_s: float = 0.0
    events: list = field(default_factory=list)  # (start_time, duration, state)

    def on_screen_pct(self) -> float:
        return 100.0 * self.on_screen_s / self.tracked_s if self.tracked_s else 100.0


class GazeTracker:
    def __init__(self, model: Path = FACE_MODEL):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

        self.mp = mp
        self.landmarker = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=RunningMode.VIDEO,
            num_faces=2,
            output_face_blendshapes=True,
        ))
        self._ts = -1
        self.reset()

    def reset(self) -> None:
        self.calib: list[np.ndarray] = []
        self.base: np.ndarray | None = None
        self.smooth: np.ndarray | None = None
        self.state = "CALIBRATING"
        self._candidate, self._candidate_since = None, 0.0
        self._away_since: float | None = None
        self._last_t: float | None = None
        self.faces = 0
        self.signals = np.zeros(4)  # eye_h, eye_v, head_h, head_v (relative to centre)
        self.stats = GazeStats(started=time.monotonic())
        self.calib_until = time.monotonic() + CALIBRATE_S

    # ------------------------------------------------------------ measurement
    def _measure(self, frame_bgr: np.ndarray, now: float):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        self._ts = max(int(now * 1000), self._ts + 1)
        res = self.landmarker.detect_for_video(image, self._ts)
        self.faces = len(res.face_landmarks)
        if not res.face_landmarks:
            return None, False
        lm = res.face_landmarks[0]
        P = lambda i: np.array([lm[i].x, lm[i].y])  # noqa: E731

        def eye_offset(iris, corners):
            a, b = P(corners[0]), P(corners[1])
            width = np.linalg.norm(b - a) + 1e-6
            return (P(iris)[0] - (a[0] + b[0]) / 2) / width

        # +x in the (un-mirrored) camera image = the person's LEFT
        eye_h = (eye_offset(IRIS_A, EYE_A) + eye_offset(IRIS_B, EYE_B)) / 2
        face_w = abs(P(CHEEK_B)[0] - P(CHEEK_A)[0]) + 1e-6
        head_h = (P(NOSE)[0] - (P(CHEEK_A)[0] + P(CHEEK_B)[0]) / 2) / face_w
        # vertical: + = up
        bs = {c.category_name: c.score for c in res.face_blendshapes[0]} if res.face_blendshapes else {}
        eye_v = ((bs.get("eyeLookUpLeft", 0) + bs.get("eyeLookUpRight", 0))
                 - (bs.get("eyeLookDownLeft", 0) + bs.get("eyeLookDownRight", 0))) / 2
        span = P(MOUTH)[1] - P(EYE_TOP)[1] + 1e-6
        head_v = -((P(NOSE)[1] - P(EYE_TOP)[1]) / span)
        blink = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2 > 0.5
        return np.array([eye_h, eye_v, head_h, head_v]), blink

    # ------------------------------------------------------------ state
    def _classify(self, s: np.ndarray) -> str:
        eye_h, eye_v, head_h, head_v = s
        # eyes counter-rotate when you turn your head but keep watching the screen,
        # so judge the combined direction, not head and eyes separately
        horiz = eye_h + head_h * HEAD_WEIGHT
        if abs(horiz) > GAZE_H_T or abs(head_h) > HEAD_EXTREME:
            return "LEFT" if (horiz if abs(horiz) > GAZE_H_T else head_h) > 0 else "RIGHT"
        if eye_v > EYE_V_UP_T or head_v > HEAD_V_T:
            return "UP"
        if eye_v < -EYE_V_DOWN_T or head_v < -HEAD_V_T:
            return "DOWN"
        return "ON SCREEN"

    def process(self, frame_bgr: np.ndarray, now: float) -> None:
        raw, blink = self._measure(frame_bgr, now)
        dt = 0.0 if self._last_t is None else min(now - self._last_t, 0.5)
        self._last_t = now

        if raw is None:
            observed = "NO FACE"
        elif self.base is None:
            if not blink:
                self.calib.append(raw)
            if now >= self.calib_until and len(self.calib) >= 5:
                self.base = np.median(np.array(self.calib), axis=0)
                self.smooth = np.zeros(4)
                print("  [gaze] calibrated - tracking")
            return
        else:
            if blink:  # blinks are not look-aways: keep the previous reading
                return
            rel = raw - self.base
            self.smooth = rel if self.smooth is None else self.smooth * 0.6 + rel * 0.4
            self.signals = self.smooth
            observed = self._classify(self.smooth)

        if self.base is None:
            return
        # debounce: the observed state must hold DEBOUNCE_S before it becomes the state
        if observed != self._candidate:
            self._candidate, self._candidate_since = observed, now
        if observed != self.state and now - self._candidate_since >= DEBOUNCE_S:
            self._change(observed, now)

        st = self.stats
        st.tracked_s += dt
        if self.state == "ON SCREEN":
            st.on_screen_s += dt

    def _change(self, new: str, now: float) -> None:
        old = self.state
        if old not in ("ON SCREEN", "CALIBRATING") and self._away_since is not None:
            dur = now - self._away_since
            if dur >= LOOKAWAY_MIN_S:
                st = self.stats
                st.lookaways += 1
                st.longest_s = max(st.longest_s, dur)
                st.events.append((self._away_since - st.started, dur, old))
                print(f"  [gaze] {old.lower()} for {dur:.1f}s")
        self._away_since = now if new != "ON SCREEN" else None
        self.state = new


class GazeWorker:
    """Runs GazeTracker on a background thread on the newest frame (older frames dropped)."""

    def __init__(self):
        self.tracker = GazeTracker()
        self.enabled = False
        self._lock = threading.Lock()
        self._frame = None
        self._event = threading.Event()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def toggle(self) -> str:
        self.enabled = not self.enabled
        if self.enabled:
            self.tracker.reset()
            return "gaze tracking ON - look at your screen for 2 s to calibrate"
        st = self.tracker.stats
        return (f"gaze tracking OFF - on screen {st.on_screen_pct():.0f}%, "
                f"{st.lookaways} look-aways, longest {st.longest_s:.1f}s")

    def submit(self, frame_bgr: np.ndarray) -> None:
        if not self.enabled:
            return
        small = cv2.resize(frame_bgr, (640, int(640 * frame_bgr.shape[0] / frame_bgr.shape[1])))
        with self._lock:
            self._frame = small
        self._event.set()

    def _run(self) -> None:
        while not self._stop:
            self._event.wait(0.5)
            self._event.clear()
            with self._lock:
                frame, self._frame = self._frame, None
            if frame is not None and self.enabled:
                self.tracker.process(frame, time.monotonic())

    def close(self) -> None:
        self._stop = True
        self._event.set()
        self._thread.join(timeout=2)

    # ------------------------------------------------------------ HUD (preview only)
    def draw_hud(self, preview: np.ndarray) -> None:
        """Draw on the MIRRORED preview: the person's left appears on the left of the preview."""
        if not self.enabled:
            return
        t = self.tracker
        h, w = preview.shape[:2]
        colours = {"ON SCREEN": (80, 200, 80), "CALIBRATING": (200, 200, 200), "NO FACE": (60, 60, 230)}
        col = colours.get(t.state, (0, 165, 255))
        x0, y0 = 12, h - 110
        cv2.rectangle(preview, (x0 - 6, y0 - 28), (x0 + 430, y0 + 92), (0, 0, 0), -1)
        label = t.state if t.state != "CALIBRATING" else "CALIBRATING - look at your screen"
        if t.faces > 1:
            label += f"  ({t.faces} faces!)"
        cv2.putText(preview, f"GAZE: {label}", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)
        st = t.stats
        cv2.putText(preview, f"on screen {st.on_screen_pct():.0f}%   look-aways {st.lookaways}   "
                             f"longest {st.longest_s:.1f}s", (x0, y0 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        e_h, e_v, h_h, h_v = t.signals
        cv2.putText(preview, f"eye h {e_h:+.2f} v {e_v:+.2f}   head h {h_h:+.2f} v {h_v:+.2f}",
                    (x0, y0 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
        # direction dot: mirrored preview, so the person's left (+h) is drawn to the left
        cx, cy, r = x0 + 400, y0 + 40, 26
        cv2.circle(preview, (cx, cy), r, (90, 90, 90), 1, cv2.LINE_AA)
        dx = int(np.clip(-(e_h + h_h * HEAD_WEIGHT) / 0.3, -1, 1) * r)
        dy = int(np.clip(-(e_v) / 0.5, -1, 1) * r)
        cv2.circle(preview, (cx + dx, cy + dy), 6, col, -1, cv2.LINE_AA)
