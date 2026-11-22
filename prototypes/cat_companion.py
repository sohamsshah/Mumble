"""A cartoon cat that sits beside you on camera and reacts to your face, live.

FaceTracker turns a webcam frame into a FaceState (head tilt/turn/nod, blink, mouth, smile).
Cat reads FaceState every frame and draws itself onto the frame:

  - copies your head tilt, turn and nods (smoothed, so it feels alive, not robotic)
  - blinks when you blink, opens its mouth + "meow!" when you open yours
  - its eyes follow where your face is
  - you smile  -> hearts float up
  - you leave  -> after a few seconds it falls asleep (Zzz); come back -> it wakes up startled
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from confetti_cam import overlay_bgra

ROOT = Path(__file__).resolve().parent
FACE_MODEL = ROOT / "models" / "face_landmarker.task"

# Colours are BGRA (OpenCV order). Orange tabby.
FUR = (70, 150, 245, 255)
STRIPE = (35, 100, 205, 255)
OUTLINE = (45, 45, 70, 255)
EAR_IN = (185, 165, 255, 255)
EYE = (90, 205, 120, 255)
PUPIL = (25, 25, 25, 255)
NOSE = (150, 120, 245, 255)
WHITE = (250, 250, 250, 255)
MOUTH = (60, 40, 110, 255)
TONGUE = (140, 110, 240, 255)
BLUSH = (170, 150, 255, 255)
HEART = (170, 90, 255)


# --------------------------------------------------------------------------- face tracking

@dataclass
class FaceState:
    found: bool = False
    cx: float = 0.5      # face centre, 0..1 in the (un-mirrored) frame
    cy: float = 0.5
    roll: float = 0.0    # degrees; positive = head tilted clockwise as seen in the frame
    yaw: float = 0.0     # -1..1; positive = nose turned toward the frame's right edge
    pitch: float = 0.55  # nose height within the face (0 top .. 1 chin); rises when nodding down
    blink: float = 0.0   # 0..1
    jaw: float = 0.0     # 0..1 mouth open
    smile: float = 0.0   # 0..1


class FaceTracker:
    def __init__(self, model: Path = FACE_MODEL):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (FaceLandmarker, FaceLandmarkerOptions,
                                                   RunningMode)

        self.landmarker = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        ))
        self.last_ts_ms = -1

    def process(self, mp_image, now: float) -> FaceState:
        ts = max(int(now * 1000), self.last_ts_ms + 1)  # must be strictly increasing
        self.last_ts_ms = ts
        return self.to_state(self.landmarker.detect_for_video(mp_image, ts), mp_image.width,
                             mp_image.height)

    @staticmethod
    def to_state(result, img_w: int, img_h: int) -> FaceState:
        if not result.face_landmarks:
            return FaceState()
        lms = result.face_landmarks[0]
        pts = np.array([(p.x * img_w, p.y * img_h) for p in lms], np.float32)

        # 33 / 263 = outer eye corners, 1 = nose tip, 234 / 454 = cheeks, 10 = forehead, 152 = chin
        eye_a, eye_b = pts[33], pts[263]
        roll = math.degrees(math.atan2(eye_b[1] - eye_a[1], eye_b[0] - eye_a[0]))

        # undo the tilt before measuring turn / nod, so tilting doesn't read as turning
        c, s = math.cos(math.radians(-roll)), math.sin(math.radians(-roll))
        rot = np.array([[c, -s], [s, c]], np.float32)
        nose, cl, cr, top, chin = (rot @ pts[i] for i in (1, 234, 454, 10, 152))
        width, height = cr[0] - cl[0], chin[1] - top[1]
        yaw = ((nose[0] - cl[0]) / width - 0.5) * 2 if abs(width) > 1 else 0.0
        pitch = (nose[1] - top[1]) / height if abs(height) > 1 else 0.55

        bs = {}
        if result.face_blendshapes:
            bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
        return FaceState(
            found=True,
            cx=float(pts[:, 0].mean() / img_w), cy=float(pts[:, 1].mean() / img_h),
            roll=roll, yaw=float(np.clip(yaw, -1, 1)), pitch=float(pitch),
            blink=max(bs.get("eyeBlinkLeft", 0), bs.get("eyeBlinkRight", 0)),
            jaw=bs.get("jawOpen", 0.0),
            smile=(bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2,
        )


# --------------------------------------------------------------------------- drawing helpers

def _ease_out_back(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    c1 = 1.70158
    return 1 + (c1 + 1) * (t - 1) ** 3 + c1 * (t - 1) ** 2


def render_head(blink: float, jaw: float, smile: float, look: tuple[float, float],
                asleep: bool, startled: bool) -> np.ndarray:
    """Cat head on a transparent 400x400 BGRA canvas, facing forward."""
    c = np.zeros((400, 400, 4), np.uint8)
    aa = cv2.LINE_AA

    # ears (flattened back when startled)
    drop = 35 if startled else 0
    for sx in (1, -1):
        def x(v): return int(200 + sx * (v - 200))
        outer = np.array([(x(68), 160), (x(92) - sx * drop, 18 + drop), (x(175), 92)], np.int32)
        inner = np.array([(x(90), 140), (x(100) - sx * drop, 50 + drop), (x(155), 100)], np.int32)
        cv2.fillPoly(c, [outer], FUR, aa)
        cv2.polylines(c, [outer], True, OUTLINE, 6, aa)
        cv2.fillPoly(c, [inner], EAR_IN, aa)

    # head
    cv2.ellipse(c, (200, 215), (152, 128), 0, 0, 360, FUR, -1, aa)
    cv2.ellipse(c, (200, 215), (152, 128), 0, 0, 360, OUTLINE, 6, aa)
    for (x0, y0), (x1, y1) in (((200, 92), (200, 130)), ((165, 98), (174, 128)),
                               ((235, 98), (226, 128))):
        cv2.line(c, (x0, y0), (x1, y1), STRIPE, 11, aa)
    # muzzle
    cv2.ellipse(c, (200, 272), (62, 40), 0, 0, 360, (215, 225, 250, 255), -1, aa)

    if smile > 0.3:
        for bx in (118, 282):
            cv2.circle(c, (bx, 262), int(14 + 10 * smile), BLUSH, -1, aa)

    # eyes
    for ex in (140, 260):
        if asleep or blink > 0.6:
            cv2.ellipse(c, (ex, 205), (30, 14), 0, 10, 170, OUTLINE, 6, aa)  # closed "u" lids
            continue
        ey = 46 if startled else int(38 * (1 - blink) + 4)
        cv2.ellipse(c, (ex, 200), (32, ey), 0, 0, 360, EYE, -1, aa)
        cv2.ellipse(c, (ex, 200), (32, ey), 0, 0, 360, OUTLINE, 5, aa)
        px = int(ex + np.clip(look[0], -1, 1) * 14)
        py = int(200 + np.clip(look[1], -1, 1) * min(12, ey * 0.4))
        pw = 7 if startled else 11
        cv2.ellipse(c, (px, py), (pw, max(3, ey - 8)), 0, 0, 360, PUPIL, -1, aa)
        cv2.circle(c, (px + 7, py - max(2, ey // 3)), 6, WHITE, -1, aa)

    # nose
    cv2.fillPoly(c, [np.array([(185, 244), (215, 244), (200, 260)], np.int32)], NOSE, aa)

    # mouth
    if jaw > 0.25:
        mh = int(10 + 45 * min(jaw, 1.0))
        cv2.ellipse(c, (200, 285 + mh // 3), (26, mh), 0, 0, 360, MOUTH, -1, aa)
        cv2.ellipse(c, (200, 285 + mh // 3 + mh // 2), (16, mh // 3 + 2), 0, 0, 360, TONGUE, -1, aa)
        cv2.ellipse(c, (200, 285 + mh // 3), (26, mh), 0, 0, 360, OUTLINE, 4, aa)
    else:
        r = int(13 + 6 * smile)
        cv2.line(c, (200, 260), (200, 268), OUTLINE, 4, aa)
        cv2.ellipse(c, (200 - r, 268), (r, int(10 + 6 * smile)), 0, 0, 180, OUTLINE, 4, aa)
        cv2.ellipse(c, (200 + r, 268), (r, int(10 + 6 * smile)), 0, 0, 180, OUTLINE, 4, aa)

    # whiskers
    for sx in (1, -1):
        for y0, y1 in ((262, 245), (274, 278), (286, 310)):
            cv2.line(c, (200 - sx * 78, y0), (200 - sx * 165, y1), OUTLINE, 3, aa)
    return c


def _draw_heart(frame: np.ndarray, x: int, y: int, size: int) -> None:
    r = max(2, size // 2)
    cv2.circle(frame, (x - r // 2 - 1, y), r // 2 + 1, HEART, -1, cv2.LINE_AA)
    cv2.circle(frame, (x + r // 2 + 1, y), r // 2 + 1, HEART, -1, cv2.LINE_AA)
    tri = np.array([(x - r - 1, y + 1), (x + r + 1, y + 1), (x, y + r + r // 2)], np.int32)
    cv2.fillPoly(frame, [tri], HEART, cv2.LINE_AA)


def _outlined_text(frame, text, org, scale, thickness, colour=(255, 255, 255)):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, (40, 40, 60), thickness + 4,
                cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, colour, thickness, cv2.LINE_AA)


# --------------------------------------------------------------------------- the cat

class Cat:
    SLEEP_AFTER = 4.0      # seconds without a face before it dozes off
    STARTLE_FOR = 0.9

    def __init__(self, side: str = "auto"):
        self.side = side          # "auto" | "left" | "right" (in the frame others see)
        self.on = False
        self.on_since = 0.0
        self.right = True
        # smoothed pose / expression
        self.roll = self.yaw = self.nod = 0.0
        self.blink = self.jaw = self.smile = 0.0
        self.look = (0.0, 0.0)
        self.pitch_base: float | None = None
        # behaviour state
        self.last_seen = -1e9
        self.asleep = False
        self.woke_at = -1e9
        self.idle_blink_at = 0.0
        self.blink_until = 0.0
        self.hearts: list[list[float]] = []   # x, y, age, size
        self.zs: list[list[float]] = []       # x, y, age
        self.next_heart = 0.0
        self.next_z = 0.0
        self.head_xy = (0, 0)

    def toggle(self, now: float, face: FaceState) -> None:
        self.on = not self.on
        if self.on:
            self.on_since = now
            self.last_seen = now
            self.asleep = False
            if self.side == "auto":   # sit on the side away from your face
                self.right = (face.cx < 0.5) if face.found else True
            else:
                self.right = self.side == "right"

    # ----------------------------------------------------------------- behaviour
    def update(self, face: FaceState, now: float, dt: float, frame_w: int, frame_h: int) -> None:
        if not self.on:
            return
        k_pose = 1 - math.exp(-dt * 10)    # pose follows you with a slight, lifelike lag
        k_fast = 1 - math.exp(-dt * 30)    # blinks / mouth need to be snappy

        if face.found:
            self.last_seen = now
            if self.asleep:
                self.asleep = False
                self.woke_at = now
            if self.pitch_base is None:
                self.pitch_base = face.pitch
            # slowly learn your "neutral" nose height so nods are measured relative to it
            self.pitch_base += (face.pitch - self.pitch_base) * (1 - math.exp(-dt * 0.15))
            nod = float(np.clip((face.pitch - self.pitch_base) / 0.07, -1, 1))

            self.roll += (face.roll - self.roll) * k_pose
            self.yaw += (face.yaw - self.yaw) * k_pose
            self.nod += (nod - self.nod) * k_pose
            self.blink += (face.blink - self.blink) * k_fast
            self.jaw += (face.jaw - self.jaw) * k_fast
            self.smile += (face.smile - self.smile) * k_pose

            # eyes look toward your face
            hx, hy = self.head_xy
            dx = face.cx * frame_w - hx
            dy = face.cy * frame_h - hy
            n = math.hypot(dx, dy) or 1.0
            target = (dx / n, dy / n * 0.6)
        else:
            if now - self.last_seen > self.SLEEP_AFTER:
                self.asleep = True
            for name in ("roll", "yaw", "nod", "jaw", "smile"):
                setattr(self, name, getattr(self, name) * (1 - k_pose))
            # blink on its own every few seconds
            if now > self.idle_blink_at:
                self.idle_blink_at = now + random.uniform(2.5, 5.0)
                self.blink_until = now + 0.15
            self.blink = 1.0 if now < self.blink_until else self.blink * (1 - k_fast)
            target = (0.0, 0.0)
        self.look = (self.look[0] + (target[0] - self.look[0]) * k_pose,
                     self.look[1] + (target[1] - self.look[1]) * k_pose)

        u = frame_h / 720
        if self.smile > 0.5 and now > self.next_heart and not self.asleep:
            self.next_heart = now + 0.3
            hx, hy = self.head_xy
            self.hearts.append([hx + random.uniform(-90, 90) * u, hy - 90 * u, 0.0,
                                random.uniform(22, 38) * u])
        if self.asleep and now > self.next_z:
            self.next_z = now + 0.9
            hx, hy = self.head_xy
            self.zs.append([hx + (60 if self.right else -60) * u, hy - 80 * u, 0.0])

    # ----------------------------------------------------------------- drawing
    def draw(self, frame: np.ndarray, now: float, dt: float) -> None:
        if not self.on:
            return
        h, w = frame.shape[:2]
        u = h / 720
        pop = _ease_out_back((now - self.on_since) / 0.45)
        if pop <= 0.01:
            return
        side = 1 if self.right else -1
        cx = int(w - 200 * u) if self.right else int(200 * u)
        startled = now - self.woke_at < self.STARTLE_FOR

        # tail (behind the body), swishing
        speed, swing = (1.0, 0.15) if self.asleep else (2.2, 0.45)
        base = np.array([cx - side * 120 * u * pop, h - 30 * u], np.float32)
        pts = []
        for i in range(12):
            t = i / 11
            ang = -math.pi / 2 - side * (0.6 + t * 0.9) + math.sin(now * speed - t * 2) * swing * t
            pts.append(base + np.array([math.cos(ang), math.sin(ang)]) * t * 190 * u * pop)
        pts = np.array(pts, np.int32)
        cv2.polylines(frame, [pts], False, OUTLINE[:3], max(2, int(36 * u * pop)), cv2.LINE_AA)
        cv2.polylines(frame, [pts], False, FUR[:3], max(1, int(26 * u * pop)), cv2.LINE_AA)

        # body + chest + paws
        by = int(h + 40 * u)
        ax, ay = int(150 * u * pop), int(200 * u * pop)
        cv2.ellipse(frame, (cx, by), (ax + 4, ay + 4), 0, 0, 360, OUTLINE[:3], -1, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, by), (ax, ay), 0, 0, 360, FUR[:3], -1, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, by + int(10 * u)), (int(70 * u * pop), int(150 * u * pop)), 0, 0,
                    360, (215, 225, 250), -1, cv2.LINE_AA)
        for px in (-45, 45):
            cv2.ellipse(frame, (cx + int(px * u * pop), h - int(12 * u)),
                        (int(36 * u * pop), int(24 * u * pop)), 0, 180, 360, FUR[:3], -1,
                        cv2.LINE_AA)
            cv2.ellipse(frame, (cx + int(px * u * pop), h - int(12 * u)),
                        (int(36 * u * pop), int(24 * u * pop)), 0, 180, 360, OUTLINE[:3],
                        max(1, int(4 * u)), cv2.LINE_AA)

        # head: copy your pose
        roll = self.roll + (18 * side if self.asleep else 0)
        nod = 0.6 if self.asleep else self.nod
        size = int(270 * u * pop)
        head = render_head(self.blink, self.jaw, self.smile, self.look, self.asleep, startled)
        turn = 1 - 0.18 * abs(self.yaw)                    # fake 3D: narrower when turned
        head = cv2.resize(head, (max(2, int(size * turn)), max(2, int(size * (1 - 0.08 * abs(nod))))),
                          interpolation=cv2.INTER_AREA)
        hh, hw = head.shape[:2]
        d = int(math.hypot(hw, hh)) + 2
        canvas = np.zeros((d, d, 4), np.uint8)
        canvas[(d - hh) // 2:(d - hh) // 2 + hh, (d - hw) // 2:(d - hw) // 2 + hw] = head
        m = cv2.getRotationMatrix2D((d / 2, d / 2), -roll, 1.0)
        canvas = cv2.warpAffine(canvas, m, (d, d), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        hx = int(cx + self.yaw * 45 * u)
        hy = int(h - 250 * u * pop + nod * 35 * u)
        self.head_xy = (hx, hy)
        overlay_bgra(frame, canvas, hx, hy, 1.0)

        # reactions
        bubble_x = hx - side * int(130 * u)
        if startled:
            _outlined_text(frame, "!", (hx + side * int(110 * u), hy - int(110 * u)), 2.4 * u,
                           max(2, int(5 * u)), (80, 220, 255))
        if self.jaw > 0.45 and not self.asleep:
            self._bubble(frame, "meow!", bubble_x, hy - int(150 * u), u)

        for hrt in self.hearts:
            hrt[2] += dt
            hrt[1] -= 110 * u * dt
            sway = math.sin(hrt[2] * 5) * 10 * u
            _draw_heart(frame, int(hrt[0] + sway), int(hrt[1]), int(hrt[3] * (1 - hrt[2] / 1.8)))
        self.hearts = [hh_ for hh_ in self.hearts if hh_[2] < 1.6]

        for z in self.zs:
            z[2] += dt
            z[1] -= 45 * u * dt
            z[0] += 20 * u * dt * (1 if self.right else -1)
            _outlined_text(frame, "z", (int(z[0]), int(z[1])), (0.8 + z[2] * 0.6) * u,
                           max(1, int(3 * u)))
        self.zs = [z for z in self.zs if z[2] < 2.4]

    @staticmethod
    def _bubble(frame, text, x, y, u):
        scale, th = 1.1 * u, max(1, int(2 * u))
        (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, th)
        pad = int(14 * u)
        x0, y0 = x - tw // 2 - pad, y - tht - pad
        x1, y1 = x + tw // 2 + pad, y + pad
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (45, 45, 70), max(1, int(3 * u)), cv2.LINE_AA)
        cv2.putText(frame, text, (x - tw // 2, y), cv2.FONT_HERSHEY_DUPLEX, scale, (45, 45, 70), th,
                    cv2.LINE_AA)
