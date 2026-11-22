"""On-camera widgets. TimerWidget: a countdown / stopwatch pill drawn on your video.

Keyboard control (in the preview window):
  1-9   start a countdown of 1-9 minutes (0 = 10 minutes)
  w     no timer -> start a stopwatch;  running -> pause;  paused -> resume
  + / - add / remove one minute (countdown)
  x     clear the timer
"""

from __future__ import annotations

import numpy as np

FONT_CANDIDATES = [
    "/System/Library/Fonts/SFNSMono.ttf",      # monospaced digits: the pill doesn't jiggle
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
]

# RGBA colours
BG_RUNNING = (20, 20, 24, 190)
BG_WARNING = (196, 120, 0, 215)     # countdown: last minute
BG_DONE = (200, 40, 40, 225)        # countdown reached zero
BG_PAUSED = (40, 90, 200, 210)
DOT_RUNNING = (255, 70, 70, 255)

DONE_SHOW_S = 15.0  # "TIME'S UP" flashes this long, then the timer hides


def _font(size: int):
    from PIL import ImageFont
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fmt(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class TimerWidget:
    def __init__(self, frame_w: int, frame_h: int, label: str = ""):
        self.w, self.h = frame_w, frame_h
        self.label = label
        self.state = "hidden"        # hidden | running | paused
        self.countdown_s = 0.0       # > 0: countdown from this; 0: stopwatch
        self.elapsed_before = 0.0    # time accumulated before the current run (pauses)
        self.run_started = 0.0
        size = max(18, int(frame_h * 0.055))
        self.font = _font(size)
        self.small = _font(int(size * 0.55))
        self.pad = int(size * 0.45)
        self._cache_key = None
        self._cache = None

    # ------------------------------------------------------------ control
    def elapsed(self, now: float) -> float:
        running = now - self.run_started if self.state == "running" else 0.0
        return self.elapsed_before + running

    def remaining(self, now: float) -> float:
        return self.countdown_s - self.elapsed(now)

    def start_countdown(self, now: float, minutes: float) -> str:
        self.countdown_s = minutes * 60
        self.elapsed_before, self.run_started, self.state = 0.0, now, "running"
        return f"countdown {fmt(self.countdown_s)} started"

    def toggle(self, now: float) -> str:
        """No timer -> stopwatch. Running -> pause. Paused -> resume."""
        if self.state == "hidden":
            self.countdown_s = 0.0
            self.elapsed_before, self.run_started, self.state = 0.0, now, "running"
            return "stopwatch started"
        if self.state == "running":
            self.elapsed_before, self.state = self.elapsed(now), "paused"
            return f"paused at {fmt(self._shown(now))}"
        self.run_started, self.state = now, "running"
        return "resumed"

    def adjust(self, now: float, minutes: float) -> str:
        if self.state == "hidden":
            return "no timer running (press 1-9 to start a countdown)"
        if not self.countdown_s:
            # turning a stopwatch into a countdown would be confusing; adjust the count instead
            self.elapsed_before = max(0.0, self.elapsed_before + minutes * 60)
        else:
            self.countdown_s = max(0.0, self.countdown_s + minutes * 60)
        return f"timer now {fmt(self._shown(now))}"

    def clear(self) -> str:
        self.state = "hidden"
        return "timer cleared"

    def _shown(self, now: float) -> float:
        if self.countdown_s:
            return max(0.0, self.remaining(now)) + 0.999  # countdown shows 5:00 .. 0:01
        return self.elapsed(now)

    # ------------------------------------------------------------ drawing
    def sprite(self, now: float) -> np.ndarray | None:
        if self.state == "hidden":
            return None
        done = bool(self.countdown_s) and self.remaining(now) <= 0
        if done and -self.remaining(now) > DONE_SHOW_S:
            self.state = "hidden"
            return None

        text = fmt(self._shown(now))
        dot = False
        if done:
            bg, text, sub = BG_DONE, "TIME'S UP", ""
        elif self.state == "paused":
            bg, sub = BG_PAUSED, "paused"
        elif self.countdown_s and self.remaining(now) <= 60:
            bg, sub, dot = BG_WARNING, self.label, True
        else:
            bg, sub, dot = BG_RUNNING, self.label, True
        blink = done and int(now * 2) % 2 == 0
        dot_on = dot and int(now * 2) % 2 == 0  # blinking "live" dot while running

        key = (text, sub, bg, dot, dot_on, blink)
        if key == self._cache_key:
            return self._cache
        self._cache_key, self._cache = key, self._render(text, sub, bg, dot, dot_on, blink)
        return self._cache

    def _render(self, text, sub, bg, dot, dot_on, blink) -> np.ndarray:
        from PIL import Image, ImageDraw
        p = self.pad
        tw = int(self.font.getlength(text))
        th = self.font.size
        dot_r = int(th * 0.22) if dot else 0
        dot_w = (2 * dot_r + p) if dot else 0
        sub_h = int(self.small.size * 1.2) if sub else 0
        bw = p * 2 + dot_w + max(tw, int(self.small.getlength(sub)) if sub else 0)
        bh = p * 2 + th + sub_h
        img = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        fill = bg if not blink else tuple(int(c * 0.6) if i < 3 else c for i, c in enumerate(bg))
        d.rounded_rectangle((0, 0, bw - 1, bh - 1), radius=int(bh * 0.3), fill=fill)
        x = p
        if dot:
            cy = p + th // 2 + 2
            colour = DOT_RUNNING if dot_on else (255, 255, 255, 70)
            d.ellipse((x, cy - dot_r, x + 2 * dot_r, cy + dot_r), fill=colour)
            x += dot_w
        d.text((x, p - int(th * 0.08)), text, font=self.font, fill=(255, 255, 255, 255))
        if sub:
            d.text((x, p + th + 2), sub, font=self.small, fill=(255, 255, 255, 200))
        return np.array(img)[:, :, [2, 1, 0, 3]].copy()  # RGBA -> BGRA

    def position(self, sprite: np.ndarray) -> tuple[int, int]:
        """Centre point: top-right corner with a margin."""
        sh, sw = sprite.shape[:2]
        m = int(self.h * 0.04)
        return self.w - m - sw // 2, m + sh // 2
