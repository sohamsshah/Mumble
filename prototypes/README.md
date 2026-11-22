> **Prototypes (not maintained).** These are the experiments Mumble grew out of. Run from this folder after
> `uv pip install --python ../.venv -r requirements.txt` (adds MediaPipe): `./download_models.sh`, then `./run.command` (or `../.venv/bin/python confetti_cam.py`).
> `party_cam.py` / `cat_companion.py` need the face model too. `spikes/` are early ScreenCaptureKit capture tests.
> Paths below that say `models/` or `logos/` are relative to this `prototypes/` folder.

# Confetti Cam

Show a 👍 to your webcam and your own logos burst across your video as confetti.
Everyone in the Zoom call sees it, because the app works as a camera.

```
webcam → MediaPipe gesture recognizer → logo confetti drawn on frames → OBS Virtual Camera → Zoom
```

## One-time setup

1. **Install OBS** (only its virtual camera driver is used):
   ```bash
   brew install --cask obs
   ```
   Open OBS once, click **Start Virtual Camera**, approve the system extension
   (System Settings › General › Login Items & Extensions › Camera Extensions, or the prompt),
   then **Stop Virtual Camera** and quit OBS. A restart may be needed.
   OBS does not need to run after this.

2. **Python environment** (Python 3.12; mediapipe has no 3.14 wheels. OpenCV is pinned
   below 5 because 5.x has no prebuilt wheel for macOS 14 and fails to compile):
   ```bash
   uv venv --python 3.12 .venv && uv pip install --python .venv -r requirements.txt
   ```
   The model `models/gesture_recognizer.task` is already downloaded. If it's missing:
   ```bash
   curl -L -o models/gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
   ```

3. **Add your logos**: drop PNGs (transparent backgrounds look best) into `logos/`.
   With no logos, the app uses coloured placeholders.

## Run

```bash
.venv/bin/python confetti_cam.py
```

- macOS will ask for camera permission for your terminal app the first time.
- In Zoom: **Settings › Video › Camera › OBS Virtual Camera**.
- Hold 👍 for ~0.3 s → confetti. 3 s cooldown between bursts.
- Preview window keys: `c` fires confetti manually, `q` quits.
  The preview's green text is only in the preview, not sent to Zoom.

Useful options:

```bash
.venv/bin/python confetti_cam.py --gesture Victory --count 100 --logo-size 120
.venv/bin/python confetti_cam.py --no-vcam     # preview only, no OBS needed
```

Gestures available: `Thumb_Up`, `Thumb_Down`, `Victory`, `Open_Palm`, `Pointing_Up`, `Closed_Fist`, `ILoveYou`.

## Live captions

Press `t` in the preview window to turn live captions on or off. What you say appears as
subtitles at the bottom of your video, so everyone in the call sees them.

- Uses Apple's Speech framework (`captions.py`). Terminal needs **Microphone** and
  **Speech Recognition** permission; macOS asks on first run.
- **On-device vs server:** if the Mac has the on-device speech model, audio never leaves the Mac.
  Otherwise Apple's server recognition is used (audio goes to Apple; the app restarts each request
  before Apple's ~1 minute cap). To get the on-device model: System Settings › Keyboard ›
  Dictation › On, then restart the app. The log says which mode is active.
- **Smarter captions:** recognition is told it's sentence-style speech, and is biased toward the
  words in `vocab.txt` (fixes things like "I contact" → "eye contact", "media pipe" → "MediaPipe").
  Add your product names, people and jargon there; edits apply to the next caption, no restart.
  `--caption-pause 2.5` sets how long a pause starts a new caption (longer = more context).
- Other languages: `--locale en-IN`, `--locale hi-IN`, etc. Start with captions on: `--captions`.
  Disable entirely: `--no-captions`.

## Timer widget

A countdown or stopwatch in the top-right of your video, so everyone in the call sees it.
Keyboard only: click the preview window first, then:

| Key | Action |
|---|---|
| `1`–`9` | Start a countdown of 1–9 minutes (`0` = 10 minutes) |
| `w` | No timer: start a stopwatch. Running: pause. Paused: resume |
| `+` / `-` | One minute more / less |
| `x` | Clear the timer |

- The countdown turns amber in the last minute and flashes red "TIME'S UP" at zero (hides after 15 s).
- Paused timers turn blue and stay until you resume or clear them.
- `--timer-label "Q&A"` adds small text under the time.

## Gaze / attention tracking (preview only)

Press `g` to start. Look at your screen normally for 2 s while it calibrates your "centre".
Then the preview shows where you're looking (ON SCREEN / LEFT / RIGHT / UP / DOWN / NO FACE,
from your own point of view), a direction dot, and live stats: % time on screen, number of
look-aways (over 1 s) and the longest one. Look-aways are also printed in the Terminal.
Press `g` again to stop and print a summary. Nothing here is sent to Zoom.

Tuning: the grey numbers in the HUD are the raw signals. Thresholds are at the top of `gaze.py`.

## All controls

| Input | Action |
|---|---|
| 👍 gesture or `c` | Logo confetti |
| `t` | Live captions on/off |
| `1`–`9`, `0`, `w`, `+`, `-`, `x` | Timer (see above) |
| `q` | Quit |

Keys only work while the preview window is focused (not while you're clicking in Zoom).

## Notes

- Zoom mirrors your self-view only; others see the video the right way round.
- If Zoom shows a black/"OBS" placeholder image, the app isn't running, or it's in `--no-vcam` mode.

## Party Cam: confetti + a cat beside you

`party_cam.py` adds a cartoon cat that sits beside you and reacts to your face live.
It also needs `models/face_landmarker.task`:
```bash
curl -L -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

```bash
.venv/bin/python party_cam.py          # ✌️ toggles the cat, 👍 fires confetti
.venv/bin/python party_cam.py --cat    # start with the cat already on
```

What the cat does:
- copies your head tilt, turn and nods; its eyes follow your face
- blinks when you blink; open your mouth → it opens its mouth + "meow!"
- smile → hearts float up
- leave the frame for 4 s → it falls asleep (Zzz); come back → it wakes up startled (!)

Preview keys: `k` cat on/off, `c` confetti, `q` quit. `--cat-side left|right` fixes its side
(default: the side away from your face, chosen when the cat appears).
