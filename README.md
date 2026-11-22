# Mumble

**Live captions on your own camera, for any video call.** Speak, and subtitles appear at the bottom of your
webcam video, so everyone in Zoom, Google Meet or Teams can read what you say. No plugin, no bot, no meeting
integration: Mumble is a virtual camera.

```
mic ───► Apple Speech (on-device when available) ───► caption text
webcam ──► frame ──► caption bar drawn on the frame ◄──┘ ──► virtual camera ──► Zoom / Meet / Teams
```

- **Private by default:** uses Apple's on-device speech model when the Mac has it, so audio never leaves the Mac.
- **Smarter than plain dictation:** sentence-style recognition with punctuation, biased toward your own
  vocabulary (`vocab.txt`, live-reloaded: "eye contact", not "I contact").
- **Readable:** 2-line subtitle bar that updates word by word; a new caption after 2.5 s of silence.

## Requirements
- macOS 13+ (Apple Silicon or Intel), Python 3.12
- [OBS Studio](https://obsproject.com), only for its virtual camera driver (OBS itself doesn't need to run)

## Setup (one time)
1. **Virtual camera:** `brew install --cask obs`. Open OBS, click **Start Virtual Camera**, approve the system
   extension, then **Stop Virtual Camera** and quit OBS.
2. **Python environment:**
   ```bash
   uv venv --python 3.12 .venv && uv pip install --python .venv -r requirements.txt
   ```
3. **Permissions:** run Mumble from **Terminal.app**. macOS asks for Camera, Microphone and Speech Recognition
   for Terminal on the first run. For the best privacy and accuracy, turn on System Settings › Keyboard ›
   **Dictation**, which downloads the on-device speech model.

## Use
```bash
./run.command               # or: .venv/bin/python mumble.py
./run.command --captions    # start with captions on
```
In Zoom: Settings › Video › Camera › **OBS Virtual Camera**. Click the Mumble preview window, then:

| Key | Action |
|---|---|
| `t` | captions on / off |
| `q` / `Esc` | quit |

| Option | |
|---|---|
| `--locale en-IN` | recognition language (`en-US`, `en-GB`, `hi-IN`, …) |
| `--vocab my_words.txt` | phrases to bias toward (default `vocab.txt`; edits apply to the next caption) |
| `--caption-pause 2.5` | seconds of silence that start a new caption |
| `--camera-name FaceTime` | choose the webcam |
| `--no-vcam` | preview only (no OBS needed) |

The preview is mirrored like a selfie; people in the call see normal, readable text.

## Troubleshooting
| Symptom | Fix |
|---|---|
| Zoom shows black or the OBS logo | Quit OBS (it holds the virtual camera) and keep Mumble running |
| "Camera/Microphone not authorized" | Run from Terminal.app; enable Terminal in System Settings › Privacy & Security, then reopen Terminal |
| Captions stop or never start | The log says whether it's on-device or server mode; check Speech Recognition permission |
| Wrong camera opens | `--camera-name FaceTime` (Mumble never auto-picks a virtual camera) |

## Repository
```
mumble.py                  the app: webcam → captions → virtual camera
captions.py                CaptionEngine (Apple Speech) + CaptionRenderer (subtitle bar)
vocab.txt                  your words and phrases
docs/IMPLEMENTATION_GUIDE.md   full spec, design, pitfalls and source; enough to rebuild Mumble from scratch
docs/PROTOTYPE_NOTES.md    how it started
prototypes/                earlier experiments: 👍 logo confetti, timer, gaze HUD, cat/party cams (unmaintained)
```

## Roadmap ideas
Global hotkey, Whisper/LLM-corrected captions, live translation, transcript export, Windows/Linux support.
See §9 of the [implementation guide](docs/IMPLEMENTATION_GUIDE.md).

## License
[MIT](LICENSE)
