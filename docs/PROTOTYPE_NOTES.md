# Prototype notes: how captions started (inside Confetti Cam)

> **Start here instead:** [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) is the self-contained
> build guide (spec, APIs, pitfalls, complete source). `mumble.py` (repo root) is the captions-only app; the notes below
> describe the original multi-feature prototype (`prototypes/confetti_cam.py`).

Status: **working prototype**, split out of the Vigil repo on 2026-11-23 so Vigil could ship on its own.
Everything needed to run it is in this folder.

## What it does

Speak, and subtitles appear at the bottom of **your own camera video** in Zoom (or Meet, Teams…),
so everyone in the call can read them. It's not a Zoom feature or plugin: the app is a virtual camera.

```
Mac mic ──► Apple Speech (SFSpeechRecognizer) ──► caption text ──► subtitle bar drawn on frames
webcam  ──────────────────────────────────────────────────────────► OBS Virtual Camera ──► Zoom
```

## Run it

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv -r requirements.txt
open -a Terminal run.command        # must run from Terminal.app: macOS gives the permissions to it
```
In Zoom: Settings › Video › Camera › **OBS Virtual Camera** (OBS must be installed once:
`brew install --cask obs`, start and stop its virtual camera once to approve the extension; OBS itself
doesn't need to run). Click the preview window and press **`t`** to turn captions on/off. `--captions` starts with them on.
`--no-vigil` silences the (harmless) "Vigil bridge unavailable" line.

## Files

| File | What |
|---|---|
| `captions.py` | **The feature.** `CaptionEngine` (mic → Apple Speech → current text, thread-safe) and `CaptionRenderer` (text → cached BGRA subtitle sprite). Self-contained, ~300 lines. |
| `vocab.txt` | Words/phrases to bias recognition toward. Live-reloaded: edits apply to the next caption. |
| `confetti_cam.py` | The host app: webcam → overlays → virtual camera. Captions are wired in at the `captions`/`renderer`/`draw_captions` bits (search `caption`). It also has 👍 confetti, timer, gaze. |
| `widgets.py`, `gaze.py` | Timer and gaze prototype, only imported by the host. Can be deleted with their call sites. |
| `models/` | MediaPipe models the host loads (gesture, face landmarks). Not needed by captions. |

## How the engine works (and why)

- **On-device when possible.** `supportsOnDeviceRecognition()` → audio never leaves the Mac. To get the
  model: System Settings › Keyboard › Dictation › On. Otherwise Apple's server recognition is used.
- **Server mode has a ~1 min cap per request**, so the watchdog recycles the request every 50 s
  (`SERVER_MAX`).
- **Utterance splitting:** after 2.5 s without new words (`SILENCE_SPLIT`, `--caption-pause`), the
  current request ends and a new one starts, so captions don't grow forever. The finished caption stays
  on screen for 3 s (`SHOW_AFTER`). A generation counter (`_gen`) drops late callbacks from old tasks.
- **"Smarter" captions (option A, done):** `TaskHintDictation` (sentence-style speech, not commands),
  `contextualStrings` from `vocab.txt` (max ~100 phrases, which fixed "I contact" → "eye contact"),
  `addsPunctuation`.
- Results are delivered on a background `NSOperationQueue`, so no main run loop is needed.
- Rendering: PIL draws the last 2 wrapped lines on a rounded translucent bar. The sprite is cached
  until the text changes, so it costs nearly nothing per frame. It's drawn **un-mirrored** on the
  output so viewers can read it.

## Gotchas we hit

- **Run from Terminal.app**, not from an IDE or agent helper: macOS ties Camera, Microphone and Speech
  permissions to the launching app. `run.command` does this.
- **Camera order:** OpenCV indexes AVFoundation devices sorted by `uniqueID`, not by the order
  AVFoundation lists them. `list_cameras()` handles it. Getting this wrong looked like "captions break the camera".
- If Zoom shows black: OBS itself is running and holding the virtual camera. Quit OBS.
- Speech setup can take a few seconds the first time (permission prompts). It's done before the camera opens.

## Known limitations

- Captions show **your** speech only (the Mac's mic), not the other participants'.
- Accuracy is Apple dictation quality: fine for conversation, weak on jargon not in `vocab.txt`.
- Keyboard controls only work while the preview window has focus.
- Server mode sends audio to Apple. Say so if you sell it.

## Ideas / next steps

1. **Strip the host down**: a `caption_cam.py` with only webcam → captions → virtual cam (drop
   confetti/timer/gaze/gestures). About a 150-line file.
2. **Global hotkey** (e.g. ⌃⌥C) so captions toggle without focusing the preview (pyobjc `NSEvent`
   global monitor, needs Accessibility permission).
3. **Option C, AI captions:** stream to Whisper (faster-whisper `base.en`, int8 runs fine on CPU, already
   proven in Vigil) for better jargon and punctuation, or an LLM pass to fix homophones in context.
4. **Translated captions** (speak English, show Hindi/Spanish): Apple has no streaming translation. Use
   Whisper translate or an LLM.
5. **Transcript export** at the end of a call (the engine already has every final utterance).
6. Performance: the host runs gesture recognition every frame (~0.8 of a CPU core). With captions only
   that drops to almost nothing. If gestures stay, run them at ~8 fps.
