#!/usr/bin/env bash
# MediaPipe models used by the prototypes (Mumble itself needs none). ~20 MB, Apache 2.0, from Google.
set -euo pipefail
cd "$(dirname "$0")" && mkdir -p models
base=https://storage.googleapis.com/mediapipe-models
get() { [ -f "models/$1" ] && echo "✓ $1" || { echo "↓ $1"; curl -fsSL -o "models/$1" "$2"; }; }
get gesture_recognizer.task "$base/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
get face_landmarker.task    "$base/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
get hand_landmarker.task    "$base/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
