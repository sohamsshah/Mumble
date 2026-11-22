#!/bin/zsh
# Original prototype: 👍 logo confetti + captions (t) + timer (1-9, w, +, -, x) + gaze HUD (g).
# Needs the MediaPipe models: ./download_models.sh
cd "$(dirname "$0")"
../.venv/bin/python -u confetti_cam.py "$@"
echo; echo "Confetti Cam stopped. Press Enter to close."; read
