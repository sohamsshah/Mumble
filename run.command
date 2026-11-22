#!/bin/zsh
# Double-click in Finder (or: open -a Terminal run.command) so macOS grants Camera/Microphone/Speech
# permissions to Terminal. Extra arguments are passed through, e.g. ./run.command --captions
cd "$(dirname "$0")"
.venv/bin/python -u mumble.py "$@"
echo; echo "Mumble stopped. Press Enter to close."; read
