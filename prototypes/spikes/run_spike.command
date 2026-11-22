#!/bin/zsh
cd "$(dirname "$0")/.."
.venv/bin/python -u spikes/sck_capture_spike.py 6 > spikes/out/spike.log 2>&1
echo "exit=$?" >> spikes/out/spike.log
