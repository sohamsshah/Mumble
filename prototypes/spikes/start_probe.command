#!/bin/zsh
cd "$(dirname "$0")/.."
.venv/bin/python -u spikes/start_probe.py > spikes/out/probe.log 2>&1
echo "exit=$?" >> spikes/out/probe.log
