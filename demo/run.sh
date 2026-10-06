#!/usr/bin/env bash
# Example data -> fingerprint -> analysis -> figures (demo/out/).  bash demo/run.sh [remake]
set -euo pipefail; cd "$(dirname "$0")/.."
if [ "${1:-}" = remake ]; then python3 example/make_example.py; fi
python3 demo/fingerprint.py && python3 demo/analyse.py && python3 demo/plot.py && python3 demo/schematic.py
