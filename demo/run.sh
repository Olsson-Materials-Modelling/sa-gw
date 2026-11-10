#!/usr/bin/env bash
# Example data -> fingerprint -> analysis -> figures (demo/out/).  bash demo/run.sh [remake] [figures]: remake regenerates the example data, figures refreshes demo/figures/
set -euo pipefail; cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
remake=""; figures=""; for arg in "$@"; do case "$arg" in remake) remake=1 ;; figures) figures=figures ;; *) echo "unknown argument: $arg" >&2; exit 2 ;; esac; done
if [ -n "$remake" ]; then python3 example/make_example.py; fi
python3 demo/fingerprint.py && python3 demo/analyse.py && python3 demo/plot.py $figures && python3 demo/schematic.py $figures
