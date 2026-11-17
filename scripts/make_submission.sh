#!/usr/bin/env bash
# Regenerate the submission CSV from the match data + the committed weights (no training).
#   bash scripts/make_submission.sh [models_dir=models/st1] [name=submission] [--rebuild]
# Same as: python -m possession.pipeline --models <models_dir> --name <name>
# Needs the challenge data bundle (configs/paths.yaml or MTAH_ROOT). Writes outputs/<name>.csv.
set -e
cd "$(dirname "$0")/.."
MODELS=${1:-models/st1}; NAME=${2:-submission}; shift $(( $# < 2 ? $# : 2 ))
PY=${PY:-.venv/Scripts/python}; [ -x "$PY" ] || PY=python
$PY -u -m possession.pipeline --models "$MODELS" --name "$NAME" "$@"
