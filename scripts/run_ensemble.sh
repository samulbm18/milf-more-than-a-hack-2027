#!/usr/bin/env bash
# PossNet ensemble: folds x seeds, sequential on one GPU. Weights: runs/<stamp>_<name>/{best,last}.pt
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}; [ -x "$PY" ] || PY=python
PREFIX=${1:-possnet2}
for seed in 0 1 2; do
  for fold in 0 1 2 3; do
    name=${PREFIX}_f${fold}_s${seed}
    if ls -d runs/*_${name} >/dev/null 2>&1 && ls runs/*_${name}/oof.npz >/dev/null 2>&1; then
      echo "skip $name"; continue
    fi
    $PY -u -m possession.train --fold $fold --seed $seed --name $name > runs/${name}.log 2>&1 \
      && echo "done $name $(tail -1 runs/${name}.log)" || echo "FAILED $name"
  done
done
echo ENSEMBLE_FINISHED
