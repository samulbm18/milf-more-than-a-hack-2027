#!/usr/bin/env bash
# Train the 4 folds of one seed (env SEED, default 0) on a label variant. Usage: [SEED=n] run_variant.sh <prefix> <steps.npz|-> [train args...]
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}; [ -x "$PY" ] || PY=python
PREFIX=$1; STEPS=$2; shift 2; SEED=${SEED:-0}
EXTRA=""; [ "$STEPS" != "-" ] && EXTRA="--steps $STEPS"
for fold in 0 1 2 3; do
  name=${PREFIX}_f${fold}_s${SEED}
  if ls runs/*_${name}/oof.npz >/dev/null 2>&1; then echo "skip $name"; continue; fi
  $PY -u -m possession.train --fold $fold --seed $SEED --name $name $EXTRA "$@" > runs/${name}.log 2>&1 \
    && echo "done $name $(tail -1 runs/${name}.log)" || echo "FAILED $name"
done
echo VARIANT_FINISHED $PREFIX
