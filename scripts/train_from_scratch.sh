#!/usr/bin/env bash
# Retrain the submitted networks from the data alone (~3 h on one 4 GB GPU). GPU training is not
# bit-deterministic: the result is statistically equivalent, not identical - the committed
# weights in models/st1 are what make the CSV reproducible.
#   1. rule labels (last touch)  -> PossNet v3, 4 folds x 3 seeds          runs/*_possnet3_*
#   2. the machine decodes the match with that ensemble; its confident decisions (posterior >= 0.8)
#      become the new targets                                             cache/selftrain_steps.npz
#   3. self-trained PossNet st1, 4 folds x 3 seeds                         runs/*_st1_*
#   4. copy best.pt + config.json to models/st1_retrained and build the CSV with it
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}; [ -x "$PY" ] || PY=python
for m in frames events proxy features; do $PY -u -m possession.$m; done
bash scripts/run_ensemble.sh possnet3
$PY -u scripts/make_selftrain_labels.py possnet3 cache/selftrain_steps.npz
for s in 0 1 2; do SEED=$s bash scripts/run_variant.sh st1 cache/selftrain_steps.npz; done
for r in runs/*_st1_f?_s?; do
  n=$(basename "$r" | sed 's/^[0-9]*_[0-9]*_//'); mkdir -p models/st1_retrained/$n
  cp "$r/best.pt" "$r/config.json" models/st1_retrained/$n/
done
$PY -u -m possession.pipeline --models models/st1_retrained --name retrained
