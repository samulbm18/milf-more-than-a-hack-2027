"""Inference with committed weights: out-of-fold ensemble of PossNet models (no training needed).

    python -m possession.infer --models models/st1 --out outputs/st1_infer_oof_ensemble.npz

Each model directory holds best.pt + config.json (as written by train.py). Every graded step is
predicted only by the models whose training region did NOT contain it (the models of its own
validation fold), so no frame is labelled by a network that was fitted on its own pseudo-labels.
Steps outside the graded halves get the average of all models. Output has the oof.npz layout
(logq, border) that decode.py --source reads.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from . import paths
from . import train as T


def model_dirs(root):
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(root, "*", "best.pt")))


def run(models_root, out):
    dirs = model_dirs(models_root)
    if not dirs:
        raise SystemExit("no models (*/best.pt) under %s" % models_root)
    if T.DEV == "cpu":
        torch.use_deterministic_algorithms(True, warn_only=True)
        if os.environ.get("POSS_THREADS"):
            torch.set_num_threads(int(os.environ["POSS_THREADS"]))
    print("device %s, %s" % (T.DEV, str(T.DTYPE).replace("torch.", "")), flush=True)
    M = T.Match()
    g = M.S["graded"]
    nq = nb = den = None
    for d in dirs:
        meta = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
        cfg, args = meta["cfg"], meta["args"]
        L = cfg["data"]["window_steps"]
        regions, _ = T.fold_regions(M, cfg["data"]["folds"], cfg["data"]["buffer_s"] * 5,
                                    cfg["data"]["exclude_public_from_train"] and not args.get("all"))
        val = regions[args["fold"]][1]
        model = T.build_model(cfg)
        model.load_state_dict(torch.load(os.path.join(d, "best.pt"), map_location=T.DEV))
        M.set_mask(np.zeros_like(M.real_vis25))  # the real visibility, as the decoder sees it
        logq, border = T.predict(model, M, cfg, L)
        q, b = np.exp(logq.cpu().numpy()), border.cpu().numpy()
        w = np.where(g, val, True).astype(float)
        if nq is None:
            nq, nb, den = np.zeros_like(q), np.zeros(len(w)), np.zeros(len(w))
        nq += w[:, None] * q
        nb += w * b
        den += w
        print("model %s (fold %d)" % (os.path.basename(d), args["fold"]), flush=True)
    if (den == 0).any():
        raise SystemExit("%d graded steps have no out-of-fold model: every fold needs >= 1 model" % (den == 0).sum())
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, logq=np.log(nq / den[:, None]), border=nb / den)
    print("wrote", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(paths.REPO, "models", "st1"))
    ap.add_argument("--out", default=os.path.join(paths.OUTPUTS, "infer_oof_ensemble.npz"))
    a = ap.parse_args()
    run(a.models, a.out)


if __name__ == "__main__":
    main()
