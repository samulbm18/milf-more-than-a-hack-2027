"""Decoder grid with PossNet out-of-fold symbols. Usage: grid_net.py <oof.npz> [em_iters]"""
import itertools
import json
import os
import sys

import numpy as np

from possession import decode as D
from possession import decoder as dc
from possession import paths

src = sys.argv[1]
em = int(sys.argv[2]) if len(sys.argv) > 2 else 0
q = D.q_from_oof(src)
S = np.load(os.path.join(paths.CACHE, "t0_steps.npz"))
sym = S["sym"]
prior = np.bincount(sym[sym >= 0], minlength=q.shape[1]) / (sym >= 0).sum()
inp = D.Inputs(q, prior)
if q.shape[1] == 2:
    grid = dict(k=[1, 3], beta_vis=[0.3, 1.0], beta_hid=[0.3, 1.0], mean_pos_s=[3.0, 10.0],
                eta_in=[0.01, 0.1], lam_vis=[0.0])
else:
    grid = dict(k=[1, 6], beta_vis=[0.1, 0.3], beta_hid=[0.1, 0.3, 0.6], lam_vis=[0.0, 0.5], eta_out=[0.05, 1.0])
rows = []
for vals in itertools.product(*grid.values()):
    p = dict(dc.DEFAULT, **dict(zip(grid, vals)))
    labels, _, _ = D.decode(inp, p, em_iters=em)
    sc = D.score_folds(labels, inp.gt)
    st = D.match_stats(labels)
    rows.append((sc["mean_fold_spell"], sc["full"]["spell_f1"], sc["full"]["macro_f1"], sc["full"]["n_pred"],
                 st["share"]["D"], st["possessions_per_min"], dict(zip(grid, vals))))
    print(json.dumps(rows[-1]), flush=True)
rows.sort(key=lambda r: -r[0])
print("TOP")
for r in rows[:10]:
    print(json.dumps(r))
