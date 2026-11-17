"""Small grid for decoder v0 (proxy symbols). Selection by the mean of the two public folds."""
import itertools
import json
import sys

import numpy as np

from possession import decode as D
from possession import decoder as dc
from possession import proxy

q = D.q_from_proxy(proxy.load()["sym"])
inp = D.Inputs(q, np.full(q.shape[1], 1 / q.shape[1]))
grid = dict(k=[1, 6], beta_vis=[0.15, 0.3, 0.6], beta_hid=[0.5], lam_vis=[0.0, 0.5], mean_ds_s=[8.0, 15.0])
rows = []
for vals in itertools.product(*grid.values()):
    p = dict(dc.DEFAULT, **dict(zip(grid, vals)))
    labels, _, _ = D.decode(inp, p, em_iters=2)
    sc = D.score_folds(labels, inp.gt)
    st = D.match_stats(labels)
    rows.append((sc["mean_fold_spell"], sc["full"]["spell_f1"], sc["full"]["macro_f1"], st["share"]["D"],
                 st["possessions_per_min"], dict(zip(grid, vals))))
    print(json.dumps(rows[-1]), flush=True)
rows.sort(key=lambda r: -r[0])
print("TOP")
for r in rows[:8]:
    print(json.dumps(r))
