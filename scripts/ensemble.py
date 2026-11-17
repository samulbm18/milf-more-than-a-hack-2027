"""Out-of-fold PossNet ensemble. Usage: ensemble.py <prefix> [<prefix> ...]

Every graded step lies in the validation block of exactly one fold; it gets the average (in
probability) of the models of that fold only, i.e. models that never trained on it. Ungraded
steps get every model. The public window is excluded from all training, so this ensemble behaves
the same there as on the rest of the match.
Writes outputs/<prefixes>_oof_ensemble.npz with logq [T5, n_sym] and border [T5].
"""
import glob
import os
import sys

import numpy as np

from possession import paths

prefixes = sys.argv[1:]
runs = sorted(r for p in prefixes for r in glob.glob(os.path.join(paths.RUNS, "*_%s_f*_s*" % p, "oof.npz")))
S = np.load(os.path.join(paths.CACHE, "t0_steps.npz"))
g = S["graded"]
num_q = num_b = den = None
for r in runs:
    z = np.load(r)
    w = np.where(g, z["val"], True).astype(float)
    q = np.exp(z["logq"])
    if num_q is None:
        num_q, num_b, den = np.zeros_like(q), np.zeros(len(w)), np.zeros(len(w))
    num_q += w[:, None] * q
    num_b += w * z["border"]
    den += w
q = num_q / den[:, None]
out = os.path.join(paths.OUTPUTS, "%s_oof_ensemble.npz" % "+".join(prefixes))
np.savez(out, logq=np.log(q), border=num_b / den)
print("%d models | members per graded step: min %d max %d | %s" % (len(runs), den[g].min(), den[g].max(), out))
