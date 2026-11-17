"""Self-training targets: the machine decodes the match with a trained network ensemble and its
confident decisions (posterior >= 0.8) become the training labels of the next network.

    python scripts/make_selftrain_labels.py [<runs prefix, default possnet3> [<out steps.npz>]]

Reads runs/*_<prefix>_f*_s*/oof.npz and the caches; writes a steps file for `possession.train --steps`
(default cache/selftrain_steps.npz). Used by scripts/train_from_scratch.sh.
"""
import glob
import os
import sys
import time

import numpy as np

from possession import decode as D
from possession import decoder as dc
from possession import frames as fr
from possession import paths, proxy
from possession import sensors as SN

CONF = 0.8       # only confident machine decisions are taught
EM_ITERS = 3


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def oof_ensemble(prefix):
    """Out-of-fold ensemble of every fold/seed run of `prefix`: each graded step is averaged only over
    the models that did not train on it."""
    out = os.path.join(paths.OUTPUTS, "%s_oof_ensemble.npz" % prefix)
    if os.path.exists(out):
        return out
    runs = sorted(glob.glob(os.path.join(paths.RUNS, "*_%s_f*_s*" % prefix, "oof.npz")))
    if not runs:
        sys.exit("no runs found for prefix %r in %s" % (prefix, paths.RUNS))
    g = np.load(paths.cache("t0_steps.npz"))["graded"]
    nq = nb = den = None
    for r in runs:
        z = np.load(r)
        w = np.where(g, z["val"], True).astype(float)
        q = np.exp(z["logq"])
        if nq is None:
            nq, nb, den = np.zeros_like(q), np.zeros(len(w)), np.zeros(len(w))
        nq += w[:, None] * q
        nb += w * z["border"]
        den += w
    os.makedirs(paths.OUTPUTS, exist_ok=True)
    np.savez(out, logq=np.log(nq / den[:, None]), border=nb / den)
    return out


def machine_inputs(src):
    sym = np.load(paths.cache("t0_steps.npz"))["sym"]
    prior = np.bincount(sym[sym >= 0], minlength=2) / (sym >= 0).sum()
    inp = D.Inputs(D.q_from_oof(src), prior, None)
    inp.border = D.border_from_oof(src)
    inp.sensors = SN.build(inp, fr.load(), alpha=500.0, use=SN.MACHINE_USE, cap=4.0)
    return inp


def make_steps(sym, weight, holder, out_path):
    """Steps cache with the new 25 Hz labels (sym / weight / holder); everything else from the base."""
    S = dict(np.load(paths.cache("t0_steps.npz")))
    steps = S["step_frame"]
    h = holder[steps]
    hit = (S["p_slot"] == h[:, None]) & (h[:, None] >= 0)
    has = hit.any(1)
    ht = np.full(len(steps), -1, np.int16)
    ht[has] = hit[has].argmax(1)
    S.update(sym=sym[steps], w=weight[steps], holder_tok=ht)
    np.savez(out_path, **S)
    return out_path


if __name__ == "__main__":
    src_prefix = sys.argv[1] if len(sys.argv) > 1 else "possnet3"
    dst = sys.argv[2] if len(sys.argv) > 2 else paths.cache("selftrain_steps.npz")
    inp = machine_inputs(oof_ensemble(src_prefix))
    lab, post, _ = D.decode(inp, dict(dc.DEFAULT, **D.MACHINE_P), em_iters=EM_ITERS)
    conf = post.max(1)
    sym = np.full(len(lab), -1, np.int8)
    sym[lab == "H"], sym[lab == "A"] = 0, 1
    keep = (sym >= 0) & (conf >= CONF)
    sym[~keep] = -1
    weight = np.where(keep, conf, 0).astype(np.float32)
    P = proxy.load()
    holder = np.where(P["sym"] == sym, P["holder"], -1).astype(np.int16)  # pointer target only where both agree
    make_steps(sym, weight, holder, dst)
    log("pseudo-labels: share labelled %.3f -> %s" % ((sym >= 0).mean(), dst))
